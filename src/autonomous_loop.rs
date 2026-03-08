//! Autonomous loop — core agent runtime for autonomous execution
//!
//! Implements the LLM -> parse -> validate -> execute -> feed results -> loop
//! cycle with sandbox validation, user interaction, and task board integration.

use crate::agent_policy::{ActionDescriptor, ActionType, AgentPolicy};
use crate::agent_sandbox::SandboxValidator;
use crate::agentic_loop::{LoopMessage, LoopRole};
use crate::mode_manager::OperationMode;
use crate::task_board::{BoardCommand, TaskBoard};
use crate::unified_tools::{ToolCall, ToolRegistry};
use crate::user_interaction::{InteractionManager, NotifyLevel, UserQuery, UserResponse};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// AgentState
// ============================================================================

/// Current state of the autonomous agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    /// Not running.
    Idle,
    /// Actively executing iterations.
    Running,
    /// Waiting for user to approve an action.
    WaitingApproval(String),
    /// Waiting for user to answer a question.
    WaitingForUser(String),
    /// Paused by external command.
    Paused,
    /// Finished successfully with final output.
    Completed(String),
    /// Terminated with an error.
    Failed(String),
}

// ============================================================================
// IterationOutcome
// ============================================================================

/// Outcome of a single iteration of the loop.
#[derive(Debug, Clone, PartialEq)]
pub enum IterationOutcome {
    /// Continue to the next iteration.
    Continue,
    /// Agent is done; final answer produced.
    Done(String),
    /// Agent needs to ask the user a question.
    AskUser(String),
    /// Agent needs approval for an action.
    NeedsApproval(String),
    /// An error occurred.
    Error(String),
}

// ============================================================================
// ParsedToolCall
// ============================================================================

/// A tool call parsed from the LLM response text.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: HashMap<String, String>,
}

// ============================================================================
// AgentResult
// ============================================================================

/// Final result of an autonomous agent run.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final output / answer.
    pub output: String,
    /// How many iterations the loop executed.
    pub iterations: usize,
    /// Names of tools that were called.
    pub tools_called: Vec<String>,
    /// Estimated cost in USD.
    pub cost: f64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

// ============================================================================
// CostConfig
// ============================================================================

/// Configuration for cost tracking during agent execution.
pub struct CostConfig {
    /// Default cost per tool call in USD.
    pub default_cost_per_call: f64,
    /// Per-tool cost overrides (tool_name -> cost_usd).
    pub tool_costs: HashMap<String, f64>,
    /// Optional callback to compute cost dynamically.
    pub cost_callback: Option<Arc<dyn Fn(&str, &HashMap<String, String>) -> f64 + Send + Sync>>,
}

impl CostConfig {
    pub fn new() -> Self {
        Self {
            default_cost_per_call: 0.001,
            tool_costs: HashMap::new(),
            cost_callback: None,
        }
    }

    /// Compute the cost for a tool call.
    pub fn cost_for(&self, tool_name: &str, arguments: &HashMap<String, String>) -> f64 {
        if let Some(ref cb) = self.cost_callback {
            return cb(tool_name, arguments);
        }
        self.tool_costs
            .get(tool_name)
            .copied()
            .unwrap_or(self.default_cost_per_call)
    }
}

impl Default for CostConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AutonomousAgentConfig
// ============================================================================

/// Configuration for an autonomous agent.
pub struct AutonomousAgentConfig {
    /// Name of the agent.
    pub name: String,
    /// Maximum number of loop iterations before forced stop.
    pub max_iterations: usize,
    /// System prompt injected at the start of the conversation.
    pub system_prompt: String,
    /// Cost tracking configuration.
    pub cost_config: CostConfig,
}

// ============================================================================
// AutonomousAgent
// ============================================================================

/// The core autonomous agent runtime.
///
/// Drives the loop: generate response -> parse tool calls -> validate in
/// sandbox -> execute via registry -> feed results back -> repeat.
/// Message type for inter-agent communication via mailbox.
#[derive(Debug, Clone)]
pub struct InterAgentMessage {
    /// Sender agent identifier.
    pub from: String,
    /// Message content.
    pub content: String,
    /// Timestamp (millis since UNIX epoch).
    pub timestamp: u64,
}

pub struct AutonomousAgent {
    config: AutonomousAgentConfig,
    policy: AgentPolicy,
    sandbox: Arc<RwLock<SandboxValidator>>,
    tool_registry: ToolRegistry,
    conversation: Vec<LoopMessage>,
    response_generator: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>,
    mode: OperationMode,
    state: AgentState,
    interaction: Option<Arc<InteractionManager>>,
    task_board: Option<Arc<RwLock<TaskBoard>>>,
    current_task_id: Option<String>,
    iteration: usize,
    total_cost: f64,
    start_time: u64,
    tools_called_log: Vec<String>,
    /// Cancellation token — if set to true, agent stops at next iteration.
    cancellation_token: Option<Arc<AtomicBool>>,
    /// Mailbox for receiving inter-agent messages (checked between iterations).
    mailbox: Option<std::sync::mpsc::Receiver<InterAgentMessage>>,
    /// Index of the planning hint message in conversation (for cleanup).
    planning_hint_idx: Option<usize>,
}

impl AutonomousAgent {
    /// Start building an agent with the given name and response generator.
    pub fn builder(
        name: impl Into<String>,
        response_generator: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>,
    ) -> AutonomousAgentBuilder {
        AutonomousAgentBuilder::new(name, response_generator)
    }

    /// Run the agent on a task. This is the main entry point.
    ///
    /// The loop proceeds as follows:
    /// 1. Set state to Running.
    /// 2. Inject system prompt and user task into conversation.
    /// 3. For each iteration:
    ///    a. Call the response generator with the conversation so far.
    ///    b. Parse tool calls from the response.
    ///    c. If no tool calls, treat the response as the final answer.
    ///    d. If a tool call is `ask_user`, use the InteractionManager.
    ///    e. For every other tool call: validate with sandbox, execute via
    ///       registry, push the result as a Tool message.
    ///    f. Update the task board if configured.
    ///    g. Check the iteration limit.
    /// 4. Return an `AgentResult`.
    pub fn run(&mut self, task: &str) -> Result<AgentResult, String> {
        // Cannot run if paused — caller must resume first
        if self.state == AgentState::Paused {
            return Err("Agent is paused".into());
        }
        self.state = AgentState::Running;
        self.iteration = 0;
        self.total_cost = 0.0;
        self.start_time = now_millis();
        self.tools_called_log.clear();

        // Inject system prompt
        if !self.config.system_prompt.is_empty() {
            self.conversation.push(LoopMessage {
                role: LoopRole::System,
                content: self.config.system_prompt.clone(),
                tool_calls: None,
                tool_results: None,
            });
        }

        // Inject user task
        self.conversation.push(LoopMessage {
            role: LoopRole::User,
            content: task.to_string(),
            tool_calls: None,
            tool_results: None,
        });

        // Notify interaction manager
        if let Some(ref im) = self.interaction {
            im.notify(
                &self.config.name,
                &format!("Starting task: {}", task),
                NotifyLevel::Info,
            );
        }

        // Main loop
        loop {
            if self.iteration >= self.config.max_iterations {
                self.state = AgentState::Failed("Max iterations reached".into());
                return Err("Max iterations reached".into());
            }

            // Check cancellation token
            if let Some(ref token) = self.cancellation_token {
                if token.load(Ordering::Relaxed) {
                    self.state = AgentState::Failed("Cancelled".into());
                    let elapsed = now_millis() - self.start_time;
                    return Ok(AgentResult {
                        output: "Agent cancelled".to_string(),
                        iterations: self.iteration,
                        tools_called: self.collect_tools_called(),
                        cost: self.total_cost,
                        duration_ms: elapsed,
                    });
                }
            }

            // Process mailbox messages (inject as system messages)
            if let Some(ref mailbox) = self.mailbox {
                while let Ok(msg) = mailbox.try_recv() {
                    self.conversation.push(LoopMessage {
                        role: LoopRole::System,
                        content: format!("[Message from {}]: {}", msg.from, msg.content),
                        tool_calls: None,
                        tool_results: None,
                    });
                }
            }

            match self.state {
                AgentState::Paused => {
                    return Err("Agent is paused".into());
                }
                AgentState::Completed(_) | AgentState::Failed(_) => break,
                _ => {}
            }

            let outcome = self.run_iteration();

            match outcome {
                IterationOutcome::Continue => continue,
                IterationOutcome::Done(answer) => {
                    self.state = AgentState::Completed(answer.clone());
                    let elapsed = now_millis() - self.start_time;
                    return Ok(AgentResult {
                        output: answer,
                        iterations: self.iteration,
                        tools_called: self.collect_tools_called(),
                        cost: self.total_cost,
                        duration_ms: elapsed,
                    });
                }
                IterationOutcome::AskUser(question) => {
                    if let Some(ref im) = self.interaction {
                        let resp = im.ask(&self.config.name, UserQuery::free_text(question));
                        match resp {
                            UserResponse::Text(text) => {
                                self.conversation.push(LoopMessage {
                                    role: LoopRole::User,
                                    content: text,
                                    tool_calls: None,
                                    tool_results: None,
                                });
                                self.state = AgentState::Running;
                            }
                            _ => {
                                self.state = AgentState::Failed("User cancelled".into());
                                return Err("User cancelled".into());
                            }
                        }
                    } else {
                        self.state =
                            AgentState::Failed("No interaction manager for ask_user".into());
                        return Err("No interaction manager for ask_user".into());
                    }
                }
                IterationOutcome::NeedsApproval(desc) => {
                    self.state = AgentState::WaitingApproval(desc.clone());
                    return Err(format!("Needs approval: {}", desc));
                }
                IterationOutcome::Error(e) => {
                    self.state = AgentState::Failed(e.clone());
                    return Err(e);
                }
            }
        }

        // Fallback if the loop ended without a clean return
        let elapsed = now_millis() - self.start_time;
        if let AgentState::Completed(ref answer) = self.state {
            Ok(AgentResult {
                output: answer.clone(),
                iterations: self.iteration,
                tools_called: self.collect_tools_called(),
                cost: self.total_cost,
                duration_ms: elapsed,
            })
        } else {
            Err("Agent ended without producing a result".into())
        }
    }

    /// Run a single iteration of the loop.
    pub fn run_iteration(&mut self) -> IterationOutcome {
        self.iteration += 1;

        // 1. Generate response
        let response = (self.response_generator)(&self.conversation);

        // 2. Add assistant message
        self.conversation.push(LoopMessage {
            role: LoopRole::Assistant,
            content: response.clone(),
            tool_calls: None,
            tool_results: None,
        });

        // 3. Parse tool calls
        let parsed = parse_tool_calls(&response);

        // 4. If no tool calls, treat response as final answer
        if parsed.is_empty() {
            return IterationOutcome::Done(response);
        }

        // 5. Process each tool call
        for tc in &parsed {
            // Special case: ask_user
            if tc.name == "ask_user" {
                let question = tc
                    .arguments
                    .get("question")
                    .cloned()
                    .unwrap_or_else(|| "What would you like?".into());
                return IterationOutcome::AskUser(question);
            }

            // Validate with sandbox
            let action = ActionDescriptor::new(ActionType::ToolCall, &tc.name);
            {
                let mut sandbox = match self.sandbox.write() {
                    Ok(s) => s,
                    Err(_) => {
                        return IterationOutcome::Error("Sandbox lock poisoned".into());
                    }
                };
                if let Err(e) = sandbox.validate(&action) {
                    // Push error as tool result
                    self.conversation.push(LoopMessage {
                        role: LoopRole::Tool,
                        content: format!("Sandbox denied {}: {}", tc.name, e),
                        tool_calls: None,
                        tool_results: None,
                    });
                    return IterationOutcome::Error(format!("Sandbox denied {}: {}", tc.name, e));
                }
            }

            // Build a ToolCall for the registry
            let mut arguments = HashMap::new();
            for (k, v) in &tc.arguments {
                arguments.insert(k.clone(), serde_json::json!(v));
            }
            let tool_call = ToolCall::new(&tc.name, arguments);

            // Execute
            match self.tool_registry.execute(&tool_call) {
                Ok(output) => {
                    self.tools_called_log.push(tc.name.clone());

                    // Record cost using configurable cost tracking
                    let call_cost = self.config.cost_config.cost_for(&tc.name, &tc.arguments);
                    self.total_cost += call_cost;
                    if let Ok(mut sandbox) = self.sandbox.write() {
                        sandbox.record_cost(call_cost);
                    }

                    // Push tool result into conversation
                    self.conversation.push(LoopMessage {
                        role: LoopRole::Tool,
                        content: format!("[Tool: {}] {}", tc.name, output.content),
                        tool_calls: None,
                        tool_results: None,
                    });
                }
                Err(e) => {
                    self.conversation.push(LoopMessage {
                        role: LoopRole::Tool,
                        content: format!("[Tool: {} Error] {}", tc.name, e),
                        tool_calls: None,
                        tool_results: None,
                    });
                }
            }
        }

        // 6. Update task board progress
        if let (Some(ref board), Some(ref task_id)) = (&self.task_board, &self.current_task_id) {
            let progress = self.iteration as f64 / self.config.max_iterations as f64;
            let action = parsed
                .last()
                .map(|tc| format!("Called {}", tc.name))
                .unwrap_or_default();
            if let Ok(mut b) = board.write() {
                let _ = b.execute_command(BoardCommand::UpdateProgress {
                    task_id: task_id.clone(),
                    progress: progress.min(0.99),
                    action,
                });
            }
        }

        IterationOutcome::Continue
    }

    /// Pause the agent. The next iteration will return early.
    pub fn pause(&mut self) {
        self.state = AgentState::Paused;
    }

    /// Resume a paused agent back to Running.
    pub fn resume(&mut self) {
        if self.state == AgentState::Paused {
            self.state = AgentState::Running;
        }
    }

    /// Abort the agent with a failure.
    pub fn abort(&mut self) {
        self.state = AgentState::Failed("Aborted by caller".into());
    }

    /// Get the current state.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Get the agent's configuration.
    pub fn config(&self) -> &AutonomousAgentConfig {
        &self.config
    }

    /// Get the agent's policy.
    pub fn policy(&self) -> &AgentPolicy {
        &self.policy
    }

    /// Get the agent's operation mode.
    pub fn mode(&self) -> OperationMode {
        self.mode
    }

    /// Get the conversation history.
    pub fn conversation(&self) -> &[LoopMessage] {
        &self.conversation
    }

    /// Get the cancellation token, if set.
    pub fn cancellation_token(&self) -> Option<&Arc<AtomicBool>> {
        self.cancellation_token.as_ref()
    }

    /// Get current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get accumulated cost.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Get agent name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Collect all tool names that were called during the run.
    fn collect_tools_called(&self) -> Vec<String> {
        self.tools_called_log.clone()
    }
}

// ============================================================================
// parse_tool_calls
// ============================================================================

/// Parse tool calls from the LLM response text.
///
/// Supports three formats (tried in order):
/// 1. JSON array: `[{"name": "tool", "arguments": {"k": "v"}}]`
/// 2. OpenAI-style: response contains `"tool_calls": [{"function": {"name": "x", "arguments": "..."}}]`
/// 3. XML tool_use: `<tool_use><name>x</name><arguments>{"k":"v"}</arguments></tool_use>`
pub fn parse_tool_calls(response: &str) -> Vec<ParsedToolCall> {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try JSON array format
    if let Some(calls) = try_parse_json_array(trimmed) {
        if !calls.is_empty() {
            return calls;
        }
    }

    // Try OpenAI-style format
    if let Some(calls) = try_parse_openai_style(trimmed) {
        if !calls.is_empty() {
            return calls;
        }
    }

    // Try XML tool_use format
    if let Some(calls) = try_parse_xml_tool_use(trimmed) {
        if !calls.is_empty() {
            return calls;
        }
    }

    Vec::new()
}

fn try_parse_json_array(text: &str) -> Option<Vec<ParsedToolCall>> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end <= start {
        return None;
    }
    let candidate = &text[start..=end];
    let arr: Vec<serde_json::Value> = serde_json::from_str(candidate).ok()?;
    let mut calls = Vec::new();
    for item in &arr {
        if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
            let arguments = extract_arguments(item.get("arguments"));
            calls.push(ParsedToolCall {
                name: name.to_string(),
                arguments,
            });
        }
    }
    Some(calls)
}

fn try_parse_openai_style(text: &str) -> Option<Vec<ParsedToolCall>> {
    // Look for "tool_calls" key in a JSON object
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    let candidate = &text[start..=end];
    let obj: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let tool_calls = obj.get("tool_calls")?.as_array()?;
    let mut calls = Vec::new();
    for tc in tool_calls {
        let func = tc.get("function")?;
        let name = func.get("name")?.as_str()?;
        // arguments can be a string (JSON-encoded) or an object
        let arguments = if let Some(args_str) = func.get("arguments").and_then(|v| v.as_str()) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args_str) {
                extract_arguments(Some(&parsed))
            } else {
                HashMap::new()
            }
        } else {
            extract_arguments(func.get("arguments"))
        };
        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments,
        });
    }
    Some(calls)
}

fn try_parse_xml_tool_use(text: &str) -> Option<Vec<ParsedToolCall>> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find("<tool_use>") {
        let abs_start = search_from + start;
        let end_tag = "</tool_use>";
        if let Some(end) = text[abs_start..].find(end_tag) {
            let block = &text[abs_start..abs_start + end + end_tag.len()];
            if let Some(call) = parse_single_xml_tool(block) {
                calls.push(call);
            }
            search_from = abs_start + end + end_tag.len();
        } else {
            break;
        }
    }
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn parse_single_xml_tool(block: &str) -> Option<ParsedToolCall> {
    let name = extract_xml_tag(block, "name")?;
    let args_str = extract_xml_tag(block, "arguments").unwrap_or_default();
    let arguments = if !args_str.is_empty() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&args_str) {
            extract_arguments(Some(&val))
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };
    Some(ParsedToolCall { name, arguments })
}

fn extract_xml_tag(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = text.find(&open)? + open.len();
    let end = text.find(&close)?;
    if end > start {
        Some(text[start..end].trim().to_string())
    } else {
        None
    }
}

fn extract_arguments(val: Option<&serde_json::Value>) -> HashMap<String, String> {
    let mut arguments = HashMap::new();
    if let Some(obj) = val.and_then(|v| v.as_object()) {
        for (k, v) in obj {
            let s = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            arguments.insert(k.clone(), s);
        }
    }
    arguments
}

// ============================================================================
// AutonomousAgentBuilder
// ============================================================================

/// Builder for constructing an `AutonomousAgent` with a fluent API.
pub struct AutonomousAgentBuilder {
    name: String,
    max_iterations: usize,
    system_prompt: String,
    cost_config: CostConfig,
    policy: AgentPolicy,
    sandbox: Option<Arc<RwLock<SandboxValidator>>>,
    tool_registry: ToolRegistry,
    response_generator: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>,
    mode: OperationMode,
    interaction: Option<Arc<InteractionManager>>,
    task_board: Option<Arc<RwLock<TaskBoard>>>,
    current_task_id: Option<String>,
    cancellation_token: Option<Arc<AtomicBool>>,
    mailbox: Option<std::sync::mpsc::Receiver<InterAgentMessage>>,
}

impl AutonomousAgentBuilder {
    pub fn new(
        name: impl Into<String>,
        response_generator: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>,
    ) -> Self {
        Self {
            name: name.into(),
            max_iterations: 50,
            system_prompt: String::new(),
            cost_config: CostConfig::default(),
            policy: AgentPolicy::default(),
            sandbox: None,
            tool_registry: ToolRegistry::new(),
            response_generator,
            mode: OperationMode::Autonomous,
            interaction: None,
            task_board: None,
            current_task_id: None,
            cancellation_token: None,
            mailbox: None,
        }
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_cost_config(mut self, config: CostConfig) -> Self {
        self.cost_config = config;
        self
    }

    pub fn policy(mut self, policy: AgentPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn sandbox(mut self, sandbox: Arc<RwLock<SandboxValidator>>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    pub fn tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    pub fn mode(mut self, mode: OperationMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn interaction(mut self, manager: Arc<InteractionManager>) -> Self {
        self.interaction = Some(manager);
        self
    }

    pub fn task_board(mut self, board: Arc<RwLock<TaskBoard>>, task_id: impl Into<String>) -> Self {
        self.task_board = Some(board);
        self.current_task_id = Some(task_id.into());
        self
    }

    /// Set a cancellation token for cooperative cancellation.
    pub fn cancellation_token(mut self, token: Arc<AtomicBool>) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Set a mailbox receiver for inter-agent messages.
    pub fn mailbox(mut self, rx: std::sync::mpsc::Receiver<InterAgentMessage>) -> Self {
        self.mailbox = Some(rx);
        self
    }

    pub fn build(self) -> AutonomousAgent {
        let sandbox = self
            .sandbox
            .unwrap_or_else(|| Arc::new(RwLock::new(SandboxValidator::new(self.policy.clone()))));

        AutonomousAgent {
            config: AutonomousAgentConfig {
                name: self.name,
                max_iterations: self.max_iterations,
                system_prompt: self.system_prompt,
                cost_config: self.cost_config,
            },
            policy: self.policy,
            sandbox,
            tool_registry: self.tool_registry,
            conversation: Vec::new(),
            response_generator: self.response_generator,
            mode: self.mode,
            state: AgentState::Idle,
            interaction: self.interaction,
            task_board: self.task_board,
            current_task_id: self.current_task_id,
            iteration: 0,
            total_cost: 0.0,
            start_time: 0,
            tools_called_log: Vec::new(),
            cancellation_token: self.cancellation_token,
            mailbox: self.mailbox,
            planning_hint_idx: None,
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_policy::{AgentPolicy, AgentPolicyBuilder, AutoApproveAll};
    use crate::agent_sandbox::SandboxValidator;
    use crate::task_board::TaskBoard;
    use crate::task_planning::StepPriority;
    use crate::unified_tools::{ToolBuilder, ToolOutput, ToolRegistry};
    use crate::user_interaction::{AutoApproveHandler, InteractionManager};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Helper: build a simple agent with the given generator.
    fn make_agent(gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>) -> AutonomousAgent {
        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));
        AutonomousAgent::builder("test-agent", gen)
            .max_iterations(10)
            .system_prompt("You are a test agent.")
            .policy(policy)
            .sandbox(sandbox)
            .build()
    }

    // -----------------------------------------------------------------------
    // 1. test_builder_basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_builder_basic() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| "hello".to_string());

        let agent = AutonomousAgent::builder("my-agent", gen)
            .max_iterations(20)
            .system_prompt("System prompt")
            .mode(OperationMode::Programming)
            .build();

        assert_eq!(agent.config.name, "my-agent");
        assert_eq!(agent.config.max_iterations, 20);
        assert_eq!(agent.config.system_prompt, "System prompt");
        assert_eq!(agent.mode, OperationMode::Programming);
        assert_eq!(agent.state, AgentState::Idle);
    }

    // -----------------------------------------------------------------------
    // 2. test_agent_state_lifecycle
    // -----------------------------------------------------------------------
    #[test]
    fn test_agent_state_lifecycle() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| "final answer".to_string());
        let mut agent = make_agent(gen);

        assert_eq!(*agent.state(), AgentState::Idle);

        let result = agent.run("Do something");
        assert!(result.is_ok());
        assert!(matches!(*agent.state(), AgentState::Completed(_)));

        if let AgentState::Completed(ref answer) = *agent.state() {
            assert_eq!(answer, "final answer");
        }
    }

    // -----------------------------------------------------------------------
    // 3. test_run_simple_task
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_simple_task() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| "The answer is 42.".to_string());
        let mut agent = make_agent(gen);

        let result = agent.run("What is the meaning of life?").unwrap();
        assert_eq!(result.output, "The answer is 42.");
        assert_eq!(result.iterations, 1);
        assert!(result.tools_called.is_empty());
        assert!(result.duration_ms < 5000);
    }

    // -----------------------------------------------------------------------
    // 4. test_run_with_tool_calls
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_with_tool_calls() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_msgs| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                // First call: return a tool call
                r#"Let me calculate that.
[{"name": "calculate", "arguments": {"expression": "2+2"}}]"#
                    .to_string()
            } else {
                // Second call: return final answer
                "The result is 4.".to_string()
            }
        });

        // Set up registry with a calculate tool
        let mut registry = ToolRegistry::new();
        let def = ToolBuilder::new("calculate", "Evaluate math")
            .required_string("expression", "Math expression")
            .build();
        registry.register(
            def,
            Arc::new(|call: &ToolCall| {
                let expr = call.get_string("expression").unwrap_or("0");
                // Simple: just return the expression as-is for the test
                Ok(ToolOutput::text(format!("Result: {}", expr)))
            }),
        );

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("calc-agent", gen)
            .max_iterations(10)
            .policy(policy)
            .sandbox(sandbox)
            .tool_registry(registry)
            .build();

        let result = agent.run("What is 2+2?").unwrap();
        assert_eq!(result.output, "The result is 4.");
        assert_eq!(result.iterations, 2);
        assert!(result.tools_called.contains(&"calculate".to_string()));
        assert!(result.cost > 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. test_parse_tool_calls_json
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_tool_calls_json() {
        let input = r#"[{"name": "search", "arguments": {"query": "rust lang"}}]"#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments.get("query").unwrap(), "rust lang");
    }

    // -----------------------------------------------------------------------
    // 6. test_parse_tool_calls_empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_tool_calls_empty() {
        assert!(parse_tool_calls("").is_empty());
        assert!(parse_tool_calls("Just some text without tool calls.").is_empty());
        assert!(parse_tool_calls("   ").is_empty());
    }

    // -----------------------------------------------------------------------
    // 7. test_parse_tool_calls_embedded
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_tool_calls_embedded() {
        let input = r#"I need to search for that.
[{"name": "web_search", "arguments": {"query": "rust async"}}]
Let me process the results."#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments.get("query").unwrap(), "rust async");
    }

    // -----------------------------------------------------------------------
    // 8. test_sandbox_denies_action
    // -----------------------------------------------------------------------
    #[test]
    fn test_sandbox_denies_action() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| r#"[{"name": "forbidden_tool", "arguments": {}}]"#.to_string());

        // Policy that denies "forbidden_tool"
        let policy = AgentPolicyBuilder::new()
            .deny_tool("forbidden_tool")
            .build();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy.clone())));

        let mut agent = AutonomousAgent::builder("deny-agent", gen)
            .max_iterations(5)
            .policy(policy)
            .sandbox(sandbox)
            .build();

        let result = agent.run("Use the forbidden tool");
        assert!(result.is_err());
        assert!(matches!(*agent.state(), AgentState::Failed(_)));
    }

    // -----------------------------------------------------------------------
    // 9. test_max_iterations_limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_iterations_limit() {
        // Generator that always returns tool calls, never a final answer
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| r#"[{"name": "noop", "arguments": {}}]"#.to_string());

        let mut registry = ToolRegistry::new();
        let def = ToolBuilder::new("noop", "Do nothing").build();
        registry.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("loop-agent", gen)
            .max_iterations(3)
            .policy(policy)
            .sandbox(sandbox)
            .tool_registry(registry)
            .build();

        let result = agent.run("Loop forever");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Max iterations"));
    }

    // -----------------------------------------------------------------------
    // 10. test_ask_user_tool
    // -----------------------------------------------------------------------
    #[test]
    fn test_ask_user_tool() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                r#"[{"name": "ask_user", "arguments": {"question": "What color?"}}]"#.to_string()
            } else {
                "The color is blue.".to_string()
            }
        });

        let handler = Arc::new(AutoApproveHandler::with_default_text("blue"));
        let im = Arc::new(InteractionManager::new(handler, 30));

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("ask-agent", gen)
            .max_iterations(10)
            .policy(policy)
            .sandbox(sandbox)
            .interaction(im)
            .build();

        let result = agent.run("What is the user's favorite color?").unwrap();
        assert_eq!(result.output, "The color is blue.");
        assert_eq!(result.iterations, 2);
    }

    // -----------------------------------------------------------------------
    // 11. test_task_board_progress
    // -----------------------------------------------------------------------
    #[test]
    fn test_task_board_progress() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                r#"[{"name": "noop", "arguments": {}}]"#.to_string()
            } else {
                "Done.".to_string()
            }
        });

        let mut registry = ToolRegistry::new();
        let def = ToolBuilder::new("noop", "Do nothing").build();
        registry.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        let board = Arc::new(RwLock::new(TaskBoard::new("Test Board")));

        // Add a task to the board
        let task_id;
        {
            let mut b = board.write().unwrap();
            b.execute_command(BoardCommand::AddTask {
                title: "Agent task".into(),
                description: "Test task".into(),
                priority: StepPriority::Medium,
            })
            .unwrap();
            task_id = b.plan().steps[0].id.clone();
            b.execute_command(BoardCommand::StartTask {
                id: task_id.clone(),
            })
            .unwrap();
        }

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("board-agent", gen)
            .max_iterations(10)
            .policy(policy)
            .sandbox(sandbox)
            .tool_registry(registry)
            .task_board(board.clone(), task_id.clone())
            .build();

        let result = agent.run("Complete the task").unwrap();
        assert_eq!(result.output, "Done.");

        // Check that progress was updated
        let b = board.read().unwrap();
        let state = b.execution_state(&task_id);
        assert!(state.is_some());
        assert!(state.unwrap().progress > 0.0);
    }

    // -----------------------------------------------------------------------
    // 12. test_pause_resume
    // -----------------------------------------------------------------------
    #[test]
    fn test_pause_resume() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| "answer".to_string());
        let mut agent = make_agent(gen);

        // Pause before running
        agent.pause();
        assert_eq!(*agent.state(), AgentState::Paused);

        // Run should fail because paused
        let result = agent.run("task");
        assert!(result.is_err());

        // Resume
        agent.resume();
        assert_eq!(*agent.state(), AgentState::Running);
    }

    // -----------------------------------------------------------------------
    // 13. test_abort
    // -----------------------------------------------------------------------
    #[test]
    fn test_abort() {
        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> =
            Arc::new(|_| "answer".to_string());
        let mut agent = make_agent(gen);

        agent.abort();
        assert!(matches!(*agent.state(), AgentState::Failed(_)));

        if let AgentState::Failed(ref msg) = *agent.state() {
            assert_eq!(msg, "Aborted by caller");
        }
    }

    // -----------------------------------------------------------------------
    // 14. test_cost_tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_cost_tracking() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n < 3 {
                r#"[{"name": "noop", "arguments": {}}]"#.to_string()
            } else {
                "Done.".to_string()
            }
        });

        let mut registry = ToolRegistry::new();
        let def = ToolBuilder::new("noop", "Do nothing").build();
        registry.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("cost-agent", gen)
            .max_iterations(10)
            .policy(policy)
            .sandbox(Arc::clone(&sandbox))
            .tool_registry(registry)
            .build();

        let result = agent.run("Do some work").unwrap();
        // 3 tool calls * 0.001 each = 0.003
        assert!((result.cost - 0.003).abs() < 0.0001);
        assert_eq!(result.iterations, 4); // 3 tool iterations + 1 final

        // Sandbox should also have recorded cost
        let sb = sandbox.read().unwrap();
        assert!((sb.total_cost() - 0.003).abs() < 0.0001);
    }

    // -----------------------------------------------------------------------
    // 15. test_agent_result_structure
    // -----------------------------------------------------------------------
    #[test]
    fn test_agent_result_structure() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                r#"[{"name": "echo", "arguments": {"text": "hello"}}]"#.to_string()
            } else {
                "Final output".to_string()
            }
        });

        let mut registry = ToolRegistry::new();
        let def = ToolBuilder::new("echo", "Echo text")
            .required_string("text", "Text to echo")
            .build();
        registry.register(
            def,
            Arc::new(|call: &ToolCall| {
                let text = call.get_string("text").unwrap_or("");
                Ok(ToolOutput::text(text.to_string()))
            }),
        );

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("result-agent", gen)
            .max_iterations(10)
            .system_prompt("You are a test agent.")
            .policy(policy)
            .sandbox(sandbox)
            .tool_registry(registry)
            .build();

        let result = agent.run("Echo hello").unwrap();

        assert_eq!(result.output, "Final output");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tools_called, vec!["echo".to_string()]);
        assert!(result.cost > 0.0);
        assert!(result.duration_ms < 5000);

        // Conversation should contain system + user + assistant + tool + assistant
        let conv = agent.conversation();
        assert!(conv.len() >= 4);
        assert_eq!(conv[0].role, LoopRole::System);
        assert_eq!(conv[1].role, LoopRole::User);
        assert_eq!(conv[2].role, LoopRole::Assistant);
        assert_eq!(conv[3].role, LoopRole::Tool);
    }

    // -----------------------------------------------------------------------
    // 16. test_parse_openai_style
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_openai_style() {
        let input = r#"{"tool_calls": [{"function": {"name": "search", "arguments": "{\"query\": \"rust\"}"}}]}"#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments.get("query").unwrap(), "rust");
    }

    // -----------------------------------------------------------------------
    // 17. test_parse_xml_tool_use
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_xml_tool_use() {
        let input = r#"I will search for that.
<tool_use><name>web_search</name><arguments>{"query": "rust async"}</arguments></tool_use>"#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments.get("query").unwrap(), "rust async");
    }

    // -----------------------------------------------------------------------
    // 18. test_parse_multiple_xml_tools
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_multiple_xml_tools() {
        let input = r#"<tool_use><name>read_file</name><arguments>{"path": "/tmp/a.txt"}</arguments></tool_use>
<tool_use><name>read_file</name><arguments>{"path": "/tmp/b.txt"}</arguments></tool_use>"#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments.get("path").unwrap(), "/tmp/a.txt");
        assert_eq!(calls[1].arguments.get("path").unwrap(), "/tmp/b.txt");
    }

    // -----------------------------------------------------------------------
    // 19. test_parse_openai_style_object_args
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_openai_style_object_args() {
        let input = r#"{"tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "2+2"}}}]}"#;
        let calls = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calculate");
        assert_eq!(calls[0].arguments.get("expression").unwrap(), "2+2");
    }

    // -----------------------------------------------------------------------
    // 20. test_cost_config_default
    // -----------------------------------------------------------------------
    #[test]
    fn test_cost_config_default() {
        let config = CostConfig::new();
        let args = HashMap::new();
        assert!((config.cost_for("any_tool", &args) - 0.001).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 21. test_cost_config_per_tool
    // -----------------------------------------------------------------------
    #[test]
    fn test_cost_config_per_tool() {
        let mut config = CostConfig::new();
        config.tool_costs.insert("expensive".to_string(), 0.05);
        let args = HashMap::new();
        assert!((config.cost_for("expensive", &args) - 0.05).abs() < 1e-10);
        assert!((config.cost_for("other", &args) - 0.001).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 22. test_cost_config_callback
    // -----------------------------------------------------------------------
    #[test]
    fn test_cost_config_callback() {
        let config = CostConfig {
            default_cost_per_call: 0.001,
            tool_costs: HashMap::new(),
            cost_callback: Some(Arc::new(
                |name: &str, _args: &HashMap<String, String>| {
                    if name == "gpt4" {
                        0.1
                    } else {
                        0.01
                    }
                },
            )),
        };
        let args = HashMap::new();
        assert!((config.cost_for("gpt4", &args) - 0.1).abs() < 1e-10);
        assert!((config.cost_for("other", &args) - 0.01).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 23. test_tools_called_tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_tools_called_tracking() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let gen: Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => r#"[{"name": "tool_a", "arguments": {}}]"#.to_string(),
                1 => r#"[{"name": "tool_b", "arguments": {}}]"#.to_string(),
                _ => "Done.".to_string(),
            }
        });

        let mut registry = ToolRegistry::new();
        let def_a = ToolBuilder::new("tool_a", "Tool A").build();
        registry.register(def_a, Arc::new(|_| Ok(ToolOutput::text("a_result"))));
        let def_b = ToolBuilder::new("tool_b", "Tool B").build();
        registry.register(def_b, Arc::new(|_| Ok(ToolOutput::text("b_result"))));

        let policy = AgentPolicy::autonomous();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy.clone(),
            Arc::new(AutoApproveAll),
        )));

        let mut agent = AutonomousAgent::builder("track-agent", gen)
            .max_iterations(10)
            .policy(policy)
            .sandbox(sandbox)
            .tool_registry(registry)
            .build();

        let result = agent.run("Use both tools").unwrap();
        assert_eq!(
            result.tools_called,
            vec!["tool_a".to_string(), "tool_b".to_string()]
        );
    }
}
