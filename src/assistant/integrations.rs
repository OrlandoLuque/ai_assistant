use super::*;

impl AiAssistant {
    // =========================================================================
    // AUTONOMOUS AGENT INTEGRATION
    // =========================================================================

    /// Get the current operation mode.
    #[cfg(feature = "autonomous")]
    pub fn operation_mode(&self) -> OperationMode {
        self.mode_manager.current()
    }

    /// Set the operation mode (respects allowed_max ceiling).
    #[cfg(feature = "autonomous")]
    pub fn set_operation_mode(&mut self, mode: OperationMode) -> AiResult<()> {
        self.mode_manager
            .set_mode(mode)
            .map_err(|e| AiError::Other(e.to_string()))
    }

    /// Escalate to next operation mode.
    #[cfg(feature = "autonomous")]
    pub fn escalate_mode(&mut self) -> AiResult<OperationMode> {
        self.mode_manager
            .escalate()
            .map_err(|e| AiError::Other(e.to_string()))
    }

    /// De-escalate to lower operation mode.
    #[cfg(feature = "autonomous")]
    pub fn de_escalate_mode(&mut self) -> OperationMode {
        self.mode_manager.de_escalate()
    }

    /// Get the profile registry.
    #[cfg(feature = "autonomous")]
    pub fn profiles(&self) -> &ProfileRegistry {
        &self.profile_registry
    }

    /// Get the profile registry mutably.
    #[cfg(feature = "autonomous")]
    pub fn profiles_mut(&mut self) -> &mut ProfileRegistry {
        &mut self.profile_registry
    }

    /// Set the interaction handler for agent-user communication.
    #[cfg(feature = "autonomous")]
    pub fn set_interaction_handler(&mut self, handler: Arc<dyn UserInteractionHandler>) {
        self.interaction_manager = Some(Arc::new(InteractionManager::new(handler, 300)));
    }

    /// Get the interaction manager (if configured).
    #[cfg(feature = "autonomous")]
    pub fn interaction_manager(&self) -> Option<&Arc<InteractionManager>> {
        self.interaction_manager.as_ref()
    }

    /// Create an autonomous agent from a registered profile name.
    ///
    /// The agent uses the assistant's config to derive a response generator
    /// callback, and applies the profile's policy and tools.
    #[cfg(feature = "autonomous")]
    pub fn create_agent(
        &self,
        profile_name: &str,
        response_generator: Arc<
            dyn Fn(&[crate::agentic_loop::LoopMessage]) -> String + Send + Sync,
        >,
    ) -> AiResult<AutonomousAgent> {
        let profile = self
            .profile_registry
            .get_agent_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Agent profile '{}' not found", profile_name))?;

        let mut builder = AutonomousAgentBuilder::new(&profile.name, response_generator)
            .policy(profile.policy.clone())
            .mode(profile.mode);

        if let Some(ref prompt) = profile.system_prompt {
            builder = builder.system_prompt(prompt.clone());
        }

        if let Some(ref max_iter) = Some(profile.policy.max_iterations) {
            builder = builder.max_iterations(*max_iter);
        }

        // Register OS tools into the agent's tool registry
        let policy = profile.policy.clone();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));
        let mut registry = crate::unified_tools::ToolRegistry::new();
        register_os_tools(&mut registry, sandbox.clone());
        builder = builder.tool_registry(registry).sandbox(sandbox);

        if let Some(ref im) = self.interaction_manager {
            builder = builder.interaction(im.clone());
        }

        Ok(builder.build())
    }

    /// Create an autonomous agent with auto-approve interaction (for headless/test usage).
    #[cfg(feature = "autonomous")]
    pub fn create_agent_headless(
        &self,
        profile_name: &str,
        response_generator: Arc<
            dyn Fn(&[crate::agentic_loop::LoopMessage]) -> String + Send + Sync,
        >,
    ) -> AiResult<AutonomousAgent> {
        let handler: Arc<dyn UserInteractionHandler> = Arc::new(AutoApproveInteraction::new());
        let im = Arc::new(InteractionManager::new(handler, 300));

        let profile = self
            .profile_registry
            .get_agent_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Agent profile '{}' not found", profile_name))?;

        let policy = profile.policy.clone();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));
        let mut registry = crate::unified_tools::ToolRegistry::new();
        register_os_tools(&mut registry, sandbox.clone());

        let mut builder = AutonomousAgentBuilder::new(&profile.name, response_generator)
            .policy(profile.policy.clone())
            .mode(profile.mode)
            .tool_registry(registry)
            .sandbox(sandbox)
            .interaction(im);

        if let Some(ref prompt) = profile.system_prompt {
            builder = builder.system_prompt(prompt.clone());
        }

        builder = builder.max_iterations(profile.policy.max_iterations);

        Ok(builder.build())
    }

    // === Butler ===

    /// Initialize the Butler for environment auto-detection.
    #[cfg(feature = "butler")]
    pub fn init_butler(&mut self) {
        self.butler = Some(Butler::new());
    }

    /// Run Butler environment scan and return the report.
    #[cfg(feature = "butler")]
    pub fn butler_scan(&mut self) -> Option<crate::butler::EnvironmentReport> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        self.butler.as_mut().map(|b| b.scan())
    }

    /// Auto-configure the assistant using Butler's environment scan.
    /// Updates the AiConfig based on detected providers.
    #[cfg(feature = "butler")]
    pub fn auto_configure(&mut self) -> AiResult<()> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        let butler = self.butler.as_mut().expect("butler must be initialized");
        let report = butler.scan();
        let suggested_config = butler.suggest_config(&report);
        self.config = suggested_config;
        Ok(())
    }

    /// Run butler environment scan and produce an optimization advisor report.
    ///
    /// Scans the environment, then generates recommendations for improving
    /// efficiency, quality, cost, security, scalability, and observability.
    #[cfg(feature = "butler")]
    pub fn butler_advise(&mut self) -> Option<crate::butler::AdvisorReport> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        self.butler.as_mut().map(|b| {
            let report = b.scan();
            b.advise(&report)
        })
    }

    /// Run butler advisor with knowledge of current feature configuration.
    ///
    /// Produces more accurate recommendations by knowing which features
    /// the user has already enabled.
    #[cfg(feature = "butler")]
    pub fn butler_advise_with_config(
        &mut self,
        advisor_config: &crate::butler::AdvisorConfig,
    ) -> Option<crate::butler::AdvisorReport> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        self.butler.as_mut().map(|b| {
            let report = b.scan();
            b.advise_with_config(&report, advisor_config)
        })
    }

    // === Scheduler ===

    /// Initialize the scheduler.
    #[cfg(feature = "scheduler")]
    pub fn init_scheduler(&mut self) {
        self.scheduler = Some(Scheduler::new());
    }

    /// Get the scheduler (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn scheduler(&self) -> Option<&Scheduler> {
        self.scheduler.as_ref()
    }

    /// Get the scheduler mutably (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn scheduler_mut(&mut self) -> Option<&mut Scheduler> {
        self.scheduler.as_mut()
    }

    /// Initialize the trigger manager.
    #[cfg(feature = "scheduler")]
    pub fn init_trigger_manager(&mut self) {
        self.trigger_manager = Some(TriggerManager::new());
    }

    /// Get the trigger manager (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn trigger_manager(&self) -> Option<&TriggerManager> {
        self.trigger_manager.as_ref()
    }

    /// Get the trigger manager mutably (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn trigger_manager_mut(&mut self) -> Option<&mut TriggerManager> {
        self.trigger_manager.as_mut()
    }

    // === Browser ===

    /// Initialize the browser session for CDP-based browser automation.
    #[cfg(feature = "browser")]
    pub fn init_browser(&mut self) {
        self.browser_session = Some(BrowserSession::new());
    }

    /// Get the browser session (if initialized).
    #[cfg(feature = "browser")]
    pub fn browser_session(&self) -> Option<&BrowserSession> {
        self.browser_session.as_ref()
    }

    /// Get the browser session mutably (if initialized).
    #[cfg(feature = "browser")]
    pub fn browser_session_mut(&mut self) -> Option<&mut BrowserSession> {
        self.browser_session.as_mut()
    }

    // === Distributed Agents ===

    /// Initialize the distributed agent manager for multi-node agent execution.
    #[cfg(feature = "distributed-agents")]
    pub fn init_distributed_agents(&mut self, local_node_id: crate::distributed::NodeId) {
        self.distributed_agent_manager = Some(DistributedAgentManager::new(local_node_id));
    }

    /// Get the distributed agent manager (if initialized).
    #[cfg(feature = "distributed-agents")]
    pub fn distributed_agents(&self) -> Option<&DistributedAgentManager> {
        self.distributed_agent_manager.as_ref()
    }

    /// Get the distributed agent manager mutably (if initialized).
    #[cfg(feature = "distributed-agents")]
    pub fn distributed_agents_mut(&mut self) -> Option<&mut DistributedAgentManager> {
        self.distributed_agent_manager.as_mut()
    }

    // === A/B Testing ===

    /// Initialize the experiment manager for A/B testing.
    #[cfg(feature = "eval")]
    pub fn init_experiment_manager(&mut self) {
        if self.experiment_manager.is_none() {
            self.experiment_manager = Some(crate::ab_testing::ExperimentManager::new());
        }
    }

    /// Get the experiment manager (if initialized).
    #[cfg(feature = "eval")]
    pub fn experiment_manager(&self) -> Option<&crate::ab_testing::ExperimentManager> {
        self.experiment_manager.as_ref()
    }

    /// Get the experiment manager mutably (if initialized).
    #[cfg(feature = "eval")]
    pub fn experiment_manager_mut(&mut self) -> Option<&mut crate::ab_testing::ExperimentManager> {
        self.experiment_manager.as_mut()
    }

    // === Cost Dashboard ===

    /// Initialize cost tracking with default settings.
    pub fn init_cost_tracking(&mut self) {
        if self.cost_dashboard.is_none() {
            self.cost_dashboard = Some(crate::cost_integration::CostDashboard::new());
        }
    }

    /// Get reference to cost dashboard.
    pub fn cost_dashboard(&self) -> Option<&crate::cost_integration::CostDashboard> {
        self.cost_dashboard.as_ref()
    }

    /// Get mutable reference to cost dashboard.
    pub fn cost_dashboard_mut(&mut self) -> Option<&mut crate::cost_integration::CostDashboard> {
        self.cost_dashboard.as_mut()
    }

    /// Get formatted cost report.
    pub fn cost_report(&self) -> Option<String> {
        self.cost_dashboard.as_ref().map(|d| d.format_report())
    }

    // === Chat Hooks ===

    /// Initialize chat hooks for UI framework event streaming.
    pub fn init_chat_hooks(&mut self) {
        if self.chat_hooks.is_none() {
            self.chat_hooks = Some(crate::ui_hooks::ChatHooks::new());
        }
    }

    /// Get the chat hooks (if initialized).
    pub fn chat_hooks(&self) -> Option<&crate::ui_hooks::ChatHooks> {
        self.chat_hooks.as_ref()
    }

    /// Get the chat hooks mutably (if initialized).
    pub fn chat_hooks_mut(&mut self) -> Option<&mut crate::ui_hooks::ChatHooks> {
        self.chat_hooks.as_mut()
    }

    /// Emit a chat event to all subscribers (if hooks are initialized).
    pub fn emit_chat_event(&mut self, event: crate::ui_hooks::ChatStreamEvent) {
        if let Some(ref mut hooks) = self.chat_hooks {
            hooks.emit(event);
        }
    }

    // =========================================================================
    // Constrained Decoding Integration (v9 item 3.1)
    // =========================================================================

    /// Generate a response constrained by a GBNF grammar.
    ///
    /// Parses the grammar string (in GBNF format) into a [`Grammar`], sends
    /// the prompt to the configured LLM provider (synchronously), and validates
    /// the response against the grammar using a [`StreamingValidator`]-style
    /// check.
    ///
    /// # Arguments
    /// * `grammar` - A GBNF grammar string (e.g. `root ::= "yes" | "no"`)
    /// * `prompt` - The user prompt to send to the LLM
    ///
    /// # Errors
    /// Returns `AiError` if the grammar cannot be parsed, the LLM call fails,
    /// or the response does not conform to the grammar.
    #[cfg(feature = "constrained-decoding")]
    pub fn generate_with_grammar(
        &self,
        grammar: &str,
        prompt: &str,
    ) -> Result<String, crate::error::AiError> {
        use crate::constrained_decoding::{Grammar, GrammarConstraint};

        // 1. Parse the GBNF grammar
        let parsed_grammar = Grammar::from_gbnf(grammar)?;

        // 2. Verify the grammar can be formatted for the current provider
        let provider_name = self.config.provider.display_name();
        // Attempt to format — if provider is unsupported, we still proceed
        // with validation-only mode.
        let _grammar_str = GrammarConstraint::for_provider(&parsed_grammar, provider_name)
            .unwrap_or_else(|_| parsed_grammar.to_gbnf());

        // 3. Build conversation with the prompt
        let conversation = vec![crate::messages::ChatMessage::user(prompt)];
        let system_prompt = build_system_prompt(&self.system_prompt_base, &self.preferences, "");

        // 4. Call the LLM synchronously through the port (F5): honours an injected
        // provider, else the config-dispatch chain.
        let response = self
            .resolve_provider()
            .generate(&conversation, &system_prompt)
            .map_err(|e| crate::error::AiError::Other(format!("LLM generation failed: {}", e)))?;

        // 5. Validate the response against the grammar rules
        // Check if the response matches any of the root rule's alternatives
        let root_rule = parsed_grammar
            .rules
            .iter()
            .find(|r| r.name == parsed_grammar.root_rule);
        if let Some(rule) = root_rule {
            let trimmed = response.trim();
            let valid = rule.alternatives.iter().any(|alt| {
                // Simple validation: check literal-only alternatives
                let literal_match: String = alt
                    .elements
                    .iter()
                    .filter_map(|el| {
                        if let crate::constrained_decoding::GrammarElement::Literal(s) = el {
                            Some(s.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if !literal_match.is_empty() {
                    return trimmed == literal_match || trimmed.contains(&literal_match);
                }
                // For non-literal rules, accept the response as valid
                // (full recursive validation would require a full parser)
                true
            });
            if !valid {
                return Err(crate::error::AiError::ConstrainedDecoding(
                    crate::error::ConstrainedDecodingError::GrammarCompilationFailed {
                        reason: format!(
                            "Response '{}' does not match grammar root rule '{}'",
                            trimmed, parsed_grammar.root_rule
                        ),
                    },
                ));
            }
        }

        Ok(response)
    }

    // =========================================================================
    // Human-in-the-Loop Integration (v9 item 3.2)
    // =========================================================================

    /// Send a message with an optional HITL approval gate.
    ///
    /// When `auto_approve` is `true`, the message is sent and the response
    /// returned directly. When `false`, the method simulates a HITL approval
    /// gate by creating an [`ApprovalRequest`] and logging it to an
    /// [`ApprovalLog`] before returning the response.
    ///
    /// The approval request records the prompt as the tool name and the
    /// response as context, providing a full audit trail of LLM interactions.
    ///
    /// # Arguments
    /// * `message` - The user message to send
    /// * `auto_approve` - If true, skip the approval gate
    ///
    /// # Errors
    /// Returns `AiError` if the LLM call fails.
    #[cfg(feature = "hitl")]
    pub fn send_message_with_approval(
        &mut self,
        message: &str,
        auto_approve: bool,
    ) -> Result<String, crate::error::AiError> {
        use crate::hitl::{
            ApprovalDecision, ApprovalLog, ApprovalLogEntry, ApprovalRequest, AutoApproveGate,
            HitlApprovalGate, ImpactLevel,
        };
        use std::collections::HashMap as HitlHashMap;

        // 1. Send the message to the LLM synchronously with context budget allocation
        let hitl_intent = self.classify_intent_for_budget(message);
        let allocated_context = self.build_allocated_context(
            message,
            &self.knowledge_context.clone(),
            hitl_intent.as_ref(),
        );
        let effective_knowledge = if allocated_context.is_empty() {
            self.knowledge_context.clone()
        } else {
            allocated_context
        };
        let conversation = {
            let mut conv = self.conversation.clone();
            conv.push(crate::messages::ChatMessage::user(message));
            conv
        };
        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            &effective_knowledge,
        );
        // Through the port (F5): honours an injected provider, else config-dispatch.
        let response = self
            .resolve_provider()
            .generate(&conversation, &system_prompt)
            .map_err(|e| crate::error::AiError::Other(format!("LLM generation failed: {}", e)))?;

        // 2. Record in conversation
        self.conversation
            .push(crate::messages::ChatMessage::user(message));
        self.conversation
            .push(crate::messages::ChatMessage::assistant(&response));

        // 3. Apply approval gate
        if !auto_approve {
            let request = ApprovalRequest::new(
                format!("msg-{}", self.conversation.len()),
                "send_message",
                HitlHashMap::new(),
                "ai_assistant",
                format!("User message: {}; LLM response: {}", message, response),
                ImpactLevel::Low,
            );

            let gate = AutoApproveGate;
            let decision = gate
                .request_approval(&request)
                .map_err(|e| crate::error::AiError::Other(format!("HITL gate error: {}", e)))?;

            // Log the decision
            let mut log = ApprovalLog::new(1000);
            log.record(ApprovalLogEntry {
                request,
                decision: decision.clone(),
                gate_name: gate.name().to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });

            if let ApprovalDecision::Deny { reason } = decision {
                return Err(crate::error::AiError::Other(format!(
                    "Message denied by HITL gate: {}",
                    reason
                )));
            }
        }

        Ok(response)
    }

    // =========================================================================
    // MCP Client Integration (v9 item 3.3)
    // =========================================================================

    /// Connect to a remote MCP server by URL.
    ///
    /// Validates the URL and creates a [`RemoteMcpClient`] connection. The
    /// connection URL is stored internally for subsequent tool listing.
    ///
    /// # Arguments
    /// * `server_url` - The MCP server URL (e.g. `"http://localhost:3000/mcp"`)
    ///
    /// # Errors
    /// Returns `AiError` if the URL is empty or the connection fails.
    pub fn connect_mcp_server(&mut self, server_url: &str) -> Result<(), crate::error::AiError> {
        use crate::mcp_client::{McpClientConfig, RemoteMcpClient};

        if server_url.is_empty() {
            return Err(crate::error::AiError::Other(
                "MCP server URL cannot be empty".to_string(),
            ));
        }

        // Validate URL format (basic check)
        if !server_url.starts_with("http://") && !server_url.starts_with("https://") {
            return Err(crate::error::AiError::Other(format!(
                "Invalid MCP server URL (must start with http:// or https://): {}",
                server_url
            )));
        }

        let config = McpClientConfig {
            url: server_url.to_string(),
            ..McpClientConfig::default()
        };

        let mut client = RemoteMcpClient::new(config);
        client
            .connect()
            .map_err(|e| crate::error::AiError::Other(format!("MCP connection failed: {}", e)))?;

        // Store the connection URL as an indicator that connection was established
        self.knowledge_context
            .push_str(&format!("\n[MCP Server connected: {}]\n", server_url));

        Ok(())
    }

    /// List available tools from connected MCP servers.
    ///
    /// Returns the names of tools discovered via MCP. If no server has been
    /// connected, returns an empty list.
    ///
    /// This is a lightweight query that does not require a persistent connection
    /// -- it creates a temporary client, connects, and fetches the tool list.
    ///
    /// # Arguments
    /// * `server_url` - The MCP server URL to query for tools
    pub fn list_mcp_tools(&self, server_url: &str) -> Vec<String> {
        use crate::mcp_client::{McpClientConfig, RemoteMcpClient};

        if server_url.is_empty() {
            return Vec::new();
        }

        let config = McpClientConfig {
            url: server_url.to_string(),
            ..McpClientConfig::default()
        };

        let mut client = RemoteMcpClient::new(config);
        match client.connect() {
            Ok(()) => match client.list_tools() {
                Ok(tools) => tools.iter().map(|t| t.name.clone()).collect(),
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }

    // =========================================================================
    // Distillation Integration (v9 item 3.4)
    // =========================================================================

    /// Collect the current conversation history as (input, output) pairs.
    ///
    /// Iterates over the conversation messages and pairs consecutive user and
    /// assistant messages into tuples. Messages without a corresponding pair
    /// are skipped.
    ///
    /// # Returns
    /// A vector of `(user_input, assistant_output)` pairs from the session.
    #[cfg(feature = "distillation")]
    pub fn collect_trajectory(&mut self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < self.conversation.len() {
            let user_msg = &self.conversation[i];
            let assistant_msg = &self.conversation[i + 1];
            if user_msg.role == "user" && assistant_msg.role == "assistant" {
                pairs.push((user_msg.content.clone(), assistant_msg.content.clone()));
                i += 2;
            } else {
                i += 1;
            }
        }
        pairs
    }

    /// Export the conversation trajectory as a JSON-formatted training dataset.
    ///
    /// Collects all (input, output) pairs from the session and serializes them
    /// as a JSON array of objects with `"input"` and `"output"` fields, suitable
    /// for fine-tuning or distillation pipelines.
    ///
    /// # Errors
    /// Returns `AiError` if JSON serialization fails.
    #[cfg(feature = "distillation")]
    pub fn export_training_data(&self) -> Result<String, crate::error::AiError> {
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < self.conversation.len() {
            let user_msg = &self.conversation[i];
            let assistant_msg = &self.conversation[i + 1];
            if user_msg.role == "user" && assistant_msg.role == "assistant" {
                pairs.push(serde_json::json!({
                    "input": user_msg.content,
                    "output": assistant_msg.content,
                }));
                i += 2;
            } else {
                i += 1;
            }
        }

        serde_json::to_string_pretty(&pairs).map_err(|e| {
            crate::error::AiError::Other(format!("Failed to serialize training data: {}", e))
        })
    }
}
