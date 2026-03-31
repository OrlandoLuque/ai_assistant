//! Agent Methodology — defines HOW an agent approaches tasks.
//!
//! Controls the workflow phases (ANALYZE→PLAN→VALIDATE→EXECUTE→REVIEW→CONCLUDE),
//! reasoning strategies, review triggers, quality gates, recovery strategies,
//! and communication style. Serializable as JSON for sharing between users.

use serde::{Deserialize, Serialize};

// ============================================================================
// Core Methodology
// ============================================================================

/// Complete methodology definition for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentMethodology {
    /// Name of this methodology (for presets).
    pub name: String,
    /// How to approach tasks.
    pub approach: TaskApproach,
    /// How to reason before acting.
    pub reasoning: ReasoningStrategy,
    /// When and how to create plans.
    pub planning: PlanningPolicy,
    /// When to review work.
    pub review: ReviewPolicy,
    /// What to do when things fail.
    pub recovery: RecoveryStrategy,
    /// How to communicate with the user.
    pub communication: CommunicationStyle,
    /// Risk tolerance for decision making.
    pub risk_tolerance: RiskTolerance,
    /// Workflow phases configuration.
    pub workflow: WorkflowProtocol,
    /// Review trigger conditions.
    pub review_triggers: ReviewTriggers,
    /// Whether this methodology is immutable once the agent starts.
    pub immutable_after_start: bool,
}

/// How the agent approaches a task at the highest level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TaskApproach {
    /// Always plan before acting. Safest.
    PlanFirst,
    /// Act immediately, plan only if stuck.
    ActFirst,
    /// Alternate between planning and acting in short cycles.
    Iterative,
}

/// How the agent reasons before making decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReasoningStrategy {
    /// Step-by-step chain of thought.
    ChainOfThought,
    /// Step back, consider the bigger picture, then proceed.
    StepBack,
    /// Act, observe result, reflect, adjust.
    Reflection,
    /// Minimal reasoning, act directly.
    Direct,
}

/// When to create task plans.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PlanningPolicy {
    /// Always create a plan, no matter how simple.
    Always,
    /// Only plan if estimated complexity exceeds threshold.
    OnlyIfComplex { complexity_threshold: f32 },
    /// Never plan, just execute.
    Never,
}

/// When to review completed work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReviewPolicy {
    /// Review after every phase.
    AfterEachPhase,
    /// Review only at milestones (plan step completions).
    AfterMilestones,
    /// Review only at the end.
    AtEnd,
    /// Never review (fast mode).
    Never,
}

/// What to do when an operation fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RecoveryStrategy {
    /// Retry the same operation (with max attempts).
    Retry { max_attempts: usize },
    /// Try an alternative tool or approach.
    AlternativeTool,
    /// Ask the user what to do.
    AskUser,
    /// Abort the task immediately.
    Abort,
    /// Chain: try retry, then alternative, then ask user.
    Cascade {
        retry_attempts: usize,
        then_alternative: bool,
        then_ask_user: bool,
    },
}

/// How the agent communicates with the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommunicationStyle {
    /// Minimal output, only results.
    Concise,
    /// Balanced explanations.
    Balanced,
    /// Detailed step-by-step explanations.
    Detailed,
    /// Ask for confirmation frequently.
    AskOften,
    /// Report only at the end.
    ReportAtEnd,
}

/// How much risk the agent is willing to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RiskTolerance {
    /// Always confirm before destructive actions.
    Conservative,
    /// Confirm for high-risk, proceed for low-risk.
    Balanced,
    /// Proceed unless explicitly dangerous.
    Bold,
}

// ============================================================================
// Workflow Protocol
// ============================================================================

/// Defines which workflow phases are active and their configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowProtocol {
    /// ANALYZE: understand the task, classify complexity.
    pub analyze: PhaseConfig,
    /// PLAN: decompose into steps if needed.
    pub plan: PhaseConfig,
    /// VALIDATE: verify plan makes sense before executing.
    pub validate: PhaseConfig,
    /// EXECUTE: perform the work.
    pub execute: PhaseConfig,
    /// REVIEW: check results after execution.
    pub review: PhaseConfig,
    /// CONCLUDE: cleanup, summary, persist state.
    pub conclude: PhaseConfig,
}

/// Configuration for a single workflow phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Whether this phase is enabled.
    pub enabled: bool,
    /// Whether this phase is mandatory (cannot be skipped).
    pub mandatory: bool,
    /// Maximum time allowed for this phase (seconds, 0 = unlimited).
    pub max_duration_secs: u64,
}

impl PhaseConfig {
    pub fn mandatory() -> Self {
        Self {
            enabled: true,
            mandatory: true,
            max_duration_secs: 0,
        }
    }

    pub fn optional() -> Self {
        Self {
            enabled: true,
            mandatory: false,
            max_duration_secs: 0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            mandatory: false,
            max_duration_secs: 0,
        }
    }
}

// ============================================================================
// Review Triggers
// ============================================================================

/// Conditions that trigger a review cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewTriggers {
    /// Review after every N iterations.
    pub after_n_iterations: Option<usize>,
    /// Review after completing a plan milestone.
    pub after_milestone: bool,
    /// Review after any tool failure.
    pub after_tool_failure: bool,
    /// Review if cumulative cost exceeds this (USD).
    pub after_cost_threshold: Option<f64>,
    /// Review if elapsed time exceeds this (seconds).
    pub after_time_threshold_secs: Option<u64>,
    /// Review when a new user message arrives mid-execution.
    pub on_user_interrupt: bool,
    /// Periodic self-evaluation of progress.
    pub periodic_self_check: bool,
    /// Self-check interval (iterations).
    pub self_check_interval: usize,
}

impl Default for ReviewTriggers {
    fn default() -> Self {
        Self {
            after_n_iterations: Some(10),
            after_milestone: true,
            after_tool_failure: true,
            after_cost_threshold: Some(0.50),
            after_time_threshold_secs: Some(300),
            on_user_interrupt: true,
            periodic_self_check: false,
            self_check_interval: 5,
        }
    }
}

// ============================================================================
// Quality Gates
// ============================================================================

/// A quality check that must pass before proceeding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    /// Gate name for diagnostics.
    pub name: String,
    /// What to check.
    pub check: GateCheck,
    /// What to do if the check fails.
    pub on_fail: GateAction,
}

/// Types of quality checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GateCheck {
    /// Output must not be empty.
    OutputNotEmpty,
    /// Output must contain certain keywords.
    ContainsKeywords { keywords: Vec<String> },
    /// No errors were logged during execution.
    NoErrors,
    /// Cumulative cost within budget.
    CostWithinBudget { max_usd: f64 },
    /// Elapsed time within limit.
    TimeWithinLimit { max_secs: u64 },
    /// LLM evaluates quality (expensive but precise).
    LlmJudge { criteria: String, min_score: f32 },
}

/// Action to take when a quality gate fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GateAction {
    /// Retry the phase (up to max attempts).
    Retry { max_attempts: usize },
    /// Ask the user what to do.
    AskUser,
    /// Abort the task.
    Abort,
    /// Continue with a warning.
    ContinueWithWarning { message: String },
}

// ============================================================================
// Presets
// ============================================================================

impl AgentMethodology {
    /// Careful preset — all phases mandatory, review every step, conservative.
    pub fn careful() -> Self {
        Self {
            name: "Careful".to_string(),
            approach: TaskApproach::PlanFirst,
            reasoning: ReasoningStrategy::Reflection,
            planning: PlanningPolicy::Always,
            review: ReviewPolicy::AfterEachPhase,
            recovery: RecoveryStrategy::Cascade {
                retry_attempts: 2,
                then_alternative: true,
                then_ask_user: true,
            },
            communication: CommunicationStyle::Detailed,
            risk_tolerance: RiskTolerance::Conservative,
            workflow: WorkflowProtocol {
                analyze: PhaseConfig::mandatory(),
                plan: PhaseConfig::mandatory(),
                validate: PhaseConfig::mandatory(),
                execute: PhaseConfig::mandatory(),
                review: PhaseConfig::mandatory(),
                conclude: PhaseConfig::mandatory(),
            },
            review_triggers: ReviewTriggers {
                after_n_iterations: Some(5),
                after_milestone: true,
                after_tool_failure: true,
                after_cost_threshold: Some(0.10),
                after_time_threshold_secs: Some(60),
                on_user_interrupt: true,
                periodic_self_check: true,
                self_check_interval: 3,
            },
            immutable_after_start: true,
        }
    }

    /// Balanced preset — plan if complex, review at milestones.
    pub fn balanced() -> Self {
        Self {
            name: "Balanced".to_string(),
            approach: TaskApproach::Iterative,
            reasoning: ReasoningStrategy::ChainOfThought,
            planning: PlanningPolicy::OnlyIfComplex {
                complexity_threshold: 0.5,
            },
            review: ReviewPolicy::AfterMilestones,
            recovery: RecoveryStrategy::Cascade {
                retry_attempts: 1,
                then_alternative: true,
                then_ask_user: false,
            },
            communication: CommunicationStyle::Balanced,
            risk_tolerance: RiskTolerance::Balanced,
            workflow: WorkflowProtocol {
                analyze: PhaseConfig::mandatory(),
                plan: PhaseConfig::optional(),
                validate: PhaseConfig::optional(),
                execute: PhaseConfig::mandatory(),
                review: PhaseConfig::optional(),
                conclude: PhaseConfig::mandatory(),
            },
            review_triggers: ReviewTriggers::default(),
            immutable_after_start: false,
        }
    }

    /// Fast preset — no planning, act directly, review only on failure.
    pub fn fast() -> Self {
        Self {
            name: "Fast".to_string(),
            approach: TaskApproach::ActFirst,
            reasoning: ReasoningStrategy::Direct,
            planning: PlanningPolicy::Never,
            review: ReviewPolicy::Never,
            recovery: RecoveryStrategy::Retry { max_attempts: 1 },
            communication: CommunicationStyle::Concise,
            risk_tolerance: RiskTolerance::Bold,
            workflow: WorkflowProtocol {
                analyze: PhaseConfig::disabled(),
                plan: PhaseConfig::disabled(),
                validate: PhaseConfig::disabled(),
                execute: PhaseConfig::mandatory(),
                review: PhaseConfig::disabled(),
                conclude: PhaseConfig::optional(),
            },
            review_triggers: ReviewTriggers {
                after_n_iterations: None,
                after_milestone: false,
                after_tool_failure: true,
                after_cost_threshold: None,
                after_time_threshold_secs: None,
                on_user_interrupt: true,
                periodic_self_check: false,
                self_check_interval: 0,
            },
            immutable_after_start: false,
        }
    }

    /// Research preset — always plan, reason deeply, review every source.
    pub fn research() -> Self {
        Self {
            name: "Research".to_string(),
            approach: TaskApproach::PlanFirst,
            reasoning: ReasoningStrategy::StepBack,
            planning: PlanningPolicy::Always,
            review: ReviewPolicy::AfterEachPhase,
            recovery: RecoveryStrategy::AlternativeTool,
            communication: CommunicationStyle::Detailed,
            risk_tolerance: RiskTolerance::Conservative,
            workflow: WorkflowProtocol {
                analyze: PhaseConfig::mandatory(),
                plan: PhaseConfig::mandatory(),
                validate: PhaseConfig::mandatory(),
                execute: PhaseConfig::mandatory(),
                review: PhaseConfig::mandatory(),
                conclude: PhaseConfig::mandatory(),
            },
            review_triggers: ReviewTriggers {
                after_n_iterations: Some(3),
                after_milestone: true,
                after_tool_failure: true,
                after_cost_threshold: Some(0.25),
                after_time_threshold_secs: Some(120),
                on_user_interrupt: true,
                periodic_self_check: true,
                self_check_interval: 3,
            },
            immutable_after_start: true,
        }
    }

    /// Check if a specific workflow phase should run.
    pub fn should_run_phase(&self, phase: WorkflowPhase) -> bool {
        let config = match phase {
            WorkflowPhase::Analyze => &self.workflow.analyze,
            WorkflowPhase::Plan => &self.workflow.plan,
            WorkflowPhase::Validate => &self.workflow.validate,
            WorkflowPhase::Execute => &self.workflow.execute,
            WorkflowPhase::Review => &self.workflow.review,
            WorkflowPhase::Conclude => &self.workflow.conclude,
        };
        config.enabled
    }

    /// Check if a review should be triggered based on current state.
    pub fn should_review(
        &self,
        iteration: usize,
        milestone_completed: bool,
        tool_failed: bool,
        cost_usd: f64,
        elapsed_secs: u64,
        user_interrupted: bool,
    ) -> bool {
        let t = &self.review_triggers;

        if let Some(n) = t.after_n_iterations {
            if n > 0 && iteration % n == 0 {
                return true;
            }
        }
        if t.after_milestone && milestone_completed {
            return true;
        }
        if t.after_tool_failure && tool_failed {
            return true;
        }
        if let Some(threshold) = t.after_cost_threshold {
            if cost_usd >= threshold {
                return true;
            }
        }
        if let Some(threshold) = t.after_time_threshold_secs {
            if elapsed_secs >= threshold {
                return true;
            }
        }
        if t.on_user_interrupt && user_interrupted {
            return true;
        }
        if t.periodic_self_check && t.self_check_interval > 0 {
            if iteration % t.self_check_interval == 0 {
                return true;
            }
        }

        false
    }
}

impl Default for AgentMethodology {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Workflow phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowPhase {
    Analyze,
    Plan,
    Validate,
    Execute,
    Review,
    Conclude,
}

impl std::fmt::Display for WorkflowPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analyze => write!(f, "ANALYZE"),
            Self::Plan => write!(f, "PLAN"),
            Self::Validate => write!(f, "VALIDATE"),
            Self::Execute => write!(f, "EXECUTE"),
            Self::Review => write!(f, "REVIEW"),
            Self::Conclude => write!(f, "CONCLUDE"),
        }
    }
}

// ============================================================================
// LLM Enhancement: Task Decomposition (V68)
// ============================================================================

/// A single step from LLM-based task decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// Description of the step.
    pub description: String,
    /// Estimated complexity.
    pub estimated_complexity: String,
}

/// Result of task decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDecomposition {
    /// The decomposed steps.
    pub steps: Vec<TaskStep>,
}

/// Configuration for task decomposition.
#[derive(Debug, Clone)]
pub struct TaskDecompositionConfig {
    /// Use LLM to decompose tasks into steps.
    /// When false (default), uses heuristic sentence-splitting.
    pub llm_enhanced: bool,
}

impl Default for TaskDecompositionConfig {
    fn default() -> Self {
        Self {
            llm_enhanced: false,
        }
    }
}

/// Decomposes tasks into steps, optionally enhanced by LLM.
pub struct TaskDecomposer {
    pub config: TaskDecompositionConfig,
}

impl TaskDecomposer {
    pub fn new(config: TaskDecompositionConfig) -> Self {
        Self { config }
    }

    /// Build a prompt for LLM-based task decomposition.
    ///
    /// Returns None if LLM enhancement is disabled.
    pub fn build_decomposition_prompt(&self, task: &str) -> Option<String> {
        if !self.config.llm_enhanced {
            return None;
        }

        let prompt = format!(
            "Break this task into steps. \
             Return JSON: {{\"steps\":[{{\"description\":\"...\",\"estimated_complexity\":\"low|medium|high\"}}]}}\n\n\
             Task: {}",
            crate::llm_enhance::prompt_wrap(task)
        );

        Some(prompt)
    }

    /// Parse LLM response for task decomposition.
    pub fn parse_decomposition_response(response: &str) -> Option<TaskDecomposition> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(steps_arr) = val.get("steps").and_then(|v| v.as_array()) {
                    let steps: Vec<TaskStep> = steps_arr
                        .iter()
                        .filter_map(|s| {
                            let desc = s.get("description").and_then(|d| d.as_str())?;
                            let complexity = s
                                .get("estimated_complexity")
                                .and_then(|c| c.as_str())
                                .unwrap_or("medium");
                            Some(TaskStep {
                                description: desc.to_string(),
                                estimated_complexity: complexity.to_string(),
                            })
                        })
                        .collect();
                    if !steps.is_empty() {
                        return Some(TaskDecomposition { steps });
                    }
                }
            }
        }
        None
    }

    /// Decompose a task into steps with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for
    /// intelligent decomposition. Otherwise uses simple sentence splitting.
    pub fn decompose_task_with_llm(
        &self,
        task: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> TaskDecomposition {
        // Heuristic baseline: split by sentences/periods/semicolons
        let sentences: Vec<&str> = task
            .split(|c: char| c == '.' || c == ';' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let heuristic_steps: Vec<TaskStep> = if sentences.len() <= 1 {
            vec![TaskStep {
                description: task.to_string(),
                estimated_complexity: "medium".to_string(),
            }]
        } else {
            sentences
                .iter()
                .map(|s| TaskStep {
                    description: s.to_string(),
                    estimated_complexity: if s.len() > 100 {
                        "high".to_string()
                    } else if s.len() > 40 {
                        "medium".to_string()
                    } else {
                        "low".to_string()
                    },
                })
                .collect()
        };

        let heuristic = TaskDecomposition {
            steps: heuristic_steps,
        };

        // Try LLM enhancement
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_decomposition_prompt(task) {
                    if let Ok(response) = enhancer.generate(&prompt, 500) {
                        if let Some(decomposition) = Self::parse_decomposition_response(&response) {
                            return decomposition;
                        }
                    }
                }
            }
        }

        heuristic
    }
}

impl Default for TaskDecomposer {
    fn default() -> Self {
        Self::new(TaskDecompositionConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_careful_all_phases_mandatory() {
        let m = AgentMethodology::careful();
        assert!(m.should_run_phase(WorkflowPhase::Analyze));
        assert!(m.should_run_phase(WorkflowPhase::Plan));
        assert!(m.should_run_phase(WorkflowPhase::Validate));
        assert!(m.should_run_phase(WorkflowPhase::Execute));
        assert!(m.should_run_phase(WorkflowPhase::Review));
        assert!(m.should_run_phase(WorkflowPhase::Conclude));
        assert!(m.immutable_after_start);
    }

    #[test]
    fn test_fast_skips_most_phases() {
        let m = AgentMethodology::fast();
        assert!(!m.should_run_phase(WorkflowPhase::Analyze));
        assert!(!m.should_run_phase(WorkflowPhase::Plan));
        assert!(!m.should_run_phase(WorkflowPhase::Validate));
        assert!(m.should_run_phase(WorkflowPhase::Execute));
        assert!(!m.should_run_phase(WorkflowPhase::Review));
    }

    #[test]
    fn test_review_trigger_iterations() {
        let m = AgentMethodology::careful();
        // careful has after_n_iterations=5, self_check_interval=3
        assert!(m.should_review(5, false, false, 0.0, 0, false)); // 5 % 5 == 0
        assert!(m.should_review(3, false, false, 0.0, 0, false)); // 3 % 3 == 0 (self-check)
        assert!(!m.should_review(1, false, false, 0.0, 0, false)); // 1 % 5 != 0, 1 % 3 != 0
        assert!(!m.should_review(4, false, false, 0.0, 0, false)); // 4 % 5 != 0, 4 % 3 != 0
    }

    #[test]
    fn test_review_trigger_milestone() {
        let m = AgentMethodology::balanced();
        assert!(m.should_review(1, true, false, 0.0, 0, false));
    }

    #[test]
    fn test_review_trigger_tool_failure() {
        let m = AgentMethodology::balanced();
        assert!(m.should_review(1, false, true, 0.0, 0, false));
    }

    #[test]
    fn test_review_trigger_cost() {
        let m = AgentMethodology::balanced();
        assert!(m.should_review(1, false, false, 0.60, 0, false));
        assert!(!m.should_review(1, false, false, 0.10, 0, false));
    }

    #[test]
    fn test_review_trigger_time() {
        let m = AgentMethodology::balanced();
        assert!(m.should_review(1, false, false, 0.0, 400, false));
    }

    #[test]
    fn test_review_trigger_user_interrupt() {
        let m = AgentMethodology::balanced();
        assert!(m.should_review(1, false, false, 0.0, 0, true));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let m = AgentMethodology::careful();
        let json = serde_json::to_string(&m).unwrap();
        let restored: AgentMethodology = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "Careful");
        assert!(restored.immutable_after_start);
    }

    #[test]
    fn test_default_is_balanced() {
        let m = AgentMethodology::default();
        assert_eq!(m.name, "Balanced");
    }

    #[test]
    fn test_quality_gate_check_variants() {
        let gate = QualityGate {
            name: "output_check".to_string(),
            check: GateCheck::OutputNotEmpty,
            on_fail: GateAction::Retry { max_attempts: 3 },
        };
        assert_eq!(gate.name, "output_check");
    }

    #[test]
    fn test_workflow_phase_display() {
        assert_eq!(WorkflowPhase::Analyze.to_string(), "ANALYZE");
        assert_eq!(WorkflowPhase::Conclude.to_string(), "CONCLUDE");
    }

    #[test]
    fn test_research_preset() {
        let m = AgentMethodology::research();
        assert_eq!(m.approach, TaskApproach::PlanFirst);
        assert_eq!(m.reasoning, ReasoningStrategy::StepBack);
        assert!(m.immutable_after_start);
    }

    // ── V68: LLM Enhancement tests ──────────────────────────────────

    #[test]
    fn test_decompose_task_heuristic_without_llm() {
        let config = TaskDecompositionConfig {
            llm_enhanced: false,
        };
        let decomposer = TaskDecomposer::new(config);
        let result = decomposer.decompose_task_with_llm(
            "Analyze the data. Build a report. Send it to the team",
            None,
        );
        assert_eq!(result.steps.len(), 3);
        assert!(result.steps[0].description.contains("Analyze"));
    }

    #[test]
    fn test_decompose_task_with_mock_llm() {
        let config = TaskDecompositionConfig {
            llm_enhanced: true,
        };
        let decomposer = TaskDecomposer::new(config);
        let mock = crate::llm_enhance::MockLlm::new(
            "{\"steps\":[{\"description\":\"Gather requirements\",\"estimated_complexity\":\"low\"},{\"description\":\"Implement solution\",\"estimated_complexity\":\"high\"}]}",
        );
        let result = decomposer.decompose_task_with_llm(
            "Build a web application",
            Some(&mock),
        );
        assert_eq!(result.steps.len(), 2, "Expected 2 LLM steps, got: {}", result.steps.len());
        assert!(result.steps[0].description.contains("Gather"));
        assert_eq!(result.steps[1].estimated_complexity, "high");
    }

    #[test]
    fn test_decompose_task_llm_fallback_on_failure() {
        let config = TaskDecompositionConfig {
            llm_enhanced: true,
        };
        let decomposer = TaskDecomposer::new(config);
        let failing = crate::llm_enhance::FailingMockLlm;
        let result = decomposer.decompose_task_with_llm(
            "Do something complex",
            Some(&failing),
        );
        // Should fall back to heuristic (not crash)
        assert!(!result.steps.is_empty());
        assert!(result.steps[0].description.contains("Do something complex"));
    }
}
