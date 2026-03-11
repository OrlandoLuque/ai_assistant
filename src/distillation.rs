//! Trace-to-Distillation Pipeline: trajectory collection, scoring, dataset building,
//! and data flywheel orchestration.
//!
//! Implements v5 roadmap Phase 5 (items 5.1, 5.2, 5.3, 5.4):
//! - 5.1 Agent Trajectory Collector
//! - 5.2 Trajectory Scorer & Filter
//! - 5.3 Distillation Dataset Builder
//! - 5.4 Data Flywheel Orchestrator
//!
//! Feature-gated behind `distillation`. The outer `#[cfg]` guard ensures this
//! entire module compiles away when the feature is not enabled.

#[cfg(feature = "distillation")]
mod inner {
    use std::collections::{HashMap, HashSet};
    use std::fmt;

    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    use crate::error::{AiError, DistillationError};

    // ========================================================================
    // 5.1 — Agent Trajectory Collector
    // ========================================================================

    /// Unique identifier for a trajectory being collected.
    pub type TrajectoryId = String;

    /// The type of action performed in a trajectory step.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum StepType {
        /// An LLM inference call.
        LlmCall {
            model: String,
            temperature: f32,
        },
        /// A tool invocation.
        ToolUse {
            tool_name: String,
        },
        /// A decision point where one option was chosen among alternatives.
        Decision {
            options: Vec<String>,
            chosen: String,
        },
        /// An observation from the environment.
        Observation {
            source: String,
        },
    }

    impl fmt::Display for StepType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                StepType::LlmCall { model, .. } => write!(f, "llm_call:{}", model),
                StepType::ToolUse { tool_name } => write!(f, "tool_use:{}", tool_name),
                StepType::Decision { chosen, .. } => write!(f, "decision:{}", chosen),
                StepType::Observation { source } => write!(f, "observation:{}", source),
            }
        }
    }

    /// The final outcome of a trajectory.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum TrajectoryOutcome {
        /// The task completed successfully.
        Success { score: f64 },
        /// The task failed.
        Failure { reason: String },
        /// The task partially completed.
        Partial { score: f64, reason: String },
    }

    /// A single step within a trajectory.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrajectoryStep {
        pub step_number: usize,
        pub step_type: StepType,
        pub input: String,
        pub output: String,
        pub tokens_used: usize,
        pub latency_ms: u64,
        pub timestamp: DateTime<Utc>,
        pub metadata: HashMap<String, String>,
    }

    /// A complete agent trajectory from task start to outcome.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Trajectory {
        pub id: String,
        pub agent_id: String,
        pub task_description: String,
        pub steps: Vec<TrajectoryStep>,
        pub outcome: TrajectoryOutcome,
        pub total_tokens: usize,
        pub total_latency_ms: u64,
        pub created_at: DateTime<Utc>,
        pub metadata: HashMap<String, String>,
    }

    /// Collects agent trajectories as they execute.
    #[derive(Debug)]
    pub struct TrajectoryCollector {
        /// In-progress trajectories (not yet finished).
        active: HashMap<TrajectoryId, Trajectory>,
        /// Completed trajectories.
        completed: Vec<Trajectory>,
    }

    impl TrajectoryCollector {
        /// Create a new empty collector.
        pub fn new() -> Self {
            Self {
                active: HashMap::new(),
                completed: Vec::new(),
            }
        }

        /// Start collecting a new trajectory for the given agent and task.
        /// Returns the trajectory ID.
        pub fn start_trajectory(&mut self, agent_id: &str, task: &str) -> TrajectoryId {
            let id = Uuid::new_v4().to_string();
            let trajectory = Trajectory {
                id: id.clone(),
                agent_id: agent_id.to_string(),
                task_description: task.to_string(),
                steps: Vec::new(),
                outcome: TrajectoryOutcome::Failure {
                    reason: "in progress".to_string(),
                },
                total_tokens: 0,
                total_latency_ms: 0,
                created_at: Utc::now(),
                metadata: HashMap::new(),
            };
            self.active.insert(id.clone(), trajectory);
            id
        }

        /// Add a step to an active trajectory.
        pub fn add_step(&mut self, id: &str, step: TrajectoryStep) -> Result<(), AiError> {
            let trajectory = self.active.get_mut(id).ok_or_else(|| {
                AiError::Distillation(DistillationError::CollectionFailed {
                    reason: format!("Trajectory '{}' not found or already finished", id),
                })
            })?;
            trajectory.total_tokens += step.tokens_used;
            trajectory.total_latency_ms += step.latency_ms;
            trajectory.steps.push(step);
            Ok(())
        }

        /// Finish an active trajectory with the given outcome, moving it to completed.
        pub fn finish_trajectory(
            &mut self,
            id: &str,
            outcome: TrajectoryOutcome,
        ) -> Result<(), AiError> {
            let mut trajectory = self.active.remove(id).ok_or_else(|| {
                AiError::Distillation(DistillationError::CollectionFailed {
                    reason: format!("Trajectory '{}' not found or already finished", id),
                })
            })?;
            trajectory.outcome = outcome;
            self.completed.push(trajectory);
            Ok(())
        }

        /// Get a reference to a completed trajectory by ID.
        pub fn get_trajectory(&self, id: &str) -> Option<&Trajectory> {
            self.completed.iter().find(|t| t.id == id)
        }

        /// List all completed trajectories.
        pub fn list_trajectories(&self) -> Vec<&Trajectory> {
            self.completed.iter().collect()
        }

        /// Remove all completed trajectories.
        pub fn clear(&mut self) {
            self.completed.clear();
        }
    }

    // ========================================================================
    // Trajectory Stores
    // ========================================================================

    /// Persistent storage backend for trajectories.
    pub trait TrajectoryStore: Send + Sync {
        /// Save a trajectory to the store.
        fn save(&mut self, trajectory: &Trajectory) -> Result<(), AiError>;
        /// Load a trajectory by ID.
        fn load(&self, id: &str) -> Result<Option<Trajectory>, AiError>;
        /// List all stored trajectory IDs.
        fn list(&self) -> Result<Vec<String>, AiError>;
        /// Delete a trajectory by ID.
        fn delete(&mut self, id: &str) -> Result<(), AiError>;
        /// Return the number of stored trajectories.
        fn count(&self) -> usize;
    }

    /// In-memory trajectory store backed by a HashMap.
    #[derive(Debug)]
    pub struct InMemoryTrajectoryStore {
        data: HashMap<String, Trajectory>,
    }

    impl InMemoryTrajectoryStore {
        pub fn new() -> Self {
            Self {
                data: HashMap::new(),
            }
        }
    }

    impl TrajectoryStore for InMemoryTrajectoryStore {
        fn save(&mut self, trajectory: &Trajectory) -> Result<(), AiError> {
            self.data.insert(trajectory.id.clone(), trajectory.clone());
            Ok(())
        }

        fn load(&self, id: &str) -> Result<Option<Trajectory>, AiError> {
            Ok(self.data.get(id).cloned())
        }

        fn list(&self) -> Result<Vec<String>, AiError> {
            Ok(self.data.keys().cloned().collect())
        }

        fn delete(&mut self, id: &str) -> Result<(), AiError> {
            self.data.remove(id);
            Ok(())
        }

        fn count(&self) -> usize {
            self.data.len()
        }
    }

    /// Append-only JSONL file store for trajectories.
    ///
    /// Each trajectory is stored as a single JSON line. On load, the file is
    /// scanned line-by-line. This is simple and efficient for append workloads.
    #[derive(Debug)]
    pub struct JsonlTrajectoryStore {
        path: String,
        /// In-memory index built on construction / after writes.
        index: HashMap<String, Trajectory>,
    }

    impl JsonlTrajectoryStore {
        /// Create or open a JSONL trajectory store at the given path.
        pub fn new(path: &str) -> Result<Self, AiError> {
            let mut store = Self {
                path: path.to_string(),
                index: HashMap::new(),
            };
            store.rebuild_index()?;
            Ok(store)
        }

        /// Rebuild the in-memory index by reading the entire file.
        /// Rejects files larger than 100 MB to prevent OOM on corrupted/bloated stores.
        fn rebuild_index(&mut self) -> Result<(), AiError> {
            self.index.clear();
            // Check file size before reading to prevent unbounded memory allocation
            if let Ok(meta) = std::fs::metadata(&self.path) {
                const MAX_STORE_SIZE: u64 = 100 * 1024 * 1024; // 100 MB
                if meta.len() > MAX_STORE_SIZE {
                    return Err(AiError::Distillation(DistillationError::StorageError {
                        operation: "read".to_string(),
                        reason: format!(
                            "Trajectory store too large ({} bytes, max {})",
                            meta.len(),
                            MAX_STORE_SIZE
                        ),
                    }));
                }
            }
            let content = match std::fs::read_to_string(&self.path) {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => {
                    return Err(AiError::Distillation(DistillationError::StorageError {
                        operation: "read".to_string(),
                        reason: e.to_string(),
                    }));
                }
            };
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let trajectory: Trajectory = serde_json::from_str(trimmed).map_err(|e| {
                    AiError::Distillation(DistillationError::StorageError {
                        operation: "deserialize".to_string(),
                        reason: e.to_string(),
                    })
                })?;
                self.index.insert(trajectory.id.clone(), trajectory);
            }
            Ok(())
        }
    }

    impl TrajectoryStore for JsonlTrajectoryStore {
        fn save(&mut self, trajectory: &Trajectory) -> Result<(), AiError> {
            use std::io::Write;
            let json = serde_json::to_string(trajectory).map_err(|e| {
                AiError::Distillation(DistillationError::StorageError {
                    operation: "serialize".to_string(),
                    reason: e.to_string(),
                })
            })?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(|e| {
                    AiError::Distillation(DistillationError::StorageError {
                        operation: "open".to_string(),
                        reason: e.to_string(),
                    })
                })?;
            writeln!(file, "{}", json).map_err(|e| {
                AiError::Distillation(DistillationError::StorageError {
                    operation: "write".to_string(),
                    reason: e.to_string(),
                })
            })?;
            self.index.insert(trajectory.id.clone(), trajectory.clone());
            Ok(())
        }

        fn load(&self, id: &str) -> Result<Option<Trajectory>, AiError> {
            Ok(self.index.get(id).cloned())
        }

        fn list(&self) -> Result<Vec<String>, AiError> {
            Ok(self.index.keys().cloned().collect())
        }

        fn delete(&mut self, id: &str) -> Result<(), AiError> {
            self.index.remove(id);
            // Rewrite the file without the deleted trajectory
            let entries: Vec<&Trajectory> = self.index.values().collect();
            let mut content = String::new();
            for t in entries {
                let line = serde_json::to_string(t).map_err(|e| {
                    AiError::Distillation(DistillationError::StorageError {
                        operation: "serialize".to_string(),
                        reason: e.to_string(),
                    })
                })?;
                content.push_str(&line);
                content.push('\n');
            }
            std::fs::write(&self.path, content).map_err(|e| {
                AiError::Distillation(DistillationError::StorageError {
                    operation: "rewrite".to_string(),
                    reason: e.to_string(),
                })
            })?;
            Ok(())
        }

        fn count(&self) -> usize {
            self.index.len()
        }
    }

    // ========================================================================
    // 5.2 — Trajectory Scorer & Filter
    // ========================================================================

    /// Scores a trajectory on a 0.0–1.0 scale.
    pub trait TrajectoryScorer: Send + Sync {
        /// Score a trajectory. Returns a value between 0.0 and 1.0.
        fn score(&self, trajectory: &Trajectory) -> f64;
        /// The name of this scorer (for logging/debugging).
        fn name(&self) -> &str;
    }

    /// Scores based on the final outcome: Success=1.0, Failure=0.0, Partial=score.
    #[derive(Debug, Clone)]
    pub struct OutcomeScorer;

    impl TrajectoryScorer for OutcomeScorer {
        fn score(&self, trajectory: &Trajectory) -> f64 {
            match &trajectory.outcome {
                TrajectoryOutcome::Success { score } => score.clamp(0.0, 1.0),
                TrajectoryOutcome::Failure { .. } => 0.0,
                TrajectoryOutcome::Partial { score, .. } => score.clamp(0.0, 1.0),
            }
        }

        fn name(&self) -> &str {
            "outcome"
        }
    }

    /// Penalizes long trajectories based on step count and token usage.
    #[derive(Debug, Clone)]
    pub struct EfficiencyScorer {
        /// Maximum number of steps before score reaches 0. Default: 20.
        pub max_steps: usize,
        /// Penalty per 1000 tokens used. Default: 0.1.
        pub token_penalty_per_1k: f64,
    }

    impl EfficiencyScorer {
        pub fn new() -> Self {
            Self {
                max_steps: 20,
                token_penalty_per_1k: 0.1,
            }
        }

        pub fn with_max_steps(mut self, max_steps: usize) -> Self {
            self.max_steps = max_steps;
            self
        }

        pub fn with_token_penalty(mut self, penalty: f64) -> Self {
            self.token_penalty_per_1k = penalty;
            self
        }
    }

    impl TrajectoryScorer for EfficiencyScorer {
        fn score(&self, trajectory: &Trajectory) -> f64 {
            let step_score = 1.0
                - (trajectory.steps.len() as f64 / self.max_steps as f64).min(1.0);
            let token_penalty =
                (trajectory.total_tokens as f64 / 1000.0) * self.token_penalty_per_1k;
            (step_score - token_penalty).clamp(0.0, 1.0)
        }

        fn name(&self) -> &str {
            "efficiency"
        }
    }

    /// Scores based on diversity of step types and tools used.
    /// More diverse trajectories (using different tools and step types) score higher.
    #[derive(Debug, Clone)]
    pub struct DiversityScorer;

    impl TrajectoryScorer for DiversityScorer {
        fn score(&self, trajectory: &Trajectory) -> f64 {
            if trajectory.steps.is_empty() {
                return 0.0;
            }
            let mut step_type_names: HashSet<String> = HashSet::new();
            let mut tool_names: HashSet<String> = HashSet::new();

            for step in &trajectory.steps {
                match &step.step_type {
                    StepType::LlmCall { .. } => {
                        step_type_names.insert("llm_call".to_string());
                    }
                    StepType::ToolUse { tool_name } => {
                        step_type_names.insert("tool_use".to_string());
                        tool_names.insert(tool_name.clone());
                    }
                    StepType::Decision { .. } => {
                        step_type_names.insert("decision".to_string());
                    }
                    StepType::Observation { .. } => {
                        step_type_names.insert("observation".to_string());
                    }
                }
            }

            // Diversity = (unique types + unique tools) / (4 types + step count as upper bound)
            let unique_count = step_type_names.len() + tool_names.len();
            let max_possible = 4 + trajectory.steps.len();
            (unique_count as f64 / max_possible as f64).min(1.0)
        }

        fn name(&self) -> &str {
            "diversity"
        }
    }

    /// Combines multiple scorers with configurable weights.
    pub struct CompositeScorer {
        scorers: Vec<(f64, Box<dyn TrajectoryScorer>)>,
    }

    impl CompositeScorer {
        pub fn new() -> Self {
            Self {
                scorers: Vec::new(),
            }
        }

        /// Add a scorer with the given weight.
        pub fn add(mut self, weight: f64, scorer: Box<dyn TrajectoryScorer>) -> Self {
            self.scorers.push((weight, scorer));
            self
        }
    }

    impl fmt::Debug for CompositeScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CompositeScorer")
                .field(
                    "scorers",
                    &self
                        .scorers
                        .iter()
                        .map(|(w, s)| format!("{}:{:.2}", s.name(), w))
                        .collect::<Vec<_>>(),
                )
                .finish()
        }
    }

    impl TrajectoryScorer for CompositeScorer {
        fn score(&self, trajectory: &Trajectory) -> f64 {
            if self.scorers.is_empty() {
                return 0.0;
            }
            let total_weight: f64 = self.scorers.iter().map(|(w, _)| w).sum();
            if total_weight <= 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = self
                .scorers
                .iter()
                .map(|(w, s)| w * s.score(trajectory))
                .sum();
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        }

        fn name(&self) -> &str {
            "composite"
        }
    }

    /// Which outcome types are required to pass the filter.
    #[derive(Debug, Clone, PartialEq)]
    pub enum RequiredOutcome {
        /// Only successful trajectories.
        SuccessOnly,
        /// Only failed trajectories.
        FailureOnly,
        /// Any outcome.
        Any,
    }

    /// Filters trajectories based on score, step count, token count, and outcome type.
    pub struct TrajectoryFilter {
        pub min_score: f64,
        pub max_steps: Option<usize>,
        pub max_tokens: Option<usize>,
        pub required_outcome: Option<RequiredOutcome>,
        scorers: Vec<Box<dyn TrajectoryScorer>>,
    }

    impl TrajectoryFilter {
        pub fn new(min_score: f64) -> Self {
            Self {
                min_score,
                max_steps: None,
                max_tokens: None,
                required_outcome: None,
                scorers: Vec::new(),
            }
        }

        pub fn with_max_steps(mut self, max: usize) -> Self {
            self.max_steps = Some(max);
            self
        }

        pub fn with_max_tokens(mut self, max: usize) -> Self {
            self.max_tokens = Some(max);
            self
        }

        pub fn with_required_outcome(mut self, outcome: RequiredOutcome) -> Self {
            self.required_outcome = Some(outcome);
            self
        }

        pub fn with_scorer(mut self, scorer: Box<dyn TrajectoryScorer>) -> Self {
            self.scorers.push(scorer);
            self
        }

        /// Compute the aggregate score for a trajectory using all configured scorers.
        fn compute_score(&self, trajectory: &Trajectory) -> f64 {
            if self.scorers.is_empty() {
                return 1.0; // No scorers means all pass score check
            }
            let sum: f64 = self.scorers.iter().map(|s| s.score(trajectory)).sum();
            sum / self.scorers.len() as f64
        }

        /// Check if the trajectory's outcome matches the required outcome.
        fn matches_outcome(&self, trajectory: &Trajectory) -> bool {
            match &self.required_outcome {
                None => true,
                Some(RequiredOutcome::Any) => true,
                Some(RequiredOutcome::SuccessOnly) => {
                    matches!(&trajectory.outcome, TrajectoryOutcome::Success { .. })
                }
                Some(RequiredOutcome::FailureOnly) => {
                    matches!(&trajectory.outcome, TrajectoryOutcome::Failure { .. })
                }
            }
        }

        /// Filter the given trajectories, returning references to those that pass all criteria.
        pub fn filter<'a>(&self, trajectories: &'a [Trajectory]) -> Vec<&'a Trajectory> {
            trajectories
                .iter()
                .filter(|t| {
                    // Check outcome requirement
                    if !self.matches_outcome(t) {
                        return false;
                    }
                    // Check max steps
                    if let Some(max) = self.max_steps {
                        if t.steps.len() > max {
                            return false;
                        }
                    }
                    // Check max tokens
                    if let Some(max) = self.max_tokens {
                        if t.total_tokens > max {
                            return false;
                        }
                    }
                    // Check score
                    let score = self.compute_score(t);
                    score >= self.min_score
                })
                .collect()
        }
    }

    impl fmt::Debug for TrajectoryFilter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TrajectoryFilter")
                .field("min_score", &self.min_score)
                .field("max_steps", &self.max_steps)
                .field("max_tokens", &self.max_tokens)
                .field("required_outcome", &self.required_outcome)
                .field("scorers_count", &self.scorers.len())
                .finish()
        }
    }

    /// A filtered dataset of trajectories, ready for dataset building.
    #[derive(Debug, Clone)]
    pub struct TrajectoryDataset {
        trajectories: Vec<Trajectory>,
    }

    impl TrajectoryDataset {
        /// Create a dataset from a vector of trajectories.
        pub fn new(trajectories: Vec<Trajectory>) -> Self {
            Self { trajectories }
        }

        /// Number of trajectories in the dataset.
        pub fn len(&self) -> usize {
            self.trajectories.len()
        }

        /// Whether the dataset is empty.
        pub fn is_empty(&self) -> bool {
            self.trajectories.is_empty()
        }

        /// Iterate over trajectories.
        pub fn iter(&self) -> impl Iterator<Item = &Trajectory> {
            self.trajectories.iter()
        }

        /// Split into train and test sets. `ratio` is the fraction for training (0.0–1.0).
        pub fn split_train_test(self, ratio: f64) -> (Self, Self) {
            let ratio = ratio.clamp(0.0, 1.0);
            let split_index = (self.trajectories.len() as f64 * ratio).round() as usize;
            let (train, test) = self.trajectories.split_at(
                split_index.min(self.trajectories.len()),
            );
            (Self::new(train.to_vec()), Self::new(test.to_vec()))
        }

        /// Shuffle the dataset using a simple Fisher-Yates shuffle with a basic RNG.
        pub fn shuffle(&mut self) {
            // Use a simple deterministic-ish shuffle based on trajectory IDs and timestamps.
            // For production use, a proper RNG would be better, but we avoid extra deps.
            let len = self.trajectories.len();
            if len <= 1 {
                return;
            }
            // Simple hash-based shuffle: sort by hash of (id + index)
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut indices: Vec<usize> = (0..len).collect();
            indices.sort_by(|a, b| {
                let mut ha = DefaultHasher::new();
                self.trajectories[*a].id.hash(&mut ha);
                a.hash(&mut ha);
                let hash_a = ha.finish();

                let mut hb = DefaultHasher::new();
                self.trajectories[*b].id.hash(&mut hb);
                b.hash(&mut hb);
                let hash_b = hb.finish();

                hash_a.cmp(&hash_b)
            });
            let original = self.trajectories.clone();
            for (new_pos, &old_pos) in indices.iter().enumerate() {
                self.trajectories[new_pos] = original[old_pos].clone();
            }
        }

        /// Remove duplicate trajectories (by ID).
        pub fn dedup(&mut self) {
            let mut seen = HashSet::new();
            self.trajectories.retain(|t| seen.insert(t.id.clone()));
        }

        /// Get the underlying trajectories.
        pub fn trajectories(&self) -> &[Trajectory] {
            &self.trajectories
        }
    }

    // ========================================================================
    // 5.3 — Distillation Dataset Builder
    // ========================================================================

    /// The output format for the fine-tuning dataset.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum DatasetFormat {
        /// OpenAI fine-tuning JSONL (messages array).
        OpenAIJsonl,
        /// Stanford Alpaca format (instruction, input, output).
        Alpaca,
        /// ShareGPT multi-turn conversation format.
        ShareGPT,
        /// Custom format with a user-supplied template.
        Custom { template: String },
    }

    impl fmt::Display for DatasetFormat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                DatasetFormat::OpenAIJsonl => write!(f, "openai_jsonl"),
                DatasetFormat::Alpaca => write!(f, "alpaca"),
                DatasetFormat::ShareGPT => write!(f, "sharegpt"),
                DatasetFormat::Custom { .. } => write!(f, "custom"),
            }
        }
    }

    /// How to flatten a multi-step trajectory into a training example.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum FlatteningStrategy {
        /// Use only the last LLM response as output.
        LastStepOnly,
        /// Represent all steps as a multi-turn conversation.
        AllSteps,
        /// Concatenate all step outputs into a single summary.
        SummaryOnly,
    }

    /// Configuration for dataset building.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DatasetConfig {
        pub format: DatasetFormat,
        pub system_prompt: Option<String>,
        pub max_examples: Option<usize>,
        pub include_tool_calls: bool,
        pub flattening: FlatteningStrategy,
    }

    impl DatasetConfig {
        pub fn new(format: DatasetFormat) -> Self {
            Self {
                format,
                system_prompt: None,
                max_examples: None,
                include_tool_calls: true,
                flattening: FlatteningStrategy::LastStepOnly,
            }
        }

        pub fn with_system_prompt(mut self, prompt: &str) -> Self {
            self.system_prompt = Some(prompt.to_string());
            self
        }

        pub fn with_max_examples(mut self, max: usize) -> Self {
            self.max_examples = Some(max);
            self
        }

        pub fn with_flattening(mut self, strategy: FlatteningStrategy) -> Self {
            self.flattening = strategy;
            self
        }

        pub fn with_include_tool_calls(mut self, include: bool) -> Self {
            self.include_tool_calls = include;
            self
        }
    }

    /// A single training example in the dataset.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DatasetEntry {
        /// System prompt (optional).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system: Option<String>,
        /// The input/instruction.
        pub input: String,
        /// The expected output.
        pub output: String,
        /// For multi-turn formats, the full messages list.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub messages: Vec<DatasetMessage>,
    }

    /// A single message in a multi-turn conversation training example.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DatasetMessage {
        pub role: String,
        pub content: String,
    }

    /// Builds fine-tuning datasets from trajectory data.
    #[derive(Debug)]
    pub struct DatasetBuilder;

    impl DatasetBuilder {
        /// Build dataset entries from a TrajectoryDataset using the given config.
        pub fn build(
            dataset: &TrajectoryDataset,
            config: &DatasetConfig,
        ) -> Result<Vec<DatasetEntry>, AiError> {
            let mut entries = Vec::new();
            let limit = config.max_examples.unwrap_or(usize::MAX);

            for trajectory in dataset.iter() {
                if entries.len() >= limit {
                    break;
                }

                let filtered_steps: Vec<&TrajectoryStep> = if config.include_tool_calls {
                    trajectory.steps.iter().collect()
                } else {
                    trajectory
                        .steps
                        .iter()
                        .filter(|s| !matches!(s.step_type, StepType::ToolUse { .. }))
                        .collect()
                };

                let entry = match &config.flattening {
                    FlatteningStrategy::LastStepOnly => {
                        let last_llm_output = filtered_steps
                            .iter()
                            .rev()
                            .find(|s| matches!(s.step_type, StepType::LlmCall { .. }))
                            .map(|s| s.output.clone())
                            .unwrap_or_default();

                        DatasetEntry {
                            system: config.system_prompt.clone(),
                            input: trajectory.task_description.clone(),
                            output: last_llm_output,
                            messages: Vec::new(),
                        }
                    }
                    FlatteningStrategy::AllSteps => {
                        let mut messages = Vec::new();
                        if let Some(sys) = &config.system_prompt {
                            messages.push(DatasetMessage {
                                role: "system".to_string(),
                                content: sys.clone(),
                            });
                        }
                        messages.push(DatasetMessage {
                            role: "user".to_string(),
                            content: trajectory.task_description.clone(),
                        });
                        for step in &filtered_steps {
                            match &step.step_type {
                                StepType::LlmCall { .. } => {
                                    messages.push(DatasetMessage {
                                        role: "assistant".to_string(),
                                        content: step.output.clone(),
                                    });
                                }
                                StepType::ToolUse { tool_name } => {
                                    messages.push(DatasetMessage {
                                        role: "assistant".to_string(),
                                        content: format!(
                                            "[tool:{}] {}",
                                            tool_name, step.output
                                        ),
                                    });
                                }
                                StepType::Observation { source } => {
                                    messages.push(DatasetMessage {
                                        role: "user".to_string(),
                                        content: format!(
                                            "[observation:{}] {}",
                                            source, step.output
                                        ),
                                    });
                                }
                                StepType::Decision { chosen, .. } => {
                                    messages.push(DatasetMessage {
                                        role: "assistant".to_string(),
                                        content: format!(
                                            "[decision:{}] {}",
                                            chosen, step.output
                                        ),
                                    });
                                }
                            }
                        }

                        let combined_output = filtered_steps
                            .iter()
                            .filter(|s| matches!(s.step_type, StepType::LlmCall { .. }))
                            .map(|s| s.output.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");

                        DatasetEntry {
                            system: config.system_prompt.clone(),
                            input: trajectory.task_description.clone(),
                            output: combined_output,
                            messages,
                        }
                    }
                    FlatteningStrategy::SummaryOnly => {
                        let concatenated = filtered_steps
                            .iter()
                            .map(|s| s.output.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");

                        DatasetEntry {
                            system: config.system_prompt.clone(),
                            input: trajectory.task_description.clone(),
                            output: concatenated,
                            messages: Vec::new(),
                        }
                    }
                };

                entries.push(entry);
            }

            Ok(entries)
        }

        /// Serialize dataset entries to a JSONL string in the specified format.
        pub fn to_jsonl(
            entries: &[DatasetEntry],
            format: &DatasetFormat,
        ) -> Result<String, AiError> {
            let mut lines = Vec::new();

            for entry in entries {
                let json_value = match format {
                    DatasetFormat::OpenAIJsonl => {
                        let mut messages = Vec::new();
                        if let Some(sys) = &entry.system {
                            messages.push(serde_json::json!({
                                "role": "system",
                                "content": sys,
                            }));
                        }
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": &entry.input,
                        }));
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": &entry.output,
                        }));
                        serde_json::json!({ "messages": messages })
                    }
                    DatasetFormat::Alpaca => {
                        let mut obj = serde_json::json!({
                            "instruction": &entry.input,
                            "input": "",
                            "output": &entry.output,
                        });
                        if let Some(sys) = &entry.system {
                            obj["instruction"] =
                                serde_json::Value::String(format!("{}\n\n{}", sys, entry.input));
                        }
                        obj
                    }
                    DatasetFormat::ShareGPT => {
                        let mut conversations = Vec::new();
                        if !entry.messages.is_empty() {
                            for msg in &entry.messages {
                                let from = match msg.role.as_str() {
                                    "system" => "system",
                                    "user" => "human",
                                    "assistant" => "gpt",
                                    other => other,
                                };
                                conversations.push(serde_json::json!({
                                    "from": from,
                                    "value": &msg.content,
                                }));
                            }
                        } else {
                            if let Some(sys) = &entry.system {
                                conversations.push(serde_json::json!({
                                    "from": "system",
                                    "value": sys,
                                }));
                            }
                            conversations.push(serde_json::json!({
                                "from": "human",
                                "value": &entry.input,
                            }));
                            conversations.push(serde_json::json!({
                                "from": "gpt",
                                "value": &entry.output,
                            }));
                        }
                        serde_json::json!({ "conversations": conversations })
                    }
                    DatasetFormat::Custom { template } => {
                        let rendered = template
                            .replace("{{system}}", entry.system.as_deref().unwrap_or(""))
                            .replace("{{input}}", &entry.input)
                            .replace("{{output}}", &entry.output);
                        // Try to parse as JSON; if it fails, wrap as a string value
                        serde_json::from_str::<serde_json::Value>(&rendered)
                            .unwrap_or_else(|_| serde_json::json!({ "text": rendered }))
                    }
                };

                let line = serde_json::to_string(&json_value).map_err(|e| {
                    AiError::Distillation(DistillationError::DatasetBuildFailed {
                        format: format.to_string(),
                        reason: e.to_string(),
                    })
                })?;
                lines.push(line);
            }

            Ok(lines.join("\n"))
        }
    }

    // ========================================================================
    // 5.4 — Data Flywheel Orchestrator
    // ========================================================================

    /// Configuration for the data flywheel.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FlywheelConfig {
        /// How many hours back to consider trajectories. Default: 24.
        pub collection_window_hours: u64,
        /// Minimum trajectories required to run a cycle. Default: 50.
        pub min_trajectories: usize,
        /// Score threshold for trajectory selection. Default: 0.7.
        pub score_threshold: f64,
        /// Output dataset format.
        pub format: DatasetFormat,
        /// Path to write the dataset file.
        pub output_path: String,
        /// Maximum examples per cycle.
        pub max_examples_per_cycle: Option<usize>,
    }

    impl FlywheelConfig {
        pub fn new(output_path: &str) -> Self {
            Self {
                collection_window_hours: 24,
                min_trajectories: 50,
                score_threshold: 0.7,
                format: DatasetFormat::OpenAIJsonl,
                output_path: output_path.to_string(),
                max_examples_per_cycle: None,
            }
        }
    }

    /// Status of a flywheel cycle.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum CycleStatus {
        /// Cycle is currently running.
        Running,
        /// Cycle completed successfully.
        Completed,
        /// Cycle failed.
        Failed { reason: String },
    }

    /// Record of a single flywheel cycle execution.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FlywheelCycle {
        pub cycle_id: String,
        pub started_at: DateTime<Utc>,
        pub completed_at: Option<DateTime<Utc>>,
        pub trajectories_evaluated: usize,
        pub trajectories_selected: usize,
        pub dataset_entries: usize,
        pub output_path: Option<String>,
        pub status: CycleStatus,
    }

    /// Trigger invoked when a flywheel cycle produces a ready dataset.
    pub trait FlywheelTrigger: Send + Sync {
        /// Called when a dataset file is ready.
        fn on_dataset_ready(&self, path: &str, stats: &FlywheelCycle) -> Result<(), AiError>;
    }

    /// A simple trigger that logs the cycle result.
    #[derive(Debug)]
    pub struct LogTrigger;

    impl FlywheelTrigger for LogTrigger {
        fn on_dataset_ready(&self, path: &str, stats: &FlywheelCycle) -> Result<(), AiError> {
            log::info!(
                "DataFlywheel cycle '{}' complete: {} entries written to '{}' ({} trajectories evaluated, {} selected)",
                stats.cycle_id,
                stats.dataset_entries,
                path,
                stats.trajectories_evaluated,
                stats.trajectories_selected,
            );
            Ok(())
        }
    }

    /// A trigger that calls a webhook URL with cycle stats.
    #[derive(Debug)]
    pub struct WebhookTrigger {
        pub url: String,
    }

    impl WebhookTrigger {
        pub fn new(url: &str) -> Self {
            Self {
                url: url.to_string(),
            }
        }
    }

    impl FlywheelTrigger for WebhookTrigger {
        fn on_dataset_ready(&self, path: &str, stats: &FlywheelCycle) -> Result<(), AiError> {
            let payload = serde_json::json!({
                "cycle_id": stats.cycle_id,
                "dataset_entries": stats.dataset_entries,
                "trajectories_evaluated": stats.trajectories_evaluated,
                "trajectories_selected": stats.trajectories_selected,
                "output_path": path,
            });

            ureq::post(&self.url)
                .set("Content-Type", "application/json")
                .send_string(&payload.to_string())
                .map_err(|e| {
                    AiError::Distillation(DistillationError::FlywheelFailed {
                        cycle_id: stats.cycle_id.clone(),
                        reason: format!("Webhook call failed: {}", e),
                    })
                })?;

            Ok(())
        }
    }

    /// Orchestrates the full collect -> score -> filter -> build -> trigger cycle.
    pub struct DataFlywheel {
        config: FlywheelConfig,
        store: Box<dyn TrajectoryStore>,
        trigger: Box<dyn FlywheelTrigger>,
        history: Vec<FlywheelCycle>,
    }

    impl DataFlywheel {
        /// Create a new data flywheel.
        pub fn new(
            config: FlywheelConfig,
            store: Box<dyn TrajectoryStore>,
            trigger: Box<dyn FlywheelTrigger>,
        ) -> Self {
            Self {
                config,
                store,
                trigger,
                history: Vec::new(),
            }
        }

        /// Run a single flywheel cycle.
        pub fn run_cycle(&mut self) -> Result<FlywheelCycle, AiError> {
            let cycle_id = Uuid::new_v4().to_string();
            let mut cycle = FlywheelCycle {
                cycle_id: cycle_id.clone(),
                started_at: Utc::now(),
                completed_at: None,
                trajectories_evaluated: 0,
                trajectories_selected: 0,
                dataset_entries: 0,
                output_path: None,
                status: CycleStatus::Running,
            };

            // Step 1: Load all trajectories from the store
            let ids = self.store.list().map_err(|e| {
                AiError::Distillation(DistillationError::FlywheelFailed {
                    cycle_id: cycle_id.clone(),
                    reason: format!("Failed to list trajectories: {}", e),
                })
            })?;

            let mut trajectories = Vec::new();
            for id in &ids {
                if let Some(t) = self.store.load(id).map_err(|e| {
                    AiError::Distillation(DistillationError::FlywheelFailed {
                        cycle_id: cycle_id.clone(),
                        reason: format!("Failed to load trajectory '{}': {}", id, e),
                    })
                })? {
                    trajectories.push(t);
                }
            }

            cycle.trajectories_evaluated = trajectories.len();

            // Check minimum trajectory count
            if trajectories.len() < self.config.min_trajectories {
                cycle.status = CycleStatus::Failed {
                    reason: format!(
                        "Insufficient trajectories: {} found, {} required",
                        trajectories.len(),
                        self.config.min_trajectories
                    ),
                };
                cycle.completed_at = Some(Utc::now());
                self.history.push(cycle.clone());
                return Err(AiError::Distillation(
                    DistillationError::NoValidTrajectories {
                        min_score: self.config.score_threshold,
                        total_checked: trajectories.len(),
                    },
                ));
            }

            // Step 2: Score and filter
            let filter = TrajectoryFilter::new(self.config.score_threshold)
                .with_scorer(Box::new(OutcomeScorer));

            let filtered: Vec<&Trajectory> = filter.filter(&trajectories);
            cycle.trajectories_selected = filtered.len();

            if filtered.is_empty() {
                cycle.status = CycleStatus::Failed {
                    reason: "No trajectories passed the score threshold".to_string(),
                };
                cycle.completed_at = Some(Utc::now());
                self.history.push(cycle.clone());
                return Err(AiError::Distillation(
                    DistillationError::NoValidTrajectories {
                        min_score: self.config.score_threshold,
                        total_checked: trajectories.len(),
                    },
                ));
            }

            // Step 3: Build dataset
            let dataset =
                TrajectoryDataset::new(filtered.into_iter().cloned().collect());

            let dataset_config = DatasetConfig {
                format: self.config.format.clone(),
                system_prompt: None,
                max_examples: self.config.max_examples_per_cycle,
                include_tool_calls: true,
                flattening: FlatteningStrategy::LastStepOnly,
            };

            let entries = DatasetBuilder::build(&dataset, &dataset_config)?;
            cycle.dataset_entries = entries.len();

            // Step 4: Write output
            let jsonl = DatasetBuilder::to_jsonl(&entries, &self.config.format)?;
            std::fs::write(&self.config.output_path, &jsonl).map_err(|e| {
                AiError::Distillation(DistillationError::FlywheelFailed {
                    cycle_id: cycle_id.clone(),
                    reason: format!("Failed to write dataset: {}", e),
                })
            })?;
            cycle.output_path = Some(self.config.output_path.clone());

            // Step 5: Trigger callback
            self.trigger
                .on_dataset_ready(&self.config.output_path, &cycle)
                .map_err(|e| {
                    AiError::Distillation(DistillationError::FlywheelFailed {
                        cycle_id: cycle_id.clone(),
                        reason: format!("Trigger failed: {}", e),
                    })
                })?;

            cycle.status = CycleStatus::Completed;
            cycle.completed_at = Some(Utc::now());
            self.history.push(cycle.clone());

            Ok(cycle)
        }

        /// Get the history of all flywheel cycles.
        pub fn get_history(&self) -> &[FlywheelCycle] {
            &self.history
        }
    }

    impl fmt::Debug for DataFlywheel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("DataFlywheel")
                .field("config", &self.config)
                .field("history_len", &self.history.len())
                .finish()
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        // -- Helpers --

        fn make_step(n: usize, step_type: StepType) -> TrajectoryStep {
            TrajectoryStep {
                step_number: n,
                step_type,
                input: format!("input_{}", n),
                output: format!("output_{}", n),
                tokens_used: 100,
                latency_ms: 50,
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            }
        }

        fn make_trajectory(
            id: &str,
            steps: Vec<TrajectoryStep>,
            outcome: TrajectoryOutcome,
        ) -> Trajectory {
            let total_tokens: usize = steps.iter().map(|s| s.tokens_used).sum();
            let total_latency_ms: u64 = steps.iter().map(|s| s.latency_ms).sum();
            Trajectory {
                id: id.to_string(),
                agent_id: "test-agent".to_string(),
                task_description: format!("Task for {}", id),
                steps,
                outcome,
                total_tokens,
                total_latency_ms,
                created_at: Utc::now(),
                metadata: HashMap::new(),
            }
        }

        fn simple_success_trajectory(id: &str, num_steps: usize) -> Trajectory {
            let steps: Vec<TrajectoryStep> = (0..num_steps)
                .map(|i| {
                    make_step(
                        i,
                        StepType::LlmCall {
                            model: "test-model".to_string(),
                            temperature: 0.7,
                        },
                    )
                })
                .collect();
            make_trajectory(id, steps, TrajectoryOutcome::Success { score: 0.9 })
        }

        fn diverse_trajectory(id: &str) -> Trajectory {
            let steps = vec![
                make_step(
                    0,
                    StepType::LlmCall {
                        model: "gpt-4".to_string(),
                        temperature: 0.7,
                    },
                ),
                make_step(
                    1,
                    StepType::ToolUse {
                        tool_name: "search".to_string(),
                    },
                ),
                make_step(
                    2,
                    StepType::Observation {
                        source: "web".to_string(),
                    },
                ),
                make_step(
                    3,
                    StepType::Decision {
                        options: vec!["a".to_string(), "b".to_string()],
                        chosen: "a".to_string(),
                    },
                ),
                make_step(
                    4,
                    StepType::ToolUse {
                        tool_name: "calculator".to_string(),
                    },
                ),
            ];
            make_trajectory(id, steps, TrajectoryOutcome::Success { score: 0.95 })
        }

        // ================================================================
        // TrajectoryCollector tests
        // ================================================================

        #[test]
        fn test_collector_lifecycle() {
            let mut collector = TrajectoryCollector::new();
            let id = collector.start_trajectory("agent-1", "Summarize a document");

            let step = make_step(
                0,
                StepType::LlmCall {
                    model: "llama3".to_string(),
                    temperature: 0.5,
                },
            );
            collector.add_step(&id, step).expect("add_step");

            collector
                .finish_trajectory(&id, TrajectoryOutcome::Success { score: 1.0 })
                .expect("finish");

            let t = collector.get_trajectory(&id).expect("should exist");
            assert_eq!(t.agent_id, "agent-1");
            assert_eq!(t.steps.len(), 1);
            assert_eq!(t.total_tokens, 100);
            assert!(matches!(t.outcome, TrajectoryOutcome::Success { .. }));
        }

        #[test]
        fn test_collector_concurrent_trajectories() {
            let mut collector = TrajectoryCollector::new();

            let id1 = collector.start_trajectory("agent-1", "Task A");
            let id2 = collector.start_trajectory("agent-2", "Task B");

            collector
                .add_step(
                    &id1,
                    make_step(
                        0,
                        StepType::LlmCall {
                            model: "m".to_string(),
                            temperature: 0.5,
                        },
                    ),
                )
                .expect("step 1");
            collector
                .add_step(
                    &id2,
                    make_step(
                        0,
                        StepType::ToolUse {
                            tool_name: "calc".to_string(),
                        },
                    ),
                )
                .expect("step 2");
            collector
                .add_step(
                    &id2,
                    make_step(
                        1,
                        StepType::LlmCall {
                            model: "m".to_string(),
                            temperature: 0.5,
                        },
                    ),
                )
                .expect("step 3");

            collector
                .finish_trajectory(&id1, TrajectoryOutcome::Success { score: 0.8 })
                .expect("finish 1");
            collector
                .finish_trajectory(
                    &id2,
                    TrajectoryOutcome::Failure {
                        reason: "timeout".to_string(),
                    },
                )
                .expect("finish 2");

            assert_eq!(collector.list_trajectories().len(), 2);
            assert_eq!(
                collector.get_trajectory(&id1).expect("t1").steps.len(),
                1
            );
            assert_eq!(
                collector.get_trajectory(&id2).expect("t2").steps.len(),
                2
            );
        }

        #[test]
        fn test_collector_add_step_to_unknown_trajectory() {
            let mut collector = TrajectoryCollector::new();
            let step = make_step(
                0,
                StepType::LlmCall {
                    model: "m".to_string(),
                    temperature: 0.5,
                },
            );
            let result = collector.add_step("nonexistent", step);
            assert!(result.is_err());
        }

        #[test]
        fn test_collector_finish_unknown_trajectory() {
            let mut collector = TrajectoryCollector::new();
            let result = collector
                .finish_trajectory("nonexistent", TrajectoryOutcome::Success { score: 1.0 });
            assert!(result.is_err());
        }

        #[test]
        fn test_collector_clear() {
            let mut collector = TrajectoryCollector::new();
            let id = collector.start_trajectory("a", "t");
            collector
                .finish_trajectory(&id, TrajectoryOutcome::Success { score: 1.0 })
                .expect("finish");
            assert_eq!(collector.list_trajectories().len(), 1);
            collector.clear();
            assert_eq!(collector.list_trajectories().len(), 0);
        }

        #[test]
        fn test_collector_accumulates_tokens_and_latency() {
            let mut collector = TrajectoryCollector::new();
            let id = collector.start_trajectory("a", "t");

            let mut step1 = make_step(
                0,
                StepType::LlmCall {
                    model: "m".to_string(),
                    temperature: 0.5,
                },
            );
            step1.tokens_used = 200;
            step1.latency_ms = 100;

            let mut step2 = make_step(
                1,
                StepType::LlmCall {
                    model: "m".to_string(),
                    temperature: 0.5,
                },
            );
            step2.tokens_used = 300;
            step2.latency_ms = 150;

            collector.add_step(&id, step1).expect("s1");
            collector.add_step(&id, step2).expect("s2");
            collector
                .finish_trajectory(&id, TrajectoryOutcome::Success { score: 1.0 })
                .expect("finish");

            let t = collector.get_trajectory(&id).expect("t");
            assert_eq!(t.total_tokens, 500);
            assert_eq!(t.total_latency_ms, 250);
        }

        // ================================================================
        // Scorer tests
        // ================================================================

        #[test]
        fn test_outcome_scorer_success() {
            let scorer = OutcomeScorer;
            let t = simple_success_trajectory("t1", 3);
            assert!((scorer.score(&t) - 0.9).abs() < 1e-9);
        }

        #[test]
        fn test_outcome_scorer_failure() {
            let scorer = OutcomeScorer;
            let t = make_trajectory(
                "t-fail",
                vec![make_step(
                    0,
                    StepType::LlmCall {
                        model: "m".to_string(),
                        temperature: 0.5,
                    },
                )],
                TrajectoryOutcome::Failure {
                    reason: "bad".to_string(),
                },
            );
            assert!((scorer.score(&t) - 0.0).abs() < 1e-9);
        }

        #[test]
        fn test_outcome_scorer_partial() {
            let scorer = OutcomeScorer;
            let t = make_trajectory(
                "t-partial",
                vec![],
                TrajectoryOutcome::Partial {
                    score: 0.6,
                    reason: "incomplete".to_string(),
                },
            );
            assert!((scorer.score(&t) - 0.6).abs() < 1e-9);
        }

        #[test]
        fn test_outcome_scorer_name() {
            assert_eq!(OutcomeScorer.name(), "outcome");
        }

        #[test]
        fn test_efficiency_scorer_few_steps() {
            let scorer = EfficiencyScorer::new();
            // 2 steps out of 20 max, 200 tokens
            let t = simple_success_trajectory("t1", 2);
            let score = scorer.score(&t);
            // step_score = 1.0 - 2/20 = 0.9
            // token_penalty = (200/1000)*0.1 = 0.02
            // score = 0.9 - 0.02 = 0.88
            assert!((score - 0.88).abs() < 1e-9);
        }

        #[test]
        fn test_efficiency_scorer_many_steps() {
            let scorer = EfficiencyScorer::new().with_max_steps(5);
            let t = simple_success_trajectory("t1", 10);
            let score = scorer.score(&t);
            // step_score = 1.0 - min(10/5, 1.0) = 0.0
            // token_penalty = (1000/1000)*0.1 = 0.1
            // score = max(0.0 - 0.1, 0.0) = 0.0
            assert!((score - 0.0).abs() < 1e-9);
        }

        #[test]
        fn test_efficiency_scorer_custom_penalty() {
            let scorer = EfficiencyScorer::new()
                .with_max_steps(10)
                .with_token_penalty(0.0); // no token penalty
            let t = simple_success_trajectory("t1", 5);
            let score = scorer.score(&t);
            // step_score = 1.0 - 5/10 = 0.5, no token penalty
            assert!((score - 0.5).abs() < 1e-9);
        }

        #[test]
        fn test_diversity_scorer_diverse() {
            let scorer = DiversityScorer;
            let t = diverse_trajectory("t1");
            let score = scorer.score(&t);
            // 4 unique step types + 2 unique tools = 6 unique items
            // max_possible = 4 + 5 = 9
            // score = 6/9 = 0.666...
            assert!(score > 0.6);
            assert!(score < 0.7);
        }

        #[test]
        fn test_diversity_scorer_homogeneous() {
            let scorer = DiversityScorer;
            let t = simple_success_trajectory("t1", 5);
            let score = scorer.score(&t);
            // 1 unique step type (llm_call), 0 tools = 1
            // max = 4 + 5 = 9
            // score = 1/9 ~= 0.111
            assert!(score > 0.1);
            assert!(score < 0.15);
        }

        #[test]
        fn test_diversity_scorer_empty() {
            let scorer = DiversityScorer;
            let t = make_trajectory("t-empty", vec![], TrajectoryOutcome::Success { score: 1.0 });
            assert!((scorer.score(&t) - 0.0).abs() < 1e-9);
        }

        #[test]
        fn test_composite_scorer() {
            let scorer = CompositeScorer::new()
                .add(0.7, Box::new(OutcomeScorer))
                .add(0.3, Box::new(EfficiencyScorer::new()));

            let t = simple_success_trajectory("t1", 2);
            let score = scorer.score(&t);

            // outcome = 0.9, efficiency = 0.88
            // composite = (0.7*0.9 + 0.3*0.88) / (0.7+0.3) = (0.63 + 0.264) / 1.0 = 0.894
            assert!(score > 0.89);
            assert!(score < 0.90);
        }

        #[test]
        fn test_composite_scorer_empty() {
            let scorer = CompositeScorer::new();
            let t = simple_success_trajectory("t1", 1);
            assert!((scorer.score(&t) - 0.0).abs() < 1e-9);
        }

        #[test]
        fn test_composite_scorer_name() {
            assert_eq!(CompositeScorer::new().name(), "composite");
        }

        // ================================================================
        // Filter tests
        // ================================================================

        #[test]
        fn test_filter_by_score() {
            let trajectories = vec![
                simple_success_trajectory("t1", 2),
                make_trajectory(
                    "t2",
                    vec![make_step(
                        0,
                        StepType::LlmCall {
                            model: "m".to_string(),
                            temperature: 0.5,
                        },
                    )],
                    TrajectoryOutcome::Failure {
                        reason: "bad".to_string(),
                    },
                ),
                make_trajectory(
                    "t3",
                    vec![],
                    TrajectoryOutcome::Partial {
                        score: 0.5,
                        reason: "half".to_string(),
                    },
                ),
            ];

            let filter = TrajectoryFilter::new(0.6).with_scorer(Box::new(OutcomeScorer));
            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].id, "t1");
        }

        #[test]
        fn test_filter_by_max_steps() {
            let trajectories = vec![
                simple_success_trajectory("short", 2),
                simple_success_trajectory("long", 15),
            ];

            let filter = TrajectoryFilter::new(0.0).with_max_steps(5);
            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].id, "short");
        }

        #[test]
        fn test_filter_by_max_tokens() {
            let trajectories = vec![
                simple_success_trajectory("small", 1),  // 100 tokens
                simple_success_trajectory("large", 20), // 2000 tokens
            ];

            let filter = TrajectoryFilter::new(0.0).with_max_tokens(500);
            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].id, "small");
        }

        #[test]
        fn test_filter_by_required_outcome_success() {
            let trajectories = vec![
                simple_success_trajectory("t1", 2),
                make_trajectory(
                    "t2",
                    vec![],
                    TrajectoryOutcome::Failure {
                        reason: "bad".to_string(),
                    },
                ),
            ];

            let filter = TrajectoryFilter::new(0.0)
                .with_required_outcome(RequiredOutcome::SuccessOnly);
            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].id, "t1");
        }

        #[test]
        fn test_filter_combined_criteria() {
            let trajectories = vec![
                simple_success_trajectory("ok", 3),    // success, 3 steps, 300 tokens
                simple_success_trajectory("big", 25),  // success, 25 steps, 2500 tokens
                make_trajectory(
                    "fail",
                    vec![make_step(
                        0,
                        StepType::LlmCall {
                            model: "m".to_string(),
                            temperature: 0.5,
                        },
                    )],
                    TrajectoryOutcome::Failure {
                        reason: "err".to_string(),
                    },
                ),
            ];

            let filter = TrajectoryFilter::new(0.5)
                .with_max_steps(10)
                .with_max_tokens(1000)
                .with_required_outcome(RequiredOutcome::SuccessOnly)
                .with_scorer(Box::new(OutcomeScorer));

            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].id, "ok");
        }

        #[test]
        fn test_filter_no_scorers_all_pass() {
            let trajectories = vec![
                simple_success_trajectory("t1", 1),
                simple_success_trajectory("t2", 2),
            ];

            let filter = TrajectoryFilter::new(0.5); // No scorers -> compute_score returns 1.0
            let filtered = filter.filter(&trajectories);
            assert_eq!(filtered.len(), 2);
        }

        // ================================================================
        // TrajectoryDataset tests
        // ================================================================

        #[test]
        fn test_dataset_split_train_test() {
            let trajectories: Vec<Trajectory> = (0..10)
                .map(|i| simple_success_trajectory(&format!("t{}", i), 2))
                .collect();

            let dataset = TrajectoryDataset::new(trajectories);
            assert_eq!(dataset.len(), 10);

            let (train, test) = dataset.split_train_test(0.8);
            assert_eq!(train.len(), 8);
            assert_eq!(test.len(), 2);
        }

        #[test]
        fn test_dataset_split_edge_cases() {
            let trajectories = vec![simple_success_trajectory("t0", 1)];
            let dataset = TrajectoryDataset::new(trajectories);

            // Ratio 0.0 -> all in test
            let (train, test) = dataset.clone().split_train_test(0.0);
            assert_eq!(train.len(), 0);
            assert_eq!(test.len(), 1);

            // Ratio 1.0 -> all in train
            let (train, test) = dataset.split_train_test(1.0);
            assert_eq!(train.len(), 1);
            assert_eq!(test.len(), 0);
        }

        #[test]
        fn test_dataset_dedup() {
            let t1 = simple_success_trajectory("t1", 2);
            let t1_dup = simple_success_trajectory("t1", 3); // same ID
            let t2 = simple_success_trajectory("t2", 2);

            let mut dataset = TrajectoryDataset::new(vec![t1, t1_dup, t2]);
            assert_eq!(dataset.len(), 3);

            dataset.dedup();
            assert_eq!(dataset.len(), 2);
        }

        #[test]
        fn test_dataset_shuffle() {
            let trajectories: Vec<Trajectory> = (0..5)
                .map(|i| simple_success_trajectory(&format!("t{}", i), 1))
                .collect();
            let original_ids: Vec<String> =
                trajectories.iter().map(|t| t.id.clone()).collect();

            let mut dataset = TrajectoryDataset::new(trajectories);
            dataset.shuffle();

            // After shuffle, all IDs should still be present
            let shuffled_ids: Vec<String> =
                dataset.iter().map(|t| t.id.clone()).collect();
            for id in &original_ids {
                assert!(
                    shuffled_ids.contains(id),
                    "ID {} missing after shuffle",
                    id
                );
            }
            assert_eq!(shuffled_ids.len(), original_ids.len());
        }

        #[test]
        fn test_dataset_is_empty() {
            let empty = TrajectoryDataset::new(vec![]);
            assert!(empty.is_empty());

            let non_empty =
                TrajectoryDataset::new(vec![simple_success_trajectory("t1", 1)]);
            assert!(!non_empty.is_empty());
        }

        // ================================================================
        // InMemoryTrajectoryStore tests
        // ================================================================

        #[test]
        fn test_in_memory_store_crud() {
            let mut store = InMemoryTrajectoryStore::new();
            let t = simple_success_trajectory("t1", 2);

            store.save(&t).expect("save");
            assert_eq!(store.count(), 1);

            let loaded = store.load("t1").expect("load").expect("should exist");
            assert_eq!(loaded.id, "t1");

            let ids = store.list().expect("list");
            assert_eq!(ids.len(), 1);
            assert!(ids.contains(&"t1".to_string()));

            store.delete("t1").expect("delete");
            assert_eq!(store.count(), 0);
            assert!(store.load("t1").expect("load").is_none());
        }

        #[test]
        fn test_in_memory_store_overwrite() {
            let mut store = InMemoryTrajectoryStore::new();
            let t1 = simple_success_trajectory("t1", 2);
            store.save(&t1).expect("save");

            let t1_v2 = simple_success_trajectory("t1", 5);
            store.save(&t1_v2).expect("save v2");

            assert_eq!(store.count(), 1);
            let loaded = store.load("t1").expect("load").expect("exists");
            assert_eq!(loaded.steps.len(), 5);
        }

        // ================================================================
        // JsonlTrajectoryStore tests
        // ================================================================

        #[test]
        fn test_jsonl_store_write_and_read() {
            let dir = std::env::temp_dir().join(format!(
                "distillation_test_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let path = dir.join("trajectories.jsonl");
            let path_str = path.to_string_lossy().to_string();

            {
                let mut store =
                    JsonlTrajectoryStore::new(&path_str).expect("create store");
                let t1 = simple_success_trajectory("t1", 2);
                let t2 = simple_success_trajectory("t2", 3);
                store.save(&t1).expect("save t1");
                store.save(&t2).expect("save t2");

                assert_eq!(store.count(), 2);
                let loaded = store.load("t1").expect("load").expect("exists");
                assert_eq!(loaded.id, "t1");
            }

            // Reopen and verify persistence
            {
                let store =
                    JsonlTrajectoryStore::new(&path_str).expect("reopen store");
                assert_eq!(store.count(), 2);
                assert!(store.load("t2").expect("load").is_some());
            }

            // Clean up
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_jsonl_store_delete() {
            let dir = std::env::temp_dir().join(format!(
                "distillation_del_test_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let path = dir.join("trajectories.jsonl");
            let path_str = path.to_string_lossy().to_string();

            let mut store =
                JsonlTrajectoryStore::new(&path_str).expect("create store");
            store
                .save(&simple_success_trajectory("t1", 1))
                .expect("save t1");
            store
                .save(&simple_success_trajectory("t2", 1))
                .expect("save t2");
            assert_eq!(store.count(), 2);

            store.delete("t1").expect("delete");
            assert_eq!(store.count(), 1);
            assert!(store.load("t1").expect("load").is_none());
            assert!(store.load("t2").expect("load").is_some());

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_jsonl_store_list() {
            let dir = std::env::temp_dir().join(format!(
                "distillation_list_test_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let path = dir.join("trajectories.jsonl");
            let path_str = path.to_string_lossy().to_string();

            let mut store =
                JsonlTrajectoryStore::new(&path_str).expect("create store");
            store
                .save(&simple_success_trajectory("a", 1))
                .expect("save");
            store
                .save(&simple_success_trajectory("b", 1))
                .expect("save");
            store
                .save(&simple_success_trajectory("c", 1))
                .expect("save");

            let ids = store.list().expect("list");
            assert_eq!(ids.len(), 3);
            assert!(ids.contains(&"a".to_string()));
            assert!(ids.contains(&"b".to_string()));
            assert!(ids.contains(&"c".to_string()));

            let _ = std::fs::remove_dir_all(&dir);
        }

        // ================================================================
        // DatasetBuilder tests
        // ================================================================

        #[test]
        fn test_dataset_builder_last_step_only() {
            let t = diverse_trajectory("t1");
            let dataset = TrajectoryDataset::new(vec![t]);

            let config = DatasetConfig::new(DatasetFormat::OpenAIJsonl)
                .with_system_prompt("You are helpful.")
                .with_flattening(FlatteningStrategy::LastStepOnly);

            let entries = DatasetBuilder::build(&dataset, &config).expect("build");
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].system.as_deref(), Some("You are helpful."));
            assert!(entries[0].input.contains("Task for t1"));
            // The last LLM call step is step 0 (the only LlmCall)
            assert_eq!(entries[0].output, "output_0");
        }

        #[test]
        fn test_dataset_builder_all_steps() {
            let t = diverse_trajectory("t1");
            let dataset = TrajectoryDataset::new(vec![t]);

            let config = DatasetConfig::new(DatasetFormat::ShareGPT)
                .with_system_prompt("System")
                .with_flattening(FlatteningStrategy::AllSteps);

            let entries = DatasetBuilder::build(&dataset, &config).expect("build");
            assert_eq!(entries.len(), 1);
            // AllSteps should produce messages
            assert!(!entries[0].messages.is_empty());
            // First message should be system
            assert_eq!(entries[0].messages[0].role, "system");
            // Second should be user (task description)
            assert_eq!(entries[0].messages[1].role, "user");
        }

        #[test]
        fn test_dataset_builder_summary_only() {
            let t = simple_success_trajectory("t1", 3);
            let dataset = TrajectoryDataset::new(vec![t]);

            let config = DatasetConfig::new(DatasetFormat::Alpaca)
                .with_flattening(FlatteningStrategy::SummaryOnly);

            let entries = DatasetBuilder::build(&dataset, &config).expect("build");
            assert_eq!(entries.len(), 1);
            // Summary concatenates all outputs
            assert!(entries[0].output.contains("output_0"));
            assert!(entries[0].output.contains("output_1"));
            assert!(entries[0].output.contains("output_2"));
        }

        #[test]
        fn test_dataset_builder_exclude_tool_calls() {
            let t = diverse_trajectory("t1");
            let dataset = TrajectoryDataset::new(vec![t]);

            let config = DatasetConfig::new(DatasetFormat::OpenAIJsonl)
                .with_flattening(FlatteningStrategy::SummaryOnly)
                .with_include_tool_calls(false);

            let entries = DatasetBuilder::build(&dataset, &config).expect("build");
            assert_eq!(entries.len(), 1);
            // output_1 and output_4 are tool_use steps, should be excluded
            assert!(!entries[0].output.contains("output_1"));
            assert!(!entries[0].output.contains("output_4"));
            // But LLM, observation, and decision outputs should be present
            assert!(entries[0].output.contains("output_0"));
            assert!(entries[0].output.contains("output_2"));
            assert!(entries[0].output.contains("output_3"));
        }

        #[test]
        fn test_dataset_builder_max_examples() {
            let trajectories: Vec<Trajectory> = (0..10)
                .map(|i| simple_success_trajectory(&format!("t{}", i), 1))
                .collect();
            let dataset = TrajectoryDataset::new(trajectories);

            let config = DatasetConfig::new(DatasetFormat::OpenAIJsonl)
                .with_max_examples(3);

            let entries = DatasetBuilder::build(&dataset, &config).expect("build");
            assert_eq!(entries.len(), 3);
        }

        // ================================================================
        // JSONL serialization tests
        // ================================================================

        #[test]
        fn test_to_jsonl_openai_format() {
            let entry = DatasetEntry {
                system: Some("sys".to_string()),
                input: "hello".to_string(),
                output: "world".to_string(),
                messages: Vec::new(),
            };

            let jsonl =
                DatasetBuilder::to_jsonl(&[entry], &DatasetFormat::OpenAIJsonl)
                    .expect("to_jsonl");
            let parsed: serde_json::Value =
                serde_json::from_str(&jsonl).expect("parse");
            let messages = parsed["messages"].as_array().expect("messages array");
            assert_eq!(messages.len(), 3);
            assert_eq!(messages[0]["role"], "system");
            assert_eq!(messages[1]["role"], "user");
            assert_eq!(messages[2]["role"], "assistant");
        }

        #[test]
        fn test_to_jsonl_alpaca_format() {
            let entry = DatasetEntry {
                system: None,
                input: "translate".to_string(),
                output: "translated".to_string(),
                messages: Vec::new(),
            };

            let jsonl =
                DatasetBuilder::to_jsonl(&[entry], &DatasetFormat::Alpaca)
                    .expect("to_jsonl");
            let parsed: serde_json::Value =
                serde_json::from_str(&jsonl).expect("parse");
            assert_eq!(parsed["instruction"], "translate");
            assert_eq!(parsed["output"], "translated");
        }

        #[test]
        fn test_to_jsonl_sharegpt_format() {
            let entry = DatasetEntry {
                system: Some("system prompt".to_string()),
                input: "question".to_string(),
                output: "answer".to_string(),
                messages: vec![
                    DatasetMessage {
                        role: "system".to_string(),
                        content: "system prompt".to_string(),
                    },
                    DatasetMessage {
                        role: "user".to_string(),
                        content: "question".to_string(),
                    },
                    DatasetMessage {
                        role: "assistant".to_string(),
                        content: "answer".to_string(),
                    },
                ],
            };

            let jsonl =
                DatasetBuilder::to_jsonl(&[entry], &DatasetFormat::ShareGPT)
                    .expect("to_jsonl");
            let parsed: serde_json::Value =
                serde_json::from_str(&jsonl).expect("parse");
            let convos = parsed["conversations"]
                .as_array()
                .expect("conversations array");
            assert_eq!(convos.len(), 3);
            assert_eq!(convos[0]["from"], "system");
            assert_eq!(convos[1]["from"], "human");
            assert_eq!(convos[2]["from"], "gpt");
        }

        #[test]
        fn test_to_jsonl_custom_format() {
            let entry = DatasetEntry {
                system: None,
                input: "hello".to_string(),
                output: "world".to_string(),
                messages: Vec::new(),
            };

            let format = DatasetFormat::Custom {
                template: r#"{"prompt":"{{input}}","completion":"{{output}}"}"#.to_string(),
            };

            let jsonl =
                DatasetBuilder::to_jsonl(&[entry], &format).expect("to_jsonl");
            let parsed: serde_json::Value =
                serde_json::from_str(&jsonl).expect("parse");
            assert_eq!(parsed["prompt"], "hello");
            assert_eq!(parsed["completion"], "world");
        }

        // ================================================================
        // Flywheel tests
        // ================================================================

        #[test]
        fn test_flywheel_full_cycle() {
            let dir = std::env::temp_dir().join(format!(
                "flywheel_test_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let output_path = dir.join("dataset.jsonl");
            let output_str = output_path.to_string_lossy().to_string();

            let mut store = InMemoryTrajectoryStore::new();

            // Create enough trajectories (min_trajectories = 3 for test)
            for i in 0..5 {
                let t = simple_success_trajectory(&format!("t{}", i), 2);
                store.save(&t).expect("save");
            }

            let config = FlywheelConfig {
                collection_window_hours: 24,
                min_trajectories: 3,
                score_threshold: 0.5,
                format: DatasetFormat::OpenAIJsonl,
                output_path: output_str.clone(),
                max_examples_per_cycle: None,
            };

            let mut flywheel = DataFlywheel::new(
                config,
                Box::new(store),
                Box::new(LogTrigger),
            );

            let cycle = flywheel.run_cycle().expect("run_cycle");
            assert_eq!(cycle.status, CycleStatus::Completed);
            assert_eq!(cycle.trajectories_evaluated, 5);
            assert!(cycle.trajectories_selected > 0);
            assert!(cycle.dataset_entries > 0);
            assert!(cycle.output_path.is_some());
            assert!(cycle.completed_at.is_some());

            // Verify the file was written
            let content = std::fs::read_to_string(&output_str).expect("read output");
            assert!(!content.is_empty());

            // Check history
            assert_eq!(flywheel.get_history().len(), 1);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_flywheel_insufficient_trajectories() {
            let dir = std::env::temp_dir().join(format!(
                "flywheel_insuff_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let output_path = dir.join("dataset.jsonl");
            let output_str = output_path.to_string_lossy().to_string();

            let mut store = InMemoryTrajectoryStore::new();
            store
                .save(&simple_success_trajectory("t1", 1))
                .expect("save");

            let config = FlywheelConfig {
                collection_window_hours: 24,
                min_trajectories: 50, // way more than we have
                score_threshold: 0.5,
                format: DatasetFormat::OpenAIJsonl,
                output_path: output_str,
                max_examples_per_cycle: None,
            };

            let mut flywheel = DataFlywheel::new(
                config,
                Box::new(store),
                Box::new(LogTrigger),
            );

            let result = flywheel.run_cycle();
            assert!(result.is_err());

            // History should still record the failed cycle
            assert_eq!(flywheel.get_history().len(), 1);
            assert!(matches!(
                flywheel.get_history()[0].status,
                CycleStatus::Failed { .. }
            ));

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_flywheel_trigger_callback() {
            use std::sync::{Arc, Mutex};

            let dir = std::env::temp_dir().join(format!(
                "flywheel_trigger_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let output_path = dir.join("dataset.jsonl");
            let output_str = output_path.to_string_lossy().to_string();

            // Custom trigger that records calls
            struct RecordingTrigger {
                calls: Arc<Mutex<Vec<(String, usize)>>>,
            }

            impl FlywheelTrigger for RecordingTrigger {
                fn on_dataset_ready(
                    &self,
                    path: &str,
                    stats: &FlywheelCycle,
                ) -> Result<(), AiError> {
                    self.calls
                        .lock()
                        .expect("lock")
                        .push((path.to_string(), stats.dataset_entries));
                    Ok(())
                }
            }

            let calls: Arc<Mutex<Vec<(String, usize)>>> =
                Arc::new(Mutex::new(Vec::new()));

            let mut store = InMemoryTrajectoryStore::new();
            for i in 0..10 {
                store
                    .save(&simple_success_trajectory(&format!("t{}", i), 1))
                    .expect("save");
            }

            let config = FlywheelConfig {
                collection_window_hours: 24,
                min_trajectories: 5,
                score_threshold: 0.5,
                format: DatasetFormat::OpenAIJsonl,
                output_path: output_str.clone(),
                max_examples_per_cycle: None,
            };

            let trigger = RecordingTrigger {
                calls: Arc::clone(&calls),
            };

            let mut flywheel = DataFlywheel::new(
                config,
                Box::new(store),
                Box::new(trigger),
            );

            flywheel.run_cycle().expect("run_cycle");

            let recorded = calls.lock().expect("lock");
            assert_eq!(recorded.len(), 1);
            assert_eq!(recorded[0].0, output_str);
            assert!(recorded[0].1 > 0);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_flywheel_no_passing_trajectories() {
            let dir = std::env::temp_dir().join(format!(
                "flywheel_nopass_{}",
                Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            let output_path = dir.join("dataset.jsonl");
            let output_str = output_path.to_string_lossy().to_string();

            let mut store = InMemoryTrajectoryStore::new();
            // All failures -> OutcomeScorer gives 0.0, threshold is 0.9
            for i in 0..10 {
                let t = make_trajectory(
                    &format!("f{}", i),
                    vec![make_step(
                        0,
                        StepType::LlmCall {
                            model: "m".to_string(),
                            temperature: 0.5,
                        },
                    )],
                    TrajectoryOutcome::Failure {
                        reason: "bad".to_string(),
                    },
                );
                store.save(&t).expect("save");
            }

            let config = FlywheelConfig {
                collection_window_hours: 24,
                min_trajectories: 5,
                score_threshold: 0.9,
                format: DatasetFormat::OpenAIJsonl,
                output_path: output_str,
                max_examples_per_cycle: None,
            };

            let mut flywheel = DataFlywheel::new(
                config,
                Box::new(store),
                Box::new(LogTrigger),
            );

            let result = flywheel.run_cycle();
            assert!(result.is_err());

            let _ = std::fs::remove_dir_all(&dir);
        }

        // ================================================================
        // Step type display
        // ================================================================

        #[test]
        fn test_step_type_display() {
            let llm = StepType::LlmCall {
                model: "gpt-4".to_string(),
                temperature: 0.7,
            };
            assert_eq!(llm.to_string(), "llm_call:gpt-4");

            let tool = StepType::ToolUse {
                tool_name: "search".to_string(),
            };
            assert_eq!(tool.to_string(), "tool_use:search");

            let decision = StepType::Decision {
                options: vec!["a".to_string(), "b".to_string()],
                chosen: "a".to_string(),
            };
            assert_eq!(decision.to_string(), "decision:a");

            let obs = StepType::Observation {
                source: "web".to_string(),
            };
            assert_eq!(obs.to_string(), "observation:web");
        }

        // ================================================================
        // DatasetFormat display
        // ================================================================

        #[test]
        fn test_dataset_format_display() {
            assert_eq!(DatasetFormat::OpenAIJsonl.to_string(), "openai_jsonl");
            assert_eq!(DatasetFormat::Alpaca.to_string(), "alpaca");
            assert_eq!(DatasetFormat::ShareGPT.to_string(), "sharegpt");
            assert_eq!(
                DatasetFormat::Custom {
                    template: "t".to_string()
                }
                .to_string(),
                "custom"
            );
        }

        // ================================================================
        // Trajectory serialization roundtrip
        // ================================================================

        #[test]
        fn test_trajectory_serde_roundtrip() {
            let t = diverse_trajectory("round-trip");
            let json = serde_json::to_string(&t).expect("serialize");
            let deserialized: Trajectory =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized.id, "round-trip");
            assert_eq!(deserialized.steps.len(), 5);
        }

        #[test]
        fn test_dataset_entry_serde_roundtrip() {
            let entry = DatasetEntry {
                system: Some("sys".to_string()),
                input: "in".to_string(),
                output: "out".to_string(),
                messages: vec![DatasetMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
            };
            let json = serde_json::to_string(&entry).expect("serialize");
            let deserialized: DatasetEntry =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized.system.as_deref(), Some("sys"));
            assert_eq!(deserialized.messages.len(), 1);
        }

        #[test]
        fn test_flywheel_config_defaults() {
            let config = FlywheelConfig::new("/tmp/out.jsonl");
            assert_eq!(config.collection_window_hours, 24);
            assert_eq!(config.min_trajectories, 50);
            assert!((config.score_threshold - 0.7).abs() < 1e-9);
            assert_eq!(config.format, DatasetFormat::OpenAIJsonl);
            assert!(config.max_examples_per_cycle.is_none());
        }
    }
}

#[cfg(feature = "distillation")]
pub use inner::*;
