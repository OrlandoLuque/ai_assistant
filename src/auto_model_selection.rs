//! Automatic model selection based on task characteristics
//!
//! This module provides intelligent routing of requests to the
//! most appropriate model based on task type, complexity, and requirements.
//!
//! # Features
//!
//! - **Task classification**: Automatically classify query type
//! - **Model matching**: Match tasks to best-suited models
//! - **Cost optimization**: Balance quality and cost
//! - **Performance tracking**: Learn from outcomes

use std::collections::HashMap;
use std::time::Duration;

/// Configuration for automatic model selection
#[derive(Debug, Clone)]
pub struct AutoSelectConfig {
    /// Available models
    pub models: Vec<ModelProfile>,
    /// Default model if no match
    pub default_model: String,
    /// Optimize for cost
    pub optimize_cost: bool,
    /// Optimize for speed
    pub optimize_speed: bool,
    /// Minimum quality threshold
    pub min_quality: f64,
    /// Enable learning from outcomes
    pub enable_learning: bool,
}

impl Default for AutoSelectConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            default_model: String::new(),
            optimize_cost: false,
            optimize_speed: false,
            min_quality: 0.7,
            enable_learning: true,
        }
    }
}

/// Profile of a model's capabilities
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Provider
    pub provider: String,
    /// Capabilities
    pub capabilities: ModelCapabilities,
    /// Cost per 1K tokens (input)
    pub cost_input: f64,
    /// Cost per 1K tokens (output)
    pub cost_output: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// Quality score for various tasks
    pub task_quality: HashMap<TaskType, f64>,
    /// Maximum context length
    pub max_context: usize,
    /// Is model available
    pub available: bool,
}

/// Model capabilities
#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    /// Supports code generation
    pub code: bool,
    /// Supports math/reasoning
    pub math: bool,
    /// Supports creative writing
    pub creative: bool,
    /// Supports multi-language
    pub multilingual: bool,
    /// Supports function calling
    pub function_calling: bool,
    /// Supports vision
    pub vision: bool,
    /// Context window size
    pub context_size: usize,
    /// Supports streaming
    pub streaming: bool,
}

/// Types of tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Code generation
    Coding,
    /// Code review/explanation
    CodeReview,
    /// Mathematical reasoning
    Math,
    /// Creative writing
    Creative,
    /// Translation
    Translation,
    /// Summarization
    Summarization,
    /// Question answering
    QA,
    /// Chat/conversation
    Chat,
    /// Data analysis
    DataAnalysis,
    /// Technical writing
    TechnicalWriting,
    /// Classification
    Classification,
    /// Unknown/general
    General,
}

impl TaskType {
    /// Detect task type from prompt
    pub fn from_prompt(prompt: &str) -> Self {
        let lower = prompt.to_lowercase();

        // Code-related
        if lower.contains("write code")
            || lower.contains("implement")
            || lower.contains("function that")
            || lower.contains("```")
        {
            return TaskType::Coding;
        }

        if lower.contains("review this code")
            || lower.contains("explain this code")
            || lower.contains("what does this")
        {
            return TaskType::CodeReview;
        }

        // Math
        if lower.contains("calculate")
            || lower.contains("solve")
            || lower.contains("equation")
            || lower.contains("math")
        {
            return TaskType::Math;
        }

        // Creative
        if lower.contains("write a story")
            || lower.contains("creative")
            || lower.contains("poem")
            || lower.contains("fiction")
        {
            return TaskType::Creative;
        }

        // Translation
        if lower.contains("translate")
            || lower.contains("translation")
            || lower.contains("to english")
            || lower.contains("to spanish")
        {
            return TaskType::Translation;
        }

        // Summarization
        if lower.contains("summarize") || lower.contains("summary") || lower.contains("tl;dr") {
            return TaskType::Summarization;
        }

        // Classification
        if lower.contains("classify")
            || lower.contains("categorize")
            || lower.contains("which category")
        {
            return TaskType::Classification;
        }

        // Data analysis
        if lower.contains("analyze") || lower.contains("data") || lower.contains("statistics") {
            return TaskType::DataAnalysis;
        }

        TaskType::General
    }
}

/// Selection result
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected model
    pub model_id: String,
    /// Model profile
    pub profile: ModelProfile,
    /// Task type detected
    pub task_type: TaskType,
    /// Confidence in selection
    pub confidence: f64,
    /// Reason for selection
    pub reason: String,
    /// Alternative models
    pub alternatives: Vec<String>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated latency
    pub estimated_latency: Duration,
}

/// Automatic model selector
pub struct AutoModelSelector {
    config: AutoSelectConfig,
    /// Performance history
    history: HashMap<(String, TaskType), PerformanceRecord>,
}

/// Performance record for learning
#[derive(Debug, Clone, Default)]
struct PerformanceRecord {
    total: usize,
    successes: usize,
    total_latency: Duration,
    total_quality: f64,
}

impl AutoModelSelector {
    /// Create a new selector
    pub fn new(config: AutoSelectConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
        }
    }

    /// Add a model profile
    pub fn add_model(&mut self, profile: ModelProfile) {
        self.config.models.push(profile);
    }

    /// Select best model for a prompt
    pub fn select(&self, prompt: &str, requirements: Option<&Requirements>) -> SelectionResult {
        let task_type = TaskType::from_prompt(prompt);
        let prompt_length = prompt.len();

        // Filter available models
        let available: Vec<_> = self
            .config
            .models
            .iter()
            .filter(|m| m.available)
            .filter(|m| {
                // Check requirements if provided
                if let Some(req) = requirements {
                    if req.needs_code && !m.capabilities.code {
                        return false;
                    }
                    if req.needs_vision && !m.capabilities.vision {
                        return false;
                    }
                    if req.needs_function_calling && !m.capabilities.function_calling {
                        return false;
                    }
                    if let Some(max_cost) = req.max_cost {
                        if m.cost_input > max_cost {
                            return false;
                        }
                    }
                    if let Some(max_latency) = req.max_latency {
                        if m.avg_latency > max_latency {
                            return false;
                        }
                    }
                    if let Some(min_context) = req.min_context {
                        if m.max_context < min_context {
                            return false;
                        }
                    }
                }
                true
            })
            .collect();

        if available.is_empty() {
            // Return default model
            let default = self
                .config
                .models
                .iter()
                .find(|m| m.id == self.config.default_model)
                .cloned()
                .unwrap_or_else(|| ModelProfile {
                    id: self.config.default_model.clone(),
                    name: "Default".to_string(),
                    provider: "unknown".to_string(),
                    capabilities: ModelCapabilities::default(),
                    cost_input: 0.0,
                    cost_output: 0.0,
                    avg_latency: Duration::from_secs(1),
                    task_quality: HashMap::new(),
                    max_context: 4096,
                    available: true,
                });

            return SelectionResult {
                model_id: default.id.clone(),
                profile: default,
                task_type,
                confidence: 0.5,
                reason: "No suitable models available, using default".to_string(),
                alternatives: Vec::new(),
                estimated_cost: 0.0,
                estimated_latency: Duration::from_secs(1),
            };
        }

        // Score each model
        let mut scored: Vec<(f64, &ModelProfile)> = available
            .iter()
            .map(|m| (self.score_model(m, task_type, prompt_length), *m))
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let best = scored[0].1;
        let score = scored[0].0;

        // Get alternatives
        let alternatives: Vec<_> = scored
            .iter()
            .skip(1)
            .take(3)
            .map(|(_, m)| m.id.clone())
            .collect();

        // Estimate cost
        let estimated_tokens = (prompt_length / 4 + 500) as f64 / 1000.0;
        let estimated_cost =
            estimated_tokens * best.cost_input + estimated_tokens * best.cost_output;

        SelectionResult {
            model_id: best.id.clone(),
            profile: best.clone(),
            task_type,
            confidence: score,
            reason: self.explain_selection(best, task_type),
            alternatives,
            estimated_cost,
            estimated_latency: best.avg_latency,
        }
    }

    /// Score a model for a task
    fn score_model(&self, model: &ModelProfile, task_type: TaskType, prompt_length: usize) -> f64 {
        let mut score = 0.5;

        // Task quality score
        if let Some(&quality) = model.task_quality.get(&task_type) {
            score += quality * 0.4;
        }

        // Historical performance
        if let Some(record) = self.history.get(&(model.id.clone(), task_type)) {
            if record.total > 0 {
                let success_rate = record.successes as f64 / record.total as f64;
                score += success_rate * 0.2;
            }
        }

        // Cost optimization
        if self.config.optimize_cost {
            let cost_factor = 1.0 / (model.cost_input + model.cost_output + 1.0);
            score += cost_factor * 0.1;
        }

        // Speed optimization
        if self.config.optimize_speed {
            let speed_factor = 1.0 / (model.avg_latency.as_secs_f64() + 1.0);
            score += speed_factor * 0.1;
        }

        // Context fit
        let context_fit = (model.max_context as f64 / (prompt_length as f64 + 1.0)).min(1.0);
        score += context_fit * 0.1;

        score.min(1.0)
    }

    /// Explain selection
    fn explain_selection(&self, model: &ModelProfile, task_type: TaskType) -> String {
        let mut reasons = Vec::new();

        if let Some(&quality) = model.task_quality.get(&task_type) {
            if quality > 0.8 {
                reasons.push(format!("High quality for {:?} tasks", task_type));
            }
        }

        if model.cost_input < 0.01 {
            reasons.push("Cost-effective".to_string());
        }

        if model.avg_latency < Duration::from_secs(2) {
            reasons.push("Fast response time".to_string());
        }

        if reasons.is_empty() {
            "Best available option".to_string()
        } else {
            reasons.join(", ")
        }
    }

    /// Record outcome for learning
    pub fn record_outcome(
        &mut self,
        model_id: &str,
        task_type: TaskType,
        success: bool,
        quality: f64,
        latency: Duration,
    ) {
        if !self.config.enable_learning {
            return;
        }

        let key = (model_id.to_string(), task_type);
        let record = self
            .history
            .entry(key)
            .or_insert_with(PerformanceRecord::default);

        record.total += 1;
        if success {
            record.successes += 1;
        }
        record.total_latency += latency;
        record.total_quality += quality;
    }

    /// Get model stats
    pub fn model_stats(&self, model_id: &str) -> HashMap<TaskType, ModelStats> {
        let mut stats = HashMap::new();

        for ((id, task_type), record) in &self.history {
            if id == model_id && record.total > 0 {
                stats.insert(
                    *task_type,
                    ModelStats {
                        total_requests: record.total,
                        success_rate: record.successes as f64 / record.total as f64,
                        avg_quality: record.total_quality / record.total as f64,
                        avg_latency: record.total_latency / record.total as u32,
                    },
                );
            }
        }

        stats
    }
}

impl Default for AutoModelSelector {
    fn default() -> Self {
        Self::new(AutoSelectConfig::default())
    }
}

/// Requirements for model selection
#[derive(Debug, Clone, Default)]
pub struct Requirements {
    /// Needs code capability
    pub needs_code: bool,
    /// Needs vision capability
    pub needs_vision: bool,
    /// Needs function calling
    pub needs_function_calling: bool,
    /// Maximum cost
    pub max_cost: Option<f64>,
    /// Maximum latency
    pub max_latency: Option<Duration>,
    /// Minimum context size
    pub min_context: Option<usize>,
    /// Preferred provider
    pub preferred_provider: Option<String>,
}

impl Requirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn needs_code(mut self) -> Self {
        self.needs_code = true;
        self
    }

    pub fn needs_vision(mut self) -> Self {
        self.needs_vision = true;
        self
    }

    pub fn max_cost(mut self, cost: f64) -> Self {
        self.max_cost = Some(cost);
        self
    }

    pub fn max_latency(mut self, latency: Duration) -> Self {
        self.max_latency = Some(latency);
        self
    }
}

/// Stats for a model
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Total requests
    pub total_requests: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub avg_quality: f64,
    /// Average latency
    pub avg_latency: Duration,
}

/// Builder for selector configuration
pub struct AutoSelectConfigBuilder {
    config: AutoSelectConfig,
}

impl AutoSelectConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AutoSelectConfig::default(),
        }
    }

    pub fn add_model(mut self, profile: ModelProfile) -> Self {
        self.config.models.push(profile);
        self
    }

    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    pub fn optimize_cost(mut self) -> Self {
        self.config.optimize_cost = true;
        self
    }

    pub fn optimize_speed(mut self) -> Self {
        self.config.optimize_speed = true;
        self
    }

    pub fn min_quality(mut self, quality: f64) -> Self {
        self.config.min_quality = quality;
        self
    }

    pub fn build(self) -> AutoSelectConfig {
        self.config
    }
}

impl Default for AutoSelectConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── WS8: Model Selector Enhancements ───────────────────────────────────────

/// Cost information for a specific model
#[derive(Debug, Clone)]
pub struct ModelCostEntry {
    /// Model identifier
    pub model_id: String,
    /// Cost per million input tokens
    pub input_cost_per_million: f64,
    /// Cost per million output tokens
    pub output_cost_per_million: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Total number of requests made
    pub total_requests: u64,
}

impl ModelCostEntry {
    /// Create a new cost entry with default stats
    pub fn new(model_id: &str, input_cost: f64, output_cost: f64) -> Self {
        Self {
            model_id: model_id.to_string(),
            input_cost_per_million: input_cost,
            output_cost_per_million: output_cost,
            avg_latency_ms: 0,
            success_rate: 1.0,
            total_requests: 0,
        }
    }
}

/// Registry tracking cost and performance data for multiple models
#[derive(Debug, Clone)]
pub struct ModelCostRegistry {
    entries: HashMap<String, ModelCostEntry>,
}

impl ModelCostRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a model cost entry
    pub fn register(&mut self, entry: ModelCostEntry) {
        self.entries.insert(entry.model_id.clone(), entry);
    }

    /// Look up a model by id
    pub fn get(&self, model_id: &str) -> Option<&ModelCostEntry> {
        self.entries.get(model_id)
    }

    /// Update running statistics after a request completes
    pub fn update_stats(&mut self, model_id: &str, latency_ms: u64, success: bool) {
        if let Some(entry) = self.entries.get_mut(model_id) {
            let n = entry.total_requests;
            entry.total_requests += 1;
            // Running average for latency
            if n == 0 {
                entry.avg_latency_ms = latency_ms;
            } else {
                entry.avg_latency_ms = ((entry.avg_latency_ms as f64 * n as f64
                    + latency_ms as f64)
                    / (n + 1) as f64) as u64;
            }
            // Running average for success rate
            let success_val = if success { 1.0 } else { 0.0 };
            if n == 0 {
                entry.success_rate = success_val;
            } else {
                entry.success_rate = (entry.success_rate * n as f64 + success_val) / (n + 1) as f64;
            }
        }
    }

    /// Find the cheapest model for a given token estimate
    pub fn cheapest_for(&self, estimated_tokens: usize) -> Option<&ModelCostEntry> {
        self.entries.values().min_by(|a, b| {
            let cost_a = a.input_cost_per_million * estimated_tokens as f64 / 1_000_000.0;
            let cost_b = b.input_cost_per_million * estimated_tokens as f64 / 1_000_000.0;
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return model ids whose estimated cost fits within the remaining budget
    pub fn within_budget(&self, budget_remaining: f64, estimated_tokens: usize) -> Vec<String> {
        self.entries
            .values()
            .filter(|e| {
                let cost = e.input_cost_per_million * estimated_tokens as f64 / 1_000_000.0;
                cost <= budget_remaining
            })
            .map(|e| e.model_id.clone())
            .collect()
    }

    /// Number of registered models
    pub fn model_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ModelCostRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy for ordering fallback models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Use models in the order they were added
    Sequential,
    /// Order by ascending input cost
    CostAscending,
    /// Order by ascending average latency
    LatencyAscending,
    /// Order by descending quality / success rate
    QualityDescending,
}

/// An ordered chain of fallback models
#[derive(Debug, Clone)]
pub struct FallbackChain {
    models: Vec<String>,
    #[allow(dead_code)]
    strategy: FallbackStrategy,
}

impl FallbackChain {
    /// Create a new fallback chain with the given strategy
    pub fn new(strategy: FallbackStrategy) -> Self {
        Self {
            models: Vec::new(),
            strategy,
        }
    }

    /// Append a model to the chain
    pub fn add_model(&mut self, model_id: &str) -> &mut Self {
        self.models.push(model_id.to_string());
        self
    }

    /// Set the full list of models (builder pattern)
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.models = models;
        self
    }

    /// Return the next model that has not been tried yet
    pub fn next(&self, tried: &[String]) -> Option<String> {
        self.models.iter().find(|m| !tried.contains(m)).cloned()
    }

    /// Sort models by ascending input cost using the registry
    pub fn sort_by_cost(&mut self, registry: &ModelCostRegistry) {
        self.models.sort_by(|a, b| {
            let cost_a = registry
                .get(a)
                .map(|e| e.input_cost_per_million)
                .unwrap_or(f64::MAX);
            let cost_b = registry
                .get(b)
                .map(|e| e.input_cost_per_million)
                .unwrap_or(f64::MAX);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort models by ascending average latency using the registry
    pub fn sort_by_latency(&mut self, registry: &ModelCostRegistry) {
        self.models.sort_by(|a, b| {
            let lat_a = registry
                .get(a)
                .map(|e| e.avg_latency_ms)
                .unwrap_or(u64::MAX);
            let lat_b = registry
                .get(b)
                .map(|e| e.avg_latency_ms)
                .unwrap_or(u64::MAX);
            lat_a.cmp(&lat_b)
        });
    }

    /// Number of models in the chain
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the chain is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Record of a single model invocation
#[derive(Debug, Clone)]
pub struct ModelInvocation {
    /// Which model was used
    pub model_id: String,
    /// The type of task
    pub task_type: TaskType,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Number of tokens consumed
    pub tokens_used: usize,
    /// Whether the invocation succeeded
    pub success: bool,
    /// Optional quality score (0.0 to 1.0)
    pub quality_score: Option<f64>,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
}

impl ModelInvocation {
    /// Create a new invocation record
    pub fn new(
        model_id: &str,
        task_type: TaskType,
        latency_ms: u64,
        tokens_used: usize,
        success: bool,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            task_type,
            latency_ms,
            tokens_used,
            success,
            quality_score: None,
            timestamp_ms: 0,
        }
    }

    /// Attach a quality score
    pub fn with_quality(mut self, score: f64) -> Self {
        self.quality_score = Some(score);
        self
    }

    /// Attach a timestamp
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }
}

/// Aggregated performance statistics for a single model
#[derive(Debug, Clone)]
pub struct ModelPerformanceStats {
    /// Model identifier
    pub model_id: String,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average quality score (only from invocations with a quality score)
    pub avg_quality: f64,
    /// Total number of recorded invocations
    pub total_invocations: u64,
}

/// Sliding-window tracker of model invocations
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    history: Vec<ModelInvocation>,
    window_size: usize,
}

impl PerformanceTracker {
    /// Create a tracker with the given sliding window size
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::new(),
            window_size,
        }
    }

    /// Record an invocation, evicting the oldest if the window is exceeded
    pub fn record(&mut self, invocation: ModelInvocation) {
        self.history.push(invocation);
        while self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }

    /// Compute aggregated stats for a specific model
    pub fn model_stats(&self, model_id: &str) -> Option<ModelPerformanceStats> {
        let entries: Vec<_> = self
            .history
            .iter()
            .filter(|inv| inv.model_id == model_id)
            .collect();

        if entries.is_empty() {
            return None;
        }

        let total = entries.len() as u64;
        let avg_latency_ms =
            entries.iter().map(|e| e.latency_ms as f64).sum::<f64>() / total as f64;
        let success_count = entries.iter().filter(|e| e.success).count() as f64;
        let success_rate = success_count / total as f64;

        let quality_entries: Vec<f64> = entries.iter().filter_map(|e| e.quality_score).collect();
        let avg_quality = if quality_entries.is_empty() {
            0.0
        } else {
            quality_entries.iter().sum::<f64>() / quality_entries.len() as f64
        };

        Some(ModelPerformanceStats {
            model_id: model_id.to_string(),
            avg_latency_ms,
            success_rate,
            avg_quality,
            total_invocations: total,
        })
    }

    /// Find the model with the highest average quality for a task type.
    /// Only considers invocations that have a quality_score.
    pub fn best_model_for(&self, task_type: TaskType) -> Option<String> {
        let mut model_quality: HashMap<String, (f64, usize)> = HashMap::new();

        for inv in &self.history {
            if inv.task_type == task_type {
                if let Some(q) = inv.quality_score {
                    let entry = model_quality
                        .entry(inv.model_id.clone())
                        .or_insert((0.0, 0));
                    entry.0 += q;
                    entry.1 += 1;
                }
            }
        }

        model_quality
            .into_iter()
            .max_by(|a, b| {
                let avg_a = a.1 .0 / a.1 .1 as f64;
                let avg_b = b.1 .0 / b.1 .1 as f64;
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id)
    }

    /// Return all models sorted by success rate, descending
    pub fn reliability_ranking(&self) -> Vec<(String, f64)> {
        let mut model_stats: HashMap<String, (usize, usize)> = HashMap::new();

        for inv in &self.history {
            let entry = model_stats.entry(inv.model_id.clone()).or_insert((0, 0));
            entry.0 += 1;
            if inv.success {
                entry.1 += 1;
            }
        }

        let mut ranking: Vec<(String, f64)> = model_stats
            .into_iter()
            .map(|(id, (total, successes))| (id, successes as f64 / total as f64))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranking
    }

    /// Number of invocations currently in history
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

/// Combines cost registry, performance tracking, and fallback into a single selector
pub struct SmartSelector {
    /// Cost registry for budget calculations
    pub cost_registry: ModelCostRegistry,
    /// Performance tracker for quality-based selection
    pub tracker: PerformanceTracker,
    /// Fallback chain when budget/quality criteria are not met
    pub fallback: FallbackChain,
    /// Remaining budget for model invocations
    pub budget_remaining: f64,
}

impl SmartSelector {
    /// Create a new smart selector
    pub fn new(
        cost_registry: ModelCostRegistry,
        tracker: PerformanceTracker,
        fallback: FallbackChain,
        budget: f64,
    ) -> Self {
        Self {
            cost_registry,
            tracker,
            fallback,
            budget_remaining: budget,
        }
    }

    /// Select the best model considering budget, quality, and fallback
    pub fn select(&self, task_type: TaskType, estimated_tokens: usize) -> Option<String> {
        let within_budget = self
            .cost_registry
            .within_budget(self.budget_remaining, estimated_tokens);

        if !within_budget.is_empty() {
            // If the tracker knows the best model for this task and it is within budget, use it
            if let Some(best) = self.tracker.best_model_for(task_type) {
                if within_budget.contains(&best) {
                    return Some(best);
                }
            }

            // Otherwise pick the cheapest within budget
            if let Some(cheapest) = self.cost_registry.cheapest_for(estimated_tokens) {
                if within_budget.contains(&cheapest.model_id) {
                    return Some(cheapest.model_id.clone());
                }
            }

            // If cheapest_for returned something not in within_budget (shouldn't happen), pick first
            return within_budget.into_iter().next();
        }

        // Nothing within budget — use fallback
        self.fallback.next(&[])
    }

    /// Record an invocation result in both the tracker and the cost registry
    pub fn record_result(&mut self, invocation: ModelInvocation) {
        let model_id = invocation.model_id.clone();
        let latency = invocation.latency_ms;
        let success = invocation.success;
        self.tracker.record(invocation);
        self.cost_registry.update_stats(&model_id, latency, success);
    }

    /// Subtract a cost from the remaining budget
    pub fn adjust_budget(&mut self, spent: f64) {
        self.budget_remaining -= spent;
    }

    /// Current remaining budget
    pub fn remaining_budget(&self) -> f64 {
        self.budget_remaining
    }
}

// ============================================================================
// PipelineRouter (Item 7.1)
// ============================================================================

/// Task types for routing decisions.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineTaskType {
    Chat,
    Summarization,
    CodeGeneration,
    Translation,
    Classification,
    Extraction,
    Creative,
    Analysis,
}

/// A routing rule that maps task patterns to preferred providers.
pub struct RoutingRule {
    pub task_pattern: String,
    pub preferred_provider: String,
    pub preferred_model: String,
    pub priority: u32,
    pub max_cost_per_token: Option<f64>,
}

/// The result of a routing decision.
pub struct PipelineRoutingDecision {
    pub provider: String,
    pub model: String,
    pub reason: String,
    pub estimated_cost_per_1k_tokens: f64,
}

/// Routes requests to the optimal provider/model based on task type, cost, and quality.
pub struct PipelineRouter {
    pub rules: Vec<RoutingRule>,
    pub default_provider: String,
    pub default_model: String,
    pub cost_weight: f64,
    pub quality_weight: f64,
    pub speed_weight: f64,
}

impl PipelineRouter {
    pub fn new(default_provider: &str, default_model: &str) -> Self {
        Self {
            rules: Vec::new(),
            default_provider: default_provider.to_string(),
            default_model: default_model.to_string(),
            cost_weight: 0.33,
            quality_weight: 0.34,
            speed_weight: 0.33,
        }
    }

    pub fn with_weights(mut self, cost: f64, quality: f64, speed: f64) -> Self {
        self.cost_weight = cost;
        self.quality_weight = quality;
        self.speed_weight = speed;
        self
    }

    pub fn add_rule(mut self, rule: RoutingRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Route a request to the best provider/model.
    pub fn route(&self, task_type: &PipelineTaskType, _prompt: &str) -> PipelineRoutingDecision {
        let task_keyword = match task_type {
            PipelineTaskType::Chat => "chat",
            PipelineTaskType::Summarization => "summar",
            PipelineTaskType::CodeGeneration => "code",
            PipelineTaskType::Translation => "translat",
            PipelineTaskType::Classification => "classif",
            PipelineTaskType::Extraction => "extract",
            PipelineTaskType::Creative => "creative",
            PipelineTaskType::Analysis => "analy",
        };

        // Find matching rule by task pattern
        let mut best_rule: Option<&RoutingRule> = None;
        for rule in &self.rules {
            if rule.task_pattern.to_lowercase().contains(task_keyword)
                || task_keyword.contains(&rule.task_pattern.to_lowercase())
            {
                match best_rule {
                    None => best_rule = Some(rule),
                    Some(current) if rule.priority > current.priority => best_rule = Some(rule),
                    _ => {}
                }
            }
        }

        match best_rule {
            Some(rule) => PipelineRoutingDecision {
                provider: rule.preferred_provider.clone(),
                model: rule.preferred_model.clone(),
                reason: format!("Matched rule for task pattern '{}'", rule.task_pattern),
                estimated_cost_per_1k_tokens: 0.005,
            },
            None => PipelineRoutingDecision {
                provider: self.default_provider.clone(),
                model: self.default_model.clone(),
                reason: "Default routing (no matching rule)".to_string(),
                estimated_cost_per_1k_tokens: 0.003,
            },
        }
    }

    /// Simple heuristic to classify task type from prompt text.
    pub fn classify_task(prompt: &str) -> PipelineTaskType {
        let lower = prompt.to_lowercase();
        if lower.contains("summarize") || lower.contains("summary") || lower.contains("tldr") {
            PipelineTaskType::Summarization
        } else if lower.contains("translate") || lower.contains("translation") {
            PipelineTaskType::Translation
        } else if lower.contains("code") || lower.contains("function") || lower.contains("implement")
            || lower.contains("program") || lower.contains("script")
        {
            PipelineTaskType::CodeGeneration
        } else if lower.contains("classify") || lower.contains("categorize") || lower.contains("label") {
            PipelineTaskType::Classification
        } else if lower.contains("extract") || lower.contains("parse") || lower.contains("find all") {
            PipelineTaskType::Extraction
        } else if lower.contains("write a story") || lower.contains("creative") || lower.contains("poem") {
            PipelineTaskType::Creative
        } else if lower.contains("analyze") || lower.contains("analysis") || lower.contains("compare") {
            PipelineTaskType::Analysis
        } else {
            PipelineTaskType::Chat
        }
    }
}

// ============================================================================
// CacheablePrompt (Item 7.2)
// ============================================================================

/// A prompt segment that can be marked as static (cacheable) or dynamic.
pub enum PromptSegment {
    Static { content: String, cache_key: String },
    Dynamic { content: String },
}

/// Marks segments of a prompt as cacheable or dynamic for provider cache optimization.
pub struct CacheablePrompt {
    pub segments: Vec<PromptSegment>,
}

impl CacheablePrompt {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn add_static(mut self, content: &str, cache_key: &str) -> Self {
        self.segments.push(PromptSegment::Static {
            content: content.to_string(),
            cache_key: cache_key.to_string(),
        });
        self
    }

    pub fn add_dynamic(mut self, content: &str) -> Self {
        self.segments.push(PromptSegment::Dynamic {
            content: content.to_string(),
        });
        self
    }

    pub fn to_full_prompt(&self) -> String {
        self.segments
            .iter()
            .map(|s| match s {
                PromptSegment::Static { content, .. } => content.as_str(),
                PromptSegment::Dynamic { content } => content.as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Hash of static segments only — same static content means same fingerprint.
    pub fn cache_fingerprint(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for seg in &self.segments {
            if let PromptSegment::Static { content, cache_key } = seg {
                cache_key.hash(&mut hasher);
                content.hash(&mut hasher);
            }
        }
        format!("{:016x}", hasher.finish())
    }

    /// Anthropic cache_control format blocks.
    pub fn to_anthropic_cache_control(&self) -> Vec<serde_json::Value> {
        self.segments
            .iter()
            .map(|s| match s {
                PromptSegment::Static { content, .. } => serde_json::json!({
                    "type": "text",
                    "text": content,
                    "cache_control": { "type": "ephemeral" }
                }),
                PromptSegment::Dynamic { content } => serde_json::json!({
                    "type": "text",
                    "text": content
                }),
            })
            .collect()
    }

    /// Percentage of tokens that are in static segments.
    pub fn static_ratio(&self) -> f64 {
        let mut static_len = 0usize;
        let mut total_len = 0usize;
        for seg in &self.segments {
            let len = match seg {
                PromptSegment::Static { content, .. } => content.len(),
                PromptSegment::Dynamic { content } => content.len(),
            };
            total_len += len;
            if matches!(seg, PromptSegment::Static { .. }) {
                static_len += len;
            }
        }
        if total_len == 0 {
            0.0
        } else {
            static_len as f64 / total_len as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_detection() {
        assert_eq!(
            TaskType::from_prompt("Write code that sorts an array"),
            TaskType::Coding
        );
        assert_eq!(TaskType::from_prompt("Calculate 2 + 2"), TaskType::Math);
        assert_eq!(
            TaskType::from_prompt("Translate to Spanish"),
            TaskType::Translation
        );
        assert_eq!(
            TaskType::from_prompt("Summarize this article"),
            TaskType::Summarization
        );
    }

    #[test]
    fn test_model_selection() {
        let mut selector = AutoModelSelector::default();

        let mut task_quality = HashMap::new();
        task_quality.insert(TaskType::Coding, 0.9);

        selector.add_model(ModelProfile {
            id: "coder".to_string(),
            name: "Code Model".to_string(),
            provider: "test".to_string(),
            capabilities: ModelCapabilities {
                code: true,
                ..Default::default()
            },
            cost_input: 0.001,
            cost_output: 0.002,
            avg_latency: Duration::from_millis(500),
            task_quality,
            max_context: 8192,
            available: true,
        });

        // Use "write code" which triggers Coding task type
        let result = selector.select("Write code for a function that sorts an array", None);
        assert_eq!(result.model_id, "coder");
        assert_eq!(result.task_type, TaskType::Coding);
    }

    #[test]
    fn test_requirements() {
        let reqs = Requirements::new()
            .needs_code()
            .max_cost(0.01)
            .max_latency(Duration::from_secs(5));

        assert!(reqs.needs_code);
        assert_eq!(reqs.max_cost, Some(0.01));
    }

    #[test]
    fn test_learning() {
        let mut selector = AutoModelSelector::new(AutoSelectConfig {
            enable_learning: true,
            ..Default::default()
        });

        selector.record_outcome(
            "model1",
            TaskType::Coding,
            true,
            0.9,
            Duration::from_secs(1),
        );
        selector.record_outcome(
            "model1",
            TaskType::Coding,
            true,
            0.8,
            Duration::from_secs(2),
        );

        let stats = selector.model_stats("model1");
        assert!(stats.contains_key(&TaskType::Coding));
    }

    // ─── WS8 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_cost_registry_crud() {
        let mut reg = ModelCostRegistry::new();
        assert_eq!(reg.model_count(), 0);

        reg.register(ModelCostEntry::new("gpt4", 30.0, 60.0));
        reg.register(ModelCostEntry::new("gpt3", 1.5, 2.0));

        assert_eq!(reg.model_count(), 2);
        let entry = reg.get("gpt4").unwrap();
        assert_eq!(entry.input_cost_per_million, 30.0);
        assert_eq!(entry.output_cost_per_million, 60.0);
        assert_eq!(entry.total_requests, 0);
        assert!((entry.success_rate - 1.0).abs() < f64::EPSILON);

        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_cost_registry_within_budget() {
        let mut reg = ModelCostRegistry::new();
        reg.register(ModelCostEntry::new("cheap", 1.0, 2.0));
        reg.register(ModelCostEntry::new("mid", 10.0, 20.0));
        reg.register(ModelCostEntry::new("expensive", 100.0, 200.0));

        // Budget = 0.02, tokens = 1000
        // cheap cost = 1.0 * 1000 / 1_000_000 = 0.001 ✓
        // mid cost   = 10.0 * 1000 / 1_000_000 = 0.01  ✓
        // expensive  = 100.0 * 1000 / 1_000_000 = 0.1   ✗
        let mut within = reg.within_budget(0.02, 1000);
        within.sort();
        assert_eq!(within.len(), 2);
        assert!(within.contains(&"cheap".to_string()));
        assert!(within.contains(&"mid".to_string()));
    }

    #[test]
    fn test_cost_registry_cheapest_for() {
        let mut reg = ModelCostRegistry::new();
        reg.register(ModelCostEntry::new("a", 5.0, 10.0));
        reg.register(ModelCostEntry::new("b", 1.0, 3.0));
        reg.register(ModelCostEntry::new("c", 20.0, 40.0));

        let cheapest = reg.cheapest_for(5000).unwrap();
        assert_eq!(cheapest.model_id, "b");
    }

    #[test]
    fn test_cost_registry_update_stats() {
        let mut reg = ModelCostRegistry::new();
        reg.register(ModelCostEntry::new("m1", 5.0, 10.0));

        // First request: latency 100, success
        reg.update_stats("m1", 100, true);
        let e = reg.get("m1").unwrap();
        assert_eq!(e.total_requests, 1);
        assert_eq!(e.avg_latency_ms, 100);
        assert!((e.success_rate - 1.0).abs() < f64::EPSILON);

        // Second request: latency 200, failure
        reg.update_stats("m1", 200, false);
        let e = reg.get("m1").unwrap();
        assert_eq!(e.total_requests, 2);
        assert_eq!(e.avg_latency_ms, 150); // (100 + 200) / 2
        assert!((e.success_rate - 0.5).abs() < f64::EPSILON);

        // Third request: latency 300, success
        reg.update_stats("m1", 300, true);
        let e = reg.get("m1").unwrap();
        assert_eq!(e.total_requests, 3);
        assert_eq!(e.avg_latency_ms, 200); // (100+200+300)/3
                                           // success_rate = (1+0+1)/3 ≈ 0.6667
        assert!((e.success_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fallback_chain_sequential() {
        let mut chain = FallbackChain::new(FallbackStrategy::Sequential);
        chain.add_model("a");
        chain.add_model("b");
        chain.add_model("c");

        assert_eq!(chain.len(), 3);
        assert!(!chain.is_empty());

        // No models tried yet
        assert_eq!(chain.next(&[]), Some("a".to_string()));
        // "a" already tried
        assert_eq!(chain.next(&["a".to_string()]), Some("b".to_string()));
        // "a" and "b" tried
        assert_eq!(
            chain.next(&["a".to_string(), "b".to_string()]),
            Some("c".to_string())
        );
        // All tried
        assert_eq!(
            chain.next(&["a".to_string(), "b".to_string(), "c".to_string()]),
            None
        );
    }

    #[test]
    fn test_fallback_chain_sort_by_cost() {
        let mut reg = ModelCostRegistry::new();
        reg.register(ModelCostEntry::new("expensive", 50.0, 100.0));
        reg.register(ModelCostEntry::new("cheap", 1.0, 2.0));
        reg.register(ModelCostEntry::new("mid", 10.0, 20.0));

        let mut chain = FallbackChain::new(FallbackStrategy::CostAscending).with_models(vec![
            "expensive".to_string(),
            "cheap".to_string(),
            "mid".to_string(),
        ]);

        chain.sort_by_cost(&reg);

        // After sorting: cheap, mid, expensive
        assert_eq!(chain.next(&[]), Some("cheap".to_string()));
        assert_eq!(chain.next(&["cheap".to_string()]), Some("mid".to_string()));
        assert_eq!(
            chain.next(&["cheap".to_string(), "mid".to_string()]),
            Some("expensive".to_string())
        );
    }

    #[test]
    fn test_performance_tracker_record_stats() {
        let mut tracker = PerformanceTracker::new(100);

        tracker
            .record(ModelInvocation::new("m1", TaskType::Coding, 100, 500, true).with_quality(0.9));
        tracker
            .record(ModelInvocation::new("m1", TaskType::Coding, 200, 600, true).with_quality(0.7));

        let stats = tracker.model_stats("m1").unwrap();
        assert_eq!(stats.total_invocations, 2);
        assert!((stats.avg_latency_ms - 150.0).abs() < f64::EPSILON);
        assert!((stats.success_rate - 1.0).abs() < f64::EPSILON);
        assert!((stats.avg_quality - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_tracker_best_model() {
        let mut tracker = PerformanceTracker::new(100);

        // m1 has quality 0.7 average for Coding
        tracker
            .record(ModelInvocation::new("m1", TaskType::Coding, 100, 500, true).with_quality(0.7));
        // m2 has quality 0.95 average for Coding
        tracker.record(
            ModelInvocation::new("m2", TaskType::Coding, 150, 400, true).with_quality(0.95),
        );
        // m2 also has a lower quality entry for Math — should not affect Coding
        tracker
            .record(ModelInvocation::new("m2", TaskType::Math, 120, 300, true).with_quality(0.5));

        let best = tracker.best_model_for(TaskType::Coding).unwrap();
        assert_eq!(best, "m2");

        let best_math = tracker.best_model_for(TaskType::Math).unwrap();
        assert_eq!(best_math, "m2"); // only m2 has Math quality entries
    }

    #[test]
    fn test_performance_tracker_reliability() {
        let mut tracker = PerformanceTracker::new(100);

        // m1: 2 successes out of 2 => 1.0
        tracker.record(ModelInvocation::new("m1", TaskType::Coding, 100, 500, true));
        tracker.record(ModelInvocation::new("m1", TaskType::Coding, 100, 500, true));
        // m2: 1 success out of 2 => 0.5
        tracker.record(ModelInvocation::new("m2", TaskType::Coding, 100, 500, true));
        tracker.record(ModelInvocation::new(
            "m2",
            TaskType::Coding,
            100,
            500,
            false,
        ));
        // m3: 0 successes out of 1 => 0.0
        tracker.record(ModelInvocation::new(
            "m3",
            TaskType::Coding,
            100,
            500,
            false,
        ));

        let ranking = tracker.reliability_ranking();
        assert_eq!(ranking.len(), 3);
        assert_eq!(ranking[0].0, "m1");
        assert!((ranking[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(ranking[1].0, "m2");
        assert!((ranking[1].1 - 0.5).abs() < f64::EPSILON);
        assert_eq!(ranking[2].0, "m3");
        assert!((ranking[2].1 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_tracker_window_eviction() {
        let mut tracker = PerformanceTracker::new(5);

        for i in 0..10 {
            tracker.record(ModelInvocation::new(
                &format!("m{}", i),
                TaskType::General,
                100,
                500,
                true,
            ));
        }

        assert_eq!(tracker.history_len(), 5);
        // Only the last 5 (m5..m9) should remain
        assert!(tracker.model_stats("m0").is_none());
        assert!(tracker.model_stats("m4").is_none());
        assert!(tracker.model_stats("m5").is_some());
        assert!(tracker.model_stats("m9").is_some());
    }

    #[test]
    fn test_smart_selector_budget_aware() {
        let mut reg = ModelCostRegistry::new();
        reg.register(ModelCostEntry::new("cheap", 1.0, 2.0));
        reg.register(ModelCostEntry::new("expensive", 1000.0, 2000.0));

        let tracker = PerformanceTracker::new(100);
        let mut fallback = FallbackChain::new(FallbackStrategy::Sequential);
        fallback.add_model("fallback_model");

        // Budget = 0.01, tokens = 1000
        // cheap cost    = 1.0 * 1000 / 1_000_000 = 0.001 ✓
        // expensive cost = 1000 * 1000 / 1_000_000 = 1.0    ✗
        let selector = SmartSelector::new(reg, tracker, fallback, 0.01);
        let selected = selector.select(TaskType::General, 1000);
        assert_eq!(selected, Some("cheap".to_string()));
    }

    #[test]
    fn test_smart_selector_record_and_adjust() {
        let reg = ModelCostRegistry::new();
        let tracker = PerformanceTracker::new(100);
        let fallback = FallbackChain::new(FallbackStrategy::Sequential);

        let mut selector = SmartSelector::new(reg, tracker, fallback, 10.0);
        assert!((selector.remaining_budget() - 10.0).abs() < f64::EPSILON);

        // Record a result
        selector.record_result(
            ModelInvocation::new("m1", TaskType::Coding, 150, 1000, true).with_quality(0.85),
        );

        // Tracker should have 1 entry
        assert_eq!(selector.tracker.history_len(), 1);
        let stats = selector.tracker.model_stats("m1").unwrap();
        assert_eq!(stats.total_invocations, 1);

        // Adjust budget
        selector.adjust_budget(3.5);
        assert!((selector.remaining_budget() - 6.5).abs() < f64::EPSILON);

        selector.adjust_budget(6.5);
        assert!((selector.remaining_budget() - 0.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // PipelineRouter tests (Item 7.1)
    // ========================================================================

    #[test]
    fn test_router_new() {
        let router = PipelineRouter::new("openai", "gpt-4o");
        assert_eq!(router.default_provider, "openai");
        assert_eq!(router.default_model, "gpt-4o");
    }

    #[test]
    fn test_router_add_rule() {
        let router = PipelineRouter::new("openai", "gpt-4o")
            .add_rule(RoutingRule {
                task_pattern: "code".to_string(),
                preferred_provider: "anthropic".to_string(),
                preferred_model: "claude-3.5-sonnet".to_string(),
                priority: 10,
                max_cost_per_token: None,
            });
        assert_eq!(router.rules.len(), 1);
    }

    #[test]
    fn test_router_route_default() {
        let router = PipelineRouter::new("openai", "gpt-4o");
        let decision = router.route(&PipelineTaskType::Chat, "hello there");
        assert_eq!(decision.provider, "openai");
        assert_eq!(decision.model, "gpt-4o");
    }

    #[test]
    fn test_router_route_with_rule() {
        let router = PipelineRouter::new("openai", "gpt-4o")
            .add_rule(RoutingRule {
                task_pattern: "code".to_string(),
                preferred_provider: "anthropic".to_string(),
                preferred_model: "claude-3.5-sonnet".to_string(),
                priority: 10,
                max_cost_per_token: None,
            });
        let decision = router.route(&PipelineTaskType::CodeGeneration, "write code");
        assert_eq!(decision.provider, "anthropic");
        assert_eq!(decision.model, "claude-3.5-sonnet");
    }

    #[test]
    fn test_router_with_weights() {
        let router = PipelineRouter::new("openai", "gpt-4o")
            .with_weights(0.5, 0.3, 0.2);
        assert!((router.cost_weight - 0.5).abs() < f64::EPSILON);
        assert!((router.quality_weight - 0.3).abs() < f64::EPSILON);
        assert!((router.speed_weight - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_routing_decision() {
        let decision = PipelineRoutingDecision {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            reason: "default".to_string(),
            estimated_cost_per_1k_tokens: 0.005,
        };
        assert_eq!(decision.provider, "openai");
        assert!(decision.estimated_cost_per_1k_tokens > 0.0);
    }

    #[test]
    fn test_classify_task_chat() {
        let task = PipelineRouter::classify_task("hello, how are you?");
        assert!(matches!(task, PipelineTaskType::Chat));
    }

    #[test]
    fn test_classify_task_code() {
        let task = PipelineRouter::classify_task("write a function to sort an array");
        assert!(matches!(task, PipelineTaskType::CodeGeneration));
    }

    #[test]
    fn test_classify_task_summarize() {
        let task = PipelineRouter::classify_task("summarize this article about AI");
        assert!(matches!(task, PipelineTaskType::Summarization));
    }

    #[test]
    fn test_classify_task_translate() {
        let task = PipelineRouter::classify_task("translate this to Spanish");
        assert!(matches!(task, PipelineTaskType::Translation));
    }

    // ========================================================================
    // CacheablePrompt tests (Item 7.2)
    // ========================================================================

    #[test]
    fn test_cacheable_prompt_new() {
        let prompt = CacheablePrompt::new();
        assert!(prompt.segments.is_empty());
    }

    #[test]
    fn test_cacheable_prompt_add_segments() {
        let prompt = CacheablePrompt::new()
            .add_static("System: You are a helpful assistant.", "sys")
            .add_dynamic("User: What is Rust?");
        assert_eq!(prompt.segments.len(), 2);
    }

    #[test]
    fn test_cacheable_prompt_full_prompt() {
        let prompt = CacheablePrompt::new()
            .add_static("System prompt.", "sys")
            .add_dynamic("User message.");
        let full = prompt.to_full_prompt();
        assert!(full.contains("System prompt."));
        assert!(full.contains("User message."));
    }

    #[test]
    fn test_cacheable_prompt_fingerprint() {
        let p1 = CacheablePrompt::new()
            .add_static("same system", "sys")
            .add_dynamic("different user 1");
        let p2 = CacheablePrompt::new()
            .add_static("same system", "sys")
            .add_dynamic("different user 2");
        // Same static content → same fingerprint
        assert_eq!(p1.cache_fingerprint(), p2.cache_fingerprint());
    }

    #[test]
    fn test_cacheable_prompt_static_ratio() {
        let prompt = CacheablePrompt::new()
            .add_static("1234567890", "key")  // 10 chars
            .add_dynamic("12345");            // 5 chars
        let ratio = prompt.static_ratio();
        assert!((ratio - 10.0 / 15.0).abs() < 0.01);
    }

    #[test]
    fn test_cacheable_prompt_anthropic_cache() {
        let prompt = CacheablePrompt::new()
            .add_static("System instructions.", "sys")
            .add_dynamic("User query.");
        let blocks = prompt.to_anthropic_cache_control();
        assert_eq!(blocks.len(), 2);
    }
}
