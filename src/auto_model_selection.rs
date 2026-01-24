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
        if lower.contains("summarize")
            || lower.contains("summary")
            || lower.contains("tl;dr")
        {
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
        if lower.contains("analyze")
            || lower.contains("data")
            || lower.contains("statistics")
        {
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
        let available: Vec<_> = self.config.models.iter()
            .filter(|m| m.available)
            .filter(|m| {
                // Check requirements if provided
                if let Some(req) = requirements {
                    if req.needs_code && !m.capabilities.code { return false; }
                    if req.needs_vision && !m.capabilities.vision { return false; }
                    if req.needs_function_calling && !m.capabilities.function_calling { return false; }
                    if req.max_cost.is_some() && m.cost_input > req.max_cost.unwrap() { return false; }
                    if req.max_latency.is_some() && m.avg_latency > req.max_latency.unwrap() { return false; }
                    if req.min_context.is_some() && m.max_context < req.min_context.unwrap() { return false; }
                }
                true
            })
            .collect();

        if available.is_empty() {
            // Return default model
            let default = self.config.models.iter()
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
        let mut scored: Vec<(f64, &ModelProfile)> = available.iter()
            .map(|m| (self.score_model(m, task_type, prompt_length), *m))
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let best = scored[0].1;
        let score = scored[0].0;

        // Get alternatives
        let alternatives: Vec<_> = scored.iter()
            .skip(1)
            .take(3)
            .map(|(_, m)| m.id.clone())
            .collect();

        // Estimate cost
        let estimated_tokens = (prompt_length / 4 + 500) as f64 / 1000.0;
        let estimated_cost = estimated_tokens * best.cost_input + estimated_tokens * best.cost_output;

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
        let record = self.history.entry(key).or_insert_with(PerformanceRecord::default);

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
                stats.insert(*task_type, ModelStats {
                    total_requests: record.total,
                    success_rate: record.successes as f64 / record.total as f64,
                    avg_quality: record.total_quality / record.total as f64,
                    avg_latency: record.total_latency / record.total as u32,
                });
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
        Self { config: AutoSelectConfig::default() }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_detection() {
        assert_eq!(TaskType::from_prompt("Write code that sorts an array"), TaskType::Coding);
        assert_eq!(TaskType::from_prompt("Calculate 2 + 2"), TaskType::Math);
        assert_eq!(TaskType::from_prompt("Translate to Spanish"), TaskType::Translation);
        assert_eq!(TaskType::from_prompt("Summarize this article"), TaskType::Summarization);
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
            capabilities: ModelCapabilities { code: true, ..Default::default() },
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

        selector.record_outcome("model1", TaskType::Coding, true, 0.9, Duration::from_secs(1));
        selector.record_outcome("model1", TaskType::Coding, true, 0.8, Duration::from_secs(2));

        let stats = selector.model_stats("model1");
        assert!(stats.contains_key(&TaskType::Coding));
    }
}
