//! Model routing for intelligent model selection
//!
//! This module provides a router that selects the optimal model based on
//! the type of task, requirements, and available models.

use crate::{AiProvider, ModelInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Task type for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TaskType {
    /// General chat/conversation
    Chat,
    /// Code generation
    Coding,
    /// Code review/analysis
    CodeReview,
    /// Creative writing
    Creative,
    /// Technical explanation
    Technical,
    /// Translation
    Translation,
    /// Summarization
    Summarization,
    /// Question answering
    QA,
    /// Math/reasoning
    Math,
    /// Data analysis
    Analysis,
    /// Image understanding (requires vision model)
    Vision,
    /// Function calling (requires function-capable model)
    FunctionCalling,
    /// Long context tasks
    LongContext,
    /// Fast response needed
    FastResponse,
}

/// Requirements for model selection
#[derive(Debug, Clone, Default)]
pub struct ModelRequirements {
    /// Required task type
    pub task_type: Option<TaskType>,
    /// Minimum context size
    pub min_context_size: Option<usize>,
    /// Must support vision
    pub requires_vision: bool,
    /// Must support function calling
    pub requires_functions: bool,
    /// Must support streaming
    pub requires_streaming: bool,
    /// Preferred providers (in order)
    pub preferred_providers: Vec<AiProvider>,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: Option<u64>,
    /// Minimum quality score (0-100)
    pub min_quality_score: Option<u32>,
}

impl ModelRequirements {
    /// Create requirements for a task type
    pub fn for_task(task_type: TaskType) -> Self {
        let mut req = Self::default();
        req.task_type = Some(task_type);

        // Set defaults based on task type
        match task_type {
            TaskType::Vision => {
                req.requires_vision = true;
            }
            TaskType::FunctionCalling => {
                req.requires_functions = true;
            }
            TaskType::LongContext => {
                req.min_context_size = Some(32000);
            }
            TaskType::FastResponse => {
                req.max_latency_ms = Some(1000);
            }
            _ => {}
        }

        req
    }

    /// Require vision capability
    pub fn with_vision(mut self) -> Self {
        self.requires_vision = true;
        self
    }

    /// Require function calling
    pub fn with_functions(mut self) -> Self {
        self.requires_functions = true;
        self
    }

    /// Set minimum context size
    pub fn with_min_context(mut self, size: usize) -> Self {
        self.min_context_size = Some(size);
        self
    }

    /// Set preferred providers
    pub fn prefer_providers(mut self, providers: Vec<AiProvider>) -> Self {
        self.preferred_providers = providers;
        self
    }
}

/// Model capabilities profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilityProfile {
    /// Model name pattern (substring match)
    pub model_pattern: String,
    /// Supported task types with quality scores (0-100)
    pub task_scores: HashMap<TaskType, u32>,
    /// Context size
    pub context_size: usize,
    /// Supports vision
    pub supports_vision: bool,
    /// Supports function calling
    pub supports_functions: bool,
    /// Supports streaming
    pub supports_streaming: bool,
    /// Average latency (ms)
    pub avg_latency_ms: u64,
    /// Overall quality score
    pub quality_score: u32,
}

impl ModelCapabilityProfile {
    /// Create a new profile
    pub fn new(pattern: &str, context_size: usize, quality_score: u32) -> Self {
        Self {
            model_pattern: pattern.to_string(),
            task_scores: HashMap::new(),
            context_size,
            supports_vision: false,
            supports_functions: false,
            supports_streaming: true,
            avg_latency_ms: 500,
            quality_score,
        }
    }

    /// Set task score
    pub fn with_task_score(mut self, task: TaskType, score: u32) -> Self {
        self.task_scores.insert(task, score.min(100));
        self
    }

    /// Enable vision
    pub fn with_vision(mut self) -> Self {
        self.supports_vision = true;
        self
    }

    /// Enable function calling
    pub fn with_functions(mut self) -> Self {
        self.supports_functions = true;
        self
    }

    /// Check if profile matches a model name
    pub fn matches(&self, model_name: &str) -> bool {
        model_name
            .to_lowercase()
            .contains(&self.model_pattern.to_lowercase())
    }

    /// Get score for a task type
    pub fn get_task_score(&self, task: TaskType) -> u32 {
        *self.task_scores.get(&task).unwrap_or(&self.quality_score)
    }
}

/// Model router for intelligent selection
pub struct ModelRouter {
    /// Known capability profiles
    profiles: Vec<ModelCapabilityProfile>,
    /// Custom scoring function
    custom_scorer: Option<Box<dyn Fn(&ModelInfo, &ModelRequirements) -> u32 + Send + Sync>>,
    /// Fallback model name
    fallback_model: Option<String>,
}

impl ModelRouter {
    /// Create a new router with default profiles
    pub fn new() -> Self {
        let mut router = Self {
            profiles: Vec::new(),
            custom_scorer: None,
            fallback_model: None,
        };

        // Add default profiles for common models
        router.add_default_profiles();
        router
    }

    /// Add default model profiles
    fn add_default_profiles(&mut self) {
        // Llama models
        self.profiles.push(
            ModelCapabilityProfile::new("llama-3", 128000, 85)
                .with_task_score(TaskType::Chat, 90)
                .with_task_score(TaskType::Coding, 85)
                .with_task_score(TaskType::Creative, 80)
                .with_functions(),
        );

        self.profiles.push(
            ModelCapabilityProfile::new("llama-2", 4096, 70)
                .with_task_score(TaskType::Chat, 75)
                .with_task_score(TaskType::Coding, 65),
        );

        // Qwen models
        self.profiles.push(
            ModelCapabilityProfile::new("qwen2.5", 32000, 88)
                .with_task_score(TaskType::Coding, 95)
                .with_task_score(TaskType::Math, 90)
                .with_task_score(TaskType::Technical, 88)
                .with_functions(),
        );

        // Mistral/Mixtral
        self.profiles.push(
            ModelCapabilityProfile::new("mistral", 32000, 82)
                .with_task_score(TaskType::Chat, 85)
                .with_task_score(TaskType::Coding, 80)
                .with_functions(),
        );

        self.profiles.push(
            ModelCapabilityProfile::new("mixtral", 32000, 85)
                .with_task_score(TaskType::Chat, 88)
                .with_task_score(TaskType::Coding, 85)
                .with_task_score(TaskType::Technical, 85)
                .with_functions(),
        );

        // CodeLlama
        self.profiles.push(
            ModelCapabilityProfile::new("codellama", 16000, 78)
                .with_task_score(TaskType::Coding, 90)
                .with_task_score(TaskType::CodeReview, 85),
        );

        // DeepSeek
        self.profiles.push(
            ModelCapabilityProfile::new("deepseek", 32000, 86)
                .with_task_score(TaskType::Coding, 92)
                .with_task_score(TaskType::Math, 88)
                .with_functions(),
        );

        // Vision models
        self.profiles.push(
            ModelCapabilityProfile::new("llava", 4096, 75)
                .with_task_score(TaskType::Vision, 85)
                .with_vision(),
        );

        self.profiles.push(
            ModelCapabilityProfile::new("moondream", 4096, 70)
                .with_task_score(TaskType::Vision, 75)
                .with_vision(),
        );

        // Phi models (fast)
        self.profiles.push(
            ModelCapabilityProfile::new("phi", 4096, 72)
                .with_task_score(TaskType::Chat, 70)
                .with_task_score(TaskType::FastResponse, 90),
        );

        // Gemma
        self.profiles.push(
            ModelCapabilityProfile::new("gemma", 8000, 78)
                .with_task_score(TaskType::Chat, 80)
                .with_task_score(TaskType::Technical, 75),
        );
    }

    /// Add a custom profile
    pub fn add_profile(&mut self, profile: ModelCapabilityProfile) {
        self.profiles.push(profile);
    }

    /// Set a custom scoring function
    pub fn set_custom_scorer<F>(&mut self, scorer: F)
    where
        F: Fn(&ModelInfo, &ModelRequirements) -> u32 + Send + Sync + 'static,
    {
        self.custom_scorer = Some(Box::new(scorer));
    }

    /// Set fallback model
    pub fn set_fallback(&mut self, model_name: &str) {
        self.fallback_model = Some(model_name.to_string());
    }

    /// Get the capability profile for a model
    pub fn get_profile(&self, model_name: &str) -> Option<&ModelCapabilityProfile> {
        self.profiles.iter().find(|p| p.matches(model_name))
    }

    /// Score a model for given requirements
    pub fn score_model(&self, model: &ModelInfo, requirements: &ModelRequirements) -> u32 {
        // Use custom scorer if available
        if let Some(ref scorer) = self.custom_scorer {
            return scorer(model, requirements);
        }

        let profile = self.get_profile(&model.name);

        let mut score = 50u32; // Base score

        if let Some(profile) = profile {
            // Task-specific score
            if let Some(task_type) = requirements.task_type {
                score = profile.get_task_score(task_type);
            } else {
                score = profile.quality_score;
            }

            // Check hard requirements
            if requirements.requires_vision && !profile.supports_vision {
                return 0;
            }
            if requirements.requires_functions && !profile.supports_functions {
                return 0;
            }
            if requirements.requires_streaming && !profile.supports_streaming {
                return 0;
            }

            // Context size requirement
            if let Some(min_ctx) = requirements.min_context_size {
                if profile.context_size < min_ctx {
                    return 0;
                }
                // Bonus for larger context
                if profile.context_size >= min_ctx * 2 {
                    score = score.saturating_add(5);
                }
            }

            // Latency requirement
            if let Some(max_latency) = requirements.max_latency_ms {
                if profile.avg_latency_ms > max_latency {
                    score = score.saturating_sub(20);
                }
            }

            // Quality requirement
            if let Some(min_quality) = requirements.min_quality_score {
                if profile.quality_score < min_quality {
                    score = score.saturating_sub(30);
                }
            }
        }

        // Provider preference bonus
        if !requirements.preferred_providers.is_empty() {
            for (idx, provider) in requirements.preferred_providers.iter().enumerate() {
                if &model.provider == provider {
                    score = score.saturating_add((10 - idx as u32 * 2).max(2));
                    break;
                }
            }
        }

        score
    }

    /// Select the best model from available models
    pub fn select_best<'a>(
        &self,
        models: &'a [ModelInfo],
        requirements: &ModelRequirements,
    ) -> Option<&'a ModelInfo> {
        if models.is_empty() {
            return None;
        }

        let mut best_model = None;
        let mut best_score = 0u32;

        for model in models {
            let score = self.score_model(model, requirements);
            if score > best_score {
                best_score = score;
                best_model = Some(model);
            }
        }

        // Return best model or fallback
        if best_model.is_none() || best_score == 0 {
            if let Some(ref fallback_name) = self.fallback_model {
                return models.iter().find(|m| m.name.contains(fallback_name));
            }
        }

        best_model
    }

    /// Get ranked models for requirements
    pub fn rank_models<'a>(
        &self,
        models: &'a [ModelInfo],
        requirements: &ModelRequirements,
    ) -> Vec<(&'a ModelInfo, u32)> {
        let mut scored: Vec<_> = models
            .iter()
            .map(|m| (m, self.score_model(m, requirements)))
            .filter(|(_, score)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored
    }

    /// Detect task type from user message
    pub fn detect_task_type(message: &str) -> TaskType {
        let msg_lower = message.to_lowercase();

        // Code-related keywords
        let code_keywords = [
            "code",
            "function",
            "implement",
            "bug",
            "error",
            "compile",
            "syntax",
            "debug",
            "class",
            "method",
            "api",
            "program",
        ];
        let code_score: usize = code_keywords
            .iter()
            .filter(|kw| msg_lower.contains(*kw))
            .count();

        // Review keywords
        if code_score > 0
            && (msg_lower.contains("review")
                || msg_lower.contains("check")
                || msg_lower.contains("analyze"))
        {
            return TaskType::CodeReview;
        }

        if code_score >= 2 {
            return TaskType::Coding;
        }

        // Creative keywords
        let creative_keywords = ["write", "story", "poem", "creative", "imagine", "fiction"];
        if creative_keywords.iter().any(|kw| msg_lower.contains(kw)) {
            return TaskType::Creative;
        }

        // Translation
        if msg_lower.contains("translate") || msg_lower.contains("translation") {
            return TaskType::Translation;
        }

        // Summarization
        if msg_lower.contains("summarize")
            || msg_lower.contains("summary")
            || msg_lower.contains("tldr")
        {
            return TaskType::Summarization;
        }

        // Math/reasoning
        let math_keywords = [
            "calculate",
            "math",
            "equation",
            "solve",
            "compute",
            "formula",
        ];
        if math_keywords.iter().any(|kw| msg_lower.contains(kw)) {
            return TaskType::Math;
        }

        // Analysis
        if msg_lower.contains("analyze")
            || msg_lower.contains("analysis")
            || msg_lower.contains("data")
        {
            return TaskType::Analysis;
        }

        // Technical explanation
        let tech_keywords = [
            "explain",
            "how does",
            "what is",
            "technical",
            "architecture",
        ];
        if tech_keywords.iter().any(|kw| msg_lower.contains(kw)) {
            return TaskType::Technical;
        }

        // Question answering
        if msg_lower.contains('?')
            || msg_lower.starts_with("what")
            || msg_lower.starts_with("who")
            || msg_lower.starts_with("when")
            || msg_lower.starts_with("where")
            || msg_lower.starts_with("why")
        {
            return TaskType::QA;
        }

        // Default to chat
        TaskType::Chat
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Routing decision with explanation
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected model
    pub model_name: String,
    /// Provider
    pub provider: AiProvider,
    /// Score
    pub score: u32,
    /// Detected task type
    pub task_type: TaskType,
    /// Reason for selection
    pub reason: String,
    /// Alternative models considered
    pub alternatives: Vec<(String, u32)>,
}

impl RoutingDecision {
    /// Create a routing decision
    pub fn new(model: &ModelInfo, score: u32, task_type: TaskType, reason: &str) -> Self {
        Self {
            model_name: model.name.clone(),
            provider: model.provider.clone(),
            score,
            task_type,
            reason: reason.to_string(),
            alternatives: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("llama-3-8b", AiProvider::Ollama),
            ModelInfo::new("qwen2.5-coder-7b", AiProvider::Ollama),
            ModelInfo::new("llava-v1.6", AiProvider::Ollama),
            ModelInfo::new("phi-3-mini", AiProvider::Ollama),
            ModelInfo::new("codellama-13b", AiProvider::Ollama),
        ]
    }

    #[test]
    fn test_task_detection() {
        // Code-related should detect Coding (need 2+ code keywords)
        assert_eq!(
            ModelRouter::detect_task_type("implement a function to sort an array"),
            TaskType::Coding
        );
        // Creative writing
        assert_eq!(
            ModelRouter::detect_task_type("Write me a creative poem about nature"),
            TaskType::Creative
        );
        // Translation
        assert_eq!(
            ModelRouter::detect_task_type("Translate this to Spanish"),
            TaskType::Translation
        );
        // Math calculation
        assert_eq!(
            ModelRouter::detect_task_type("Calculate 2 + 2"),
            TaskType::Math
        );
        // Code review
        assert_eq!(
            ModelRouter::detect_task_type("Review this code and check for bugs"),
            TaskType::CodeReview
        );
        // Summarization
        assert_eq!(
            ModelRouter::detect_task_type("Summarize this article"),
            TaskType::Summarization
        );
    }

    #[test]
    fn test_model_selection() {
        let router = ModelRouter::new();
        let models = create_test_models();

        // For coding task
        let requirements = ModelRequirements::for_task(TaskType::Coding);
        let best = router.select_best(&models, &requirements);
        assert!(best.is_some());
        let model_name = &best.unwrap().name;
        assert!(model_name.contains("qwen") || model_name.contains("codellama"));
    }

    #[test]
    fn test_vision_requirement() {
        let router = ModelRouter::new();
        let models = create_test_models();

        let requirements = ModelRequirements::for_task(TaskType::Vision);
        let best = router.select_best(&models, &requirements);

        if let Some(model) = best {
            assert!(model.name.contains("llava"));
        }
    }

    #[test]
    fn test_model_ranking() {
        let router = ModelRouter::new();
        let models = create_test_models();

        let requirements = ModelRequirements::for_task(TaskType::Chat);
        let ranked = router.rank_models(&models, &requirements);

        assert!(!ranked.is_empty());
        // First should have highest score
        if ranked.len() >= 2 {
            assert!(ranked[0].1 >= ranked[1].1);
        }
    }

    #[test]
    fn test_custom_profile() {
        let mut router = ModelRouter::new();

        let custom = ModelCapabilityProfile::new("my-model", 16000, 90)
            .with_task_score(TaskType::Coding, 100);

        router.add_profile(custom);

        let profile = router.get_profile("my-model-7b");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().get_task_score(TaskType::Coding), 100);
    }

    #[test]
    fn test_requirements_defaults() {
        let req = ModelRequirements::default();
        assert!(req.task_type.is_none());
        assert!(req.min_context_size.is_none());
        assert!(!req.requires_vision);
        assert!(!req.requires_functions);
        assert!(!req.requires_streaming);
        assert!(req.preferred_providers.is_empty());
    }

    #[test]
    fn test_requirements_for_task_sets_flags() {
        let vision_req = ModelRequirements::for_task(TaskType::Vision);
        assert!(vision_req.requires_vision);

        let func_req = ModelRequirements::for_task(TaskType::FunctionCalling);
        assert!(func_req.requires_functions);

        let long_ctx_req = ModelRequirements::for_task(TaskType::LongContext);
        assert_eq!(long_ctx_req.min_context_size, Some(32000));

        let fast_req = ModelRequirements::for_task(TaskType::FastResponse);
        assert_eq!(fast_req.max_latency_ms, Some(1000));
    }

    #[test]
    fn test_profile_matches() {
        let profile = ModelCapabilityProfile::new("llama-3", 128000, 85);
        assert!(profile.matches("llama-3-8b"));
        assert!(profile.matches("LLAMA-3-70B"));
        assert!(!profile.matches("mistral-7b"));
    }

    #[test]
    fn test_profile_task_score_fallback() {
        let profile = ModelCapabilityProfile::new("test", 4096, 75)
            .with_task_score(TaskType::Coding, 95);

        assert_eq!(profile.get_task_score(TaskType::Coding), 95);
        // Unset task falls back to quality_score
        assert_eq!(profile.get_task_score(TaskType::Chat), 75);
    }

    #[test]
    fn test_requirements_builder_methods() {
        let req = ModelRequirements::default()
            .with_vision()
            .with_functions()
            .with_min_context(64000);

        assert!(req.requires_vision);
        assert!(req.requires_functions);
        assert_eq!(req.min_context_size, Some(64000));
    }
}
