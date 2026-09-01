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

        // Vision models — local LLMs (specific llava variants come later
        // in the cloud-vision block; the generic `llava` here only matches
        // bare `llava` / `llava:13b` etc.)
        self.profiles.push(
            ModelCapabilityProfile::new("moondream", 4096, 70)
                .with_task_score(TaskType::Vision, 75)
                .with_vision(),
        );

        // Phi models (fast). phi-3.5-vision must precede generic "phi".
        self.profiles.push(
            ModelCapabilityProfile::new("phi-3.5-vision", 128_000, 80)
                .with_task_score(TaskType::Vision, 82)
                .with_task_score(TaskType::Chat, 78)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("phi", 4096, 72)
                .with_task_score(TaskType::Chat, 70)
                .with_task_score(TaskType::FastResponse, 90),
        );

        // Gemma. More specific patterns first so `find()` prefers them.
        // gemma3 (4B/12B/27B) is multimodal with 128K context.
        // Vision score is intentionally below the Qwen2.5-VL profiles: in
        // OCR / document / chart / grounding benchmarks Qwen2.5-VL leads
        // by ~10–15 points. Gemma 3 still earns a vision profile because
        // its 4B variant is one of the few decent edge / on-device VLMs
        // (better latency than Qwen2.5-VL-3B on CPU/iGPU). See
        // `curated_models.rs` for the matching "edge tier" entries.
        self.profiles.push(
            ModelCapabilityProfile::new("gemma3", 128_000, 84)
                .with_task_score(TaskType::Chat, 86)
                .with_task_score(TaskType::Technical, 82)
                .with_task_score(TaskType::Vision, 75)
                .with_task_score(TaskType::FastResponse, 84)
                .with_task_score(TaskType::LongContext, 85)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("gemma2", 8192, 80)
                .with_task_score(TaskType::Chat, 82)
                .with_task_score(TaskType::Technical, 78),
        );
        // Generic gemma (legacy v1) — last so v3/v2 win.
        self.profiles.push(
            ModelCapabilityProfile::new("gemma", 8000, 76)
                .with_task_score(TaskType::Chat, 78)
                .with_task_score(TaskType::Technical, 74),
        );

        // PrismML Bonsai (Qwen3 base, Q1_0 / ternary). Text-only, 64K ctx,
        // tiny footprint — strong for FastResponse / edge. Requires the
        // PrismML llama.cpp fork (see `Butler::model_runtime_hint`).
        // ternary-bonsai must precede bonsai so the more specific pattern wins.
        self.profiles.push(
            ModelCapabilityProfile::new("ternary-bonsai", 65_536, 75)
                .with_task_score(TaskType::Chat, 76)
                .with_task_score(TaskType::FastResponse, 92)
                .with_task_score(TaskType::LongContext, 78),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("bonsai", 65_536, 74)
                .with_task_score(TaskType::Chat, 75)
                .with_task_score(TaskType::FastResponse, 94)
                .with_task_score(TaskType::LongContext, 78),
        );

        // ===== Cloud vision-capable models =====
        // These mirror the adapters in openai_adapter.rs / anthropic_adapter.rs /
        // cloud_providers.rs so `select_best(Vision)` can pick them. Order is
        // specific-before-generic so `find()` resolves the right profile.

        // OpenAI GPT-4o (multimodal flagship)
        self.profiles.push(
            ModelCapabilityProfile::new("gpt-4o", 128_000, 95)
                .with_task_score(TaskType::Chat, 95)
                .with_task_score(TaskType::Coding, 92)
                .with_task_score(TaskType::Vision, 92)
                .with_task_score(TaskType::Analysis, 92)
                .with_task_score(TaskType::LongContext, 90)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("gpt-4-turbo", 128_000, 92)
                .with_task_score(TaskType::Chat, 92)
                .with_task_score(TaskType::Coding, 90)
                .with_task_score(TaskType::Vision, 88)
                .with_task_score(TaskType::LongContext, 90)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("gpt-4-vision", 128_000, 88)
                .with_task_score(TaskType::Vision, 88)
                .with_task_score(TaskType::Chat, 85)
                .with_vision(),
        );

        // Anthropic Claude 3 / 3.5 (vision via base64)
        self.profiles.push(
            ModelCapabilityProfile::new("claude-3.5-sonnet", 200_000, 95)
                .with_task_score(TaskType::Chat, 94)
                .with_task_score(TaskType::Coding, 96)
                .with_task_score(TaskType::Vision, 92)
                .with_task_score(TaskType::Analysis, 94)
                .with_task_score(TaskType::LongContext, 95)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("claude-3-opus", 200_000, 94)
                .with_task_score(TaskType::Chat, 92)
                .with_task_score(TaskType::Coding, 92)
                .with_task_score(TaskType::Vision, 90)
                .with_task_score(TaskType::Creative, 95)
                .with_task_score(TaskType::LongContext, 95)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("claude-3-sonnet", 200_000, 88)
                .with_task_score(TaskType::Chat, 88)
                .with_task_score(TaskType::Coding, 88)
                .with_task_score(TaskType::Vision, 85)
                .with_task_score(TaskType::LongContext, 92)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("claude-3-haiku", 200_000, 80)
                .with_task_score(TaskType::Chat, 80)
                .with_task_score(TaskType::FastResponse, 92)
                .with_task_score(TaskType::Vision, 78)
                .with_task_score(TaskType::LongContext, 90)
                .with_vision()
                .with_functions(),
        );

        // Google Gemini (1M ctx, native multimodal)
        self.profiles.push(
            ModelCapabilityProfile::new("gemini-1.5-pro", 1_000_000, 92)
                .with_task_score(TaskType::Chat, 90)
                .with_task_score(TaskType::Vision, 90)
                .with_task_score(TaskType::Analysis, 90)
                .with_task_score(TaskType::LongContext, 98)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("gemini-1.5-flash", 1_000_000, 84)
                .with_task_score(TaskType::Chat, 82)
                .with_task_score(TaskType::Vision, 84)
                .with_task_score(TaskType::FastResponse, 90)
                .with_task_score(TaskType::LongContext, 95)
                .with_vision()
                .with_functions(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("gemini-pro-vision", 32_000, 80)
                .with_task_score(TaskType::Vision, 82)
                .with_vision(),
        );

        // Mistral Pixtral (vision)
        self.profiles.push(
            ModelCapabilityProfile::new("pixtral", 128_000, 84)
                .with_task_score(TaskType::Vision, 86)
                .with_task_score(TaskType::Chat, 82)
                .with_vision(),
        );

        // Qwen2.5-VL / Qwen2-VL / Qwen-VL family. Most specific first so
        // `find()` resolves to the strongest match (qwen2.5-vl ⊂ qwen2-vl
        // ⊂ qwen-vl as substrings).
        //
        // Qwen2.5-VL is the current open-weight SOTA for vision: leads
        // OCRBench / DocVQA / ChartQA / MMMU vs. all other open VLMs and
        // matches GPT-4o on several. Native visual grounding (bbox /
        // referring expressions) and video. Apache-2.0 license. The
        // Vision score (90) deliberately exceeds gemma3 (75) and the
        // older qwen2-vl entry (86) so `select_best(Vision)` prefers it
        // when both are registered.
        self.profiles.push(
            ModelCapabilityProfile::new("qwen2.5-vl", 128_000, 88)
                .with_task_score(TaskType::Vision, 90)
                .with_task_score(TaskType::Chat, 84)
                .with_task_score(TaskType::Analysis, 86)
                .with_task_score(TaskType::LongContext, 85)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("qwen2-vl", 32_000, 84)
                .with_task_score(TaskType::Vision, 86)
                .with_task_score(TaskType::Chat, 82)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("qwen-vl", 32_000, 78)
                .with_task_score(TaskType::Vision, 80)
                .with_vision(),
        );

        // OpenBMB MiniCPM-V (small but capable)
        self.profiles.push(
            ModelCapabilityProfile::new("minicpm-v", 32_000, 76)
                .with_task_score(TaskType::Vision, 80)
                .with_task_score(TaskType::FastResponse, 82)
                .with_vision(),
        );

        // Other open vision models
        self.profiles.push(
            ModelCapabilityProfile::new("cogvlm", 8_192, 76)
                .with_task_score(TaskType::Vision, 80)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("yi-vl", 4_096, 72)
                .with_task_score(TaskType::Vision, 75)
                .with_vision(),
        );
        // llava variants — specific first, generic last. Generic `llava` is
        // appended LAST so `llava-llama3:8b` resolves to the specific profile
        // (find() returns the first match in iteration order).
        self.profiles.push(
            ModelCapabilityProfile::new("llava-llama3", 8_192, 78)
                .with_task_score(TaskType::Vision, 82)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("llava-phi3", 4_096, 74)
                .with_task_score(TaskType::Vision, 78)
                .with_task_score(TaskType::FastResponse, 84)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("llava-next", 32_000, 80)
                .with_task_score(TaskType::Vision, 84)
                .with_vision(),
        );
        self.profiles.push(
            ModelCapabilityProfile::new("bakllava", 4_096, 72)
                .with_task_score(TaskType::Vision, 76)
                .with_vision(),
        );
        // Generic llava (last so specific variants win)
        self.profiles.push(
            ModelCapabilityProfile::new("llava", 4096, 75)
                .with_task_score(TaskType::Vision, 85)
                .with_vision(),
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

    /// Get the capability profile for a model.
    ///
    /// Resolves to the **longest matching pattern** (not the first), so
    /// `llava-phi3:3.8b` resolves to `llava-phi3` rather than the generic
    /// `phi` profile. Length-based tiebreak removes the ordering hazard
    /// that simple `find()` introduces when one profile name is a substring
    /// of another (e.g. `phi` ⊂ `phi-3.5-vision` ⊂ `llava-phi3`).
    pub fn get_profile(&self, model_name: &str) -> Option<&ModelCapabilityProfile> {
        self.profiles
            .iter()
            .filter(|p| p.matches(model_name))
            .max_by_key(|p| p.model_pattern.len())
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

        scored.sort_by_key(|e| std::cmp::Reverse(e.1));
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
        let profile =
            ModelCapabilityProfile::new("test", 4096, 75).with_task_score(TaskType::Coding, 95);

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

    #[test]
    fn test_gemma3_profile_supports_vision_and_long_context() {
        let router = ModelRouter::new();
        let p = router.get_profile("gemma3:12b").expect("gemma3 profile");
        assert!(p.supports_vision, "gemma3 must be marked as vision-capable");
        assert!(p.context_size >= 128_000, "gemma3 must declare 128K ctx");
        assert!(p.get_task_score(TaskType::Vision) >= 75);
    }

    #[test]
    fn test_gemma2_profile_text_only() {
        let router = ModelRouter::new();
        let p = router.get_profile("gemma2-9b").expect("gemma2 profile");
        assert!(!p.supports_vision);
        assert_eq!(p.context_size, 8192);
    }

    #[test]
    fn test_bonsai_profile_fast_response_dominant() {
        let router = ModelRouter::new();
        let p = router
            .get_profile("Bonsai-8B-gguf")
            .expect("bonsai profile");
        // Bonsai must beat phi on FastResponse — that is its raison d'être.
        let phi = router.get_profile("phi-3-mini").expect("phi profile");
        assert!(
            p.get_task_score(TaskType::FastResponse) >= phi.get_task_score(TaskType::FastResponse)
        );
        assert_eq!(p.context_size, 65_536);
        assert!(!p.supports_vision);
    }

    #[test]
    fn test_ternary_bonsai_more_specific_than_bonsai() {
        let router = ModelRouter::new();
        // The pattern `ternary-bonsai` must be tried before `bonsai` so that
        // a ternary model resolves to the ternary profile, not the 1-bit one.
        let p = router
            .get_profile("Ternary-Bonsai-8B-gguf")
            .expect("ternary profile");
        assert_eq!(p.model_pattern, "ternary-bonsai");
    }

    #[test]
    fn test_phi_vision_resolves_to_vision_profile() {
        let router = ModelRouter::new();
        // phi-3.5-vision must beat the generic phi profile.
        let p = router
            .get_profile("phi-3.5-vision-instruct")
            .expect("phi vision profile");
        assert!(p.supports_vision);
        assert_eq!(p.model_pattern, "phi-3.5-vision");
    }

    #[test]
    fn test_cloud_vision_models_all_supported() {
        let router = ModelRouter::new();
        let cases = &[
            ("gpt-4o", "gpt-4o"),
            ("gpt-4o-mini", "gpt-4o"),
            ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
            ("gpt-4-vision-preview", "gpt-4-vision"),
            ("claude-3.5-sonnet-20241022", "claude-3.5-sonnet"),
            ("claude-3-opus-20240229", "claude-3-opus"),
            ("claude-3-sonnet-20240229", "claude-3-sonnet"),
            ("claude-3-haiku-20240307", "claude-3-haiku"),
            ("gemini-1.5-pro-latest", "gemini-1.5-pro"),
            ("gemini-1.5-flash-002", "gemini-1.5-flash"),
            ("pixtral-12b-2409", "pixtral"),
            ("Qwen2-VL-7B-Instruct", "qwen2-vl"),
            ("MiniCPM-V-2_6", "minicpm-v"),
            ("cogvlm-chat", "cogvlm"),
        ];
        for (model_name, expected_pattern) in cases {
            let p = router
                .get_profile(model_name)
                .unwrap_or_else(|| panic!("missing profile for {}", model_name));
            assert!(p.supports_vision, "{} should be vision-capable", model_name);
            assert_eq!(
                p.model_pattern, *expected_pattern,
                "{} resolved to wrong profile",
                model_name
            );
        }
    }

    #[test]
    fn test_llava_specific_variants_win_over_generic() {
        let router = ModelRouter::new();
        let p = router
            .get_profile("llava-llama3:8b")
            .expect("llava-llama3 profile");
        assert_eq!(p.model_pattern, "llava-llama3");
        let p = router
            .get_profile("llava-phi3:3.8b")
            .expect("llava-phi3 profile");
        assert_eq!(p.model_pattern, "llava-phi3");
        // bare llava still resolves
        let p = router.get_profile("llava:13b").expect("bare llava profile");
        assert_eq!(p.model_pattern, "llava");
    }

    #[test]
    fn test_gemini_long_context_dominance() {
        let router = ModelRouter::new();
        let p = router
            .get_profile("gemini-1.5-pro")
            .expect("gemini-1.5-pro profile");
        assert!(
            p.context_size >= 1_000_000,
            "gemini 1.5 must declare 1M ctx"
        );
        assert!(p.supports_vision);
        assert!(p.get_task_score(TaskType::LongContext) >= 95);
    }

    #[test]
    fn test_select_best_vision_picks_cloud_model_when_available() {
        let router = ModelRouter::new();
        let models = vec![
            ModelInfo::new("llama-3-8b", AiProvider::Ollama),
            ModelInfo::new("gpt-4o", AiProvider::OpenAI),
            ModelInfo::new("claude-3.5-sonnet", AiProvider::Anthropic),
            ModelInfo::new("llava:13b", AiProvider::Ollama),
        ];
        let req = ModelRequirements::for_task(TaskType::Vision);
        let best = router
            .select_best(&models, &req)
            .expect("vision-capable pick");
        // Either gpt-4o or claude-3.5-sonnet should win (both score ≥ 92 on Vision),
        // not llava (85). The exact winner depends on quality_score tiebreak.
        assert!(
            best.name == "gpt-4o" || best.name == "claude-3.5-sonnet",
            "expected gpt-4o or claude-3.5-sonnet, got {}",
            best.name
        );
    }

    #[test]
    fn test_qwen2_5_vl_beats_gemma3_for_vision() {
        // Locking the policy decision: among open-weight VLMs, Qwen2.5-VL
        // is preferred over Gemma 3 for vision (OCR / docs / grounding).
        // Gemma 3 stays better only for text Chat tasks. Update the
        // profiles in `register_default_profiles` carefully if this
        // ordering ever changes.
        let router = ModelRouter::new();
        let models = vec![
            ModelInfo::new("gemma3:12b", AiProvider::Ollama),
            ModelInfo::new("qwen2.5-vl:7b", AiProvider::Ollama),
        ];
        let req = ModelRequirements::for_task(TaskType::Vision);
        let best = router
            .select_best(&models, &req)
            .expect("vision-capable pick");
        assert_eq!(
            best.name, "qwen2.5-vl:7b",
            "Qwen2.5-VL must outrank Gemma 3 on Vision"
        );
    }

    #[test]
    fn test_qwen2_5_vl_profile_resolves_to_specific_match() {
        // qwen2.5-vl as a substring is also matched by qwen2-vl and
        // qwen-vl. Confirm that a model id containing "qwen2.5-vl"
        // resolves to the most specific profile (Vision: 90), not the
        // older qwen2-vl one (Vision: 86).
        let router = ModelRouter::new();
        let p = router
            .get_profile("qwen2.5-vl-7b-instruct")
            .expect("qwen2.5-vl profile");
        assert!(p.supports_vision);
        assert_eq!(p.get_task_score(TaskType::Vision), 90);
        assert!(p.context_size >= 128_000);
    }
}
