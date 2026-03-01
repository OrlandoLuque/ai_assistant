//! Per-subtask model routing configuration and multi-model generator dispatcher.
//!
//! Provides `EvalAgentConfig` for assigning different LLM models to different subtasks
//! within an agent workflow, `SearchDimension` for defining the configuration search space,
//! and `MultiModelGenerator` for routing prompts to the correct model.

use super::runner::{BenchmarkRunResult, ModelIdentifier};
use super::subtask::Subtask;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// EvalAgentConfig
// ---------------------------------------------------------------------------

/// Agent workflow configuration with per-subtask model assignments.
///
/// Each subtask (coding, reasoning, review, etc.) can use a different LLM model,
/// temperature, chain-of-thought setting, and prompt template. This enables
/// optimal cost/quality routing — e.g., use a cheap fast model for formatting
/// and an expensive powerful model for reasoning.
///
/// Named `EvalAgentConfig` to avoid collision with `AgentConfig` in agentic_loop.rs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalAgentConfig {
    /// Human-readable name for this configuration (used as identifier in reports)
    pub name: String,
    /// Default model used when no subtask-specific override exists
    pub default_model: ModelIdentifier,
    /// Per-subtask model assignments (key = `Subtask.to_string()`, e.g., "CodeGeneration")
    pub subtask_models: HashMap<String, ModelIdentifier>,
    /// Global temperature for generation
    pub temperature: f32,
    /// Per-subtask temperature overrides
    pub subtask_temperatures: HashMap<String, f32>,
    /// Whether to use chain-of-thought prompting globally
    pub chain_of_thought: bool,
    /// Per-subtask chain-of-thought overrides
    pub subtask_cot: HashMap<String, bool>,
    /// Max tokens per response (global)
    pub max_tokens: Option<usize>,
    /// Per-subtask max_tokens overrides
    pub subtask_max_tokens: HashMap<String, usize>,
    /// RAG level (0=none, 1=basic, ..., 5=full graph)
    pub rag_level: u8,
    /// Per-subtask prompt templates (use `{prompt}` and `{system}` as placeholders)
    pub subtask_templates: HashMap<String, String>,
    /// Estimated cost per LLM call, keyed by `ModelIdentifier.to_string()`
    pub cost_per_call: HashMap<String, f64>,
    /// Arbitrary metadata for tracking configuration lineage
    pub metadata: HashMap<String, String>,
}

impl Default for EvalAgentConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            default_model: ModelIdentifier {
                name: "unknown".to_string(),
                provider: "unknown".to_string(),
                variant: None,
            },
            subtask_models: HashMap::new(),
            temperature: 0.7,
            subtask_temperatures: HashMap::new(),
            chain_of_thought: false,
            subtask_cot: HashMap::new(),
            max_tokens: None,
            subtask_max_tokens: HashMap::new(),
            rag_level: 0,
            subtask_templates: HashMap::new(),
            cost_per_call: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl EvalAgentConfig {
    /// Create a new config with a name and default model.
    pub fn new(name: &str, default_model: ModelIdentifier) -> Self {
        Self {
            name: name.to_string(),
            default_model,
            ..Default::default()
        }
    }

    /// Assign a specific model to a subtask.
    pub fn with_subtask_model(mut self, subtask: &Subtask, model: ModelIdentifier) -> Self {
        self.subtask_models.insert(subtask.to_string(), model);
        self
    }

    /// Set global temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set temperature for a specific subtask.
    pub fn with_subtask_temperature(mut self, subtask: &Subtask, temp: f32) -> Self {
        self.subtask_temperatures.insert(subtask.to_string(), temp);
        self
    }

    /// Set global chain-of-thought.
    pub fn with_chain_of_thought(mut self, cot: bool) -> Self {
        self.chain_of_thought = cot;
        self
    }

    /// Set chain-of-thought for a specific subtask.
    pub fn with_subtask_cot(mut self, subtask: &Subtask, cot: bool) -> Self {
        self.subtask_cot.insert(subtask.to_string(), cot);
        self
    }

    /// Set global max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set RAG level (0-5).
    pub fn with_rag_level(mut self, level: u8) -> Self {
        self.rag_level = level;
        self
    }

    /// Set estimated cost per call for a model.
    pub fn with_model_cost(mut self, model: &ModelIdentifier, cost: f64) -> Self {
        self.cost_per_call.insert(model.to_string(), cost);
        self
    }

    /// Add metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the model assigned to a subtask, falling back to `default_model`.
    pub fn model_for_subtask(&self, subtask: &str) -> &ModelIdentifier {
        self.subtask_models.get(subtask).unwrap_or(&self.default_model)
    }

    /// Get the temperature for a subtask, falling back to global temperature.
    pub fn temperature_for_subtask(&self, subtask: &str) -> f32 {
        self.subtask_temperatures
            .get(subtask)
            .copied()
            .unwrap_or(self.temperature)
    }

    /// Get the chain-of-thought setting for a subtask, falling back to global.
    pub fn cot_for_subtask(&self, subtask: &str) -> bool {
        self.subtask_cot
            .get(subtask)
            .copied()
            .unwrap_or(self.chain_of_thought)
    }

    /// Get the max tokens for a subtask, falling back to global.
    pub fn max_tokens_for_subtask(&self, subtask: &str) -> Option<usize> {
        self.subtask_max_tokens
            .get(subtask)
            .copied()
            .or(self.max_tokens)
    }

    /// Get the estimated cost per call for a model (default 0.0 if not set).
    pub fn cost_for_model(&self, model: &ModelIdentifier) -> f64 {
        self.cost_per_call
            .get(&model.to_string())
            .copied()
            .unwrap_or(0.0)
    }

    /// Compute human-readable differences between this config and another.
    pub fn diff(&self, other: &EvalAgentConfig) -> Vec<String> {
        let mut diffs = Vec::new();

        if self.default_model != other.default_model {
            diffs.push(format!(
                "DefaultModel: {} -> {}",
                self.default_model, other.default_model
            ));
        }

        // Subtask model changes
        let all_subtasks: std::collections::HashSet<&String> = self
            .subtask_models
            .keys()
            .chain(other.subtask_models.keys())
            .collect();
        for subtask in all_subtasks {
            let old = self.subtask_models.get(subtask);
            let new = other.subtask_models.get(subtask);
            if old != new {
                let old_str = old
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| self.default_model.to_string());
                let new_str = new
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| other.default_model.to_string());
                diffs.push(format!("SubtaskModel({}): {} -> {}", subtask, old_str, new_str));
            }
        }

        if (self.temperature - other.temperature).abs() > 1e-6 {
            diffs.push(format!(
                "Temperature: {:.2} -> {:.2}",
                self.temperature, other.temperature
            ));
        }

        if self.chain_of_thought != other.chain_of_thought {
            diffs.push(format!(
                "ChainOfThought: {} -> {}",
                self.chain_of_thought, other.chain_of_thought
            ));
        }

        if self.rag_level != other.rag_level {
            diffs.push(format!(
                "RagLevel: {} -> {}",
                self.rag_level, other.rag_level
            ));
        }

        if self.max_tokens != other.max_tokens {
            diffs.push(format!(
                "MaxTokens: {:?} -> {:?}",
                self.max_tokens, other.max_tokens
            ));
        }

        // Per-subtask temperature changes
        let all_temp_subtasks: std::collections::HashSet<&String> = self
            .subtask_temperatures
            .keys()
            .chain(other.subtask_temperatures.keys())
            .collect();
        for subtask in all_temp_subtasks {
            let old = self.subtask_temperatures.get(subtask).copied();
            let new = other.subtask_temperatures.get(subtask).copied();
            if old != new {
                diffs.push(format!(
                    "SubtaskTemperature({}): {:?} -> {:?}",
                    subtask, old, new
                ));
            }
        }

        // Per-subtask CoT changes
        let all_cot_subtasks: std::collections::HashSet<&String> = self
            .subtask_cot
            .keys()
            .chain(other.subtask_cot.keys())
            .collect();
        for subtask in all_cot_subtasks {
            let old = self.subtask_cot.get(subtask).copied();
            let new = other.subtask_cot.get(subtask).copied();
            if old != new {
                diffs.push(format!(
                    "SubtaskCoT({}): {:?} -> {:?}",
                    subtask, old, new
                ));
            }
        }

        diffs
    }
}

// ---------------------------------------------------------------------------
// SearchDimension
// ---------------------------------------------------------------------------

/// A dimension in the configuration search space that can be varied.
///
/// Each dimension defines a set of candidate values to test. The search engine
/// sweeps through each dimension one at a time (coordinate descent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchDimension {
    /// Which model to use for a specific subtask
    SubtaskModel {
        subtask: String,
        candidates: Vec<ModelIdentifier>,
    },
    /// Global temperature values to test
    Temperature { values: Vec<f32> },
    /// Per-subtask temperature values
    SubtaskTemperature { subtask: String, values: Vec<f32> },
    /// Chain of thought on/off (binary, always 2 variants)
    ChainOfThought,
    /// Per-subtask chain of thought on/off
    SubtaskChainOfThought { subtask: String },
    /// RAG level values to test
    RagLevel { values: Vec<u8> },
    /// Max tokens values to test
    MaxTokens { values: Vec<usize> },
    /// Per-subtask max tokens values
    SubtaskMaxTokens { subtask: String, values: Vec<usize> },
}

impl SearchDimension {
    /// Human-readable name for this dimension.
    pub fn name(&self) -> String {
        match self {
            Self::SubtaskModel { subtask, .. } => format!("SubtaskModel({})", subtask),
            Self::Temperature { .. } => "Temperature".to_string(),
            Self::SubtaskTemperature { subtask, .. } => {
                format!("SubtaskTemperature({})", subtask)
            }
            Self::ChainOfThought => "ChainOfThought".to_string(),
            Self::SubtaskChainOfThought { subtask } => {
                format!("SubtaskChainOfThought({})", subtask)
            }
            Self::RagLevel { .. } => "RagLevel".to_string(),
            Self::MaxTokens { .. } => "MaxTokens".to_string(),
            Self::SubtaskMaxTokens { subtask, .. } => format!("SubtaskMaxTokens({})", subtask),
        }
    }

    /// Number of variant values this dimension can take.
    pub fn variant_count(&self) -> usize {
        match self {
            Self::SubtaskModel { candidates, .. } => candidates.len(),
            Self::Temperature { values } => values.len(),
            Self::SubtaskTemperature { values, .. } => values.len(),
            Self::ChainOfThought => 2,
            Self::SubtaskChainOfThought { .. } => 2,
            Self::RagLevel { values } => values.len(),
            Self::MaxTokens { values } => values.len(),
            Self::SubtaskMaxTokens { values, .. } => values.len(),
        }
    }

    /// Human-readable label for a specific variant index.
    pub fn variant_label(&self, idx: usize) -> String {
        match self {
            Self::SubtaskModel { candidates, .. } => {
                candidates.get(idx).map(|m| m.to_string()).unwrap_or_else(|| format!("idx:{}", idx))
            }
            Self::Temperature { values } => {
                values.get(idx).map(|v| format!("{:.2}", v)).unwrap_or_else(|| format!("idx:{}", idx))
            }
            Self::SubtaskTemperature { values, .. } => {
                values.get(idx).map(|v| format!("{:.2}", v)).unwrap_or_else(|| format!("idx:{}", idx))
            }
            Self::ChainOfThought => {
                if idx == 0 { "false".to_string() } else { "true".to_string() }
            }
            Self::SubtaskChainOfThought { .. } => {
                if idx == 0 { "false".to_string() } else { "true".to_string() }
            }
            Self::RagLevel { values } => {
                values.get(idx).map(|v| format!("{}", v)).unwrap_or_else(|| format!("idx:{}", idx))
            }
            Self::MaxTokens { values } => {
                values.get(idx).map(|v| format!("{}", v)).unwrap_or_else(|| format!("idx:{}", idx))
            }
            Self::SubtaskMaxTokens { values, .. } => {
                values.get(idx).map(|v| format!("{}", v)).unwrap_or_else(|| format!("idx:{}", idx))
            }
        }
    }

    /// Apply a variant to a config, producing a new config with that dimension changed.
    pub fn apply_variant(&self, config: &EvalAgentConfig, idx: usize) -> EvalAgentConfig {
        let mut new_config = config.clone();
        match self {
            Self::SubtaskModel { subtask, candidates } => {
                if let Some(model) = candidates.get(idx) {
                    new_config
                        .subtask_models
                        .insert(subtask.clone(), model.clone());
                }
            }
            Self::Temperature { values } => {
                if let Some(&temp) = values.get(idx) {
                    new_config.temperature = temp;
                }
            }
            Self::SubtaskTemperature { subtask, values } => {
                if let Some(&temp) = values.get(idx) {
                    new_config
                        .subtask_temperatures
                        .insert(subtask.clone(), temp);
                }
            }
            Self::ChainOfThought => {
                new_config.chain_of_thought = idx != 0;
            }
            Self::SubtaskChainOfThought { subtask } => {
                new_config.subtask_cot.insert(subtask.clone(), idx != 0);
            }
            Self::RagLevel { values } => {
                if let Some(&level) = values.get(idx) {
                    new_config.rag_level = level;
                }
            }
            Self::MaxTokens { values } => {
                if let Some(&max) = values.get(idx) {
                    new_config.max_tokens = Some(max);
                }
            }
            Self::SubtaskMaxTokens { subtask, values } => {
                if let Some(&max) = values.get(idx) {
                    new_config.subtask_max_tokens.insert(subtask.clone(), max);
                }
            }
        }
        new_config
    }
}

impl std::fmt::Display for SearchDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// MultiModelGenerator
// ---------------------------------------------------------------------------

/// Dispatches prompts to different LLM generators based on model identity.
///
/// Each registered model key maps to a generator callback. When `generate()` is
/// called with a model key, the corresponding generator is invoked. If the key
/// is not found, the default generator is used as fallback.
///
/// # Example
/// ```ignore
/// let mut gen = MultiModelGenerator::new(|p| Ok("default response".into()));
/// gen.register_model("openai/gpt-4", |p| call_openai(p));
/// gen.register_model("ollama/llama3", |p| call_ollama(p));
///
/// // Routes to the correct provider
/// let response = gen.generate("openai/gpt-4", "Hello?")?;
/// ```
pub struct MultiModelGenerator {
    generators: HashMap<String, Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>>,
    default_generator: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
}

impl MultiModelGenerator {
    /// Create a new generator with a default/fallback callback.
    pub fn new<F>(default_generator: F) -> Self
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        Self {
            generators: HashMap::new(),
            default_generator: Arc::new(default_generator),
        }
    }

    /// Register a generator for a specific model key (e.g., "openai/gpt-4").
    pub fn register_model<F>(&mut self, model_key: &str, generator: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        self.generators
            .insert(model_key.to_string(), Arc::new(generator));
    }

    /// Generate a response for a prompt, routing to the correct model.
    ///
    /// Falls back to `default_generator` if `model_key` is not registered.
    pub fn generate(&self, model_key: &str, prompt: &str) -> Result<String, String> {
        let gen = self
            .generators
            .get(model_key)
            .unwrap_or(&self.default_generator);
        (gen)(prompt)
    }

    /// Check if a model key has a registered generator.
    pub fn has_model(&self, model_key: &str) -> bool {
        self.generators.contains_key(model_key)
    }

    /// Number of registered model generators (excluding default).
    pub fn model_count(&self) -> usize {
        self.generators.len()
    }
}

impl std::fmt::Debug for MultiModelGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiModelGenerator")
            .field("registered_models", &self.generators.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ConfigMeasurement
// ---------------------------------------------------------------------------

/// Snapshot of quality/cost/latency measurements for a single configuration.
///
/// Produced by `ConfigSearchEngine::measure()`. Contains aggregate statistics
/// plus the full `BenchmarkRunResult` for pipeline integration with
/// `ComparisonMatrix`, `SubtaskAnalyzer`, and `ReportBuilder`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMeasurement {
    /// The configuration that was measured
    pub config: EvalAgentConfig,
    /// Mean quality score across all problems (0.0-1.0)
    pub quality: f64,
    /// Standard deviation of per-problem scores
    pub quality_std: f64,
    /// Mean latency per problem in milliseconds
    pub latency_ms: f64,
    /// Total estimated cost (from `cost_per_call`)
    pub cost: f64,
    /// Number of problems evaluated
    pub sample_count: usize,
    /// Per-subtask mean quality (key = subtask name, "General" for untagged)
    pub subtask_quality: HashMap<String, f64>,
    /// Full benchmark run result for pipeline integration
    pub run_result: Option<BenchmarkRunResult>,
}

impl ConfigMeasurement {
    /// Extract per-problem mean scores from the run result.
    ///
    /// Used by the search engine for Welch's t-test comparison.
    pub fn per_problem_scores(&self) -> Vec<f64> {
        self.run_result
            .as_ref()
            .map(|r| {
                r.results
                    .iter()
                    .map(|pr| {
                        if pr.scores.is_empty() {
                            0.0
                        } else {
                            pr.scores.iter().sum::<f64>() / pr.scores.len() as f64
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, provider: &str) -> ModelIdentifier {
        ModelIdentifier {
            name: name.to_string(),
            provider: provider.to_string(),
            variant: None,
        }
    }

    #[test]
    fn test_eval_agent_config_defaults() {
        let config = EvalAgentConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.temperature, 0.7);
        assert!(!config.chain_of_thought);
        assert_eq!(config.rag_level, 0);
        assert!(config.max_tokens.is_none());
        assert!(config.subtask_models.is_empty());
        assert!(config.cost_per_call.is_empty());
    }

    #[test]
    fn test_eval_agent_config_builder() {
        let gpt4 = model("gpt-4", "openai");
        let llama = model("llama3", "ollama");

        let config = EvalAgentConfig::new("test_config", gpt4.clone())
            .with_subtask_model(&Subtask::CodeGeneration, llama.clone())
            .with_temperature(0.3)
            .with_subtask_temperature(&Subtask::ReasoningChain, 0.1)
            .with_chain_of_thought(true)
            .with_subtask_cot(&Subtask::OutputFormatting, false)
            .with_max_tokens(1024)
            .with_rag_level(3)
            .with_model_cost(&gpt4, 0.03)
            .with_model_cost(&llama, 0.001)
            .with_metadata("version", "1");

        assert_eq!(config.name, "test_config");
        assert_eq!(config.default_model, gpt4);
        assert_eq!(config.subtask_models["CodeGeneration"], llama);
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.subtask_temperatures["ReasoningChain"], 0.1);
        assert!(config.chain_of_thought);
        assert!(!config.subtask_cot["OutputFormatting"]);
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.rag_level, 3);
        assert_eq!(config.cost_per_call.len(), 2);
        assert_eq!(config.metadata["version"], "1");
    }

    #[test]
    fn test_eval_agent_config_serialization() {
        let config = EvalAgentConfig::new("ser_test", model("m", "p"))
            .with_subtask_model(&Subtask::CodeGeneration, model("coder", "ollama"))
            .with_temperature(0.5);

        let json = serde_json::to_string(&config).expect("serialize");
        let restored: EvalAgentConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.name, "ser_test");
        assert_eq!(restored.temperature, 0.5);
        assert!(restored.subtask_models.contains_key("CodeGeneration"));
    }

    #[test]
    fn test_model_for_subtask_specific() {
        let llama = model("llama3", "ollama");
        let config = EvalAgentConfig::new("test", model("gpt-4", "openai"))
            .with_subtask_model(&Subtask::CodeGeneration, llama.clone());

        assert_eq!(config.model_for_subtask("CodeGeneration"), &llama);
    }

    #[test]
    fn test_model_for_subtask_fallback() {
        let gpt4 = model("gpt-4", "openai");
        let config = EvalAgentConfig::new("test", gpt4.clone());

        // No subtask-specific model → falls back to default
        assert_eq!(config.model_for_subtask("CodeGeneration"), &gpt4);
        assert_eq!(config.model_for_subtask("ReasoningChain"), &gpt4);
        assert_eq!(config.model_for_subtask("NonExistent"), &gpt4);
    }

    #[test]
    fn test_temperature_for_subtask() {
        let config = EvalAgentConfig::new("test", model("m", "p"))
            .with_temperature(0.7)
            .with_subtask_temperature(&Subtask::ReasoningChain, 0.2);

        assert_eq!(config.temperature_for_subtask("ReasoningChain"), 0.2);
        assert_eq!(config.temperature_for_subtask("CodeGeneration"), 0.7); // fallback
    }

    #[test]
    fn test_cot_for_subtask() {
        let config = EvalAgentConfig::new("test", model("m", "p"))
            .with_chain_of_thought(true)
            .with_subtask_cot(&Subtask::OutputFormatting, false);

        assert!(!config.cot_for_subtask("OutputFormatting"));
        assert!(config.cot_for_subtask("ReasoningChain")); // fallback to global=true
    }

    #[test]
    fn test_cost_for_model() {
        let gpt4 = model("gpt-4", "openai");
        let llama = model("llama3", "ollama");
        let config = EvalAgentConfig::new("test", gpt4.clone())
            .with_model_cost(&gpt4, 0.03);

        assert!((config.cost_for_model(&gpt4) - 0.03).abs() < 1e-10);
        assert_eq!(config.cost_for_model(&llama), 0.0); // not set → 0.0
    }

    #[test]
    fn test_search_dimension_subtask_model() {
        let dim = SearchDimension::SubtaskModel {
            subtask: "CodeGeneration".to_string(),
            candidates: vec![model("gpt-4", "openai"), model("llama3", "ollama")],
        };
        assert_eq!(dim.name(), "SubtaskModel(CodeGeneration)");
        assert_eq!(dim.variant_count(), 2);
        assert!(dim.variant_label(0).contains("gpt-4"));
        assert!(dim.variant_label(1).contains("llama3"));
    }

    #[test]
    fn test_search_dimension_temperature() {
        let dim = SearchDimension::Temperature {
            values: vec![0.0, 0.3, 0.7, 1.0],
        };
        assert_eq!(dim.name(), "Temperature");
        assert_eq!(dim.variant_count(), 4);
        assert_eq!(dim.variant_label(0), "0.00");
        assert_eq!(dim.variant_label(2), "0.70");
    }

    #[test]
    fn test_search_dimension_chain_of_thought() {
        let dim = SearchDimension::ChainOfThought;
        assert_eq!(dim.variant_count(), 2);
        assert_eq!(dim.variant_label(0), "false");
        assert_eq!(dim.variant_label(1), "true");
    }

    #[test]
    fn test_search_dimension_apply_variant() {
        let config = EvalAgentConfig::new("base", model("gpt-4", "openai"))
            .with_temperature(0.7);

        // Apply temperature variant
        let dim_temp = SearchDimension::Temperature {
            values: vec![0.0, 0.3, 1.0],
        };
        let modified = dim_temp.apply_variant(&config, 1);
        assert_eq!(modified.temperature, 0.3);
        assert_eq!(modified.name, "base"); // other fields unchanged

        // Apply subtask model variant
        let llama = model("llama3", "ollama");
        let dim_model = SearchDimension::SubtaskModel {
            subtask: "CodeGeneration".to_string(),
            candidates: vec![llama.clone()],
        };
        let modified = dim_model.apply_variant(&config, 0);
        assert_eq!(modified.subtask_models["CodeGeneration"], llama);

        // Apply CoT variant
        let dim_cot = SearchDimension::ChainOfThought;
        let modified = dim_cot.apply_variant(&config, 1);
        assert!(modified.chain_of_thought);
        let modified = dim_cot.apply_variant(&config, 0);
        assert!(!modified.chain_of_thought);
    }

    #[test]
    fn test_search_dimension_variant_label() {
        let dim = SearchDimension::RagLevel {
            values: vec![0, 1, 3, 5],
        };
        assert_eq!(dim.variant_label(0), "0");
        assert_eq!(dim.variant_label(3), "5");

        let dim = SearchDimension::MaxTokens {
            values: vec![512, 1024, 2048],
        };
        assert_eq!(dim.variant_label(1), "1024");
    }

    #[test]
    fn test_multi_model_generator_dispatch() {
        let mut gen = MultiModelGenerator::new(|_| Ok("default".to_string()));
        gen.register_model("openai/gpt-4", |_| Ok("gpt4_response".to_string()));
        gen.register_model("ollama/llama3", |_| Ok("llama_response".to_string()));

        assert_eq!(gen.generate("openai/gpt-4", "test").unwrap(), "gpt4_response");
        assert_eq!(gen.generate("ollama/llama3", "test").unwrap(), "llama_response");
        assert!(gen.has_model("openai/gpt-4"));
        assert!(!gen.has_model("anthropic/claude"));
        assert_eq!(gen.model_count(), 2);
    }

    #[test]
    fn test_multi_model_generator_fallback() {
        let gen = MultiModelGenerator::new(|_| Ok("fallback".to_string()));

        // Unknown model key → uses default
        assert_eq!(gen.generate("unknown/model", "test").unwrap(), "fallback");
        assert_eq!(gen.model_count(), 0);
    }

    #[test]
    fn test_config_diff() {
        let gpt4 = model("gpt-4", "openai");
        let llama = model("llama3", "ollama");

        let base = EvalAgentConfig::new("base", gpt4.clone())
            .with_temperature(0.7)
            .with_chain_of_thought(false);

        let modified = EvalAgentConfig::new("modified", gpt4.clone())
            .with_subtask_model(&Subtask::CodeGeneration, llama)
            .with_temperature(0.3)
            .with_chain_of_thought(true)
            .with_rag_level(3);

        let diffs = base.diff(&modified);
        assert!(diffs.iter().any(|d| d.contains("Temperature")));
        assert!(diffs.iter().any(|d| d.contains("ChainOfThought")));
        assert!(diffs.iter().any(|d| d.contains("RagLevel")));
        assert!(diffs.iter().any(|d| d.contains("SubtaskModel(CodeGeneration)")));
        assert!(diffs.len() >= 4);
    }
}
