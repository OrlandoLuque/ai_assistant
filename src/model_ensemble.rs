//! Model ensemble for combining multiple model outputs
//!
//! This module provides ensemble methods to combine outputs from
//! multiple AI models for improved accuracy and reliability.
//!
//! # Features
//!
//! - **Voting**: Majority vote for classification
//! - **Weighted averaging**: Combine with model weights
//! - **Best-of-N**: Select best response by scoring
//! - **Mixture of experts**: Route to specialized models

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for ensemble
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EnsembleConfig {
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
    /// Models in the ensemble
    pub models: Vec<EnsembleModel>,
    /// Timeout per model
    pub model_timeout: Duration,
    /// Minimum models required
    pub min_models: usize,
    /// Enable parallel execution
    pub parallel: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Voting,
            models: Vec::new(),
            model_timeout: Duration::from_secs(30),
            min_models: 1,
            parallel: true,
        }
    }
}

/// Ensemble strategies
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EnsembleStrategy {
    /// Simple majority voting
    Voting,
    /// Weighted voting
    WeightedVoting,
    /// Best response by score
    BestOfN,
    /// Average/merge responses
    Average,
    /// Cascade (use fallback on failure)
    Cascade,
    /// Route to best model per query
    Routing,
    /// Custom strategy
    Custom(String),
}

/// A model in the ensemble
#[derive(Debug, Clone)]
pub struct EnsembleModel {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Provider (ollama, openai, etc.)
    pub provider: String,
    /// Weight for voting/averaging
    pub weight: f64,
    /// Specializations (domains the model is good at)
    pub specializations: Vec<String>,
    /// Historical accuracy
    pub accuracy: f64,
    /// Average latency
    pub avg_latency: Duration,
}

impl EnsembleModel {
    /// Create a new ensemble model
    pub fn new(id: impl Into<String>, provider: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            provider: provider.into(),
            weight: 1.0,
            specializations: Vec::new(),
            accuracy: 0.8,
            avg_latency: Duration::from_secs(1),
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Add specialization
    pub fn with_specialization(mut self, spec: impl Into<String>) -> Self {
        self.specializations.push(spec.into());
        self
    }
}

/// A response from a model
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// Model that generated this
    pub model_id: String,
    /// The response text
    pub response: String,
    /// Time taken
    pub duration: Duration,
    /// Confidence score
    pub confidence: f64,
    /// Whether successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Result of ensemble execution
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final combined response
    pub response: String,
    /// Individual model responses
    pub model_responses: Vec<ModelResponse>,
    /// Winning model (for voting/best-of-N)
    pub winning_model: Option<String>,
    /// Agreement score (0-1)
    pub agreement: f64,
    /// Total time
    pub total_duration: Duration,
    /// Strategy used
    pub strategy_used: EnsembleStrategy,
}

/// Ensemble executor
pub struct Ensemble {
    config: EnsembleConfig,
    /// Response scorer
    scorer: Option<Box<dyn Fn(&str) -> f64 + Send + Sync>>,
}

impl Ensemble {
    /// Create a new ensemble
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            scorer: None,
        }
    }

    /// Set response scorer
    pub fn with_scorer<F>(mut self, scorer: F) -> Self
    where
        F: Fn(&str) -> f64 + Send + Sync + 'static,
    {
        self.scorer = Some(Box::new(scorer));
        self
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: EnsembleModel) {
        self.config.models.push(model);
    }

    /// Execute ensemble on a prompt
    pub fn execute<F>(&self, prompt: &str, generate: F) -> EnsembleResult
    where
        F: Fn(&str, &str, &str) -> Result<String, String>,
    {
        let start = Instant::now();
        let mut responses = Vec::new();

        // Generate responses from all models
        for model in &self.config.models {
            let model_start = Instant::now();
            let result = generate(prompt, &model.id, &model.provider);

            let response = match result {
                Ok(text) => {
                    let confidence = self.scorer.as_ref().map(|s| s(&text)).unwrap_or(0.8);

                    ModelResponse {
                        model_id: model.id.clone(),
                        response: text,
                        duration: model_start.elapsed(),
                        confidence,
                        success: true,
                        error: None,
                    }
                }
                Err(e) => ModelResponse {
                    model_id: model.id.clone(),
                    response: String::new(),
                    duration: model_start.elapsed(),
                    confidence: 0.0,
                    success: false,
                    error: Some(e),
                },
            };

            responses.push(response);
        }

        // Apply ensemble strategy
        let (final_response, winning_model) = match &self.config.strategy {
            EnsembleStrategy::Voting => self.voting(&responses),
            EnsembleStrategy::WeightedVoting => self.weighted_voting(&responses),
            EnsembleStrategy::BestOfN => self.best_of_n(&responses),
            EnsembleStrategy::Average => self.average(&responses),
            EnsembleStrategy::Cascade => self.cascade(&responses),
            EnsembleStrategy::Routing => self.routing(prompt, &responses),
            EnsembleStrategy::Custom(_) => self.voting(&responses), // Default to voting
        };

        // Calculate agreement
        let agreement = self.calculate_agreement(&responses);

        EnsembleResult {
            response: final_response,
            model_responses: responses,
            winning_model,
            agreement,
            total_duration: start.elapsed(),
            strategy_used: self.config.strategy.clone(),
        }
    }

    /// Simple majority voting
    fn voting(&self, responses: &[ModelResponse]) -> (String, Option<String>) {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return (String::new(), None);
        }

        // Count similar responses
        let mut response_counts: HashMap<String, (usize, String)> = HashMap::new();

        for resp in &successful {
            let normalized = resp.response.trim().to_lowercase();
            let entry = response_counts
                .entry(normalized)
                .or_insert((0, resp.model_id.clone()));
            entry.0 += 1;
        }

        // Find most common
        let winner = response_counts
            .into_iter()
            .max_by_key(|(_, (count, _))| *count);

        match winner {
            Some((_, (_, model_id))) => {
                let winning_resp = successful
                    .iter()
                    .find(|r| r.model_id == model_id)
                    .map(|r| r.response.clone())
                    .unwrap_or_default();
                (winning_resp, Some(model_id))
            }
            None => (String::new(), None),
        }
    }

    /// Weighted voting
    fn weighted_voting(&self, responses: &[ModelResponse]) -> (String, Option<String>) {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return (String::new(), None);
        }

        // Sum weights for similar responses
        let mut response_weights: HashMap<String, (f64, String)> = HashMap::new();

        for resp in &successful {
            let weight = self
                .config
                .models
                .iter()
                .find(|m| m.id == resp.model_id)
                .map(|m| m.weight)
                .unwrap_or(1.0);

            let normalized = resp.response.trim().to_lowercase();
            let entry = response_weights
                .entry(normalized)
                .or_insert((0.0, resp.model_id.clone()));
            entry.0 += weight;
        }

        // Find highest weighted
        let winner = response_weights
            .into_iter()
            .max_by(|(_, (w1, _)), (_, (w2, _))| {
                w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal)
            });

        match winner {
            Some((_, (_, model_id))) => {
                let winning_resp = successful
                    .iter()
                    .find(|r| r.model_id == model_id)
                    .map(|r| r.response.clone())
                    .unwrap_or_default();
                (winning_resp, Some(model_id))
            }
            None => (String::new(), None),
        }
    }

    /// Best of N by confidence
    fn best_of_n(&self, responses: &[ModelResponse]) -> (String, Option<String>) {
        let best = responses.iter().filter(|r| r.success).max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match best {
            Some(r) => (r.response.clone(), Some(r.model_id.clone())),
            None => (String::new(), None),
        }
    }

    /// Average/merge responses
    fn average(&self, responses: &[ModelResponse]) -> (String, Option<String>) {
        // For text, we can't really average - use best quality one
        // In practice, this would be used for numerical outputs
        self.best_of_n(responses)
    }

    /// Cascade through models
    fn cascade(&self, responses: &[ModelResponse]) -> (String, Option<String>) {
        // Use first successful response
        for resp in responses {
            if resp.success && !resp.response.is_empty() {
                return (resp.response.clone(), Some(resp.model_id.clone()));
            }
        }
        (String::new(), None)
    }

    /// Route to best model for query type
    fn routing(&self, prompt: &str, responses: &[ModelResponse]) -> (String, Option<String>) {
        let lower = prompt.to_lowercase();

        // Find model specialized for this type of query
        for model in &self.config.models {
            for spec in &model.specializations {
                if lower.contains(&spec.to_lowercase()) {
                    if let Some(resp) = responses
                        .iter()
                        .find(|r| r.model_id == model.id && r.success)
                    {
                        return (resp.response.clone(), Some(model.id.clone()));
                    }
                }
            }
        }

        // Fall back to best of N
        self.best_of_n(responses)
    }

    /// Calculate agreement between responses
    fn calculate_agreement(&self, responses: &[ModelResponse]) -> f64 {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.len() < 2 {
            return 1.0;
        }

        // Calculate pairwise similarity
        let mut similarity_sum = 0.0;
        let mut pairs = 0;

        for i in 0..successful.len() {
            for j in (i + 1)..successful.len() {
                similarity_sum +=
                    self.text_similarity(&successful[i].response, &successful[j].response);
                pairs += 1;
            }
        }

        if pairs == 0 {
            1.0
        } else {
            similarity_sum / pairs as f64
        }
    }

    /// Simple text similarity
    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let lower_a = a.to_lowercase();
        let lower_b = b.to_lowercase();
        let words_a: std::collections::HashSet<_> = lower_a.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = lower_b.split_whitespace().collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

impl Default for Ensemble {
    fn default() -> Self {
        Self::new(EnsembleConfig::default())
    }
}

/// Builder for ensemble configuration
#[non_exhaustive]
pub struct EnsembleConfigBuilder {
    config: EnsembleConfig,
}

impl EnsembleConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: EnsembleConfig::default(),
        }
    }

    pub fn strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn add_model(mut self, model: EnsembleModel) -> Self {
        self.config.models.push(model);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.model_timeout = timeout;
        self
    }

    pub fn min_models(mut self, min: usize) -> Self {
        self.config.min_models = min;
        self
    }

    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    pub fn build(self) -> EnsembleConfig {
        self.config
    }
}

impl Default for EnsembleConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Ensemble builder
pub struct EnsembleBuilder {
    config: EnsembleConfig,
    scorer: Option<Box<dyn Fn(&str) -> f64 + Send + Sync>>,
}

impl EnsembleBuilder {
    pub fn new() -> Self {
        Self {
            config: EnsembleConfig::default(),
            scorer: None,
        }
    }

    pub fn strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn add_model(mut self, model: EnsembleModel) -> Self {
        self.config.models.push(model);
        self
    }

    pub fn scorer<F>(mut self, scorer: F) -> Self
    where
        F: Fn(&str) -> f64 + Send + Sync + 'static,
    {
        self.scorer = Some(Box::new(scorer));
        self
    }

    pub fn build(self) -> Ensemble {
        let mut ensemble = Ensemble::new(self.config);
        if let Some(scorer) = self.scorer {
            ensemble.scorer = Some(scorer);
        }
        ensemble
    }
}

impl Default for EnsembleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_creation() {
        let config = EnsembleConfigBuilder::new()
            .strategy(EnsembleStrategy::Voting)
            .add_model(EnsembleModel::new("gpt-4", "openai"))
            .add_model(EnsembleModel::new("llama2", "ollama"))
            .build();

        let ensemble = Ensemble::new(config);
        assert_eq!(ensemble.config.models.len(), 2);
    }

    #[test]
    fn test_voting() {
        let ensemble = EnsembleBuilder::new()
            .strategy(EnsembleStrategy::Voting)
            .add_model(EnsembleModel::new("model1", "provider"))
            .add_model(EnsembleModel::new("model2", "provider"))
            .add_model(EnsembleModel::new("model3", "provider"))
            .build();

        let responses = vec![
            ModelResponse {
                model_id: "model1".to_string(),
                response: "The answer is 42".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.8,
                success: true,
                error: None,
            },
            ModelResponse {
                model_id: "model2".to_string(),
                response: "The answer is 42".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.7,
                success: true,
                error: None,
            },
            ModelResponse {
                model_id: "model3".to_string(),
                response: "It's 43".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.6,
                success: true,
                error: None,
            },
        ];

        let (winner, _) = ensemble.voting(&responses);
        assert!(winner.contains("42"));
    }

    #[test]
    fn test_best_of_n() {
        let ensemble = Ensemble::default();

        let responses = vec![
            ModelResponse {
                model_id: "model1".to_string(),
                response: "Low quality".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.5,
                success: true,
                error: None,
            },
            ModelResponse {
                model_id: "model2".to_string(),
                response: "High quality".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.9,
                success: true,
                error: None,
            },
        ];

        let (winner, model_id) = ensemble.best_of_n(&responses);
        assert_eq!(winner, "High quality");
        assert_eq!(model_id, Some("model2".to_string()));
    }

    #[test]
    fn test_cascade() {
        let ensemble = Ensemble::default();

        let responses = vec![
            ModelResponse {
                model_id: "model1".to_string(),
                response: String::new(),
                duration: Duration::from_secs(1),
                confidence: 0.0,
                success: false,
                error: Some("Failed".to_string()),
            },
            ModelResponse {
                model_id: "model2".to_string(),
                response: "Success!".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.8,
                success: true,
                error: None,
            },
        ];

        let (winner, model_id) = ensemble.cascade(&responses);
        assert_eq!(winner, "Success!");
        assert_eq!(model_id, Some("model2".to_string()));
    }

    #[test]
    fn test_agreement_calculation() {
        let ensemble = Ensemble::default();

        let responses = vec![
            ModelResponse {
                model_id: "m1".to_string(),
                response: "The answer is 42".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.8,
                success: true,
                error: None,
            },
            ModelResponse {
                model_id: "m2".to_string(),
                response: "The answer is 42".to_string(),
                duration: Duration::from_secs(1),
                confidence: 0.8,
                success: true,
                error: None,
            },
        ];

        let agreement = ensemble.calculate_agreement(&responses);
        assert_eq!(agreement, 1.0); // Identical responses
    }

    #[test]
    fn test_ensemble_config_defaults() {
        let config = EnsembleConfig::default();
        assert_eq!(config.strategy, EnsembleStrategy::Voting);
        assert!(config.models.is_empty());
        assert_eq!(config.model_timeout, Duration::from_secs(30));
        assert_eq!(config.min_models, 1);
        assert!(config.parallel);
    }

    #[test]
    fn test_ensemble_model_builder() {
        let model = EnsembleModel::new("gpt-4", "openai")
            .with_weight(2.0)
            .with_specialization("coding");

        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.provider, "openai");
        assert_eq!(model.weight, 2.0);
        assert_eq!(model.specializations, vec!["coding"]);
    }

    #[test]
    fn test_ensemble_add_model() {
        let mut ensemble = Ensemble::default();
        ensemble.add_model(EnsembleModel::new("m1", "ollama"));
        ensemble.add_model(EnsembleModel::new("m2", "openai"));

        // Execute with 2 models
        let result = ensemble.execute("test", |_prompt, model_id, _provider| {
            Ok(format!("Response from {}", model_id))
        });

        assert_eq!(result.model_responses.len(), 2);
        assert!(result.model_responses.iter().all(|r| r.success));
    }

    #[test]
    fn test_execute_with_all_failures() {
        let mut ensemble = Ensemble::default();
        ensemble.add_model(EnsembleModel::new("m1", "ollama"));

        let result = ensemble.execute("test", |_prompt, _model, _provider| {
            Err("model unavailable".to_string())
        });

        assert_eq!(result.model_responses.len(), 1);
        assert!(!result.model_responses[0].success);
        assert!(result.model_responses[0].error.is_some());
    }

    #[test]
    fn test_ensemble_strategy_variants() {
        let strategies = [
            EnsembleStrategy::Voting,
            EnsembleStrategy::WeightedVoting,
            EnsembleStrategy::BestOfN,
            EnsembleStrategy::Average,
            EnsembleStrategy::Cascade,
            EnsembleStrategy::Routing,
            EnsembleStrategy::Custom("mine".to_string()),
        ];
        // All variants are distinct
        for (i, a) in strategies.iter().enumerate() {
            for (j, b) in strategies.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
