//! Cost estimation for AI API usage
//!
//! This module provides cost estimation based on token usage and
//! pricing for various AI providers and models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pricing information for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Model name pattern
    pub model_pattern: String,
    /// Cost per 1M input tokens (in USD)
    pub input_cost_per_million: f64,
    /// Cost per 1M output tokens (in USD)
    pub output_cost_per_million: f64,
    /// Cost per image (if applicable)
    pub image_cost: Option<f64>,
    /// Currency (default USD)
    pub currency: String,
    /// Provider name
    pub provider: String,
}

impl ModelPricing {
    /// Create new pricing info
    pub fn new(pattern: &str, input_cost: f64, output_cost: f64) -> Self {
        Self {
            model_pattern: pattern.to_string(),
            input_cost_per_million: input_cost,
            output_cost_per_million: output_cost,
            image_cost: None,
            currency: "USD".to_string(),
            provider: "unknown".to_string(),
        }
    }

    /// Set provider
    pub fn with_provider(mut self, provider: &str) -> Self {
        self.provider = provider.to_string();
        self
    }

    /// Set image cost
    pub fn with_image_cost(mut self, cost: f64) -> Self {
        self.image_cost = Some(cost);
        self
    }

    /// Calculate cost for given token counts
    pub fn calculate(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;
        input_cost + output_cost
    }

    /// Calculate cost including images
    pub fn calculate_with_images(
        &self,
        input_tokens: usize,
        output_tokens: usize,
        images: usize,
    ) -> f64 {
        let base_cost = self.calculate(input_tokens, output_tokens);
        let image_cost = self.image_cost.unwrap_or(0.0) * images as f64;
        base_cost + image_cost
    }

    /// Check if this pricing matches a model name
    pub fn matches(&self, model_name: &str) -> bool {
        model_name
            .to_lowercase()
            .contains(&self.model_pattern.to_lowercase())
    }
}

/// Cost estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Input tokens
    pub input_tokens: usize,
    /// Output tokens
    pub output_tokens: usize,
    /// Number of images
    pub images: usize,
    /// Estimated cost
    pub cost: f64,
    /// Currency
    pub currency: String,
    /// Model used
    pub model: String,
    /// Provider
    pub provider: String,
    /// Pricing tier used
    pub pricing_tier: Option<String>,
}

impl CostEstimate {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        if self.cost < 0.01 {
            format!(
                "${:.4} {} ({} in / {} out)",
                self.cost, self.currency, self.input_tokens, self.output_tokens
            )
        } else {
            format!(
                "${:.2} {} ({} in / {} out)",
                self.cost, self.currency, self.input_tokens, self.output_tokens
            )
        }
    }

    /// Format as short string
    pub fn format_short(&self) -> String {
        if self.cost < 0.01 {
            format!("${:.4}", self.cost)
        } else if self.cost < 1.0 {
            format!("${:.3}", self.cost)
        } else {
            format!("${:.2}", self.cost)
        }
    }
}

/// Cost tracker for session/conversation costs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostTracker {
    /// Total input tokens
    pub total_input_tokens: usize,
    /// Total output tokens
    pub total_output_tokens: usize,
    /// Total images processed
    pub total_images: usize,
    /// Total cost
    pub total_cost: f64,
    /// Number of requests
    pub request_count: usize,
    /// Cost by model
    pub cost_by_model: HashMap<String, f64>,
    /// Currency
    pub currency: String,
    /// Cost history
    history: Vec<CostEstimate>,
}

impl CostTracker {
    /// Create a new cost tracker
    pub fn new() -> Self {
        Self {
            currency: "USD".to_string(),
            ..Default::default()
        }
    }

    /// Add a cost entry
    pub fn add(&mut self, estimate: CostEstimate) {
        self.total_input_tokens += estimate.input_tokens;
        self.total_output_tokens += estimate.output_tokens;
        self.total_images += estimate.images;
        self.total_cost += estimate.cost;
        self.request_count += 1;

        *self
            .cost_by_model
            .entry(estimate.model.clone())
            .or_insert(0.0) += estimate.cost;

        self.history.push(estimate);
    }

    /// Get average cost per request
    pub fn average_cost(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.total_cost / self.request_count as f64
        }
    }

    /// Get average tokens per request
    pub fn average_tokens(&self) -> (usize, usize) {
        if self.request_count == 0 {
            (0, 0)
        } else {
            (
                self.total_input_tokens / self.request_count,
                self.total_output_tokens / self.request_count,
            )
        }
    }

    /// Get cost history
    pub fn history(&self) -> &[CostEstimate] {
        &self.history
    }

    /// Clear history but keep totals
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Reset all tracking
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Format summary
    pub fn summary(&self) -> String {
        format!(
            "Total: ${:.4} | Requests: {} | Avg: ${:.4}/req | Tokens: {} in / {} out",
            self.total_cost,
            self.request_count,
            self.average_cost(),
            self.total_input_tokens,
            self.total_output_tokens
        )
    }
}

/// Cost estimator with pricing database
#[derive(Debug)]
pub struct CostEstimator {
    /// Pricing database
    pricing: Vec<ModelPricing>,
    /// Default pricing for unknown models
    default_pricing: ModelPricing,
    /// Local models are free
    local_models_free: bool,
}

impl CostEstimator {
    /// Create a new estimator with common pricing
    pub fn new() -> Self {
        let mut estimator = Self {
            pricing: Vec::new(),
            default_pricing: ModelPricing::new("default", 1.0, 2.0),
            local_models_free: true,
        };

        estimator.add_common_pricing();
        estimator
    }

    /// Add common model pricing
    fn add_common_pricing(&mut self) {
        // OpenAI models
        self.pricing.push(
            ModelPricing::new("gpt-4o", 2.50, 10.00)
                .with_provider("openai")
                .with_image_cost(0.00255),
        );
        self.pricing
            .push(ModelPricing::new("gpt-4-turbo", 10.00, 30.00).with_provider("openai"));
        self.pricing
            .push(ModelPricing::new("gpt-4", 30.00, 60.00).with_provider("openai"));
        self.pricing
            .push(ModelPricing::new("gpt-3.5-turbo", 0.50, 1.50).with_provider("openai"));

        // Anthropic models
        self.pricing
            .push(ModelPricing::new("claude-3-opus", 15.00, 75.00).with_provider("anthropic"));
        self.pricing
            .push(ModelPricing::new("claude-3-sonnet", 3.00, 15.00).with_provider("anthropic"));
        self.pricing
            .push(ModelPricing::new("claude-3-haiku", 0.25, 1.25).with_provider("anthropic"));

        // Google models
        self.pricing
            .push(ModelPricing::new("gemini-pro", 0.50, 1.50).with_provider("google"));
        self.pricing
            .push(ModelPricing::new("gemini-ultra", 7.00, 21.00).with_provider("google"));

        // Together.ai / Replicate pricing examples
        self.pricing
            .push(ModelPricing::new("llama-3-70b", 0.90, 0.90).with_provider("together"));
        self.pricing
            .push(ModelPricing::new("llama-3-8b", 0.20, 0.20).with_provider("together"));
        self.pricing
            .push(ModelPricing::new("mixtral-8x7b", 0.60, 0.60).with_provider("together"));
        self.pricing
            .push(ModelPricing::new("mistral-7b", 0.20, 0.20).with_provider("together"));

        // Groq (very fast, lower cost)
        self.pricing
            .push(ModelPricing::new("groq", 0.05, 0.08).with_provider("groq"));
    }

    /// Add custom pricing
    pub fn add_pricing(&mut self, pricing: ModelPricing) {
        self.pricing.push(pricing);
    }

    /// Set default pricing for unknown models
    pub fn set_default_pricing(&mut self, pricing: ModelPricing) {
        self.default_pricing = pricing;
    }

    /// Set whether local models (Ollama, LM Studio) are free
    pub fn set_local_free(&mut self, free: bool) {
        self.local_models_free = free;
    }

    /// Get pricing for a model
    pub fn get_pricing(&self, model_name: &str) -> &ModelPricing {
        self.pricing
            .iter()
            .find(|p| p.matches(model_name))
            .unwrap_or(&self.default_pricing)
    }

    /// Check if a model is considered local/free
    pub fn is_local_model(&self, model_name: &str, provider: &str) -> bool {
        let local_providers = ["ollama", "lm-studio", "localai", "kobold", "text-gen"];
        local_providers
            .iter()
            .any(|p| provider.to_lowercase().contains(p))
            || model_name.to_lowercase().contains("local")
    }

    /// Estimate cost for a request
    pub fn estimate(
        &self,
        model_name: &str,
        provider: &str,
        input_tokens: usize,
        output_tokens: usize,
    ) -> CostEstimate {
        // Local models are free
        if self.local_models_free && self.is_local_model(model_name, provider) {
            return CostEstimate {
                input_tokens,
                output_tokens,
                images: 0,
                cost: 0.0,
                currency: "USD".to_string(),
                model: model_name.to_string(),
                provider: provider.to_string(),
                pricing_tier: Some("local/free".to_string()),
            };
        }

        let pricing = self.get_pricing(model_name);
        let cost = pricing.calculate(input_tokens, output_tokens);

        CostEstimate {
            input_tokens,
            output_tokens,
            images: 0,
            cost,
            currency: pricing.currency.clone(),
            model: model_name.to_string(),
            provider: provider.to_string(),
            pricing_tier: Some(pricing.model_pattern.clone()),
        }
    }

    /// Estimate cost with images
    pub fn estimate_with_images(
        &self,
        model_name: &str,
        provider: &str,
        input_tokens: usize,
        output_tokens: usize,
        images: usize,
    ) -> CostEstimate {
        let mut estimate = self.estimate(model_name, provider, input_tokens, output_tokens);

        if images > 0 {
            let pricing = self.get_pricing(model_name);
            let image_cost = pricing.image_cost.unwrap_or(0.0) * images as f64;
            estimate.cost += image_cost;
            estimate.images = images;
        }

        estimate
    }

    /// Estimate cost from text (using token estimation)
    pub fn estimate_from_text(
        &self,
        model_name: &str,
        provider: &str,
        input_text: &str,
        expected_output_ratio: f32,
    ) -> CostEstimate {
        let input_tokens = crate::estimate_tokens(input_text);
        let output_tokens = (input_tokens as f32 * expected_output_ratio) as usize;
        self.estimate(model_name, provider, input_tokens, output_tokens)
    }

    /// Get all pricing info
    pub fn all_pricing(&self) -> &[ModelPricing] {
        &self.pricing
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget manager for cost limits
#[derive(Debug, Clone)]
pub struct BudgetManager {
    /// Daily budget limit
    pub daily_limit: Option<f64>,
    /// Monthly budget limit
    pub monthly_limit: Option<f64>,
    /// Per-request limit
    pub per_request_limit: Option<f64>,
    /// Spent today
    pub spent_today: f64,
    /// Spent this month
    pub spent_month: f64,
    /// Warning threshold (0.0 - 1.0)
    pub warning_threshold: f32,
}

impl Default for BudgetManager {
    fn default() -> Self {
        Self {
            daily_limit: None,
            monthly_limit: None,
            per_request_limit: None,
            spent_today: 0.0,
            spent_month: 0.0,
            warning_threshold: 0.8,
        }
    }
}

impl BudgetManager {
    /// Create a new budget manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Set daily limit
    pub fn with_daily_limit(mut self, limit: f64) -> Self {
        self.daily_limit = Some(limit);
        self
    }

    /// Set monthly limit
    pub fn with_monthly_limit(mut self, limit: f64) -> Self {
        self.monthly_limit = Some(limit);
        self
    }

    /// Set per-request limit
    pub fn with_request_limit(mut self, limit: f64) -> Self {
        self.per_request_limit = Some(limit);
        self
    }

    /// Check if a cost is within budget
    pub fn check(&self, cost: f64) -> BudgetStatus {
        // Check per-request limit
        if let Some(limit) = self.per_request_limit {
            if cost > limit {
                return BudgetStatus::Exceeded {
                    limit_type: "per_request".to_string(),
                    limit,
                    current: cost,
                };
            }
        }

        // Check daily limit
        if let Some(limit) = self.daily_limit {
            let new_total = self.spent_today + cost;
            if new_total > limit {
                return BudgetStatus::Exceeded {
                    limit_type: "daily".to_string(),
                    limit,
                    current: new_total,
                };
            }
            if new_total > limit * self.warning_threshold as f64 {
                return BudgetStatus::Warning {
                    limit_type: "daily".to_string(),
                    limit,
                    current: new_total,
                    remaining: limit - new_total,
                };
            }
        }

        // Check monthly limit
        if let Some(limit) = self.monthly_limit {
            let new_total = self.spent_month + cost;
            if new_total > limit {
                return BudgetStatus::Exceeded {
                    limit_type: "monthly".to_string(),
                    limit,
                    current: new_total,
                };
            }
            if new_total > limit * self.warning_threshold as f64 {
                return BudgetStatus::Warning {
                    limit_type: "monthly".to_string(),
                    limit,
                    current: new_total,
                    remaining: limit - new_total,
                };
            }
        }

        BudgetStatus::Ok
    }

    /// Record a cost
    pub fn record(&mut self, cost: f64) {
        self.spent_today += cost;
        self.spent_month += cost;
    }

    /// Reset daily counter (call at midnight)
    pub fn reset_daily(&mut self) {
        self.spent_today = 0.0;
    }

    /// Reset monthly counter (call at month start)
    pub fn reset_monthly(&mut self) {
        self.spent_month = 0.0;
    }

    /// Get remaining budget
    pub fn remaining(&self) -> (Option<f64>, Option<f64>) {
        let daily_remaining = self.daily_limit.map(|l| (l - self.spent_today).max(0.0));
        let monthly_remaining = self.monthly_limit.map(|l| (l - self.spent_month).max(0.0));
        (daily_remaining, monthly_remaining)
    }
}

/// Budget check status
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BudgetStatus {
    /// Within budget
    Ok,
    /// Approaching limit
    Warning {
        limit_type: String,
        limit: f64,
        current: f64,
        remaining: f64,
    },
    /// Budget exceeded
    Exceeded {
        limit_type: String,
        limit: f64,
        current: f64,
    },
}

impl BudgetStatus {
    /// Check if OK
    pub fn is_ok(&self) -> bool {
        matches!(self, BudgetStatus::Ok)
    }

    /// Check if exceeded
    pub fn is_exceeded(&self) -> bool {
        matches!(self, BudgetStatus::Exceeded { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        let pricing = ModelPricing::new("gpt-4", 30.0, 60.0);

        // 1000 tokens: $0.00003 input (30/1M * 1000) + $0.00006 output (60/1M * 1000) = $0.00009
        let cost = pricing.calculate(1000, 1000);
        // Expected: 30 * 1000 / 1_000_000 + 60 * 1000 / 1_000_000 = 0.03 + 0.06 = 0.09
        assert!((cost - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_estimator() {
        let estimator = CostEstimator::new();

        // Local model should be free
        let estimate = estimator.estimate("llama-3-8b", "ollama", 1000, 500);
        assert_eq!(estimate.cost, 0.0);

        // GPT-4 should have cost
        let estimate = estimator.estimate("gpt-4-turbo", "openai", 1000, 1000);
        assert!(estimate.cost > 0.0);
    }

    #[test]
    fn test_cost_tracker() {
        let mut tracker = CostTracker::new();

        let estimate = CostEstimate {
            input_tokens: 1000,
            output_tokens: 500,
            images: 0,
            cost: 0.05,
            currency: "USD".to_string(),
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            pricing_tier: None,
        };

        tracker.add(estimate.clone());
        tracker.add(estimate);

        assert_eq!(tracker.request_count, 2);
        assert!((tracker.total_cost - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_budget_manager() {
        let mut budget = BudgetManager::new()
            .with_daily_limit(1.0)
            .with_request_limit(0.50);

        // Should be OK
        assert!(budget.check(0.10).is_ok());

        // Should exceed per-request
        assert!(budget.check(0.60).is_exceeded());

        // Record some spending
        budget.record(0.90);

        // Should exceed daily
        assert!(budget.check(0.20).is_exceeded());
    }

    #[test]
    fn test_pricing_patterns() {
        let estimator = CostEstimator::new();

        // Should match GPT-4 pricing
        let pricing = estimator.get_pricing("gpt-4-0125-preview");
        assert!(pricing.input_cost_per_million > 0.0);

        // Should match Claude
        let pricing = estimator.get_pricing("claude-3-sonnet-20240229");
        assert_eq!(pricing.provider, "anthropic");
    }

    #[test]
    fn test_cost_estimate_format() {
        let est = CostEstimate {
            input_tokens: 1000, output_tokens: 500, images: 0,
            cost: 0.005, currency: "USD".into(), model: "m".into(),
            provider: "p".into(), pricing_tier: None,
        };
        let formatted = est.format();
        assert!(formatted.contains("1000"));
        assert!(formatted.contains("500"));
        let short = est.format_short();
        assert!(short.starts_with('$'));
    }

    #[test]
    fn test_cost_tracker_averages() {
        let mut tracker = CostTracker::new();
        for i in 0..4 {
            tracker.add(CostEstimate {
                input_tokens: 100 * (i + 1), output_tokens: 50 * (i + 1), images: 0,
                cost: 0.01 * (i + 1) as f64, currency: "USD".into(),
                model: "gpt-4".into(), provider: "openai".into(), pricing_tier: None,
            });
        }
        assert_eq!(tracker.request_count, 4);
        assert!((tracker.average_cost() - 0.025).abs() < 0.001);
        let (avg_in, avg_out) = tracker.average_tokens();
        assert_eq!(avg_in, 250); // (100+200+300+400)/4
        assert_eq!(avg_out, 125);
        assert_eq!(tracker.history().len(), 4);
    }

    #[test]
    fn test_cost_tracker_reset() {
        let mut tracker = CostTracker::new();
        tracker.add(CostEstimate {
            input_tokens: 100, output_tokens: 50, images: 0,
            cost: 0.01, currency: "USD".into(), model: "m".into(),
            provider: "p".into(), pricing_tier: None,
        });
        tracker.reset();
        assert_eq!(tracker.request_count, 0);
        assert!((tracker.total_cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_budget_warning() {
        let mut budget = BudgetManager::new()
            .with_daily_limit(10.0)
            .with_monthly_limit(100.0);
        budget.warning_threshold = 0.8;
        // Spend 8.5 of 10 daily => warning territory
        budget.record(8.5);
        let status = budget.check(0.5);
        assert!(matches!(status, BudgetStatus::Warning { .. }));
        // Check remaining
        let (daily, monthly) = budget.remaining();
        assert!((daily.unwrap() - 1.5).abs() < 0.01);
        assert!((monthly.unwrap() - 91.5).abs() < 0.01);
    }

    #[test]
    fn test_pricing_with_images() {
        let pricing = ModelPricing::new("gpt-4o", 2.50, 10.00).with_image_cost(0.01);
        let cost = pricing.calculate_with_images(1000, 1000, 3);
        // base: 2.5*0.001 + 10*0.001 = 0.0025 + 0.01 = 0.0125
        // images: 0.01*3 = 0.03
        // total: 0.0425
        assert!((cost - 0.0425).abs() < 0.0001);
    }
}
