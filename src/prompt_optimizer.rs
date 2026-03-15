//! Prompt optimization
//!
//! This module provides tools for automatically optimizing prompts
//! based on response quality and effectiveness feedback.
//!
//! # Features
//!
//! - **A/B testing**: Compare prompt variations
//! - **Auto-optimization**: Learn from feedback
//! - **Template refinement**: Improve template performance
//! - **Token efficiency**: Reduce prompt length while maintaining quality
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::prompt_optimizer::{PromptOptimizer, OptimizerConfig, PromptVariant};
//!
//! let mut optimizer = PromptOptimizer::new(OptimizerConfig::default());
//!
//! // Register prompt variants
//! optimizer.add_variant("concise", "Answer briefly: {query}");
//! optimizer.add_variant("detailed", "Provide a detailed answer to: {query}");
//!
//! // Get best variant for a query
//! let best = optimizer.select_best("What is AI?");
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Prompt variant for A/B testing
#[derive(Debug, Clone)]
pub struct PromptVariant {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Prompt template
    pub template: String,
    /// Times used
    pub use_count: u64,
    /// Success count (positive feedback)
    pub success_count: u64,
    /// Total quality score sum
    pub quality_sum: f64,
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    /// Average token count
    pub avg_tokens: usize,
    /// Is active for testing
    pub active: bool,
    /// Created at
    pub created_at: Instant,
}

impl PromptVariant {
    /// Create a new variant
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            template: template.into(),
            use_count: 0,
            success_count: 0,
            quality_sum: 0.0,
            avg_response_time_ms: 0,
            avg_tokens: 0,
            active: true,
            created_at: Instant::now(),
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.use_count == 0 {
            0.5 // Neutral for untested
        } else {
            self.success_count as f64 / self.use_count as f64
        }
    }

    /// Get average quality
    pub fn avg_quality(&self) -> f64 {
        if self.use_count == 0 {
            0.5
        } else {
            self.quality_sum / self.use_count as f64
        }
    }

    /// Calculate effectiveness score
    pub fn effectiveness(&self) -> f64 {
        let success = self.success_rate();
        let quality = self.avg_quality();

        // Combined score with exploration bonus for less-tested variants
        let exploration_bonus = if self.use_count < 10 {
            0.1 * (1.0 - self.use_count as f64 / 10.0)
        } else {
            0.0
        };

        success * 0.5 + quality * 0.5 + exploration_bonus
    }

    /// Apply template with variables
    pub fn apply(&self, variables: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in variables {
            result = result.replace(&format!("{{{}}}", key), value);
        }
        result
    }

    /// Apply with single query variable
    pub fn apply_query(&self, query: &str) -> String {
        self.template.replace("{query}", query)
    }

    /// Estimate tokens for this template
    pub fn estimate_tokens(&self, query: &str) -> usize {
        let filled = self.apply_query(query);
        filled.len() / 4 // Rough estimate
    }
}

/// Optimization configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptimizerConfig {
    /// Minimum uses before considering for best selection
    pub min_uses_for_selection: u64,
    /// Exploration rate (0-1) - probability of trying non-best variant
    pub exploration_rate: f64,
    /// Auto-deactivate variants below this success rate
    pub min_success_rate: f64,
    /// Maximum active variants
    pub max_active_variants: usize,
    /// Enable automatic variant generation
    pub auto_generate: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            min_uses_for_selection: 5,
            exploration_rate: 0.1,
            min_success_rate: 0.3,
            max_active_variants: 10,
            auto_generate: false,
        }
    }
}

/// Prompt optimizer for A/B testing and improvement
pub struct PromptOptimizer {
    config: OptimizerConfig,
    variants: HashMap<String, PromptVariant>,
    groups: HashMap<String, Vec<String>>, // Group name -> variant IDs
    feedback_history: Vec<FeedbackEntry>,
}

impl PromptOptimizer {
    /// Create a new optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            variants: HashMap::new(),
            groups: HashMap::new(),
            feedback_history: Vec::new(),
        }
    }

    /// Add a variant
    pub fn add_variant(&mut self, name: &str, template: &str) -> String {
        let variant = PromptVariant::new(name, template);
        let id = variant.id.clone();
        self.variants.insert(id.clone(), variant);
        id
    }

    /// Add variant to a group
    pub fn add_to_group(&mut self, group: &str, name: &str, template: &str) -> String {
        let id = self.add_variant(name, template);
        self.groups
            .entry(group.to_string())
            .or_default()
            .push(id.clone());
        id
    }

    /// Get variant by ID
    pub fn get_variant(&self, id: &str) -> Option<&PromptVariant> {
        self.variants.get(id)
    }

    /// Get all variants
    pub fn all_variants(&self) -> Vec<&PromptVariant> {
        self.variants.values().collect()
    }

    /// Get active variants
    pub fn active_variants(&self) -> Vec<&PromptVariant> {
        self.variants.values().filter(|v| v.active).collect()
    }

    /// Select best variant overall
    pub fn select_best(&self, _query: &str) -> Option<&PromptVariant> {
        // Check if we should explore
        if rand_float() < self.config.exploration_rate {
            return self.select_random();
        }

        self.variants
            .values()
            .filter(|v| v.active)
            .filter(|v| v.use_count >= self.config.min_uses_for_selection)
            .max_by(|a, b| {
                a.effectiveness()
                    .partial_cmp(&b.effectiveness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| self.select_random())
    }

    /// Select best from a group
    pub fn select_best_from_group(&self, group: &str, _query: &str) -> Option<&PromptVariant> {
        let variant_ids = self.groups.get(group)?;

        // Check if we should explore
        if rand_float() < self.config.exploration_rate {
            return self.select_random_from_group(group);
        }

        variant_ids
            .iter()
            .filter_map(|id| self.variants.get(id))
            .filter(|v| v.active)
            .filter(|v| v.use_count >= self.config.min_uses_for_selection)
            .max_by(|a, b| {
                a.effectiveness()
                    .partial_cmp(&b.effectiveness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| self.select_random_from_group(group))
    }

    /// Select random active variant
    pub fn select_random(&self) -> Option<&PromptVariant> {
        let active: Vec<_> = self.variants.values().filter(|v| v.active).collect();
        if active.is_empty() {
            return None;
        }
        let idx = (rand_float() * active.len() as f64) as usize;
        active.get(idx.min(active.len() - 1)).copied()
    }

    /// Select random from group
    pub fn select_random_from_group(&self, group: &str) -> Option<&PromptVariant> {
        let variant_ids = self.groups.get(group)?;
        let active: Vec<_> = variant_ids
            .iter()
            .filter_map(|id| self.variants.get(id))
            .filter(|v| v.active)
            .collect();

        if active.is_empty() {
            return None;
        }
        let idx = (rand_float() * active.len() as f64) as usize;
        active.get(idx.min(active.len() - 1)).copied()
    }

    /// Record feedback for a variant
    pub fn record_feedback(&mut self, variant_id: &str, feedback: Feedback) {
        if let Some(variant) = self.variants.get_mut(variant_id) {
            variant.use_count += 1;

            if feedback.success {
                variant.success_count += 1;
            }

            if let Some(quality) = feedback.quality_score {
                variant.quality_sum += quality;
            }

            if let Some(time_ms) = feedback.response_time_ms {
                // Running average
                variant.avg_response_time_ms =
                    (variant.avg_response_time_ms * (variant.use_count - 1) + time_ms)
                        / variant.use_count;
            }

            if let Some(tokens) = feedback.token_count {
                variant.avg_tokens = (variant.avg_tokens * (variant.use_count as usize - 1)
                    + tokens)
                    / variant.use_count as usize;
            }

            // Record in history
            self.feedback_history.push(FeedbackEntry {
                variant_id: variant_id.to_string(),
                feedback: feedback.clone(),
                timestamp: Instant::now(),
            });

            // Auto-deactivate poor performers
            if variant.use_count >= self.config.min_uses_for_selection
                && variant.success_rate() < self.config.min_success_rate
            {
                variant.active = false;
            }
        }
    }

    /// Deactivate a variant
    pub fn deactivate(&mut self, variant_id: &str) {
        if let Some(variant) = self.variants.get_mut(variant_id) {
            variant.active = false;
        }
    }

    /// Reactivate a variant
    pub fn reactivate(&mut self, variant_id: &str) {
        if let Some(variant) = self.variants.get_mut(variant_id) {
            variant.active = true;
        }
    }

    /// Get optimization statistics
    pub fn stats(&self) -> OptimizationStats {
        let variants: Vec<_> = self.variants.values().collect();

        let total_uses: u64 = variants.iter().map(|v| v.use_count).sum();
        let total_successes: u64 = variants.iter().map(|v| v.success_count).sum();

        let best = variants
            .iter()
            .filter(|v| v.active && v.use_count >= self.config.min_uses_for_selection)
            .max_by(|a, b| {
                a.effectiveness()
                    .partial_cmp(&b.effectiveness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        OptimizationStats {
            total_variants: variants.len(),
            active_variants: variants.iter().filter(|v| v.active).count(),
            total_uses,
            total_successes,
            overall_success_rate: if total_uses > 0 {
                total_successes as f64 / total_uses as f64
            } else {
                0.0
            },
            best_variant_id: best.map(|v| v.id.clone()),
            best_effectiveness: best.map(|v| v.effectiveness()).unwrap_or(0.0),
        }
    }

    /// Get variant comparison report
    pub fn comparison_report(&self) -> Vec<VariantReport> {
        self.variants
            .values()
            .map(|v| VariantReport {
                id: v.id.clone(),
                name: v.name.clone(),
                use_count: v.use_count,
                success_rate: v.success_rate(),
                avg_quality: v.avg_quality(),
                effectiveness: v.effectiveness(),
                active: v.active,
            })
            .collect()
    }

    /// Get all feedback history entries
    pub fn feedback_history(&self) -> &[FeedbackEntry] {
        &self.feedback_history
    }

    /// Get the total number of feedback entries recorded
    pub fn feedback_count(&self) -> usize {
        self.feedback_history.len()
    }

    /// Get feedback entries for a specific variant
    pub fn feedback_for_variant(&self, variant_id: &str) -> Vec<&FeedbackEntry> {
        self.feedback_history
            .iter()
            .filter(|e| e.variant_id == variant_id)
            .collect()
    }

    /// Clear feedback history (keep stats)
    pub fn clear_history(&mut self) {
        self.feedback_history.clear();
    }

    /// Reset all variants (clear stats)
    pub fn reset_stats(&mut self) {
        for variant in self.variants.values_mut() {
            variant.use_count = 0;
            variant.success_count = 0;
            variant.quality_sum = 0.0;
            variant.avg_response_time_ms = 0;
            variant.avg_tokens = 0;
            variant.active = true;
        }
        self.feedback_history.clear();
    }
}

impl Default for PromptOptimizer {
    fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }
}

/// Feedback for a prompt variant
#[derive(Debug, Clone)]
pub struct Feedback {
    /// Was the response successful/helpful
    pub success: bool,
    /// Quality score (0-1)
    pub quality_score: Option<f64>,
    /// Response time in ms
    pub response_time_ms: Option<u64>,
    /// Token count
    pub token_count: Option<usize>,
    /// User rating (1-5)
    pub user_rating: Option<u8>,
    /// Additional notes
    pub notes: Option<String>,
}

impl Feedback {
    /// Create positive feedback
    pub fn positive() -> Self {
        Self {
            success: true,
            quality_score: None,
            response_time_ms: None,
            token_count: None,
            user_rating: None,
            notes: None,
        }
    }

    /// Create negative feedback
    pub fn negative() -> Self {
        Self {
            success: false,
            quality_score: None,
            response_time_ms: None,
            token_count: None,
            user_rating: None,
            notes: None,
        }
    }

    /// Set quality score
    pub fn with_quality(mut self, score: f64) -> Self {
        self.quality_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set response time
    pub fn with_time(mut self, ms: u64) -> Self {
        self.response_time_ms = Some(ms);
        self
    }

    /// Set token count
    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.token_count = Some(tokens);
        self
    }

    /// Set user rating
    pub fn with_rating(mut self, rating: u8) -> Self {
        self.user_rating = Some(rating.clamp(1, 5));
        self.success = rating >= 3;
        self.quality_score = Some(rating as f64 / 5.0);
        self
    }
}

/// Feedback history entry
#[derive(Debug, Clone)]
pub struct FeedbackEntry {
    /// The variant that received this feedback
    pub variant_id: String,
    /// The feedback itself
    pub feedback: Feedback,
    /// When the feedback was recorded
    pub timestamp: Instant,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total variants
    pub total_variants: usize,
    /// Active variants
    pub active_variants: usize,
    /// Total uses across all variants
    pub total_uses: u64,
    /// Total successes
    pub total_successes: u64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Best performing variant ID
    pub best_variant_id: Option<String>,
    /// Best effectiveness score
    pub best_effectiveness: f64,
}

/// Report for a single variant
#[derive(Debug, Clone)]
pub struct VariantReport {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Number of uses
    pub use_count: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub avg_quality: f64,
    /// Effectiveness score
    pub effectiveness: f64,
    /// Is active
    pub active: bool,
}

/// Simple random float (0-1) without external deps
fn rand_float() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (seed as f64 / u32::MAX as f64).fract()
}

/// Prompt shortener for token efficiency
pub struct PromptShortener {
    /// Words to remove
    filler_words: Vec<&'static str>,
    /// Replacements
    replacements: Vec<(&'static str, &'static str)>,
}

impl PromptShortener {
    /// Create a new shortener
    pub fn new() -> Self {
        Self {
            filler_words: vec![
                "please",
                "kindly",
                "just",
                "simply",
                "basically",
                "actually",
                "really",
                "very",
                "quite",
                "rather",
            ],
            replacements: vec![
                ("in order to", "to"),
                ("due to the fact that", "because"),
                ("at this point in time", "now"),
                ("in the event that", "if"),
                ("with regard to", "about"),
                ("for the purpose of", "to"),
                ("in spite of the fact that", "although"),
                ("a large number of", "many"),
                ("a small number of", "few"),
            ],
        }
    }

    /// Shorten a prompt
    pub fn shorten(&self, prompt: &str) -> String {
        let mut result = prompt.to_string();

        // Apply replacements
        for (from, to) in &self.replacements {
            result = result.replace(from, to);
        }

        // Remove filler words (simple approach)
        for word in &self.filler_words {
            result = result.replace(&format!(" {} ", word), " ");
            result = result.replace(&format!(" {},", word), ",");
        }

        // Clean up extra spaces
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }

        result.trim().to_string()
    }

    /// Estimate token savings
    pub fn estimate_savings(&self, prompt: &str) -> usize {
        let original_tokens = prompt.len() / 4;
        let shortened = self.shorten(prompt);
        let new_tokens = shortened.len() / 4;
        original_tokens.saturating_sub(new_tokens)
    }
}

impl Default for PromptShortener {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_creation() {
        let variant = PromptVariant::new("test", "Hello {query}");
        assert_eq!(variant.name, "test");
        assert_eq!(variant.apply_query("world"), "Hello world");
    }

    #[test]
    fn test_optimizer_add_select() {
        let mut optimizer = PromptOptimizer::new(OptimizerConfig {
            min_uses_for_selection: 0, // Allow immediate selection
            exploration_rate: 0.0,     // No exploration
            ..Default::default()
        });

        let id1 = optimizer.add_variant("v1", "Template 1: {query}");
        let id2 = optimizer.add_variant("v2", "Template 2: {query}");

        // Record feedback to make v1 better
        for _ in 0..5 {
            optimizer.record_feedback(&id1, Feedback::positive().with_quality(0.9));
        }
        for _ in 0..5 {
            optimizer.record_feedback(&id2, Feedback::negative().with_quality(0.3));
        }

        let best = optimizer.select_best("test").unwrap();
        assert_eq!(best.id, id1);
    }

    #[test]
    fn test_feedback_recording() {
        let mut optimizer = PromptOptimizer::default();
        let id = optimizer.add_variant("test", "Test");

        optimizer.record_feedback(&id, Feedback::positive().with_quality(0.8));
        optimizer.record_feedback(&id, Feedback::positive().with_quality(0.9));

        let variant = optimizer.get_variant(&id).unwrap();
        assert_eq!(variant.use_count, 2);
        assert_eq!(variant.success_count, 2);
        assert!((variant.avg_quality() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_groups() {
        let mut optimizer = PromptOptimizer::default();

        optimizer.add_to_group("qa", "brief", "Brief: {query}");
        optimizer.add_to_group("qa", "detailed", "Detailed: {query}");
        optimizer.add_to_group("code", "python", "Python: {query}");

        let qa_best = optimizer.select_best_from_group("qa", "test");
        assert!(qa_best.is_some());
    }

    #[test]
    fn test_shortener() {
        let shortener = PromptShortener::new();

        let prompt = "Please kindly provide a response in order to help me";
        let shortened = shortener.shorten(prompt);

        assert!(shortened.len() < prompt.len());
        assert!(!shortened.contains("kindly"));
        assert!(!shortened.contains("in order to"));
    }

    #[test]
    fn test_stats() {
        let mut optimizer = PromptOptimizer::default();
        let id = optimizer.add_variant("test", "Test");

        optimizer.record_feedback(&id, Feedback::positive());
        optimizer.record_feedback(&id, Feedback::negative());

        let stats = optimizer.stats();
        assert_eq!(stats.total_variants, 1);
        assert_eq!(stats.total_uses, 2);
        assert_eq!(stats.total_successes, 1);
        assert!((stats.overall_success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_feedback_history_access() {
        let mut optimizer = PromptOptimizer::default();
        let id1 = optimizer.add_variant("v1", "Template 1");
        let id2 = optimizer.add_variant("v2", "Template 2");

        optimizer.record_feedback(&id1, Feedback::positive().with_quality(0.9));
        optimizer.record_feedback(&id1, Feedback::negative().with_quality(0.3));
        optimizer.record_feedback(&id2, Feedback::positive().with_quality(0.8));

        // Total feedback count
        assert_eq!(optimizer.feedback_count(), 3);

        // Feedback history slice
        let history = optimizer.feedback_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].variant_id, id1);
        assert!(history[0].feedback.success);

        // Filter by variant
        let v1_feedback = optimizer.feedback_for_variant(&id1);
        assert_eq!(v1_feedback.len(), 2);
        let v2_feedback = optimizer.feedback_for_variant(&id2);
        assert_eq!(v2_feedback.len(), 1);

        // Clear history
        optimizer.clear_history();
        assert_eq!(optimizer.feedback_count(), 0);
        assert!(optimizer.feedback_history().is_empty());
    }

    #[test]
    fn test_auto_deactivation_of_poor_performers() {
        let mut optimizer = PromptOptimizer::new(OptimizerConfig {
            min_uses_for_selection: 5,
            min_success_rate: 0.3,
            exploration_rate: 0.0,
            ..Default::default()
        });

        let id = optimizer.add_variant("poor", "Bad template: {query}");

        // Variant starts active
        assert!(optimizer.get_variant(&id).unwrap().active);

        // Record 5 negative feedbacks (success_rate = 0.0, below 0.3 threshold)
        for _ in 0..5 {
            optimizer.record_feedback(&id, Feedback::negative().with_quality(0.1));
        }

        // After min_uses_for_selection uses with low success rate, should be deactivated
        let variant = optimizer.get_variant(&id).unwrap();
        assert!(!variant.active);
        assert_eq!(variant.use_count, 5);
        assert_eq!(variant.success_count, 0);
        assert!((variant.success_rate() - 0.0).abs() < 0.001);

        // Active variants should now be empty
        assert!(optimizer.active_variants().is_empty());
    }

    #[test]
    fn test_variant_apply_with_variables_and_effectiveness() {
        let variant = PromptVariant::new("multi-var", "Hello {name}, your question about {topic} is: {query}");

        // apply with HashMap of multiple variables
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("topic".to_string(), "Rust".to_string());
        vars.insert("query".to_string(), "How do lifetimes work?".to_string());

        let result = variant.apply(&vars);
        assert_eq!(result, "Hello Alice, your question about Rust is: How do lifetimes work?");

        // estimate_tokens: rough estimate at len/4
        let tokens = variant.estimate_tokens("test");
        let filled = variant.apply_query("test");
        assert_eq!(tokens, filled.len() / 4);

        // effectiveness for unused variant: 0.5 success + 0.5 quality + exploration bonus
        // success_rate=0.5 (neutral), avg_quality=0.5 (neutral), exploration_bonus=0.1
        let eff = variant.effectiveness();
        assert!((eff - 0.6).abs() < 0.001);

        // comparison_report generation
        let mut optimizer = PromptOptimizer::default();
        let id = optimizer.add_variant("rpt", "Report {query}");
        optimizer.record_feedback(&id, Feedback::positive().with_quality(0.9));

        let report = optimizer.comparison_report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].name, "rpt");
        assert_eq!(report[0].use_count, 1);
        assert!(report[0].active);
        assert!(report[0].success_rate > 0.9);
    }

    #[test]
    fn test_deactivate_reactivate_and_reset_stats() {
        let mut optimizer = PromptOptimizer::new(OptimizerConfig {
            exploration_rate: 0.0,
            min_uses_for_selection: 0,
            ..Default::default()
        });

        let id1 = optimizer.add_variant("v1", "Template 1: {query}");
        let id2 = optimizer.add_variant("v2", "Template 2: {query}");

        // Record some usage
        for _ in 0..3 {
            optimizer.record_feedback(&id1, Feedback::positive().with_quality(0.9).with_time(100).with_tokens(50));
            optimizer.record_feedback(&id2, Feedback::positive().with_quality(0.5));
        }

        // Both active initially
        assert_eq!(optimizer.active_variants().len(), 2);

        // Deactivate v2
        optimizer.deactivate(&id2);
        assert!(!optimizer.get_variant(&id2).unwrap().active);
        assert_eq!(optimizer.active_variants().len(), 1);

        // Stats should reflect deactivation
        let stats = optimizer.stats();
        assert_eq!(stats.active_variants, 1);
        assert_eq!(stats.total_uses, 6);

        // Reactivate v2
        optimizer.reactivate(&id2);
        assert!(optimizer.get_variant(&id2).unwrap().active);
        assert_eq!(optimizer.active_variants().len(), 2);

        // Verify response time and token tracking on v1
        let v1 = optimizer.get_variant(&id1).unwrap();
        assert_eq!(v1.avg_response_time_ms, 100);
        assert_eq!(v1.avg_tokens, 50);

        // Reset stats
        optimizer.reset_stats();

        let v1_after = optimizer.get_variant(&id1).unwrap();
        assert_eq!(v1_after.use_count, 0);
        assert_eq!(v1_after.success_count, 0);
        assert!((v1_after.quality_sum - 0.0).abs() < 0.001);
        assert_eq!(v1_after.avg_response_time_ms, 0);
        assert_eq!(v1_after.avg_tokens, 0);
        assert!(v1_after.active); // reset_stats reactivates all variants
        assert_eq!(optimizer.feedback_count(), 0);

        // Also verify Feedback::with_rating builder
        let fb = Feedback::positive().with_rating(4);
        assert!(fb.success); // rating >= 3 means success
        assert_eq!(fb.user_rating, Some(4));
        assert!((fb.quality_score.unwrap() - 0.8).abs() < 0.001); // 4/5 = 0.8

        let fb_low = Feedback::positive().with_rating(2);
        assert!(!fb_low.success); // rating < 3 means not success
        assert_eq!(fb_low.user_rating, Some(2));
    }
}
