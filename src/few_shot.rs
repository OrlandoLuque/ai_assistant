//! Few-shot learning manager
//!
//! This module provides tools for managing few-shot examples in prompts,
//! enabling better model performance through example-based learning.
//!
//! # Features
//!
//! - **Example management**: Store and retrieve examples
//! - **Dynamic selection**: Select relevant examples based on query
//! - **Category organization**: Organize examples by task type
//! - **Quality scoring**: Track example effectiveness
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::few_shot::{FewShotManager, Example, ExampleCategory};
//!
//! let mut manager = FewShotManager::new();
//!
//! // Add examples
//! manager.add_example(Example::new(
//!     "What is the capital of France?",
//!     "The capital of France is Paris.",
//!     ExampleCategory::FactualQA,
//! ));
//!
//! // Get examples for a new query
//! let examples = manager.select_examples("What is the capital of Germany?", 3);
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Example category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExampleCategory {
    /// Factual question answering
    FactualQA,
    /// Creative writing
    Creative,
    /// Code generation
    Coding,
    /// Translation
    Translation,
    /// Summarization
    Summarization,
    /// Classification
    Classification,
    /// Extraction
    Extraction,
    /// Conversation
    Conversation,
    /// Math/reasoning
    Math,
    /// Custom category
    Custom,
}

impl ExampleCategory {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::FactualQA => "Factual Q&A",
            Self::Creative => "Creative Writing",
            Self::Coding => "Code Generation",
            Self::Translation => "Translation",
            Self::Summarization => "Summarization",
            Self::Classification => "Classification",
            Self::Extraction => "Extraction",
            Self::Conversation => "Conversation",
            Self::Math => "Math/Reasoning",
            Self::Custom => "Custom",
        }
    }
}

impl Default for ExampleCategory {
    fn default() -> Self {
        Self::Custom
    }
}

/// A few-shot example
#[derive(Debug, Clone)]
pub struct Example {
    /// Unique ID
    pub id: String,
    /// Input/question
    pub input: String,
    /// Expected output/answer
    pub output: String,
    /// Category
    pub category: ExampleCategory,
    /// Quality score (0-1)
    pub quality_score: f64,
    /// Times used
    pub use_count: u64,
    /// Success count (when feedback is provided)
    pub success_count: u64,
    /// Tags for filtering
    pub tags: Vec<String>,
    /// Created timestamp
    pub created_at: Instant,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Example {
    /// Create a new example
    pub fn new(
        input: impl Into<String>,
        output: impl Into<String>,
        category: ExampleCategory,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            input: input.into(),
            output: output.into(),
            category,
            quality_score: 1.0,
            use_count: 0,
            success_count: 0,
            tags: Vec::new(),
            created_at: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(|t| t.into()));
        self
    }

    /// Set quality score
    pub fn with_quality(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.use_count == 0 {
            1.0
        } else {
            self.success_count as f64 / self.use_count as f64
        }
    }

    /// Calculate effective score (quality * success rate)
    pub fn effective_score(&self) -> f64 {
        self.quality_score * self.success_rate()
    }

    /// Format as prompt text
    pub fn format(&self, input_prefix: &str, output_prefix: &str) -> String {
        format!(
            "{}{}\n{}{}",
            input_prefix, self.input, output_prefix, self.output
        )
    }

    /// Format with default prefixes
    pub fn format_default(&self) -> String {
        self.format("User: ", "Assistant: ")
    }

    /// Get token estimate (rough)
    pub fn estimated_tokens(&self) -> usize {
        // Rough estimate: ~4 chars per token
        (self.input.len() + self.output.len()) / 4
    }
}

/// Configuration for few-shot selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Maximum examples to select
    pub max_examples: usize,
    /// Minimum quality score threshold
    pub min_quality: f64,
    /// Weight for semantic similarity
    pub similarity_weight: f64,
    /// Weight for quality score
    pub quality_weight: f64,
    /// Weight for success rate
    pub success_weight: f64,
    /// Prefer diverse examples
    pub prefer_diversity: bool,
    /// Maximum total tokens for examples
    pub max_tokens: Option<usize>,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            max_examples: 5,
            min_quality: 0.5,
            similarity_weight: 0.5,
            quality_weight: 0.3,
            success_weight: 0.2,
            prefer_diversity: true,
            max_tokens: Some(2000),
        }
    }
}

/// Few-shot manager for example storage and selection
pub struct FewShotManager {
    examples: Vec<Example>,
    by_category: HashMap<ExampleCategory, Vec<usize>>,
    by_tag: HashMap<String, Vec<usize>>,
    config: SelectionConfig,
}

impl FewShotManager {
    /// Create a new manager
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            by_category: HashMap::new(),
            by_tag: HashMap::new(),
            config: SelectionConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: SelectionConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Add an example
    pub fn add_example(&mut self, example: Example) -> String {
        let id = example.id.clone();
        let idx = self.examples.len();
        let category = example.category;
        let tags = example.tags.clone();

        self.examples.push(example);

        // Index by category
        self.by_category.entry(category).or_default().push(idx);

        // Index by tags
        for tag in tags {
            self.by_tag.entry(tag).or_default().push(idx);
        }

        id
    }

    /// Remove an example
    pub fn remove_example(&mut self, id: &str) -> Option<Example> {
        let idx = self.examples.iter().position(|e| e.id == id)?;
        let example = self.examples.remove(idx);

        // Rebuild indices
        self.rebuild_indices();

        Some(example)
    }

    /// Get example by ID
    pub fn get_example(&self, id: &str) -> Option<&Example> {
        self.examples.iter().find(|e| e.id == id)
    }

    /// Get mutable example by ID
    pub fn get_example_mut(&mut self, id: &str) -> Option<&mut Example> {
        self.examples.iter_mut().find(|e| e.id == id)
    }

    /// Get examples by category
    pub fn get_by_category(&self, category: ExampleCategory) -> Vec<&Example> {
        self.by_category
            .get(&category)
            .map(|indices| indices.iter().map(|&i| &self.examples[i]).collect())
            .unwrap_or_default()
    }

    /// Get examples by tag
    pub fn get_by_tag(&self, tag: &str) -> Vec<&Example> {
        self.by_tag
            .get(tag)
            .map(|indices| indices.iter().map(|&i| &self.examples[i]).collect())
            .unwrap_or_default()
    }

    /// Select examples for a query
    pub fn select_examples(&self, query: &str, max: usize) -> Vec<&Example> {
        self.select_with_config(query, max, None)
    }

    /// Select examples with category filter
    pub fn select_for_category(
        &self,
        query: &str,
        category: ExampleCategory,
        max: usize,
    ) -> Vec<&Example> {
        self.select_with_config(query, max, Some(category))
    }

    /// Select examples with full config control
    pub fn select_with_config(
        &self,
        query: &str,
        max: usize,
        category: Option<ExampleCategory>,
    ) -> Vec<&Example> {
        // Filter candidates
        let candidates: Vec<_> = self
            .examples
            .iter()
            .filter(|e| e.quality_score >= self.config.min_quality)
            .filter(|e| category.map(|c| e.category == c).unwrap_or(true))
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        // Score each candidate
        let mut scored: Vec<_> = candidates
            .iter()
            .map(|e| {
                let similarity = self.compute_similarity(query, &e.input);
                let score = similarity * self.config.similarity_weight
                    + e.quality_score * self.config.quality_weight
                    + e.success_rate() * self.config.success_weight;
                (*e, score)
            })
            .collect();

        // Sort by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top examples, respecting token limit
        let mut selected = Vec::new();
        let mut total_tokens = 0;
        let max_tokens = self.config.max_tokens.unwrap_or(usize::MAX);

        for (example, _) in scored {
            if selected.len() >= max.min(self.config.max_examples) {
                break;
            }

            let tokens = example.estimated_tokens();
            if total_tokens + tokens > max_tokens {
                continue;
            }

            // Diversity check
            if self.config.prefer_diversity && !selected.is_empty() {
                let too_similar = selected
                    .iter()
                    .any(|e: &&Example| self.compute_similarity(&e.input, &example.input) > 0.8);
                if too_similar {
                    continue;
                }
            }

            total_tokens += tokens;
            selected.push(example);
        }

        selected
    }

    /// Record example usage
    pub fn record_usage(&mut self, id: &str, success: bool) {
        if let Some(example) = self.get_example_mut(id) {
            example.use_count += 1;
            if success {
                example.success_count += 1;
            }
        }
    }

    /// Update quality score based on feedback
    pub fn update_quality(&mut self, id: &str, adjustment: f64) {
        if let Some(example) = self.get_example_mut(id) {
            example.quality_score = (example.quality_score + adjustment).clamp(0.0, 1.0);
        }
    }

    /// Format examples as prompt
    pub fn format_prompt(
        &self,
        examples: &[&Example],
        input_prefix: &str,
        output_prefix: &str,
    ) -> String {
        examples
            .iter()
            .map(|e| e.format(input_prefix, output_prefix))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Format with default prefixes
    pub fn format_prompt_default(&self, examples: &[&Example]) -> String {
        self.format_prompt(examples, "User: ", "Assistant: ")
    }

    /// Get all examples
    pub fn all_examples(&self) -> &[Example] {
        &self.examples
    }

    /// Get example count
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get statistics
    pub fn stats(&self) -> FewShotStats {
        let mut stats = FewShotStats::default();
        stats.total_examples = self.examples.len();

        for example in &self.examples {
            stats.total_uses += example.use_count;
            stats.total_successes += example.success_count;

            *stats.by_category.entry(example.category).or_insert(0) += 1;

            if example.quality_score >= 0.8 {
                stats.high_quality += 1;
            }
        }

        if stats.total_uses > 0 {
            stats.overall_success_rate = stats.total_successes as f64 / stats.total_uses as f64;
        }

        stats
    }

    /// Clear all examples
    pub fn clear(&mut self) {
        self.examples.clear();
        self.by_category.clear();
        self.by_tag.clear();
    }

    fn rebuild_indices(&mut self) {
        self.by_category.clear();
        self.by_tag.clear();

        for (idx, example) in self.examples.iter().enumerate() {
            self.by_category
                .entry(example.category)
                .or_default()
                .push(idx);
            for tag in &example.tags {
                self.by_tag.entry(tag.clone()).or_default().push(idx);
            }
        }
    }

    fn compute_similarity(&self, a: &str, b: &str) -> f64 {
        // Simple word overlap similarity
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let a_words: std::collections::HashSet<_> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<_> = b_lower.split_whitespace().collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        intersection as f64 / union as f64
    }
}

impl Default for FewShotManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Few-shot statistics
#[derive(Debug, Clone, Default)]
pub struct FewShotStats {
    /// Total examples
    pub total_examples: usize,
    /// High quality examples (score >= 0.8)
    pub high_quality: usize,
    /// Total times examples were used
    pub total_uses: u64,
    /// Total successful uses
    pub total_successes: u64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Examples by category
    pub by_category: HashMap<ExampleCategory, usize>,
}

/// Pre-built example sets for common tasks
pub struct ExampleSets;

impl ExampleSets {
    /// Get coding examples
    pub fn coding() -> Vec<Example> {
        vec![
            Example::new(
                "Write a function to reverse a string in Python",
                "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
                ExampleCategory::Coding,
            ),
            Example::new(
                "Write a function to check if a number is prime",
                "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
                ExampleCategory::Coding,
            ),
        ]
    }

    /// Get Q&A examples
    pub fn qa() -> Vec<Example> {
        vec![
            Example::new(
                "What is the capital of Japan?",
                "The capital of Japan is Tokyo.",
                ExampleCategory::FactualQA,
            ),
            Example::new(
                "When was the Eiffel Tower built?",
                "The Eiffel Tower was built between 1887 and 1889, and was completed on March 31, 1889.",
                ExampleCategory::FactualQA,
            ),
        ]
    }

    /// Get translation examples
    pub fn translation() -> Vec<Example> {
        vec![
            Example::new(
                "Translate to Spanish: Hello, how are you?",
                "Hola, ¿cómo estás?",
                ExampleCategory::Translation,
            ),
            Example::new(
                "Translate to French: The weather is nice today.",
                "Le temps est beau aujourd'hui.",
                ExampleCategory::Translation,
            ),
        ]
    }

    /// Get summarization examples
    pub fn summarization() -> Vec<Example> {
        vec![
            Example::new(
                "Summarize: The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing fonts and keyboards.",
                "A pangram sentence using all alphabet letters, commonly used for testing.",
                ExampleCategory::Summarization,
            ),
        ]
    }
}

/// Builder for creating example sets
pub struct ExampleBuilder {
    examples: Vec<Example>,
}

impl ExampleBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Add an example
    pub fn add(mut self, input: &str, output: &str, category: ExampleCategory) -> Self {
        self.examples.push(Example::new(input, output, category));
        self
    }

    /// Add with tags
    pub fn add_with_tags(
        mut self,
        input: &str,
        output: &str,
        category: ExampleCategory,
        tags: &[&str],
    ) -> Self {
        let example =
            Example::new(input, output, category).with_tags(tags.iter().map(|s| s.to_string()));
        self.examples.push(example);
        self
    }

    /// Build the examples
    pub fn build(self) -> Vec<Example> {
        self.examples
    }

    /// Build and add to manager
    pub fn build_into(self, manager: &mut FewShotManager) {
        for example in self.examples {
            manager.add_example(example);
        }
    }
}

impl Default for ExampleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_creation() {
        let example = Example::new("input", "output", ExampleCategory::FactualQA)
            .with_tag("test")
            .with_quality(0.9);

        assert_eq!(example.input, "input");
        assert_eq!(example.output, "output");
        assert_eq!(example.quality_score, 0.9);
        assert!(example.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_manager_add_select() {
        let mut manager = FewShotManager::new();

        manager.add_example(Example::new("What is 2+2?", "4", ExampleCategory::Math));
        manager.add_example(Example::new("What is 3+3?", "6", ExampleCategory::Math));

        let selected = manager.select_examples("What is 5+5?", 2);
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_category_filter() {
        let mut manager = FewShotManager::new();

        manager.add_example(Example::new("math", "math", ExampleCategory::Math));
        manager.add_example(Example::new("code", "code", ExampleCategory::Coding));

        let math = manager.get_by_category(ExampleCategory::Math);
        assert_eq!(math.len(), 1);
        assert_eq!(math[0].input, "math");
    }

    #[test]
    fn test_usage_tracking() {
        let mut manager = FewShotManager::new();
        let id = manager.add_example(Example::new("test", "test", ExampleCategory::Custom));

        manager.record_usage(&id, true);
        manager.record_usage(&id, true);
        manager.record_usage(&id, false);

        let example = manager.get_example(&id).unwrap();
        assert_eq!(example.use_count, 3);
        assert_eq!(example.success_count, 2);
        assert!((example.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_format_prompt() {
        let mut manager = FewShotManager::new();
        manager.add_example(Example::new("Q1", "A1", ExampleCategory::FactualQA));
        manager.add_example(Example::new("Q2", "A2", ExampleCategory::FactualQA));

        let examples = manager.select_examples("test", 2);
        let prompt = manager.format_prompt_default(&examples);

        assert!(prompt.contains("User: Q"));
        assert!(prompt.contains("Assistant: A"));
    }

    #[test]
    fn test_example_sets() {
        let coding = ExampleSets::coding();
        assert!(!coding.is_empty());
        assert!(coding.iter().all(|e| e.category == ExampleCategory::Coding));
    }
}
