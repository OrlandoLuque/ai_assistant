//! Smart suggestions
//!
//! Generate follow-up question suggestions based on context.

use std::collections::HashSet;

/// Suggestion type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionType {
    FollowUp,
    Clarification,
    Related,
    DeepDive,
    Alternative,
}

/// A suggestion
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub text: String,
    pub suggestion_type: SuggestionType,
    pub relevance: f64,
}

impl Suggestion {
    pub fn new(text: &str, suggestion_type: SuggestionType) -> Self {
        Self {
            text: text.to_string(),
            suggestion_type,
            relevance: 1.0,
        }
    }

    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.relevance = relevance;
        self
    }
}

/// Compute the Jaccard similarity between the words in `suggestion_text` and the
/// words in the concatenated `context` strings.
///
/// Jaccard = |A intersection B| / |A union B|.
/// Returns 0.0 if both sets are empty.
pub fn compute_relevance(suggestion_text: &str, context: &[String]) -> f64 {
    let to_word_set = |s: &str| -> HashSet<String> {
        s.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    };

    let set_a = to_word_set(suggestion_text);
    let context_combined: String = context.join(" ");
    let set_b = to_word_set(&context_combined);

    let union_size = set_a.union(&set_b).count();
    if union_size == 0 {
        return 0.0;
    }

    let intersection_size = set_a.intersection(&set_b).count();
    intersection_size as f64 / union_size as f64
}

/// Suggestion generator
pub struct SuggestionGenerator {
    templates: Vec<SuggestionTemplate>,
}

impl SuggestionGenerator {
    pub fn new() -> Self {
        Self {
            templates: Self::default_templates(),
        }
    }

    fn default_templates() -> Vec<SuggestionTemplate> {
        vec![
            // Follow-up templates
            SuggestionTemplate {
                pattern: "explain",
                suggestions: vec![
                    "Can you give me an example?",
                    "How does this work in practice?",
                    "What are the common use cases?",
                ],
                suggestion_type: SuggestionType::FollowUp,
            },
            SuggestionTemplate {
                pattern: "code",
                suggestions: vec![
                    "Can you add error handling?",
                    "How would I test this?",
                    "Can you optimize this code?",
                ],
                suggestion_type: SuggestionType::FollowUp,
            },
            // Clarification templates
            SuggestionTemplate {
                pattern: "",
                suggestions: vec![
                    "Can you explain that in simpler terms?",
                    "What do you mean by that?",
                    "Can you be more specific?",
                ],
                suggestion_type: SuggestionType::Clarification,
            },
            // Deep dive templates
            SuggestionTemplate {
                pattern: "concept",
                suggestions: vec![
                    "Tell me more about the underlying theory",
                    "What are the advanced aspects?",
                    "How does this compare to alternatives?",
                ],
                suggestion_type: SuggestionType::DeepDive,
            },
        ]
    }

    /// Generate suggestions based on conversation context
    pub fn generate(&self, query: &str, response: &str, max: usize) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        let lower_query = query.to_lowercase();
        let lower_response = response.to_lowercase();

        // Build context for relevance computation
        let context: Vec<String> = vec![query.to_string(), response.to_string()];
        let has_context = !query.is_empty() || !response.is_empty();

        // Pattern-based suggestions
        for template in &self.templates {
            if template.pattern.is_empty()
                || lower_query.contains(template.pattern)
                || lower_response.contains(template.pattern)
            {
                for text in &template.suggestions {
                    let mut suggestion = Suggestion::new(text, template.suggestion_type);
                    if has_context {
                        suggestion.relevance = compute_relevance(text, &context);
                    }
                    suggestions.push(suggestion);
                }
            }
        }

        // Topic-based suggestions
        let topics = self.extract_topics(response);
        for topic in topics.iter().take(3) {
            let text = format!("Tell me more about {}", topic);
            let relevance = if has_context {
                compute_relevance(&text, &context)
            } else {
                1.0
            };
            suggestions
                .push(Suggestion::new(&text, SuggestionType::Related).with_relevance(relevance));
        }

        // Question suggestions based on response content
        if lower_response.contains("however") || lower_response.contains("although") {
            let text = "What are the alternatives?";
            let relevance = if has_context {
                compute_relevance(text, &context)
            } else {
                1.0
            };
            suggestions
                .push(Suggestion::new(text, SuggestionType::Alternative).with_relevance(relevance));
        }

        if lower_response.contains("example") {
            let text = "Can you show another example?";
            let relevance = if has_context {
                compute_relevance(text, &context)
            } else {
                1.0
            };
            suggestions
                .push(Suggestion::new(text, SuggestionType::FollowUp).with_relevance(relevance));
        }

        // Remove duplicates and limit
        let mut seen = HashSet::new();
        suggestions.retain(|s| seen.insert(s.text.clone()));
        suggestions.truncate(max);

        suggestions
    }

    fn extract_topics(&self, text: &str) -> Vec<String> {
        let mut word_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for word in text.split_whitespace() {
            let word = word
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();

            if word.len() > 5 && !is_common_word(&word) {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        let mut topics: Vec<_> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();

        topics.sort_by(|a, b| b.1.cmp(&a.1));
        topics.into_iter().take(5).map(|(word, _)| word).collect()
    }
}

impl Default for SuggestionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

struct SuggestionTemplate {
    pattern: &'static str,
    suggestions: Vec<&'static str>,
    suggestion_type: SuggestionType,
}

fn is_common_word(word: &str) -> bool {
    const COMMON: &[&str] = &[
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
        "our", "out", "have", "been", "were", "being", "their", "there", "would", "could",
        "should", "about", "which", "when", "what", "this", "that", "with", "from", "your", "they",
        "will", "more", "some", "into", "just", "also", "than",
    ];
    COMMON.contains(&word)
}

/// Context-aware suggestion config
#[derive(Debug, Clone)]
pub struct SuggestionConfig {
    pub max_suggestions: usize,
    pub include_follow_up: bool,
    pub include_clarification: bool,
    pub include_related: bool,
    pub include_deep_dive: bool,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 5,
            include_follow_up: true,
            include_clarification: true,
            include_related: true,
            include_deep_dive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_suggestions() {
        let generator = SuggestionGenerator::new();

        let suggestions = generator.generate(
            "Explain how machine learning works",
            "Machine learning is a subset of AI that enables systems to learn from data.",
            5,
        );

        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_code_suggestions() {
        let generator = SuggestionGenerator::new();

        let suggestions = generator.generate(
            "Write code to sort an array",
            "Here's a function to sort: def sort(arr): return sorted(arr)",
            5,
        );

        assert!(suggestions
            .iter()
            .any(|s| s.text.contains("error handling") || s.text.contains("test")));
    }

    #[test]
    fn test_relevance_high_for_matching_context() {
        let context = vec![
            "How do I test error handling in Rust?".to_string(),
            "You can use Result and the ? operator for error handling.".to_string(),
        ];
        let relevance = compute_relevance("Can you add error handling?", &context);
        assert!(
            relevance > 0.0,
            "Relevance should be > 0.0 when words overlap, got {}",
            relevance
        );
    }

    #[test]
    fn test_relevance_low_for_unrelated() {
        let context = vec!["Tell me about quantum physics and particle accelerators.".to_string()];
        let relevance = compute_relevance("How do I bake chocolate cookies?", &context);
        assert!(
            relevance < 0.1,
            "Relevance should be near 0.0 for unrelated content, got {}",
            relevance
        );
    }

    #[test]
    fn test_relevance_jaccard_correctness() {
        // suggestion words: {"the", "cat", "sat"}
        // context words:    {"the", "cat", "ran"}
        // intersection: {"the", "cat"} => 2
        // union: {"the", "cat", "sat", "ran"} => 4
        // Jaccard = 2/4 = 0.5
        let context = vec!["the cat ran".to_string()];
        let relevance = compute_relevance("the cat sat", &context);
        assert!(
            (relevance - 0.5).abs() < f64::EPSILON,
            "Expected Jaccard 0.5, got {}",
            relevance
        );
    }

    #[test]
    fn test_default_generator() {
        let gen = SuggestionGenerator::default();
        let suggestions = gen.generate("hello", "world", 10);
        // Clarification template matches everything (empty pattern)
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_max_truncation() {
        let gen = SuggestionGenerator::new();
        let suggestions = gen.generate("Explain how computers work", "Computers use transistors", 2);
        assert!(suggestions.len() <= 2);
    }

    #[test]
    fn test_alternative_suggestion() {
        let gen = SuggestionGenerator::new();
        let suggestions = gen.generate("Tell me about X", "X is great, however Y is also good", 10);
        assert!(suggestions.iter().any(|s| s.suggestion_type == SuggestionType::Alternative));
    }

    #[test]
    fn test_example_suggestion() {
        let gen = SuggestionGenerator::new();
        let suggestions = gen.generate("Show me", "Here is an example of the concept", 10);
        assert!(suggestions.iter().any(|s| s.text.contains("another example")));
    }

    #[test]
    fn test_suggestion_config_defaults() {
        let config = SuggestionConfig::default();
        assert_eq!(config.max_suggestions, 5);
        assert!(config.include_follow_up);
        assert!(config.include_clarification);
        assert!(config.include_related);
        assert!(config.include_deep_dive);
    }
}
