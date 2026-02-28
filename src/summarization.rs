//! Conversation summarization
//!
//! Automatic summarization of long conversations.

use std::collections::HashMap;

/// Summary configuration
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    pub max_length: usize,
    pub include_key_points: bool,
    pub include_topics: bool,
    pub include_action_items: bool,
    pub style: SummaryStyle,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            max_length: 500,
            include_key_points: true,
            include_topics: true,
            include_action_items: true,
            style: SummaryStyle::Concise,
        }
    }
}

/// Summary style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryStyle {
    Concise,
    Detailed,
    Bullet,
    Narrative,
}

/// Conversation summary
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub summary: String,
    pub key_points: Vec<String>,
    pub topics: Vec<String>,
    pub action_items: Vec<String>,
    pub message_count: usize,
    pub token_count: usize,
}

/// Summarizer for conversations
pub struct ConversationSummarizer {
    config: SummaryConfig,
}

impl ConversationSummarizer {
    pub fn new(config: SummaryConfig) -> Self {
        Self { config }
    }

    /// Generate a prompt for summarizing a conversation
    pub fn build_summary_prompt(&self, messages: &[(String, String)]) -> String {
        let mut prompt = String::new();

        prompt.push_str("Summarize the following conversation:\n\n");

        for (role, content) in messages {
            prompt.push_str(&format!("{}: {}\n", role, content));
        }

        prompt.push_str("\n---\n");

        match self.config.style {
            SummaryStyle::Concise => {
                prompt.push_str("Provide a concise summary in 2-3 sentences.\n")
            }
            SummaryStyle::Detailed => prompt.push_str("Provide a detailed summary.\n"),
            SummaryStyle::Bullet => prompt.push_str("Provide a bullet-point summary.\n"),
            SummaryStyle::Narrative => prompt.push_str("Provide a narrative summary.\n"),
        }

        if self.config.include_key_points {
            prompt.push_str("Include key points discussed.\n");
        }
        if self.config.include_topics {
            prompt.push_str("List the main topics.\n");
        }
        if self.config.include_action_items {
            prompt.push_str("Note any action items or next steps.\n");
        }

        prompt
    }

    /// Extract key points from text (simple heuristic)
    pub fn extract_key_points(&self, text: &str) -> Vec<String> {
        let mut points = Vec::new();

        for sentence in text.split('.') {
            let trimmed = sentence.trim();
            if trimmed.len() > 20
                && (trimmed.contains("important")
                    || trimmed.contains("key")
                    || trimmed.contains("must")
                    || trimmed.contains("should")
                    || trimmed.contains("need"))
            {
                points.push(format!("{}.", trimmed));
            }
        }

        points.into_iter().take(5).collect()
    }

    /// Extract topics from conversation
    pub fn extract_topics(&self, messages: &[(String, String)]) -> Vec<String> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for (_, content) in messages {
            for word in content.split_whitespace() {
                let word = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();

                if word.len() > 4 && !is_stop_word(&word) {
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut topics: Vec<_> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();

        topics.sort_by(|a, b| b.1.cmp(&a.1));
        topics.into_iter().take(5).map(|(word, _)| word).collect()
    }

    /// Create rolling summary for long conversations
    pub fn rolling_summary(
        &self,
        existing_summary: &str,
        new_messages: &[(String, String)],
    ) -> String {
        let mut prompt = String::new();

        if !existing_summary.is_empty() {
            prompt.push_str(&format!("Previous summary:\n{}\n\n", existing_summary));
        }

        prompt.push_str("New messages:\n");
        for (role, content) in new_messages {
            prompt.push_str(&format!("{}: {}\n", role, content));
        }

        prompt.push_str(
            "\nUpdate the summary to include the new information while keeping it concise.",
        );

        prompt
    }
}

impl Default for ConversationSummarizer {
    fn default() -> Self {
        Self::new(SummaryConfig::default())
    }
}

fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above", "below", "between",
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "also", "now", "and", "but",
        "or", "if", "because", "until", "while", "this", "that", "these", "those", "what", "which",
        "who", "whom", "their", "them", "they", "your", "you", "our", "its", "his", "her",
    ];

    STOP_WORDS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_prompt() {
        let summarizer = ConversationSummarizer::default();
        let messages = vec![
            ("User".to_string(), "What is AI?".to_string()),
            (
                "Assistant".to_string(),
                "AI is artificial intelligence.".to_string(),
            ),
        ];

        let prompt = summarizer.build_summary_prompt(&messages);
        assert!(prompt.contains("What is AI?"));
        assert!(prompt.contains("Summarize"));
    }

    #[test]
    fn test_default_config() {
        let config = SummaryConfig::default();
        assert_eq!(config.max_length, 500);
        assert!(config.include_key_points);
        assert!(config.include_topics);
        assert!(config.include_action_items);
        assert_eq!(config.style, SummaryStyle::Concise);
    }

    #[test]
    fn test_summary_styles_in_prompt() {
        let messages = vec![("User".to_string(), "Hello".to_string())];

        for (style, keyword) in [
            (SummaryStyle::Detailed, "detailed"),
            (SummaryStyle::Bullet, "bullet"),
            (SummaryStyle::Narrative, "narrative"),
        ] {
            let s = ConversationSummarizer::new(SummaryConfig {
                style,
                ..Default::default()
            });
            let prompt = s.build_summary_prompt(&messages);
            assert!(
                prompt.to_lowercase().contains(keyword),
                "Style {:?} should produce '{}' in prompt",
                style,
                keyword
            );
        }
    }

    #[test]
    fn test_extract_key_points() {
        let summarizer = ConversationSummarizer::default();
        let text = "The weather is nice. It is important to stay hydrated during summer. You should drink water. The sky is blue. We need to plan ahead for the winter season.";
        let points = summarizer.extract_key_points(text);
        assert!(!points.is_empty());
        assert!(points.len() <= 5);
        for p in &points {
            assert!(p.ends_with('.'));
        }
    }

    #[test]
    fn test_extract_key_points_empty() {
        let summarizer = ConversationSummarizer::default();
        let points = summarizer.extract_key_points("Short text here.");
        assert!(points.is_empty());
    }

    #[test]
    fn test_rolling_summary_with_existing() {
        let summarizer = ConversationSummarizer::default();
        let result = summarizer.rolling_summary(
            "Previous discussion about AI.",
            &[("User".to_string(), "What about ML?".to_string())],
        );
        assert!(result.contains("Previous summary:"));
        assert!(result.contains("Previous discussion about AI"));
        assert!(result.contains("What about ML?"));
    }

    #[test]
    fn test_rolling_summary_without_existing() {
        let summarizer = ConversationSummarizer::default();
        let result = summarizer.rolling_summary(
            "",
            &[("User".to_string(), "First message".to_string())],
        );
        assert!(!result.contains("Previous summary:"));
        assert!(result.contains("First message"));
    }

    #[test]
    fn test_prompt_config_flags() {
        let config = SummaryConfig {
            include_key_points: false,
            include_topics: false,
            include_action_items: false,
            ..Default::default()
        };
        let summarizer = ConversationSummarizer::new(config);
        let messages = vec![("User".to_string(), "test".to_string())];
        let prompt = summarizer.build_summary_prompt(&messages);
        assert!(!prompt.contains("key points"));
        assert!(!prompt.contains("main topics"));
        assert!(!prompt.contains("action items"));
    }

    #[test]
    fn test_stop_words_filtered() {
        let summarizer = ConversationSummarizer::default();
        let messages = vec![
            ("User".to_string(), "Explain machine learning algorithms in detail".to_string()),
            ("Assistant".to_string(), "Machine learning algorithms process data".to_string()),
        ];
        let topics = summarizer.extract_topics(&messages);
        // "machine" and "learning" should appear but stop words like "the", "in" should not
        for topic in &topics {
            assert!(topic.len() > 4);
        }
    }

    #[test]
    fn test_topic_extraction() {
        let summarizer = ConversationSummarizer::default();
        let messages = vec![
            (
                "User".to_string(),
                "Tell me about machine learning and deep learning.".to_string(),
            ),
            (
                "Assistant".to_string(),
                "Machine learning is a subset of AI. Deep learning uses neural networks."
                    .to_string(),
            ),
        ];

        let topics = summarizer.extract_topics(&messages);
        assert!(!topics.is_empty());
    }
}
