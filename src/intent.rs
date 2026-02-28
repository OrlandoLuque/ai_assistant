//! Intent classification
//!
//! Classify user intents from messages.

use std::collections::HashMap;

/// User intent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intent {
    Question,
    Command,
    Greeting,
    Farewell,
    Thanks,
    Complaint,
    Request,
    Clarification,
    Confirmation,
    Negation,
    Opinion,
    Chitchat,
    CodeRequest,
    Explanation,
    Comparison,
    Unknown,
}

impl Intent {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Question => "Question",
            Self::Command => "Command",
            Self::Greeting => "Greeting",
            Self::Farewell => "Farewell",
            Self::Thanks => "Thanks",
            Self::Complaint => "Complaint",
            Self::Request => "Request",
            Self::Clarification => "Clarification",
            Self::Confirmation => "Confirmation",
            Self::Negation => "Negation",
            Self::Opinion => "Opinion",
            Self::Chitchat => "Chitchat",
            Self::CodeRequest => "Code Request",
            Self::Explanation => "Explanation",
            Self::Comparison => "Comparison",
            Self::Unknown => "Unknown",
        }
    }
}

/// Intent classification result
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub primary: Intent,
    pub confidence: f64,
    pub all_intents: Vec<(Intent, f64)>,
}

/// Intent classifier
pub struct IntentClassifier {
    patterns: HashMap<Intent, Vec<&'static str>>,
}

impl IntentClassifier {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        patterns.insert(
            Intent::Question,
            vec![
                "what",
                "why",
                "how",
                "when",
                "where",
                "who",
                "which",
                "whose",
                "is it",
                "are there",
                "can you",
                "could you",
                "do you",
                "does it",
                "?",
            ],
        );

        patterns.insert(
            Intent::Command,
            vec![
                "do ",
                "make ",
                "create ",
                "build ",
                "write ",
                "generate ",
                "show ",
                "display ",
                "list ",
                "find ",
                "search ",
                "get ",
                "run ",
                "execute ",
                "start ",
                "stop ",
                "delete ",
                "remove ",
            ],
        );

        patterns.insert(
            Intent::Greeting,
            vec![
                "hello",
                "hi ",
                "hey ",
                "good morning",
                "good afternoon",
                "good evening",
                "greetings",
                "howdy",
            ],
        );

        patterns.insert(
            Intent::Farewell,
            vec![
                "bye",
                "goodbye",
                "see you",
                "take care",
                "good night",
                "farewell",
                "later",
            ],
        );

        patterns.insert(
            Intent::Thanks,
            vec!["thank", "thanks", "appreciate", "grateful", "cheers"],
        );

        patterns.insert(
            Intent::Complaint,
            vec![
                "doesn't work",
                "not working",
                "broken",
                "bug",
                "error",
                "wrong",
                "incorrect",
                "bad",
                "terrible",
                "awful",
            ],
        );

        patterns.insert(
            Intent::Request,
            vec![
                "please",
                "could you",
                "would you",
                "can you",
                "i need",
                "i want",
                "i'd like",
                "help me",
            ],
        );

        patterns.insert(
            Intent::Clarification,
            vec![
                "what do you mean",
                "i don't understand",
                "can you explain",
                "be more specific",
                "elaborate",
                "clarify",
            ],
        );

        patterns.insert(
            Intent::Confirmation,
            vec![
                "yes",
                "yeah",
                "yep",
                "correct",
                "right",
                "exactly",
                "sure",
                "ok",
                "okay",
                "agreed",
                "affirmative",
            ],
        );

        patterns.insert(
            Intent::Negation,
            vec![
                "no",
                "nope",
                "not",
                "never",
                "wrong",
                "incorrect",
                "disagree",
                "negative",
            ],
        );

        patterns.insert(
            Intent::CodeRequest,
            vec![
                "code",
                "function",
                "class",
                "implement",
                "programming",
                "script",
                "algorithm",
                "syntax",
                "debug",
            ],
        );

        patterns.insert(
            Intent::Explanation,
            vec![
                "explain",
                "tell me about",
                "describe",
                "what is",
                "definition",
                "meaning of",
            ],
        );

        patterns.insert(
            Intent::Comparison,
            vec![
                "compare",
                "difference",
                "versus",
                "vs",
                "better than",
                "worse than",
                "similar",
                "different",
            ],
        );

        Self { patterns }
    }

    /// Classify the intent of a message
    pub fn classify(&self, message: &str) -> IntentResult {
        let lower = message.to_lowercase();
        let mut scores: HashMap<Intent, f64> = HashMap::new();

        for (intent, patterns) in &self.patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if lower.contains(pattern) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                scores.insert(*intent, score / patterns.len() as f64);
            }
        }

        // Get all intents sorted by score
        let mut all_intents: Vec<_> = scores.into_iter().collect();
        all_intents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (primary, confidence) = all_intents
            .first()
            .cloned()
            .unwrap_or((Intent::Unknown, 0.0));

        IntentResult {
            primary,
            confidence,
            all_intents,
        }
    }

    /// Get suggested response type for intent
    pub fn suggest_response_type(&self, intent: Intent) -> &'static str {
        match intent {
            Intent::Question => "informative",
            Intent::Command => "action",
            Intent::Greeting => "greeting",
            Intent::Farewell => "farewell",
            Intent::Thanks => "acknowledgment",
            Intent::Complaint => "supportive",
            Intent::Request => "helpful",
            Intent::Clarification => "detailed_explanation",
            Intent::Confirmation => "acknowledgment",
            Intent::Negation => "clarification",
            Intent::Opinion => "balanced",
            Intent::Chitchat => "conversational",
            Intent::CodeRequest => "code",
            Intent::Explanation => "educational",
            Intent::Comparison => "analytical",
            Intent::Unknown => "general",
        }
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What is the capital of France?");
        // "What is" matches both Question and Explanation patterns
        assert!(result.primary == Intent::Question || result.primary == Intent::Explanation);
    }

    #[test]
    fn test_greeting() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Hello, how are you?");
        assert!(result.primary == Intent::Greeting || result.primary == Intent::Question);
    }

    #[test]
    fn test_code_request() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Write a function to sort an array");
        assert!(result.primary == Intent::CodeRequest || result.primary == Intent::Command);
    }

    #[test]
    fn test_farewell_intent() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("goodbye, see you later");
        assert_eq!(result.primary, Intent::Farewell);
        assert!(result.confidence > 0.0);
        // Should match "goodbye", "see you", "later" — multiple farewell patterns
        assert!(
            result.all_intents.iter().any(|(i, _)| *i == Intent::Farewell),
            "Farewell should appear in all_intents"
        );
    }

    #[test]
    fn test_thanks_intent() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("thank you so much");
        assert_eq!(result.primary, Intent::Thanks);
        assert!(result.confidence > 0.0);
        assert_eq!(result.primary.name(), "Thanks");
    }

    #[test]
    fn test_complaint_intent() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("this doesn't work, it's broken");
        assert_eq!(result.primary, Intent::Complaint);
        assert!(result.confidence > 0.0);
        // Should match "doesn't work" and "broken"
        assert!(
            result.all_intents.iter().any(|(i, _)| *i == Intent::Complaint),
        );
    }

    #[test]
    fn test_comparison_intent() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("compare Python vs Rust");
        assert_eq!(result.primary, Intent::Comparison);
        assert!(result.confidence > 0.0);
        // "compare" and "vs" are both Comparison patterns
        assert!(
            result.all_intents.iter().any(|(i, _)| *i == Intent::Comparison),
        );
    }

    #[test]
    fn test_confirmation_intent() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("yes, that's correct");
        assert_eq!(result.primary, Intent::Confirmation);
        assert!(result.confidence > 0.0);
        // "yes" and "correct" both match Confirmation patterns
        assert!(
            result.all_intents.iter().any(|(i, _)| *i == Intent::Confirmation),
        );
    }

    #[test]
    fn test_suggest_response_type() {
        let classifier = IntentClassifier::new();

        // Verify every intent variant maps to the correct response type
        assert_eq!(classifier.suggest_response_type(Intent::Question), "informative");
        assert_eq!(classifier.suggest_response_type(Intent::Command), "action");
        assert_eq!(classifier.suggest_response_type(Intent::Greeting), "greeting");
        assert_eq!(classifier.suggest_response_type(Intent::Farewell), "farewell");
        assert_eq!(classifier.suggest_response_type(Intent::Thanks), "acknowledgment");
        assert_eq!(classifier.suggest_response_type(Intent::Complaint), "supportive");
        assert_eq!(classifier.suggest_response_type(Intent::Request), "helpful");
        assert_eq!(classifier.suggest_response_type(Intent::Clarification), "detailed_explanation");
        assert_eq!(classifier.suggest_response_type(Intent::Confirmation), "acknowledgment");
        assert_eq!(classifier.suggest_response_type(Intent::Negation), "clarification");
        assert_eq!(classifier.suggest_response_type(Intent::Opinion), "balanced");
        assert_eq!(classifier.suggest_response_type(Intent::Chitchat), "conversational");
        assert_eq!(classifier.suggest_response_type(Intent::CodeRequest), "code");
        assert_eq!(classifier.suggest_response_type(Intent::Explanation), "educational");
        assert_eq!(classifier.suggest_response_type(Intent::Comparison), "analytical");
        assert_eq!(classifier.suggest_response_type(Intent::Unknown), "general");
    }

    #[test]
    fn test_empty_input_is_unknown() {
        let classifier = IntentClassifier::new();
        let intent = classifier.classify("");
        assert_eq!(intent.primary, Intent::Unknown);
    }
}
