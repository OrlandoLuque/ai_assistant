//! Intent classification
//!
//! Classify user intents from messages.

use std::collections::HashMap;

/// User intent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

/// Configuration for LLM-enhanced intent classification.
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// Use LLM to enhance intent classification.
    /// When false (default), uses heuristic pattern matching.
    pub llm_enhanced: bool,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            llm_enhanced: false,
        }
    }
}

/// Intent classifier
pub struct IntentClassifier {
    patterns: HashMap<Intent, Vec<&'static str>>,
    config: IntentConfig,
}

impl IntentClassifier {
    pub fn new() -> Self {
        Self::with_config(IntentConfig::default())
    }

    /// Create a new classifier with the given configuration.
    pub fn with_config(config: IntentConfig) -> Self {
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

        Self { patterns, config }
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

// ============================================================================
// LLM Enhancement: Intent Classification (V68)
// ============================================================================

impl IntentClassifier {
    /// Build a prompt for LLM-based intent classification.
    ///
    /// Returns None if LLM enhancement is disabled or message is empty.
    pub fn build_classify_prompt(&self, message: &str) -> Option<String> {
        if !self.config.llm_enhanced || message.is_empty() {
            return None;
        }

        Some(format!(
            "Classify the intent of this message. Return JSON: \
             {{\"intent\":\"question|command|greeting|request|complaint|chitchat\",\"confidence\":0.9}}\n\n{}",
            crate::llm_enhance::prompt_wrap(message)
        ))
    }

    /// Parse LLM response for intent classification.
    pub fn parse_classify_response(response: &str) -> Option<IntentResult> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let intent_str = val.get("intent")?.as_str()?;
                let confidence = val
                    .get("confidence")
                    .and_then(|c| c.as_f64())
                    .unwrap_or(0.8);

                let intent = match intent_str.to_lowercase().as_str() {
                    "question" => Intent::Question,
                    "command" => Intent::Command,
                    "greeting" => Intent::Greeting,
                    "request" => Intent::Request,
                    "complaint" => Intent::Complaint,
                    "chitchat" => Intent::Chitchat,
                    "farewell" => Intent::Farewell,
                    "thanks" => Intent::Thanks,
                    "clarification" => Intent::Clarification,
                    "confirmation" => Intent::Confirmation,
                    "negation" => Intent::Negation,
                    "opinion" => Intent::Opinion,
                    "code_request" | "coderequest" => Intent::CodeRequest,
                    "explanation" => Intent::Explanation,
                    "comparison" => Intent::Comparison,
                    _ => Intent::Unknown,
                };

                return Some(IntentResult {
                    primary: intent,
                    confidence,
                    all_intents: vec![(intent, confidence)],
                });
            }
        }
        None
    }

    /// Classify intent with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for
    /// classification. Otherwise falls back to heuristic classification.
    pub fn classify_with_llm(
        &self,
        message: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> IntentResult {
        // Try LLM enhancement first
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_classify_prompt(message) {
                    if let Ok(response) = enhancer.generate(&prompt, 200) {
                        if let Some(result) = Self::parse_classify_response(&response) {
                            return result;
                        }
                    }
                }
            }
        }

        // Fallback: heuristic classification
        self.classify(message)
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

    // ── V68: LLM Enhancement tests ──────────────────────────────────

    #[test]
    fn test_classify_heuristic_without_llm() {
        let config = IntentConfig {
            llm_enhanced: false,
        };
        let classifier = IntentClassifier::with_config(config);
        let result = classifier.classify_with_llm("What is the weather?", None);
        assert!(
            result.primary == Intent::Question || result.primary == Intent::Explanation,
            "Heuristic should classify question, got: {:?}",
            result.primary
        );
    }

    #[test]
    fn test_classify_with_mock_llm() {
        let config = IntentConfig {
            llm_enhanced: true,
        };
        let classifier = IntentClassifier::with_config(config);
        let mock = crate::llm_enhance::MockLlm::new(
            "{\"intent\":\"complaint\",\"confidence\":0.95}",
        );
        let result = classifier.classify_with_llm("this is broken", Some(&mock));
        assert_eq!(result.primary, Intent::Complaint);
        assert!((result.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_classify_llm_fallback_on_failure() {
        let config = IntentConfig {
            llm_enhanced: true,
        };
        let classifier = IntentClassifier::with_config(config);
        let failing = crate::llm_enhance::FailingMockLlm;
        let result = classifier.classify_with_llm("goodbye, see you later", Some(&failing));
        // Should fall back to heuristic (not crash)
        assert_eq!(result.primary, Intent::Farewell);
    }
}
