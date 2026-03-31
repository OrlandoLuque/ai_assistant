//! Emotion detection from audio and text.
//!
//! Provides an `EmotionDetector` trait for detecting emotional state from
//! audio samples or text. Implementations can use local ONNX models
//! (emotion2vec, SenseVoice) or cloud APIs (Hume AI).
//!
//! Feature-gated behind the `audio` feature flag.

use serde::{Deserialize, Serialize};

// ============================================================================
// Emotion Types
// ============================================================================

/// Detected emotional category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EmotionCategory {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Excited,
    Frustrated,
    Confused,
    Calm,
    Bored,
}

impl std::fmt::Display for EmotionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Neutral => write!(f, "neutral"),
            Self::Happy => write!(f, "happy"),
            Self::Sad => write!(f, "sad"),
            Self::Angry => write!(f, "angry"),
            Self::Fearful => write!(f, "fearful"),
            Self::Disgusted => write!(f, "disgusted"),
            Self::Surprised => write!(f, "surprised"),
            Self::Excited => write!(f, "excited"),
            Self::Frustrated => write!(f, "frustrated"),
            Self::Confused => write!(f, "confused"),
            Self::Calm => write!(f, "calm"),
            Self::Bored => write!(f, "bored"),
            _ => write!(f, "unknown"),
        }
    }
}

impl Default for EmotionCategory {
    fn default() -> Self {
        Self::Neutral
    }
}

/// Detected emotional state with confidence and intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionState {
    /// Primary detected emotion.
    pub category: EmotionCategory,
    /// Confidence in the detection (0.0 to 1.0).
    pub confidence: f32,
    /// Intensity of the emotion (0.0 = mild, 1.0 = extreme).
    pub intensity: f32,
    /// Secondary emotion (if mixed emotions detected).
    pub secondary: Option<EmotionCategory>,
    /// All emotion probabilities (for detailed analysis).
    pub probabilities: Vec<(EmotionCategory, f32)>,
}

impl EmotionState {
    /// Create a simple emotion state with just category and confidence.
    pub fn new(category: EmotionCategory, confidence: f32) -> Self {
        Self {
            category,
            confidence: confidence.clamp(0.0, 1.0),
            intensity: 0.5,
            secondary: None,
            probabilities: vec![(category, confidence)],
        }
    }

    /// Create with full probability distribution.
    pub fn from_probabilities(mut probs: Vec<(EmotionCategory, f32)>) -> Self {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (category, confidence) = probs.first().copied().unwrap_or((EmotionCategory::Neutral, 0.0));
        let secondary = probs.get(1).map(|(cat, _)| *cat);
        Self {
            category,
            confidence,
            intensity: confidence, // approximate
            secondary,
            probabilities: probs,
        }
    }

    /// Whether the emotion is "negative" (might need empathetic response).
    pub fn is_negative(&self) -> bool {
        matches!(
            self.category,
            EmotionCategory::Sad
                | EmotionCategory::Angry
                | EmotionCategory::Fearful
                | EmotionCategory::Frustrated
                | EmotionCategory::Disgusted
        )
    }

    /// Whether the emotion is "positive".
    pub fn is_positive(&self) -> bool {
        matches!(
            self.category,
            EmotionCategory::Happy | EmotionCategory::Excited | EmotionCategory::Calm
        )
    }

    /// Suggest a TTS instruction based on detected emotion (for empathetic response).
    pub fn suggest_tts_instruction(&self) -> &'static str {
        if self.confidence < 0.4 {
            return "Speak naturally.";
        }
        match self.category {
            EmotionCategory::Frustrated | EmotionCategory::Angry => {
                "Speak calmly and reassuringly. Be patient and empathetic."
            }
            EmotionCategory::Sad | EmotionCategory::Fearful => {
                "Speak gently and warmly. Be supportive and understanding."
            }
            EmotionCategory::Excited | EmotionCategory::Happy => {
                "Speak with energy and enthusiasm. Match the user's positive tone."
            }
            EmotionCategory::Confused => {
                "Speak clearly and slowly. Be helpful and structured."
            }
            EmotionCategory::Bored => {
                "Speak concisely and engagingly. Keep it interesting."
            }
            _ => "Speak naturally.",
        }
    }

    /// Generate a context string to inject into the LLM prompt.
    pub fn to_prompt_context(&self) -> String {
        if self.confidence < 0.3 {
            return String::new();
        }
        let mut ctx = format!(
            "User emotional state: {} (confidence: {:.0}%, intensity: {:.0}%)",
            self.category,
            self.confidence * 100.0,
            self.intensity * 100.0
        );
        if let Some(secondary) = self.secondary {
            ctx.push_str(&format!(", also showing signs of {}", secondary));
        }
        if self.is_negative() {
            ctx.push_str(". Respond with empathy and care.");
        }
        ctx
    }
}

impl Default for EmotionState {
    fn default() -> Self {
        Self::new(EmotionCategory::Neutral, 0.0)
    }
}

// ============================================================================
// Emotion Detector Trait
// ============================================================================

/// Trait for detecting emotions from audio or text.
pub trait EmotionDetector: Send + Sync {
    /// Detect emotion from an audio sample.
    ///
    /// # Arguments
    /// * `audio` - Raw audio bytes
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 16000)
    fn detect_from_audio(
        &self,
        audio: &[u8],
        sample_rate: u32,
    ) -> Result<EmotionState, String>;

    /// Detect emotion from text (sentiment analysis).
    fn detect_from_text(&self, text: &str) -> Result<EmotionState, String>;

    /// Detector name for diagnostics.
    fn name(&self) -> &str;

    /// Whether this detector supports audio input.
    fn supports_audio(&self) -> bool {
        true
    }

    /// Whether this detector supports text input.
    fn supports_text(&self) -> bool {
        true
    }
}

// ============================================================================
// Simple keyword-based text emotion detector (no LLM needed)
// ============================================================================

/// Simple heuristic emotion detector based on keyword patterns.
/// Useful as a free, fast fallback when no ONNX model or API is available.
pub struct KeywordEmotionDetector;

impl KeywordEmotionDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for KeywordEmotionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionDetector for KeywordEmotionDetector {
    fn detect_from_audio(
        &self,
        _audio: &[u8],
        _sample_rate: u32,
    ) -> Result<EmotionState, String> {
        // Keyword detector cannot analyze audio — return neutral
        Ok(EmotionState::default())
    }

    fn detect_from_text(&self, text: &str) -> Result<EmotionState, String> {
        let lower = text.to_lowercase();

        let patterns: Vec<(EmotionCategory, &[&str], f32)> = vec![
            (EmotionCategory::Angry, &["angry", "furious", "outraged", "hate", "terrible", "worst", "damn", "hell"], 0.7),
            (EmotionCategory::Frustrated, &["frustrated", "annoying", "stuck", "doesn't work", "broken", "ugh", "can't", "impossible"], 0.65),
            (EmotionCategory::Sad, &["sad", "depressed", "unhappy", "disappointed", "sorry", "unfortunately", "miss", "lonely"], 0.65),
            (EmotionCategory::Happy, &["happy", "great", "awesome", "wonderful", "love", "excellent", "amazing", "perfect", "thanks"], 0.6),
            (EmotionCategory::Excited, &["excited", "incredible", "fantastic", "wow", "can't wait", "!!", "omg"], 0.6),
            (EmotionCategory::Confused, &["confused", "don't understand", "what do you mean", "unclear", "lost", "huh", "?"], 0.55),
            (EmotionCategory::Fearful, &["scared", "afraid", "worried", "anxious", "nervous", "concerned", "fear"], 0.6),
        ];

        let mut best = EmotionCategory::Neutral;
        let mut best_score = 0.0f32;

        for (category, keywords, base_confidence) in &patterns {
            let matches = keywords.iter().filter(|kw| lower.contains(*kw)).count();
            if matches > 0 {
                let score = base_confidence + (matches as f32 - 1.0) * 0.1;
                if score > best_score {
                    best = *category;
                    best_score = score;
                }
            }
        }

        Ok(EmotionState::new(best, best_score.min(1.0)))
    }

    fn name(&self) -> &str {
        "KeywordEmotionDetector"
    }

    fn supports_audio(&self) -> bool {
        false
    }
}

// ============================================================================
// Audio Event Types
// ============================================================================

/// Non-speech audio events that can be detected (e.g., by SenseVoice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AudioEvent {
    Laughter,
    Crying,
    Applause,
    Coughing,
    Sigh,
    Music,
    Silence,
    Noise,
}

impl std::fmt::Display for AudioEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Laughter => write!(f, "laughter"),
            Self::Crying => write!(f, "crying"),
            Self::Applause => write!(f, "applause"),
            Self::Coughing => write!(f, "coughing"),
            Self::Sigh => write!(f, "sigh"),
            Self::Music => write!(f, "music"),
            Self::Silence => write!(f, "silence"),
            Self::Noise => write!(f, "noise"),
            _ => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// LLM Enhancement: Speaker Intent Classification (V68)
// ============================================================================

/// Speaker intent beyond emotion (classified by LLM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerIntent {
    /// The classified intent.
    pub intent: String,
    /// Urgency level.
    pub urgency: String,
}

/// Configuration for intent classification.
#[derive(Debug, Clone)]
pub struct IntentClassifierConfig {
    /// Use LLM to classify speaker intent beyond basic emotion detection.
    /// When false (default), uses heuristic keyword-based classification.
    pub llm_enhanced: bool,
}

impl Default for IntentClassifierConfig {
    fn default() -> Self {
        Self {
            llm_enhanced: false,
        }
    }
}

/// Classifies speaker intent from text, optionally enhanced by LLM.
pub struct IntentClassifier {
    pub config: IntentClassifierConfig,
}

impl IntentClassifier {
    pub fn new(config: IntentClassifierConfig) -> Self {
        Self { config }
    }

    /// Build a prompt for LLM-based intent classification.
    ///
    /// Returns None if LLM enhancement is disabled.
    pub fn build_intent_prompt(&self, text: &str) -> Option<String> {
        if !self.config.llm_enhanced {
            return None;
        }

        let prompt = format!(
            "Beyond emotion, classify the speaker's intent. \
             Return JSON: {{\"intent\":\"question|command|information|request\",\"urgency\":\"low|medium|high\"}}\n\n\
             Text: {}",
            crate::llm_enhance::prompt_wrap(text)
        );

        Some(prompt)
    }

    /// Parse LLM response for intent classification.
    pub fn parse_intent_response(response: &str) -> Option<SpeakerIntent> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let intent = val
                    .get("intent")
                    .and_then(|s| s.as_str())
                    .unwrap_or("information");
                let urgency = val
                    .get("urgency")
                    .and_then(|s| s.as_str())
                    .unwrap_or("low");
                return Some(SpeakerIntent {
                    intent: intent.to_string(),
                    urgency: urgency.to_string(),
                });
            }
        }
        None
    }

    /// Classify speaker intent with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for
    /// nuanced intent classification. Otherwise uses keyword heuristics.
    pub fn classify_intent_with_llm(
        &self,
        text: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> SpeakerIntent {
        // Heuristic baseline
        let lower = text.to_lowercase();
        let heuristic_intent = if lower.contains('?') || lower.starts_with("what")
            || lower.starts_with("how") || lower.starts_with("why")
            || lower.starts_with("when") || lower.starts_with("where")
            || lower.starts_with("who") || lower.starts_with("is ")
        {
            "question"
        } else if lower.starts_with("please") || lower.contains("could you")
            || lower.contains("can you") || lower.contains("would you")
        {
            "request"
        } else if lower.starts_with("do ") || lower.starts_with("run ")
            || lower.starts_with("stop") || lower.starts_with("start")
            || lower.starts_with("set ") || lower.starts_with("turn")
        {
            "command"
        } else {
            "information"
        };

        let heuristic_urgency = if lower.contains("urgent") || lower.contains("asap")
            || lower.contains("immediately") || lower.contains("now")
            || lower.contains('!')
        {
            "high"
        } else if lower.contains("when you can") || lower.contains("no rush") {
            "low"
        } else {
            "medium"
        };

        let heuristic = SpeakerIntent {
            intent: heuristic_intent.to_string(),
            urgency: heuristic_urgency.to_string(),
        };

        // Try LLM enhancement
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_intent_prompt(text) {
                    if let Ok(response) = enhancer.generate(&prompt, 200) {
                        if let Some(intent) = Self::parse_intent_response(&response) {
                            return intent;
                        }
                    }
                }
            }
        }

        heuristic
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new(IntentClassifierConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_state_new() {
        let state = EmotionState::new(EmotionCategory::Happy, 0.85);
        assert_eq!(state.category, EmotionCategory::Happy);
        assert!((state.confidence - 0.85).abs() < 0.01);
        assert!(state.is_positive());
        assert!(!state.is_negative());
    }

    #[test]
    fn test_emotion_state_from_probabilities() {
        let probs = vec![
            (EmotionCategory::Angry, 0.6),
            (EmotionCategory::Frustrated, 0.3),
            (EmotionCategory::Neutral, 0.1),
        ];
        let state = EmotionState::from_probabilities(probs);
        assert_eq!(state.category, EmotionCategory::Angry);
        assert_eq!(state.secondary, Some(EmotionCategory::Frustrated));
        assert!(state.is_negative());
    }

    #[test]
    fn test_emotion_tts_instruction() {
        let frustrated = EmotionState::new(EmotionCategory::Frustrated, 0.8);
        assert!(frustrated.suggest_tts_instruction().contains("calm"));

        let happy = EmotionState::new(EmotionCategory::Happy, 0.9);
        assert!(happy.suggest_tts_instruction().contains("energy"));

        let low_confidence = EmotionState::new(EmotionCategory::Angry, 0.2);
        assert!(low_confidence.suggest_tts_instruction().contains("naturally"));
    }

    #[test]
    fn test_emotion_prompt_context() {
        let state = EmotionState::new(EmotionCategory::Frustrated, 0.75);
        let ctx = state.to_prompt_context();
        assert!(ctx.contains("frustrated"));
        assert!(ctx.contains("75%"));
        assert!(ctx.contains("empathy"));
    }

    #[test]
    fn test_emotion_prompt_context_low_confidence() {
        let state = EmotionState::new(EmotionCategory::Angry, 0.1);
        let ctx = state.to_prompt_context();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_keyword_detector_happy() {
        let detector = KeywordEmotionDetector::new();
        let result = detector.detect_from_text("This is awesome! I love it!").unwrap();
        assert_eq!(result.category, EmotionCategory::Happy);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_keyword_detector_frustrated() {
        let detector = KeywordEmotionDetector::new();
        let result = detector.detect_from_text("This doesn't work, I'm stuck and it's annoying").unwrap();
        assert_eq!(result.category, EmotionCategory::Frustrated);
    }

    #[test]
    fn test_keyword_detector_neutral() {
        let detector = KeywordEmotionDetector::new();
        let result = detector.detect_from_text("Please list the files in the directory").unwrap();
        assert_eq!(result.category, EmotionCategory::Neutral);
        assert!(result.confidence < 0.1);
    }

    #[test]
    fn test_keyword_detector_no_audio() {
        let detector = KeywordEmotionDetector::new();
        assert!(!detector.supports_audio());
        assert!(detector.supports_text());
    }

    #[test]
    fn test_audio_event_display() {
        assert_eq!(AudioEvent::Laughter.to_string(), "laughter");
        assert_eq!(AudioEvent::Sigh.to_string(), "sigh");
    }

    #[test]
    fn test_emotion_category_display() {
        assert_eq!(EmotionCategory::Frustrated.to_string(), "frustrated");
        assert_eq!(EmotionCategory::Excited.to_string(), "excited");
    }

    // ── V68: LLM Enhancement tests ──────────────────────────────────

    #[test]
    fn test_classify_intent_heuristic_without_llm() {
        let config = IntentClassifierConfig {
            llm_enhanced: false,
        };
        let classifier = IntentClassifier::new(config);
        let intent = classifier.classify_intent_with_llm("What is the weather?", None);
        assert_eq!(intent.intent, "question");
        assert_eq!(intent.urgency, "medium");
    }

    #[test]
    fn test_classify_intent_with_mock_llm() {
        let config = IntentClassifierConfig {
            llm_enhanced: true,
        };
        let classifier = IntentClassifier::new(config);
        let mock = crate::llm_enhance::MockLlm::new(
            "{\"intent\":\"command\",\"urgency\":\"high\"}",
        );
        let intent = classifier.classify_intent_with_llm("Turn off the lights now!", Some(&mock));
        assert_eq!(intent.intent, "command", "Expected LLM intent, got: {}", intent.intent);
        assert_eq!(intent.urgency, "high");
    }

    #[test]
    fn test_classify_intent_llm_fallback_on_failure() {
        let config = IntentClassifierConfig {
            llm_enhanced: true,
        };
        let classifier = IntentClassifier::new(config);
        let failing = crate::llm_enhance::FailingMockLlm;
        let intent = classifier.classify_intent_with_llm("How does this work?", Some(&failing));
        // Should fall back to heuristic (not crash)
        assert_eq!(intent.intent, "question");
    }
}
