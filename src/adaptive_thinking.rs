//! Adaptive thinking module: automatically adjusts reasoning depth based on query complexity.
//!
//! ## Overview
//!
//! Different queries require different levels of reasoning effort. A greeting like "hello"
//! doesn't need chain-of-thought prompting, while "compare the trade-offs of CRDT vs OT
//! for collaborative editing" benefits from deep, structured reasoning.
//!
//! This module provides:
//! - **Query classification** via heuristics (no LLM call) into 5 depth levels
//! - **Strategy selection** that maps depth to temperature, max_tokens, RAG tier, and CoT prompts
//! - **Thinking tag parsing** for models that emit `<think>...</think>` blocks (DeepSeek-R1, QwQ)
//!
//! ## Architecture
//!
//! ```text
//! User query
//!      │
//!      ▼
//! ┌──────────────────┐
//! │  QueryClassifier  │ ← reuses IntentClassifier + structural heuristics
//! │   (no LLM call)   │
//! └────────┬─────────┘
//!          │ ThinkingStrategy
//!          ▼
//! ┌──────────────────┐
//! │  apply_strategy   │ ← modifies system_prompt + config before LLM call
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ ThinkingTagParser │ ← strips <think>...</think> from streaming output
//! └──────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use ai_assistant::adaptive_thinking::*;
//!
//! // Create a classifier with default config (disabled by default)
//! let config = AdaptiveThinkingConfig { enabled: true, ..Default::default() };
//! let classifier = QueryClassifier::new(config);
//!
//! // Classify a query
//! let strategy = classifier.classify("Compare Rust and Go for web servers");
//! assert_eq!(strategy.depth, ThinkingDepth::Complex);
//! assert!(strategy.temperature < 0.5); // Lower temperature for precise reasoning
//! ```

use crate::intent::IntentClassifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Enums
// ============================================================================

/// Thinking depth level, representing how much reasoning effort a query requires.
///
/// Ordered from least to most effort. Implements `Ord` so depths can be compared
/// and clamped to min/max bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDepth {
    /// Greetings, thanks, confirmations, trivial chitchat.
    /// No reasoning needed — quick, conversational response.
    Trivial,
    /// Single-fact questions, simple lookups, yes/no answers.
    /// Minimal reasoning — direct answer.
    Simple,
    /// Explanations, how-to questions, moderate context needed.
    /// Standard reasoning — structured response.
    Moderate,
    /// Multi-step reasoning, analysis, comparisons, synthesis.
    /// Deep reasoning with chain-of-thought prompting.
    Complex,
    /// Deep analysis, multi-concept integration, research-level queries.
    /// Maximum reasoning with explicit step-by-step thinking and verification.
    Expert,
}

impl ThinkingDepth {
    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Trivial => "Trivial",
            Self::Simple => "Simple",
            Self::Moderate => "Moderate",
            Self::Complex => "Complex",
            Self::Expert => "Expert",
        }
    }

    /// Numeric level (0–4) for arithmetic operations.
    pub fn numeric_level(&self) -> u8 {
        match self {
            Self::Trivial => 0,
            Self::Simple => 1,
            Self::Moderate => 2,
            Self::Complex => 3,
            Self::Expert => 4,
        }
    }

    /// Create a `ThinkingDepth` from a numeric level (clamped to 0–4).
    pub fn from_level(level: u8) -> Self {
        match level.min(4) {
            0 => Self::Trivial,
            1 => Self::Simple,
            2 => Self::Moderate,
            3 => Self::Complex,
            _ => Self::Expert,
        }
    }

    /// Upgrade to the next higher depth level. Returns `Expert` if already at max.
    pub fn upgrade(&self) -> Self {
        Self::from_level(self.numeric_level().saturating_add(1))
    }

    /// Downgrade to the next lower depth level. Returns `Trivial` if already at min.
    pub fn downgrade(&self) -> Self {
        Self::from_level(self.numeric_level().saturating_sub(1))
    }
}

/// Priority rule for resolving conflicts between adaptive RAG tier suggestion
/// and an explicitly configured RAG tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RagTierPriority {
    /// Adaptive thinking's suggestion takes precedence (default).
    Adaptive,
    /// User's explicit tier configuration takes precedence.
    Explicit,
    /// Use whichever tier is higher (more thorough).
    Highest,
}

// ============================================================================
// Classification signals
// ============================================================================

/// All heuristic signals extracted from a query, used to determine thinking depth.
/// Exposed for debugging and explainability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationSignals {
    /// Total word count of the query.
    pub word_count: usize,
    /// Number of question marks in the query.
    pub question_marks: usize,
    /// Whether the query contains comparison language ("compare", "vs", "difference between").
    pub has_comparison: bool,
    /// Whether the query contains analysis language ("analyze", "evaluate", "implications").
    pub has_analysis: bool,
    /// Whether the query contains code-related terms ("function", "implement", "algorithm").
    pub has_code: bool,
    /// Whether the query has multiple sub-questions or clauses.
    pub is_multi_part: bool,
    /// Number of distinct concepts/entities detected (via capitalized words + technical terms).
    pub concept_count: usize,
    /// The primary intent detected by `IntentClassifier`.
    pub detected_intent: String,
    /// Confidence of the detected intent (0.0–1.0).
    pub intent_confidence: f64,
    /// Whether expert-level patterns were detected ("comprehensive analysis", "trade-offs").
    pub has_expert_patterns: bool,
}

// ============================================================================
// Thinking strategy (output)
// ============================================================================

/// Complete strategy for processing a query, produced by `QueryClassifier::classify`.
///
/// Contains the determined depth plus all parameter adjustments to apply
/// before the LLM call: temperature, max_tokens, RAG tier hint, and
/// system prompt additions for chain-of-thought reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingStrategy {
    /// Determined thinking depth after classification and clamping.
    pub depth: ThinkingDepth,
    /// Confidence in the classification (0.0 to 1.0).
    pub confidence: f32,
    /// Suggested temperature override for generation.
    pub temperature: f32,
    /// Suggested max_tokens (None = use provider/profile default).
    pub max_tokens: Option<u32>,
    /// Suggested RAG complexity level (maps to `QueryComplexity` in rag_tiers).
    /// Values: "simple", "standard", "complex", "reasoning".
    pub rag_complexity_hint: String,
    /// System prompt addition with CoT instructions (empty if none needed).
    pub system_prompt_addition: String,
    /// Suggested ModelProfile name (None = keep current).
    pub profile_suggestion: Option<String>,
    /// Whether the ThinkingTagParser should be active for this response.
    pub parse_thinking_tags: bool,
    /// The signals that contributed to this classification (for debugging).
    pub signals: ClassificationSignals,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the adaptive thinking system.
///
/// Disabled by default for backwards compatibility. Enable with `enabled: true`
/// or use `AiAssistant::enable_adaptive_thinking()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThinkingConfig {
    /// Master enable/disable switch. Default: `false`.
    pub enabled: bool,

    /// Minimum thinking depth floor. Even trivial queries won't go below this.
    pub min_depth: ThinkingDepth,

    /// Maximum thinking depth ceiling. Even expert queries won't go above this.
    pub max_depth: ThinkingDepth,

    /// Whether to inject CoT reasoning instructions into the system prompt.
    pub inject_cot_instructions: bool,

    /// Whether to parse `<think>...</think>` tags from model output.
    pub parse_thinking_tags: bool,

    /// Whether to strip thinking tags from the visible response.
    /// Only applies when `parse_thinking_tags` is true.
    pub strip_thinking_from_response: bool,

    /// Whether thinking tag parsing happens transparently inside `poll_response`.
    /// If false, the caller must use `ThinkingTagParser` manually.
    pub transparent_thinking_parse: bool,

    /// Whether to adjust temperature based on depth.
    pub adjust_temperature: bool,

    /// Whether to suggest a RAG tier based on depth.
    pub adjust_rag_tier: bool,

    /// Priority rule when adaptive RAG tier conflicts with explicit user tier.
    pub rag_tier_priority: RagTierPriority,

    /// Whether to adjust max_tokens based on depth.
    pub adjust_max_tokens: bool,

    /// Custom temperature overrides per depth level.
    /// If None, uses built-in defaults.
    pub temperature_map: Option<HashMap<ThinkingDepth, f32>>,

    /// Custom max_tokens overrides per depth level.
    /// If None, uses built-in defaults.
    pub max_tokens_map: Option<HashMap<ThinkingDepth, u32>>,

    /// Custom CoT instructions per depth level (for language/style customization).
    /// If None, uses built-in English defaults.
    pub cot_instructions_override: Option<HashMap<ThinkingDepth, String>>,
}

impl Default for AdaptiveThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_depth: ThinkingDepth::Trivial,
            max_depth: ThinkingDepth::Expert,
            inject_cot_instructions: true,
            parse_thinking_tags: true,
            strip_thinking_from_response: true,
            transparent_thinking_parse: true,
            adjust_temperature: true,
            adjust_rag_tier: true,
            rag_tier_priority: RagTierPriority::Adaptive,
            adjust_max_tokens: true,
            temperature_map: None,
            max_tokens_map: None,
            cot_instructions_override: None,
        }
    }
}

// ============================================================================
// Query classifier
// ============================================================================

/// Heuristic-based query complexity classifier.
///
/// Uses `IntentClassifier` from `intent.rs` for intent detection, plus structural
/// analysis (word count, question patterns, comparison/analysis keywords) to
/// determine thinking depth. No LLM call is made — classification is instant.
///
/// # Example
///
/// ```rust
/// use ai_assistant::adaptive_thinking::*;
///
/// let config = AdaptiveThinkingConfig { enabled: true, ..Default::default() };
/// let classifier = QueryClassifier::new(config);
///
/// let strategy = classifier.classify("hello");
/// assert_eq!(strategy.depth, ThinkingDepth::Trivial);
///
/// let strategy = classifier.classify("Compare HashMap vs BTreeMap performance");
/// assert_eq!(strategy.depth, ThinkingDepth::Complex);
/// ```
pub struct QueryClassifier {
    config: AdaptiveThinkingConfig,
    intent_classifier: IntentClassifier,
}

// Keyword lists as constants for pattern matching
const COMPARISON_KEYWORDS: &[&str] = &[
    "compare",
    "comparison",
    "versus",
    " vs ",
    " vs.",
    "difference between",
    "differences between",
    "better than",
    "worse than",
    "advantages of",
    "disadvantages of",
    "pros and cons",
    "trade-off",
    "tradeoff",
    "trade off",
];

const ANALYSIS_KEYWORDS: &[&str] = &[
    "analyze",
    "analyse",
    "evaluate",
    "assess",
    "implications",
    "impact",
    "consequences",
    "critically",
    "in depth",
    "in-depth",
];

const CODE_KEYWORDS: &[&str] = &[
    "code",
    "function",
    "implement",
    "programming",
    "algorithm",
    "syntax",
    "debug",
    "compile",
    "runtime",
    "variable",
    "struct",
    "class",
    "method",
    "```",
    "fn ",
    "def ",
    "int ",
    "void ",
];

const EXPERT_PATTERNS: &[&str] = &[
    "comprehensive analysis",
    "critically analyze",
    "critically analyse",
    "compare and contrast",
    "evaluate the trade-offs",
    "evaluate the tradeoffs",
    "design implications",
    "architectural",
    "in-depth analysis",
    "multiple perspectives",
    "from multiple angles",
];

const MULTI_PART_INDICATORS: &[&str] = &[
    " also ",
    " additionally ",
    " moreover ",
    " furthermore ",
    " besides ",
    " as well as ",
    " on top of ",
];

const TRIVIAL_PATTERNS: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "hola",
    "thanks",
    "thank you",
    "gracias",
    "bye",
    "goodbye",
    "adiós",
    "adios",
    "ok",
    "okay",
    "sure",
    "yes",
    "no",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "buenos días",
    "buenas tardes",
    "buenas noches",
];

impl QueryClassifier {
    /// Create a new classifier with the given configuration.
    pub fn new(config: AdaptiveThinkingConfig) -> Self {
        Self {
            config,
            intent_classifier: IntentClassifier::new(),
        }
    }

    /// Classify a query and produce a complete `ThinkingStrategy`.
    ///
    /// This is the main entry point. It analyzes the query using heuristics
    /// (no LLM call), determines the appropriate thinking depth, and builds
    /// a strategy with all parameter adjustments.
    pub fn classify(&self, query: &str) -> ThinkingStrategy {
        let signals = self.analyze_signals(query);
        let raw_depth = self.determine_depth(&signals);
        let clamped = raw_depth
            .max(self.config.min_depth)
            .min(self.config.max_depth);
        self.build_strategy(clamped, signals)
    }

    /// Classify with additional context from the conversation.
    ///
    /// Takes into account conversation length and previously detected topics
    /// to potentially upgrade the depth for follow-up questions in complex
    /// conversations.
    pub fn classify_with_context(
        &self,
        query: &str,
        conversation_length: usize,
        previous_topics: &[String],
    ) -> ThinkingStrategy {
        let signals = self.analyze_signals(query);
        let mut depth = self.determine_depth(&signals);

        // Upgrade depth for follow-up questions in long, topic-rich conversations
        if conversation_length > 10 && previous_topics.len() >= 3 && depth < ThinkingDepth::Moderate
        {
            depth = ThinkingDepth::Moderate;
        }

        let clamped = depth.max(self.config.min_depth).min(self.config.max_depth);
        self.build_strategy(clamped, signals)
    }

    /// Extract all classification signals from a query.
    ///
    /// Combines structural analysis (word count, punctuation) with
    /// `IntentClassifier` results and keyword matching. Returns all signals
    /// for transparency/debugging.
    pub fn analyze_signals(&self, query: &str) -> ClassificationSignals {
        let lower = query.to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        let word_count = words.len();
        let question_marks = query.chars().filter(|c| *c == '?').count();

        let has_comparison = COMPARISON_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_analysis = ANALYSIS_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_code = CODE_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_expert_patterns = EXPERT_PATTERNS.iter().any(|kw| lower.contains(kw));

        // Multi-part detection: query has multiple clauses with connectors
        let has_multi_connector = MULTI_PART_INDICATORS.iter().any(|kw| lower.contains(kw));
        let is_multi_part = (has_multi_connector && question_marks >= 1)
            || question_marks >= 2
            || (lower.contains(" and ") && question_marks >= 1 && word_count > 15);

        // Concept count: count capitalized words (likely proper nouns/entities) plus
        // distinct technical terms. We use a simple heuristic: words starting with
        // uppercase (not at sentence start) or words that look technical.
        let concept_count = count_concepts(&words);

        let intent_result = self.intent_classifier.classify(query);
        let detected_intent = intent_result.primary.name().to_string();
        let intent_confidence = intent_result.confidence;

        ClassificationSignals {
            word_count,
            question_marks,
            has_comparison,
            has_analysis,
            has_code,
            is_multi_part,
            concept_count,
            detected_intent,
            intent_confidence,
            has_expert_patterns,
        }
    }

    /// Determine thinking depth from classification signals.
    ///
    /// Priority order (checked top to bottom):
    /// 1. **Trivial**: short trivial patterns (greetings, thanks)
    /// 2. **Expert**: expert patterns + high complexity indicators
    /// 3. **Complex**: comparisons, analysis, multi-part queries
    /// 4. **Simple**: short factual questions
    /// 5. **Moderate**: default for everything else
    fn determine_depth(&self, signals: &ClassificationSignals) -> ThinkingDepth {
        // 1. Trivial: short messages matching trivial patterns
        if signals.word_count <= 5 {
            let intent = signals.detected_intent.as_str();
            if matches!(
                intent,
                "Greeting" | "Farewell" | "Thanks" | "Confirmation" | "Negation"
            ) {
                return ThinkingDepth::Trivial;
            }
            // Also check pattern list for edge cases the intent classifier might miss
            // (e.g., "hola" might not match English-trained patterns)
            let lower_query_approximation = signals.detected_intent.to_lowercase();
            let _ = lower_query_approximation; // We check via intent above
        }

        // For very short non-intent-matched queries, still check trivial patterns
        if signals.word_count <= 3 && signals.question_marks == 0 && !signals.has_code {
            return ThinkingDepth::Trivial;
        }

        // 2. Expert: strong expert indicators
        if signals.has_expert_patterns {
            return ThinkingDepth::Expert;
        }
        if signals.has_analysis && signals.has_comparison && signals.is_multi_part {
            return ThinkingDepth::Expert;
        }
        if signals.word_count > 80 && signals.question_marks >= 2 && signals.concept_count >= 4 {
            return ThinkingDepth::Expert;
        }

        // 3. Complex: comparison, analysis, or multi-part queries
        if signals.has_comparison || (signals.has_analysis && signals.word_count > 15) {
            return ThinkingDepth::Complex;
        }
        if signals.is_multi_part && signals.word_count > 20 {
            return ThinkingDepth::Complex;
        }
        if signals.concept_count >= 3 && signals.word_count > 25 {
            return ThinkingDepth::Complex;
        }

        // 4. Simple: short factual questions
        // "What is Rust?" (short Explanation) → Simple
        // "Explain how async/await works in Rust" (long Explanation) → falls through to Moderate
        if signals.word_count <= 12 && signals.question_marks <= 1 && signals.concept_count <= 1 {
            let intent = signals.detected_intent.as_str();
            if matches!(intent, "Question" | "Command") {
                return ThinkingDepth::Simple;
            }
            // Short explanation requests are essentially factual lookups
            if intent == "Explanation" && signals.word_count <= 8 {
                return ThinkingDepth::Simple;
            }
        }

        // 5. Default: moderate
        ThinkingDepth::Moderate
    }

    /// Build a complete `ThinkingStrategy` from a determined depth and signals.
    fn build_strategy(
        &self,
        depth: ThinkingDepth,
        signals: ClassificationSignals,
    ) -> ThinkingStrategy {
        let temperature = if self.config.adjust_temperature {
            self.config
                .temperature_map
                .as_ref()
                .and_then(|m| m.get(&depth).copied())
                .unwrap_or_else(|| Self::default_temperature(depth))
        } else {
            0.7
        };

        let max_tokens = if self.config.adjust_max_tokens {
            self.config
                .max_tokens_map
                .as_ref()
                .and_then(|m| m.get(&depth).copied())
                .or_else(|| Self::default_max_tokens(depth))
        } else {
            None
        };

        let rag_complexity_hint = if self.config.adjust_rag_tier {
            Self::depth_to_rag_complexity(depth).to_string()
        } else {
            "standard".to_string()
        };

        let system_prompt_addition = if self.config.inject_cot_instructions {
            if let Some(ref overrides) = self.config.cot_instructions_override {
                overrides.get(&depth).cloned().unwrap_or_default()
            } else {
                Self::default_cot_instructions(depth).to_string()
            }
        } else {
            String::new()
        };

        let profile_suggestion = Self::default_profile_suggestion(depth).map(String::from);

        let confidence = self.calculate_confidence(&signals, depth);

        ThinkingStrategy {
            depth,
            confidence,
            temperature,
            max_tokens,
            rag_complexity_hint,
            system_prompt_addition,
            profile_suggestion,
            parse_thinking_tags: self.config.parse_thinking_tags,
            signals,
        }
    }

    /// Calculate classification confidence based on how many signals agree.
    fn calculate_confidence(&self, signals: &ClassificationSignals, depth: ThinkingDepth) -> f32 {
        let mut agreement = 0u32;
        let mut total = 0u32;

        // Intent agreement
        total += 1;
        let intent_agrees = match depth {
            ThinkingDepth::Trivial => matches!(
                signals.detected_intent.as_str(),
                "Greeting" | "Farewell" | "Thanks" | "Confirmation" | "Negation" | "Chitchat"
            ),
            ThinkingDepth::Simple => {
                matches!(signals.detected_intent.as_str(), "Question" | "Command")
            }
            ThinkingDepth::Moderate => matches!(
                signals.detected_intent.as_str(),
                "Explanation" | "Request" | "Clarification" | "Code Request"
            ),
            ThinkingDepth::Complex => matches!(
                signals.detected_intent.as_str(),
                "Comparison" | "Explanation"
            ),
            ThinkingDepth::Expert => matches!(
                signals.detected_intent.as_str(),
                "Comparison" | "Explanation"
            ),
        };
        if intent_agrees {
            agreement += 1;
        }

        // Word count agreement
        total += 1;
        let wc_agrees = match depth {
            ThinkingDepth::Trivial => signals.word_count <= 5,
            ThinkingDepth::Simple => signals.word_count <= 15,
            ThinkingDepth::Moderate => signals.word_count > 5 && signals.word_count <= 40,
            ThinkingDepth::Complex => signals.word_count > 15,
            ThinkingDepth::Expert => signals.word_count > 30,
        };
        if wc_agrees {
            agreement += 1;
        }

        // Structural agreement
        total += 1;
        let struct_agrees = match depth {
            ThinkingDepth::Trivial => {
                !signals.has_comparison && !signals.has_analysis && !signals.is_multi_part
            }
            ThinkingDepth::Simple => !signals.has_comparison && !signals.has_analysis,
            ThinkingDepth::Moderate => true, // moderate is the catch-all
            ThinkingDepth::Complex => {
                signals.has_comparison || signals.has_analysis || signals.is_multi_part
            }
            ThinkingDepth::Expert => {
                signals.has_expert_patterns || (signals.has_comparison && signals.has_analysis)
            }
        };
        if struct_agrees {
            agreement += 1;
        }

        agreement as f32 / total as f32
    }

    // === Default parameter mappings ===

    fn default_temperature(depth: ThinkingDepth) -> f32 {
        match depth {
            ThinkingDepth::Trivial => 0.8,
            ThinkingDepth::Simple => 0.7,
            ThinkingDepth::Moderate => 0.6,
            ThinkingDepth::Complex => 0.4,
            ThinkingDepth::Expert => 0.2,
        }
    }

    fn default_max_tokens(depth: ThinkingDepth) -> Option<u32> {
        match depth {
            ThinkingDepth::Trivial => Some(256),
            ThinkingDepth::Simple => Some(1024),
            ThinkingDepth::Moderate => Some(2048),
            ThinkingDepth::Complex => Some(4096),
            ThinkingDepth::Expert => None, // unlimited
        }
    }

    fn depth_to_rag_complexity(depth: ThinkingDepth) -> &'static str {
        match depth {
            ThinkingDepth::Trivial => "simple",
            ThinkingDepth::Simple => "simple",
            ThinkingDepth::Moderate => "standard",
            ThinkingDepth::Complex => "complex",
            ThinkingDepth::Expert => "reasoning",
        }
    }

    fn default_profile_suggestion(depth: ThinkingDepth) -> Option<&'static str> {
        match depth {
            ThinkingDepth::Trivial => Some("conversational"),
            ThinkingDepth::Simple => None,
            ThinkingDepth::Moderate => None,
            ThinkingDepth::Complex => Some("precise"),
            ThinkingDepth::Expert => Some("detailed"),
        }
    }

    fn default_cot_instructions(depth: ThinkingDepth) -> &'static str {
        match depth {
            ThinkingDepth::Trivial | ThinkingDepth::Simple => "",
            ThinkingDepth::Moderate => {
                "When answering, organize your response clearly with relevant context. \
                 If the topic requires explanation, provide a structured answer."
            }
            ThinkingDepth::Complex => {
                "This is a complex question. Think through it step by step:\n\
                 1. Identify the key components of the question\n\
                 2. Address each component with evidence and reasoning\n\
                 3. Synthesize your findings into a coherent answer\n\
                 4. Note any caveats or limitations\n\
                 Be thorough but stay focused on what was asked."
            }
            ThinkingDepth::Expert => {
                "This requires deep, expert-level analysis. Use rigorous step-by-step reasoning:\n\
                 1. Break down the problem into its fundamental parts\n\
                 2. Analyze each part independently with detailed evidence\n\
                 3. Consider multiple perspectives and potential counterarguments\n\
                 4. Evaluate trade-offs and implications\n\
                 5. Synthesize everything into a comprehensive, well-structured answer\n\
                 6. Explicitly state your confidence level and any assumptions\n\
                 Take your time and be thorough. Quality over brevity."
            }
        }
    }
}

/// Count distinct concepts in a word list.
///
/// Heuristic: words that start with uppercase (not at sentence start) are likely
/// proper nouns or entities. Also counts words that look technical (contain
/// underscores, camelCase, or are all-caps with length > 2).
fn count_concepts(words: &[&str]) -> usize {
    let mut concepts = std::collections::HashSet::new();

    for (i, word) in words.iter().enumerate() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        if clean.is_empty() {
            continue;
        }

        // Capitalized word not at sentence start
        let first_char = clean.chars().next().unwrap_or(' ');
        let is_sentence_start = i == 0
            || words.get(i.wrapping_sub(1)).map_or(false, |prev| {
                prev.ends_with('.') || prev.ends_with('?') || prev.ends_with('!')
            });

        if first_char.is_uppercase() && !is_sentence_start && clean.len() > 1 {
            concepts.insert(clean.to_lowercase());
        }

        // Technical terms: contains underscore, is camelCase, or ALL_CAPS
        if clean.contains('_') && clean.len() > 2 {
            concepts.insert(clean.to_lowercase());
        }
        if clean.len() > 3 && clean.chars().all(|c| c.is_uppercase() || c == '_') {
            concepts.insert(clean.to_lowercase());
        }
        // camelCase: has lowercase followed by uppercase
        let chars: Vec<char> = clean.chars().collect();
        for j in 1..chars.len() {
            if chars[j - 1].is_lowercase() && chars[j].is_uppercase() {
                concepts.insert(clean.to_lowercase());
                break;
            }
        }
    }

    concepts.len()
}

// ============================================================================
// Thinking tag parser
// ============================================================================

/// Result of parsing `<think>...</think>` tags from a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingParseResult {
    /// The visible response with thinking content stripped.
    pub visible_response: String,
    /// The extracted thinking/reasoning content (from `<think>` tags).
    pub thinking: Option<String>,
    /// Whether any `<think>` tags were found and fully closed.
    pub had_thinking_tags: bool,
    /// Number of complete `<think>...</think>` blocks found.
    pub thinking_block_count: usize,
}

/// Streaming parser for `<think>...</think>` tags.
///
/// Handles the case where tags span multiple streaming chunks by maintaining
/// an internal buffer. Each call to `process_chunk` returns only the visible
/// (non-thinking) content from that chunk.
///
/// # Example
///
/// ```rust
/// use ai_assistant::adaptive_thinking::ThinkingTagParser;
///
/// let mut parser = ThinkingTagParser::new(true);
///
/// // Simulate streaming chunks
/// let v1 = parser.process_chunk("<think>reasoning here</think>");
/// assert_eq!(v1, "");
///
/// let v2 = parser.process_chunk("The answer is 42.");
/// assert_eq!(v2, "The answer is 42.");
///
/// parser.finalize();
/// let result = parser.result();
/// assert_eq!(result.thinking.as_deref(), Some("reasoning here"));
/// assert_eq!(result.visible_response, "The answer is 42.");
/// ```
pub struct ThinkingTagParser {
    buffer: String,
    in_thinking_block: bool,
    thinking_content: String,
    visible_content: String,
    block_count: usize,
    strip_thinking: bool,
}

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";
const OPEN_TAG_LEN: usize = 7; // "<think>".len()
const CLOSE_TAG_LEN: usize = 8; // "</think>".len()

impl ThinkingTagParser {
    /// Create a new parser.
    ///
    /// If `strip_thinking` is true, thinking content is removed from the visible
    /// output. If false, thinking tags are preserved in the output (useful for
    /// debugging).
    pub fn new(strip_thinking: bool) -> Self {
        Self {
            buffer: String::new(),
            in_thinking_block: false,
            thinking_content: String::new(),
            visible_content: String::new(),
            block_count: 0,
            strip_thinking,
        }
    }

    /// Process a single streaming chunk.
    ///
    /// Returns the portion of this chunk that should be shown to the user.
    /// Thinking content is accumulated internally and can be retrieved via
    /// `result()` after `finalize()`.
    ///
    /// Handles partial tags at chunk boundaries by buffering — if a chunk
    /// ends with a prefix of `<think>` or `</think>`, those bytes are held
    /// until the next chunk arrives to determine if they form a complete tag.
    pub fn process_chunk(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut visible_output = String::new();

        loop {
            if self.in_thinking_block {
                if let Some(end_pos) = self.buffer.find(CLOSE_TAG) {
                    // Complete thinking block: extract content before </think>
                    let thinking_part = self.buffer[..end_pos].to_string();
                    self.thinking_content.push_str(&thinking_part);
                    self.block_count += 1;
                    self.buffer = self.buffer[end_pos + CLOSE_TAG_LEN..].to_string();
                    self.in_thinking_block = false;
                } else {
                    // No closing tag yet — flush safe content to thinking accumulator
                    let safe_len = self.buffer.len().saturating_sub(CLOSE_TAG_LEN);
                    if safe_len > 0 {
                        let safe_part = self.buffer[..safe_len].to_string();
                        self.thinking_content.push_str(&safe_part);
                        self.buffer = self.buffer[safe_len..].to_string();
                    }
                    break;
                }
            } else {
                if let Some(start_pos) = self.buffer.find(OPEN_TAG) {
                    // Found opening tag: everything before it is visible
                    let visible_part = self.buffer[..start_pos].to_string();
                    if self.strip_thinking {
                        visible_output.push_str(&visible_part);
                        self.visible_content.push_str(&visible_part);
                    } else {
                        visible_output.push_str(&self.buffer[..start_pos + OPEN_TAG_LEN]);
                        self.visible_content
                            .push_str(&self.buffer[..start_pos + OPEN_TAG_LEN]);
                    }
                    self.buffer = self.buffer[start_pos + OPEN_TAG_LEN..].to_string();
                    self.in_thinking_block = true;
                } else {
                    // No opening tag — flush safe content as visible
                    let safe_len = self.buffer.len().saturating_sub(OPEN_TAG_LEN);
                    if safe_len > 0 {
                        let safe_part = self.buffer[..safe_len].to_string();
                        visible_output.push_str(&safe_part);
                        self.visible_content.push_str(&safe_part);
                        self.buffer = self.buffer[safe_len..].to_string();
                    }
                    break;
                }
            }
        }

        visible_output
    }

    /// Finalize parsing after all chunks have been received.
    ///
    /// Flushes any remaining buffer content. If still inside an unclosed
    /// `<think>` block, the remaining content is treated as thinking content.
    /// Returns any remaining visible content.
    pub fn finalize(&mut self) -> String {
        let remaining = std::mem::take(&mut self.buffer);
        if self.in_thinking_block {
            // Unclosed thinking block — treat remainder as thinking
            self.thinking_content.push_str(&remaining);
            String::new()
        } else {
            self.visible_content.push_str(&remaining);
            remaining
        }
    }

    /// Get the full parse result after all chunks have been processed.
    ///
    /// Call `finalize()` first to ensure all buffered content is processed.
    pub fn result(&self) -> ThinkingParseResult {
        ThinkingParseResult {
            visible_response: self.visible_content.trim().to_string(),
            thinking: if self.thinking_content.is_empty() {
                None
            } else {
                Some(self.thinking_content.trim().to_string())
            },
            had_thinking_tags: self.block_count > 0,
            thinking_block_count: self.block_count,
        }
    }

    /// Returns true if the parser is currently inside a `<think>` block.
    pub fn is_in_thinking_block(&self) -> bool {
        self.in_thinking_block
    }
}

/// Parse thinking tags from a complete (non-streaming) response.
///
/// Convenience function for when you have the full response text and don't
/// need streaming support.
///
/// # Example
///
/// ```rust
/// use ai_assistant::adaptive_thinking::parse_thinking_tags;
///
/// let result = parse_thinking_tags(
///     "<think>Let me think...</think>The answer is 42."
/// );
/// assert_eq!(result.visible_response, "The answer is 42.");
/// assert_eq!(result.thinking.as_deref(), Some("Let me think..."));
/// ```
pub fn parse_thinking_tags(response: &str) -> ThinkingParseResult {
    let mut parser = ThinkingTagParser::new(true);
    parser.process_chunk(response);
    parser.finalize();
    parser.result()
}

/// Check if the given query would be classified as trivial.
///
/// Useful for quick checks without building a full `ThinkingStrategy`.
/// Checks against the built-in trivial pattern list.
pub fn is_trivial_query(query: &str) -> bool {
    let lower = query.to_lowercase().trim().to_string();
    let word_count = lower.split_whitespace().count();
    if word_count > 5 {
        return false;
    }
    TRIVIAL_PATTERNS
        .iter()
        .any(|p| lower == *p || lower.starts_with(&format!("{} ", p)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> AdaptiveThinkingConfig {
        AdaptiveThinkingConfig {
            enabled: true,
            ..Default::default()
        }
    }

    // === ThinkingDepth tests ===

    #[test]
    fn test_thinking_depth_ordering() {
        assert!(ThinkingDepth::Trivial < ThinkingDepth::Simple);
        assert!(ThinkingDepth::Simple < ThinkingDepth::Moderate);
        assert!(ThinkingDepth::Moderate < ThinkingDepth::Complex);
        assert!(ThinkingDepth::Complex < ThinkingDepth::Expert);
    }

    #[test]
    fn test_thinking_depth_numeric_level() {
        assert_eq!(ThinkingDepth::Trivial.numeric_level(), 0);
        assert_eq!(ThinkingDepth::Simple.numeric_level(), 1);
        assert_eq!(ThinkingDepth::Moderate.numeric_level(), 2);
        assert_eq!(ThinkingDepth::Complex.numeric_level(), 3);
        assert_eq!(ThinkingDepth::Expert.numeric_level(), 4);
    }

    #[test]
    fn test_thinking_depth_from_level() {
        assert_eq!(ThinkingDepth::from_level(0), ThinkingDepth::Trivial);
        assert_eq!(ThinkingDepth::from_level(2), ThinkingDepth::Moderate);
        assert_eq!(ThinkingDepth::from_level(4), ThinkingDepth::Expert);
        assert_eq!(ThinkingDepth::from_level(99), ThinkingDepth::Expert); // clamped
    }

    #[test]
    fn test_thinking_depth_upgrade_downgrade() {
        assert_eq!(ThinkingDepth::Trivial.upgrade(), ThinkingDepth::Simple);
        assert_eq!(ThinkingDepth::Moderate.upgrade(), ThinkingDepth::Complex);
        assert_eq!(ThinkingDepth::Expert.upgrade(), ThinkingDepth::Expert); // already max
        assert_eq!(ThinkingDepth::Expert.downgrade(), ThinkingDepth::Complex);
        assert_eq!(ThinkingDepth::Simple.downgrade(), ThinkingDepth::Trivial);
        assert_eq!(ThinkingDepth::Trivial.downgrade(), ThinkingDepth::Trivial); // already min
    }

    #[test]
    fn test_thinking_depth_display_name() {
        assert_eq!(ThinkingDepth::Trivial.display_name(), "Trivial");
        assert_eq!(ThinkingDepth::Expert.display_name(), "Expert");
    }

    #[test]
    fn test_thinking_depth_serde_round_trip() {
        let depth = ThinkingDepth::Complex;
        let json = serde_json::to_string(&depth).unwrap();
        assert_eq!(json, "\"complex\"");
        let deserialized: ThinkingDepth = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, depth);
    }

    // === QueryClassifier: trivial detection ===

    #[test]
    fn test_classify_trivial_greetings() {
        let classifier = QueryClassifier::new(enabled_config());
        assert_eq!(classifier.classify("hello").depth, ThinkingDepth::Trivial);
        assert_eq!(classifier.classify("hi").depth, ThinkingDepth::Trivial);
        assert_eq!(classifier.classify("hey").depth, ThinkingDepth::Trivial);
        assert_eq!(classifier.classify("bye").depth, ThinkingDepth::Trivial);
    }

    #[test]
    fn test_classify_trivial_thanks() {
        let classifier = QueryClassifier::new(enabled_config());
        assert_eq!(classifier.classify("thanks").depth, ThinkingDepth::Trivial);
        assert_eq!(
            classifier.classify("thank you").depth,
            ThinkingDepth::Trivial
        );
    }

    #[test]
    fn test_classify_trivial_confirmations() {
        let classifier = QueryClassifier::new(enabled_config());
        assert_eq!(classifier.classify("ok").depth, ThinkingDepth::Trivial);
        assert_eq!(classifier.classify("yes").depth, ThinkingDepth::Trivial);
        assert_eq!(classifier.classify("sure").depth, ThinkingDepth::Trivial);
    }

    // === QueryClassifier: simple detection ===

    #[test]
    fn test_classify_simple_factual() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("What is Rust?");
        assert!(
            strategy.depth <= ThinkingDepth::Simple,
            "Expected Simple or Trivial, got {:?}",
            strategy.depth
        );
    }

    #[test]
    fn test_classify_simple_short_question() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("Who created Linux?");
        assert!(
            strategy.depth <= ThinkingDepth::Simple,
            "Expected Simple or Trivial, got {:?}",
            strategy.depth
        );
    }

    // === QueryClassifier: moderate detection ===

    #[test]
    fn test_classify_moderate_explanation() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy =
            classifier.classify("Explain how async await works in Rust and when to use it");
        assert!(
            strategy.depth >= ThinkingDepth::Moderate,
            "Expected Moderate or higher, got {:?}",
            strategy.depth
        );
    }

    #[test]
    fn test_classify_moderate_how_to() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier
            .classify("How do I set up a Rust project with Cargo and configure dependencies?");
        assert!(
            strategy.depth >= ThinkingDepth::Moderate,
            "Expected Moderate or higher, got {:?}",
            strategy.depth
        );
    }

    // === QueryClassifier: complex detection ===

    #[test]
    fn test_classify_complex_comparison() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify(
            "Compare the performance of HashMap and BTreeMap in Rust and explain when to use each",
        );
        assert!(
            strategy.depth >= ThinkingDepth::Complex,
            "Expected Complex or Expert, got {:?}",
            strategy.depth
        );
    }

    #[test]
    fn test_classify_complex_multi_question() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify(
            "What are the differences between async runtimes? How do they handle I/O? Which is best for web servers?"
        );
        assert!(
            strategy.depth >= ThinkingDepth::Complex,
            "Expected Complex or Expert, got {:?}",
            strategy.depth
        );
    }

    // === QueryClassifier: expert detection ===

    #[test]
    fn test_classify_expert_deep_analysis() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify(
            "Provide a comprehensive analysis of the trade-offs between the actor model and CSP \
             concurrency paradigms. Compare and contrast their suitability for distributed systems \
             versus embedded systems, considering correctness, performance, and developer ergonomics."
        );
        assert_eq!(strategy.depth, ThinkingDepth::Expert);
    }

    // === QueryClassifier: min/max clamping ===

    #[test]
    fn test_classify_respects_min_depth() {
        let config = AdaptiveThinkingConfig {
            enabled: true,
            min_depth: ThinkingDepth::Moderate,
            ..Default::default()
        };
        let classifier = QueryClassifier::new(config);
        let strategy = classifier.classify("hello");
        assert!(
            strategy.depth >= ThinkingDepth::Moderate,
            "min_depth not respected: got {:?}",
            strategy.depth
        );
    }

    #[test]
    fn test_classify_respects_max_depth() {
        let config = AdaptiveThinkingConfig {
            enabled: true,
            max_depth: ThinkingDepth::Complex,
            ..Default::default()
        };
        let classifier = QueryClassifier::new(config);
        let strategy = classifier.classify(
            "Provide a comprehensive analysis of the trade-offs between multiple concurrency models \
             and critically analyze their implications for distributed and embedded systems"
        );
        assert!(
            strategy.depth <= ThinkingDepth::Complex,
            "max_depth not respected: got {:?}",
            strategy.depth
        );
    }

    // === ThinkingStrategy parameter tests ===

    #[test]
    fn test_strategy_temperature_decreases_with_depth() {
        let classifier = QueryClassifier::new(enabled_config());
        let trivial = classifier.classify("hi");
        let complex = classifier.classify(
            "Compare and contrast the performance implications of different memory allocators",
        );
        assert!(
            trivial.temperature > complex.temperature,
            "Trivial temp ({}) should be > Complex temp ({})",
            trivial.temperature,
            complex.temperature
        );
    }

    #[test]
    fn test_strategy_custom_temperature_map() {
        let mut temp_map = HashMap::new();
        temp_map.insert(ThinkingDepth::Trivial, 0.99);
        let config = AdaptiveThinkingConfig {
            enabled: true,
            temperature_map: Some(temp_map),
            ..Default::default()
        };
        let classifier = QueryClassifier::new(config);
        let strategy = classifier.classify("hi");
        assert!((strategy.temperature - 0.99).abs() < 0.01);
    }

    #[test]
    fn test_strategy_cot_empty_for_trivial() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("hi");
        assert!(
            strategy.system_prompt_addition.is_empty(),
            "Trivial should have empty CoT, got: '{}'",
            strategy.system_prompt_addition
        );
    }

    #[test]
    fn test_strategy_cot_has_content_for_complex() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier
            .classify("Compare the differences between X and Y and explain the implications");
        assert!(
            !strategy.system_prompt_addition.is_empty(),
            "Complex should have CoT instructions"
        );
        assert!(
            strategy.system_prompt_addition.contains("step by step")
                || strategy.system_prompt_addition.contains("structured"),
            "CoT should mention step-by-step reasoning"
        );
    }

    #[test]
    fn test_strategy_custom_cot_instructions() {
        let mut cot_map = HashMap::new();
        cot_map.insert(ThinkingDepth::Moderate, "Piensa paso a paso.".to_string());
        let config = AdaptiveThinkingConfig {
            enabled: true,
            cot_instructions_override: Some(cot_map),
            ..Default::default()
        };
        let classifier = QueryClassifier::new(config);
        let strategy = classifier.classify("Explain how iterators work in Rust and their benefits");
        if strategy.depth == ThinkingDepth::Moderate {
            assert_eq!(strategy.system_prompt_addition, "Piensa paso a paso.");
        }
    }

    #[test]
    fn test_strategy_rag_complexity_hint() {
        let classifier = QueryClassifier::new(enabled_config());
        let trivial = classifier.classify("hi");
        assert_eq!(trivial.rag_complexity_hint, "simple");

        let complex = classifier.classify(
            "Compare and contrast multiple approaches to garbage collection in programming languages"
        );
        assert!(
            complex.rag_complexity_hint == "complex" || complex.rag_complexity_hint == "reasoning",
            "Complex query should have complex/reasoning hint, got: {}",
            complex.rag_complexity_hint
        );
    }

    #[test]
    fn test_strategy_profile_suggestions() {
        let classifier = QueryClassifier::new(enabled_config());
        let trivial = classifier.classify("hello");
        assert_eq!(
            trivial.profile_suggestion.as_deref(),
            Some("conversational")
        );

        let expert = classifier.classify(
            "Provide a comprehensive analysis of the trade-offs between different concurrency models"
        );
        assert_eq!(expert.profile_suggestion.as_deref(), Some("detailed"));
    }

    // === ClassificationSignals tests ===

    #[test]
    fn test_signals_comparison_detection() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("What is the difference between HashMap and BTreeMap?");
        assert!(strategy.signals.has_comparison);
    }

    #[test]
    fn test_signals_analysis_detection() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("Analyze the implications of this design decision");
        assert!(strategy.signals.has_analysis);
    }

    #[test]
    fn test_signals_code_detection() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("Write a function to sort an array");
        assert!(strategy.signals.has_code);
    }

    #[test]
    fn test_signals_multi_part_detection() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify(
            "What is X and how does it relate to Y? Also explain the performance characteristics",
        );
        assert!(strategy.signals.is_multi_part);
    }

    #[test]
    fn test_signals_word_count() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("hello world");
        assert_eq!(strategy.signals.word_count, 2);
    }

    #[test]
    fn test_signals_question_marks() {
        let classifier = QueryClassifier::new(enabled_config());
        let strategy = classifier.classify("What? Why? How?");
        assert_eq!(strategy.signals.question_marks, 3);
    }

    // === ThinkingTagParser tests ===

    #[test]
    fn test_parse_simple_thinking_tags() {
        let result =
            parse_thinking_tags("<think>I need to reason about this.</think>The answer is 42.");
        assert!(result.had_thinking_tags);
        assert_eq!(result.thinking_block_count, 1);
        assert_eq!(result.visible_response, "The answer is 42.");
        assert_eq!(
            result.thinking.as_deref(),
            Some("I need to reason about this.")
        );
    }

    #[test]
    fn test_parse_no_thinking_tags() {
        let result = parse_thinking_tags("Just a normal response without any tags.");
        assert!(!result.had_thinking_tags);
        assert_eq!(result.thinking_block_count, 0);
        assert_eq!(
            result.visible_response,
            "Just a normal response without any tags."
        );
        assert!(result.thinking.is_none());
    }

    #[test]
    fn test_parse_multiple_thinking_blocks() {
        let result = parse_thinking_tags(
            "<think>First thought.</think>Part 1. <think>Second thought.</think>Part 2.",
        );
        assert_eq!(result.thinking_block_count, 2);
        assert!(result.visible_response.contains("Part 1."));
        assert!(result.visible_response.contains("Part 2."));
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("First thought."));
        assert!(thinking.contains("Second thought."));
    }

    #[test]
    fn test_parse_empty_thinking_block() {
        let result = parse_thinking_tags("<think></think>The answer.");
        assert!(result.had_thinking_tags);
        assert_eq!(result.thinking_block_count, 1);
        assert_eq!(result.visible_response, "The answer.");
        // Empty thinking block → thinking is None (trimmed empty string)
        assert!(result.thinking.is_none());
    }

    #[test]
    fn test_parse_unclosed_thinking_tag() {
        let result = parse_thinking_tags("<think>Started thinking but never closed");
        // No complete blocks
        assert!(!result.had_thinking_tags);
        assert_eq!(result.thinking_block_count, 0);
        // All content went to thinking accumulator (unclosed block)
        assert!(result.visible_response.is_empty());
    }

    #[test]
    fn test_parse_thinking_only() {
        let result = parse_thinking_tags("<think>All reasoning, no visible output.</think>");
        assert!(result.had_thinking_tags);
        assert_eq!(result.visible_response, "");
        assert_eq!(
            result.thinking.as_deref(),
            Some("All reasoning, no visible output.")
        );
    }

    #[test]
    fn test_streaming_parser_basic() {
        let mut parser = ThinkingTagParser::new(true);
        let v1 = parser.process_chunk("<think>reason</think>answer");
        let v2 = parser.finalize();
        let full = format!("{}{}", v1, v2);
        let result = parser.result();
        assert!(result.had_thinking_tags);
        assert!(full.contains("answer"));
        assert!(!full.contains("reason"));
    }

    #[test]
    fn test_streaming_parser_tag_across_chunks() {
        let mut parser = ThinkingTagParser::new(true);

        // Tag split across chunks
        let v1 = parser.process_chunk("Hello <th");
        let v2 = parser.process_chunk("ink>secret</thi");
        let v3 = parser.process_chunk("nk>visible");
        let v4 = parser.finalize();

        let result = parser.result();
        assert!(result.had_thinking_tags);
        assert_eq!(result.thinking_block_count, 1);
        assert_eq!(result.thinking.as_deref(), Some("secret"));
        let full_visible = format!("{}{}{}{}", v1, v2, v3, v4);
        assert!(full_visible.contains("Hello"));
        assert!(full_visible.contains("visible"));
        assert!(!full_visible.contains("secret"));
    }

    #[test]
    fn test_streaming_parser_no_tags() {
        let mut parser = ThinkingTagParser::new(true);
        let v1 = parser.process_chunk("Just a ");
        let v2 = parser.process_chunk("normal response.");
        let v3 = parser.finalize();
        let full = format!("{}{}{}", v1, v2, v3);
        assert_eq!(full, "Just a normal response.");
        assert!(!parser.result().had_thinking_tags);
    }

    #[test]
    fn test_streaming_parser_preserves_tags_when_not_stripping() {
        let mut parser = ThinkingTagParser::new(false);
        let v1 = parser.process_chunk("<think>reason</think>answer");
        let v2 = parser.finalize();
        let full = format!("{}{}", v1, v2);
        // When not stripping, the <think> tag itself should appear in visible output
        assert!(full.contains("<think>"));
        assert!(full.contains("answer"));
    }

    #[test]
    fn test_streaming_parser_is_in_thinking_block() {
        let mut parser = ThinkingTagParser::new(true);
        assert!(!parser.is_in_thinking_block());
        parser.process_chunk("<think>start");
        assert!(parser.is_in_thinking_block());
        parser.process_chunk("</think>done");
        assert!(!parser.is_in_thinking_block());
    }

    #[test]
    fn test_parse_with_angle_brackets_in_content() {
        // Ensure that angle brackets that aren't think tags don't confuse the parser
        let result = parse_thinking_tags("Use Vec<String> for the collection.");
        assert!(!result.had_thinking_tags);
        assert_eq!(
            result.visible_response,
            "Use Vec<String> for the collection."
        );
    }

    // === Config tests ===

    #[test]
    fn test_default_config_is_disabled() {
        let config = AdaptiveThinkingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.min_depth, ThinkingDepth::Trivial);
        assert_eq!(config.max_depth, ThinkingDepth::Expert);
        assert_eq!(config.rag_tier_priority, RagTierPriority::Adaptive);
        assert!(config.transparent_thinking_parse);
    }

    #[test]
    fn test_config_serde_round_trip() {
        let config = AdaptiveThinkingConfig {
            enabled: true,
            min_depth: ThinkingDepth::Simple,
            max_depth: ThinkingDepth::Complex,
            rag_tier_priority: RagTierPriority::Highest,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptiveThinkingConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.enabled);
        assert_eq!(deserialized.min_depth, ThinkingDepth::Simple);
        assert_eq!(deserialized.max_depth, ThinkingDepth::Complex);
        assert_eq!(deserialized.rag_tier_priority, RagTierPriority::Highest);
    }

    // === RagTierPriority tests ===

    #[test]
    fn test_rag_tier_priority_serde() {
        let p = RagTierPriority::Adaptive;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"adaptive\"");
        let deser: RagTierPriority = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, p);
    }

    // === is_trivial_query tests ===

    #[test]
    fn test_is_trivial_query() {
        assert!(is_trivial_query("hello"));
        assert!(is_trivial_query("thanks"));
        assert!(is_trivial_query("hola"));
        assert!(is_trivial_query("ok"));
        assert!(!is_trivial_query("How do I configure a Rust project?"));
        assert!(!is_trivial_query("Compare HashMap vs BTreeMap performance"));
    }

    // === classify_with_context tests ===

    #[test]
    fn test_classify_with_context_upgrades_short_query_in_long_conversation() {
        let classifier = QueryClassifier::new(enabled_config());

        // "ok" in a short conversation → Trivial
        let strategy = classifier.classify_with_context("ok", 2, &[]);
        assert!(strategy.depth <= ThinkingDepth::Moderate);

        // "ok" in a long conversation with many topics → at least Moderate
        let topics = vec!["rust".into(), "concurrency".into(), "performance".into()];
        let strategy = classifier.classify_with_context("yes", 15, &topics);
        assert!(
            strategy.depth >= ThinkingDepth::Moderate,
            "Long conversation context should upgrade depth, got {:?}",
            strategy.depth
        );
    }

    // === Confidence tests ===

    #[test]
    fn test_confidence_is_between_zero_and_one() {
        let classifier = QueryClassifier::new(enabled_config());
        for query in &[
            "hi",
            "What is Rust?",
            "Compare X vs Y in detail",
            "Analyze the implications",
        ] {
            let strategy = classifier.classify(query);
            assert!(
                strategy.confidence >= 0.0 && strategy.confidence <= 1.0,
                "Confidence {} out of range for query '{}'",
                strategy.confidence,
                query
            );
        }
    }
}
