//! Conversation analysis: sentiment, topics, auto-summarization

use crate::messages::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Sentiment Analysis
// ============================================================================

/// Sentiment of a message or conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Sentiment {
    VeryPositive,
    Positive,
    Neutral,
    Negative,
    VeryNegative,
}

impl Sentiment {
    /// Get numeric score (-1.0 to 1.0)
    pub fn score(&self) -> f32 {
        match self {
            Self::VeryPositive => 1.0,
            Self::Positive => 0.5,
            Self::Neutral => 0.0,
            Self::Negative => -0.5,
            Self::VeryNegative => -1.0,
        }
    }

    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::VeryPositive => "😄",
            Self::Positive => "🙂",
            Self::Neutral => "😐",
            Self::Negative => "😕",
            Self::VeryNegative => "😞",
        }
    }

    /// Create from numeric score
    pub fn from_score(score: f32) -> Self {
        if score >= 0.6 {
            Self::VeryPositive
        } else if score >= 0.2 {
            Self::Positive
        } else if score > -0.2 {
            Self::Neutral
        } else if score > -0.6 {
            Self::Negative
        } else {
            Self::VeryNegative
        }
    }
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VeryPositive => write!(f, "Very Positive"),
            Self::Positive => write!(f, "Positive"),
            Self::Neutral => write!(f, "Neutral"),
            Self::Negative => write!(f, "Negative"),
            Self::VeryNegative => write!(f, "Very Negative"),
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Overall sentiment
    pub sentiment: Sentiment,
    /// Raw score (-1.0 to 1.0)
    pub score: f32,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Detected emotions
    pub emotions: Vec<(String, f32)>,
    /// Key positive indicators found
    pub positive_indicators: Vec<String>,
    /// Key negative indicators found
    pub negative_indicators: Vec<String>,
}

/// Analyze sentiment using keyword-based approach (no AI required)
pub struct SentimentAnalyzer {
    positive_words: HashMap<&'static str, f32>,
    negative_words: HashMap<&'static str, f32>,
    intensifiers: HashMap<&'static str, f32>,
    negators: Vec<&'static str>,
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let mut positive = HashMap::new();
        let mut negative = HashMap::new();

        // Positive words with weights
        for (word, weight) in [
            ("great", 0.8),
            ("excellent", 0.9),
            ("amazing", 0.9),
            ("wonderful", 0.85),
            ("fantastic", 0.9),
            ("perfect", 1.0),
            ("good", 0.6),
            ("nice", 0.5),
            ("helpful", 0.7),
            ("thanks", 0.6),
            ("thank", 0.6),
            ("love", 0.85),
            ("awesome", 0.85),
            ("brilliant", 0.9),
            ("outstanding", 0.9),
            ("happy", 0.7),
            ("pleased", 0.7),
            ("satisfied", 0.7),
            ("enjoy", 0.7),
            ("beautiful", 0.75),
            ("impressive", 0.8),
            ("useful", 0.65),
            ("clear", 0.5),
            ("easy", 0.5),
            ("fast", 0.5),
            ("efficient", 0.6),
            ("works", 0.4),
            ("working", 0.4),
            ("solved", 0.7),
            ("fixed", 0.7),
            ("genial", 0.8),
            ("excelente", 0.9),
            ("perfecto", 1.0),
            ("gracias", 0.6),
            ("bueno", 0.6),
            ("bien", 0.5),
            ("increíble", 0.9),
            ("maravilloso", 0.85),
        ] {
            positive.insert(word, weight);
        }

        // Negative words with weights
        for (word, weight) in [
            ("bad", -0.7),
            ("terrible", -0.9),
            ("awful", -0.9),
            ("horrible", -0.9),
            ("poor", -0.6),
            ("wrong", -0.6),
            ("error", -0.5),
            ("fail", -0.7),
            ("failed", -0.7),
            ("broken", -0.8),
            ("bug", -0.5),
            ("issue", -0.4),
            ("problem", -0.5),
            ("difficult", -0.4),
            ("hard", -0.3),
            ("confusing", -0.5),
            ("confused", -0.4),
            ("frustrated", -0.7),
            ("annoying", -0.6),
            ("annoyed", -0.6),
            ("disappointing", -0.7),
            ("disappointed", -0.7),
            ("useless", -0.8),
            ("slow", -0.4),
            ("crash", -0.8),
            ("crashes", -0.8),
            ("stuck", -0.5),
            ("hate", -0.9),
            ("worst", -1.0),
            ("never", -0.3),
            ("can't", -0.3),
            ("cannot", -0.3),
            ("doesn't", -0.3),
            ("doesn't work", -0.7),
            ("malo", -0.7),
            ("terrible", -0.9),
            ("error", -0.5),
            ("falla", -0.7),
            ("problema", -0.5),
            ("difícil", -0.4),
            ("confuso", -0.5),
        ] {
            negative.insert(word, weight);
        }

        let mut intensifiers = HashMap::new();
        for (word, mult) in [
            ("very", 1.5),
            ("really", 1.4),
            ("extremely", 1.8),
            ("incredibly", 1.7),
            ("absolutely", 1.6),
            ("completely", 1.5),
            ("totally", 1.5),
            ("quite", 1.2),
            ("fairly", 1.1),
            ("somewhat", 0.8),
            ("slightly", 0.6),
            ("much", 1.3),
            ("so", 1.4),
            ("too", 1.3),
            ("muy", 1.5),
            ("realmente", 1.4),
            ("extremadamente", 1.8),
        ] {
            intensifiers.insert(word, mult);
        }

        let negators = vec![
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "couldn't",
            "shouldn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "no",
            "nunca",
            "ni",
            "nadie",
            "nada",
        ];

        Self {
            positive_words: positive,
            negative_words: negative,
            intensifiers,
            negators,
        }
    }

    /// Analyze sentiment of a single message
    pub fn analyze_message(&self, text: &str) -> SentimentAnalysis {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        let mut score = 0.0;
        let mut positive_found = Vec::new();
        let mut negative_found = Vec::new();
        let mut word_count = 0;

        let mut i = 0;
        while i < words.len() {
            let word = words[i];

            // Check for intensifier
            let intensifier = self.intensifiers.get(word).copied().unwrap_or(1.0);

            // Check for negation in previous words
            let negated = if i > 0 {
                (1..=3).any(|j| i >= j && self.negators.contains(&words[i - j]))
            } else {
                false
            };

            // Check positive words
            if let Some(&weight) = self.positive_words.get(word) {
                let adjusted = if negated { -weight * 0.5 } else { weight } * intensifier;
                score += adjusted;
                word_count += 1;
                if adjusted > 0.0 {
                    positive_found.push(word.to_string());
                } else {
                    negative_found.push(format!("not {}", word));
                }
            }

            // Check negative words
            if let Some(&weight) = self.negative_words.get(word) {
                let adjusted = if negated { -weight * 0.5 } else { weight } * intensifier;
                score += adjusted;
                word_count += 1;
                if adjusted < 0.0 {
                    negative_found.push(word.to_string());
                } else {
                    positive_found.push(format!("not {}", word));
                }
            }

            i += 1;
        }

        // Normalize score
        let normalized = if word_count > 0 {
            (score / word_count as f32).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Calculate confidence based on evidence
        let confidence = if word_count == 0 {
            0.3 // Low confidence for neutral with no evidence
        } else {
            (0.5 + (word_count as f32 * 0.1)).min(0.95)
        };

        // Detect emotions
        let mut emotions = Vec::new();
        let text_lower = text.to_lowercase();

        if text_lower.contains("thank") || text_lower.contains("gracias") {
            emotions.push(("gratitude".to_string(), 0.8));
        }
        if text_lower.contains("help") || text_lower.contains("ayuda") {
            emotions.push(("seeking_help".to_string(), 0.7));
        }
        if text_lower.contains("frustrat") || text_lower.contains("frustrad") {
            emotions.push(("frustration".to_string(), 0.85));
        }
        if text_lower.contains("confus") {
            emotions.push(("confusion".to_string(), 0.75));
        }
        if text_lower.contains("excit") || text_lower.contains("emocion") {
            emotions.push(("excitement".to_string(), 0.8));
        }
        if text_lower.contains("curious") || text_lower.contains("curios") {
            emotions.push(("curiosity".to_string(), 0.7));
        }

        SentimentAnalysis {
            sentiment: Sentiment::from_score(normalized),
            score: normalized,
            confidence,
            emotions,
            positive_indicators: positive_found,
            negative_indicators: negative_found,
        }
    }

    /// Analyze sentiment of entire conversation
    pub fn analyze_conversation(&self, messages: &[ChatMessage]) -> ConversationSentimentAnalysis {
        if messages.is_empty() {
            return ConversationSentimentAnalysis {
                overall: SentimentAnalysis {
                    sentiment: Sentiment::Neutral,
                    score: 0.0,
                    confidence: 0.0,
                    emotions: Vec::new(),
                    positive_indicators: Vec::new(),
                    negative_indicators: Vec::new(),
                },
                user_sentiment: SentimentAnalysis {
                    sentiment: Sentiment::Neutral,
                    score: 0.0,
                    confidence: 0.0,
                    emotions: Vec::new(),
                    positive_indicators: Vec::new(),
                    negative_indicators: Vec::new(),
                },
                trend: SentimentTrend::Stable,
                message_sentiments: Vec::new(),
            };
        }

        let mut all_scores = Vec::new();
        let mut user_scores = Vec::new();
        let mut message_sentiments = Vec::new();

        for msg in messages {
            let analysis = self.analyze_message(&msg.content);
            all_scores.push(analysis.score);

            if msg.role == "user" {
                user_scores.push(analysis.score);
            }

            message_sentiments.push((msg.role.clone(), analysis));
        }

        // Calculate overall
        let overall_score = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        let user_score = if user_scores.is_empty() {
            0.0
        } else {
            user_scores.iter().sum::<f32>() / user_scores.len() as f32
        };

        // Determine trend
        let trend = if user_scores.len() >= 2 {
            let first_half: f32 = user_scores[..user_scores.len() / 2].iter().sum::<f32>()
                / (user_scores.len() / 2) as f32;
            let second_half: f32 = user_scores[user_scores.len() / 2..].iter().sum::<f32>()
                / (user_scores.len() - user_scores.len() / 2) as f32;

            let diff = second_half - first_half;
            if diff > 0.2 {
                SentimentTrend::Improving
            } else if diff < -0.2 {
                SentimentTrend::Declining
            } else {
                SentimentTrend::Stable
            }
        } else {
            SentimentTrend::Stable
        };

        ConversationSentimentAnalysis {
            overall: SentimentAnalysis {
                sentiment: Sentiment::from_score(overall_score),
                score: overall_score,
                confidence: 0.7,
                emotions: Vec::new(),
                positive_indicators: Vec::new(),
                negative_indicators: Vec::new(),
            },
            user_sentiment: SentimentAnalysis {
                sentiment: Sentiment::from_score(user_score),
                score: user_score,
                confidence: 0.7,
                emotions: Vec::new(),
                positive_indicators: Vec::new(),
                negative_indicators: Vec::new(),
            },
            trend,
            message_sentiments,
        }
    }
}

/// Sentiment trend over conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SentimentTrend {
    Improving,
    Stable,
    Declining,
}

impl std::fmt::Display for SentimentTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Improving => write!(f, "Improving"),
            Self::Stable => write!(f, "Stable"),
            Self::Declining => write!(f, "Declining"),
        }
    }
}

/// Full conversation sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSentimentAnalysis {
    pub overall: SentimentAnalysis,
    pub user_sentiment: SentimentAnalysis,
    pub trend: SentimentTrend,
    pub message_sentiments: Vec<(String, SentimentAnalysis)>,
}

// ============================================================================
// Topic Detection
// ============================================================================

/// A detected topic in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Topic name/label
    pub name: String,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
    /// Keywords associated with this topic
    pub keywords: Vec<String>,
    /// Message indices where this topic appears
    pub message_indices: Vec<usize>,
}

/// Topic detector using keyword extraction
pub struct TopicDetector {
    /// Domain-specific topic definitions
    topic_keywords: HashMap<String, Vec<&'static str>>,
    /// Minimum keyword matches to consider a topic
    min_matches: usize,
}

impl Default for TopicDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicDetector {
    pub fn new() -> Self {
        let mut topics = HashMap::new();

        // General technical topics
        topics.insert(
            "programming".to_string(),
            vec![
                "code",
                "function",
                "variable",
                "bug",
                "error",
                "compile",
                "runtime",
                "debug",
                "test",
                "class",
                "method",
                "api",
                "library",
                "framework",
                "código",
                "función",
                "variable",
                "error",
                "compilar",
            ],
        );

        topics.insert(
            "help_request".to_string(),
            vec![
                "help",
                "how",
                "why",
                "what",
                "can you",
                "could you",
                "please",
                "explain",
                "show",
                "tell",
                "ayuda",
                "cómo",
                "por qué",
                "qué",
            ],
        );

        topics.insert(
            "error_troubleshooting".to_string(),
            vec![
                "error",
                "crash",
                "fail",
                "broken",
                "issue",
                "problem",
                "bug",
                "not working",
                "doesn't work",
                "won't",
                "can't",
                "error",
                "falla",
                "problema",
                "no funciona",
            ],
        );

        topics.insert(
            "configuration".to_string(),
            vec![
                "config",
                "setting",
                "setup",
                "install",
                "configure",
                "option",
                "preference",
                "enable",
                "disable",
                "path",
                "file",
                "configuración",
                "ajuste",
                "instalar",
                "opción",
            ],
        );

        topics.insert(
            "performance".to_string(),
            vec![
                "slow",
                "fast",
                "speed",
                "performance",
                "optimize",
                "memory",
                "cpu",
                "lag",
                "freeze",
                "efficient",
                "lento",
                "rápido",
                "rendimiento",
                "optimizar",
                "memoria",
            ],
        );

        topics.insert(
            "feature_request".to_string(),
            vec![
                "would be nice",
                "could you add",
                "feature",
                "suggestion",
                "idea",
                "want",
                "need",
                "wish",
                "hope",
                "request",
                "sería bueno",
                "podrías añadir",
                "característica",
                "sugerencia",
            ],
        );

        topics.insert(
            "gratitude".to_string(),
            vec![
                "thank",
                "thanks",
                "appreciate",
                "helpful",
                "great",
                "awesome",
                "perfect",
                "exactly",
                "solved",
                "gracias",
                "agradezco",
                "útil",
                "genial",
                "perfecto",
            ],
        );

        // Star Citizen specific topics
        topics.insert(
            "ships".to_string(),
            vec![
                "ship",
                "vehicle",
                "fighter",
                "bomber",
                "cargo",
                "mining",
                "exploration",
                "nave",
                "vehículo",
                "caza",
                "bombardero",
                "carga",
                "minería",
            ],
        );

        topics.insert(
            "localization".to_string(),
            vec![
                "translation",
                "translate",
                "language",
                "localization",
                "text",
                "traducción",
                "traducir",
                "idioma",
                "localización",
                "texto",
            ],
        );

        Self {
            topic_keywords: topics,
            min_matches: 2,
        }
    }

    /// Add a custom topic
    pub fn add_topic(&mut self, name: &str, keywords: Vec<&'static str>) {
        self.topic_keywords.insert(name.to_string(), keywords);
    }

    /// Remove a topic
    pub fn remove_topic(&mut self, name: &str) {
        self.topic_keywords.remove(name);
    }

    /// Detect topics in a conversation
    pub fn detect_topics(&self, messages: &[ChatMessage]) -> Vec<Topic> {
        let mut topic_matches: HashMap<String, (usize, Vec<String>, Vec<usize>)> = HashMap::new();

        for (idx, msg) in messages.iter().enumerate() {
            let content_lower = msg.content.to_lowercase();
            let words: Vec<&str> = content_lower
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() > 2)
                .collect();

            for (topic_name, keywords) in &self.topic_keywords {
                let mut matched_keywords = Vec::new();

                for keyword in keywords {
                    // Check for exact word match or phrase match
                    if words.contains(keyword) || content_lower.contains(keyword) {
                        matched_keywords.push(keyword.to_string());
                    }
                }

                if !matched_keywords.is_empty() {
                    let entry = topic_matches.entry(topic_name.clone()).or_insert((
                        0,
                        Vec::new(),
                        Vec::new(),
                    ));
                    entry.0 += matched_keywords.len();
                    for kw in matched_keywords {
                        if !entry.1.contains(&kw) {
                            entry.1.push(kw);
                        }
                    }
                    if !entry.2.contains(&idx) {
                        entry.2.push(idx);
                    }
                }
            }
        }

        // Convert to Topic structs
        let mut topics: Vec<Topic> = topic_matches
            .into_iter()
            .filter(|(_, (count, _, _))| *count >= self.min_matches)
            .map(|(name, (count, keywords, indices))| {
                let relevance = (count as f32 / (messages.len() as f32 * 2.0)).min(1.0);
                Topic {
                    name,
                    relevance,
                    keywords,
                    message_indices: indices,
                }
            })
            .collect();

        // Sort by relevance
        topics.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        topics
    }

    /// Get main topic of conversation
    pub fn get_main_topic(&self, messages: &[ChatMessage]) -> Option<Topic> {
        self.detect_topics(messages).into_iter().next()
    }

    /// Extract key terms from text (simple TF approach)
    pub fn extract_key_terms(&self, text: &str, top_n: usize) -> Vec<(String, usize)> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        // Common stop words to ignore
        let stop_words: std::collections::HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "until", "while", "el", "la", "los", "las", "un", "una",
            "de", "en", "con", "por", "para", "que", "es", "son", "está", "están", "como", "más",
            "pero", "this", "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
            "they", "my", "your", "his", "her", "our", "their",
        ]
        .iter()
        .copied()
        .collect();

        for word in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !stop_words.contains(w))
        {
            *word_counts.entry(word.to_string()).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = word_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);
        sorted
    }
}

// ============================================================================
// Auto-Summarization
// ============================================================================

/// Configuration for auto-summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SummaryConfig {
    /// Enable automatic summarization
    pub enabled: bool,
    /// Trigger summary after this many messages
    pub trigger_message_count: usize,
    /// Maximum tokens for summary
    pub max_summary_tokens: usize,
    /// Include topic analysis in summary
    pub include_topics: bool,
    /// Include sentiment analysis in summary
    pub include_sentiment: bool,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_message_count: 10,
            max_summary_tokens: 500,
            include_topics: true,
            include_sentiment: true,
        }
    }
}

/// Auto-generated session summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Brief summary text
    pub summary: String,
    /// Key points discussed
    pub key_points: Vec<String>,
    /// Main topics
    pub topics: Vec<String>,
    /// Questions asked by user
    pub user_questions: Vec<String>,
    /// Solutions/answers provided
    pub solutions_provided: Vec<String>,
    /// Overall sentiment
    pub sentiment: Sentiment,
    /// Number of messages summarized
    pub message_count: usize,
    /// When the summary was generated
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Session summarizer
pub struct SessionSummarizer {
    config: SummaryConfig,
    sentiment_analyzer: SentimentAnalyzer,
    topic_detector: TopicDetector,
}

impl SessionSummarizer {
    pub fn new(config: SummaryConfig) -> Self {
        Self {
            config,
            sentiment_analyzer: SentimentAnalyzer::new(),
            topic_detector: TopicDetector::new(),
        }
    }

    /// Generate a summary from messages (without AI - rule-based)
    pub fn summarize(&self, messages: &[ChatMessage]) -> SessionSummary {
        if messages.is_empty() {
            return SessionSummary {
                summary: "No messages in session.".to_string(),
                key_points: Vec::new(),
                topics: Vec::new(),
                user_questions: Vec::new(),
                solutions_provided: Vec::new(),
                sentiment: Sentiment::Neutral,
                message_count: 0,
                generated_at: chrono::Utc::now(),
            };
        }

        // Extract user questions
        let user_questions: Vec<String> = messages
            .iter()
            .filter(|m| m.role == "user")
            .filter(|m| {
                m.content.contains('?')
                    || m.content.to_lowercase().starts_with("how")
                    || m.content.to_lowercase().starts_with("what")
                    || m.content.to_lowercase().starts_with("why")
                    || m.content.to_lowercase().starts_with("cómo")
                    || m.content.to_lowercase().starts_with("qué")
                    || m.content.to_lowercase().starts_with("por qué")
            })
            .map(|m| {
                // Take first sentence or first 100 chars
                let content = &m.content;
                content
                    .split(|c| c == '?' || c == '.')
                    .next()
                    .map(|s| {
                        if s.len() > 100 {
                            format!("{}...", &s[..100])
                        } else {
                            s.to_string()
                        }
                    })
                    .unwrap_or_else(|| content.chars().take(100).collect())
            })
            .take(5)
            .collect();

        // Detect topics
        let topics = self.topic_detector.detect_topics(messages);
        let topic_names: Vec<String> = topics.iter().take(3).map(|t| t.name.clone()).collect();

        // Analyze sentiment
        let sentiment_analysis = self.sentiment_analyzer.analyze_conversation(messages);

        // Extract key terms for key points
        let all_text: String = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let key_terms = self.topic_detector.extract_key_terms(&all_text, 10);
        let key_points: Vec<String> = key_terms
            .iter()
            .take(5)
            .map(|(term, count)| format!("{} (mentioned {} times)", term, count))
            .collect();

        // Generate summary text
        let summary = self.generate_summary_text(messages, &topic_names, &sentiment_analysis);

        // Extract solutions (assistant messages that seem conclusive)
        let solutions: Vec<String> = messages
            .iter()
            .filter(|m| m.role == "assistant")
            .filter(|m| {
                let lower = m.content.to_lowercase();
                lower.contains("you can")
                    || lower.contains("to do this")
                    || lower.contains("the solution")
                    || lower.contains("here's how")
                    || lower.contains("puedes")
                    || lower.contains("la solución")
            })
            .map(|m| m.content.chars().take(150).collect::<String>())
            .take(3)
            .collect();

        SessionSummary {
            summary,
            key_points,
            topics: topic_names,
            user_questions,
            solutions_provided: solutions,
            sentiment: sentiment_analysis.overall.sentiment,
            message_count: messages.len(),
            generated_at: chrono::Utc::now(),
        }
    }

    fn generate_summary_text(
        &self,
        messages: &[ChatMessage],
        topics: &[String],
        sentiment: &ConversationSentimentAnalysis,
    ) -> String {
        let user_count = messages.iter().filter(|m| m.role == "user").count();
        let assistant_count = messages.iter().filter(|m| m.role == "assistant").count();

        let topic_str = if topics.is_empty() {
            "general discussion".to_string()
        } else {
            topics.join(", ")
        };

        let sentiment_str = match sentiment.overall.sentiment {
            Sentiment::VeryPositive | Sentiment::Positive => "positive",
            Sentiment::Neutral => "neutral",
            Sentiment::Negative | Sentiment::VeryNegative => "challenging",
        };

        let trend_str = match sentiment.trend {
            SentimentTrend::Improving => " with improving engagement",
            SentimentTrend::Declining => " with some difficulties",
            SentimentTrend::Stable => "",
        };

        format!(
            "Session with {} user messages and {} assistant responses. \
            Main topics: {}. Overall tone: {}{}.",
            user_count, assistant_count, topic_str, sentiment_str, trend_str
        )
    }

    /// Check if summarization should be triggered
    pub fn should_summarize(&self, message_count: usize) -> bool {
        self.config.enabled && message_count >= self.config.trigger_message_count
    }
}

// ============================================================================
// Emoticon & Emoji Detection, Classification, and Sentiment
// ============================================================================

/// Emotion category for emoticons and emoji
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EmojiCategory {
    Happy,
    Sad,
    Angry,
    Surprised,
    Love,
    Laughing,
    Thinking,
    Winking,
    Cool,
    Fearful,
    Playful,
    Neutral,
    Custom(String),
}

impl std::fmt::Display for EmojiCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Happy => write!(f, "Happy"),
            Self::Sad => write!(f, "Sad"),
            Self::Angry => write!(f, "Angry"),
            Self::Surprised => write!(f, "Surprised"),
            Self::Love => write!(f, "Love"),
            Self::Laughing => write!(f, "Laughing"),
            Self::Thinking => write!(f, "Thinking"),
            Self::Winking => write!(f, "Winking"),
            Self::Cool => write!(f, "Cool"),
            Self::Fearful => write!(f, "Fearful"),
            Self::Playful => write!(f, "Playful"),
            Self::Neutral => write!(f, "Neutral"),
            Self::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl EmojiCategory {
    /// Sentiment score for this category (-1.0 to 1.0)
    pub fn sentiment_score(&self) -> f32 {
        match self {
            Self::Happy => 0.7,
            Self::Sad => -0.7,
            Self::Angry => -0.8,
            Self::Surprised => 0.1,
            Self::Love => 0.9,
            Self::Laughing => 0.8,
            Self::Thinking => -0.1,
            Self::Winking => 0.5,
            Self::Cool => 0.6,
            Self::Fearful => -0.6,
            Self::Playful => 0.4,
            Self::Neutral => 0.0,
            Self::Custom(_) => 0.0,
        }
    }

    /// Representative emoji for this category
    pub fn emoji(&self) -> &str {
        match self {
            Self::Happy => "😊",
            Self::Sad => "😢",
            Self::Angry => "😠",
            Self::Surprised => "😮",
            Self::Love => "❤️",
            Self::Laughing => "😂",
            Self::Thinking => "🤔",
            Self::Winking => "😉",
            Self::Cool => "😎",
            Self::Fearful => "😰",
            Self::Playful => "😜",
            Self::Neutral => "😐",
            Self::Custom(_) => "❓",
        }
    }
}

/// A detected emoticon or emoji in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmoticonMatch {
    /// The matched text (e.g. ":)" or "😊")
    pub original: String,
    /// Unicode emoji equivalent
    pub emoji: String,
    /// Emotion classification
    pub category: EmojiCategory,
    /// Sentiment score for this match (-1.0 to 1.0)
    pub sentiment_score: f32,
    /// Byte offset in original text
    pub byte_offset: usize,
    /// true if Unicode emoji, false if text emoticon
    pub is_unicode: bool,
}

/// Aggregate result of emoticon/emoji analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmoticonAnalysis {
    /// All detected matches
    pub matches: Vec<EmoticonMatch>,
    /// Weighted average sentiment from emoticons/emoji
    pub overall_sentiment: f32,
    /// Most frequent category (None if no matches)
    pub dominant_category: Option<EmojiCategory>,
    /// Original text with text emoticons replaced by Unicode emoji
    pub converted_text: String,
    /// Count per category
    pub category_counts: HashMap<String, usize>,
}

/// Detects, classifies, and converts emoticons and Unicode emoji
pub struct EmoticonDetector {
    /// (text_emoticon, emoji_replacement, category) — sorted by length desc
    emoticons: Vec<(&'static str, &'static str, EmojiCategory)>,
    /// (unicode_char, emoji_str, category)
    emoji_categories: Vec<(char, &'static str, EmojiCategory)>,
}

impl Default for EmoticonDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EmoticonDetector {
    /// Create with built-in emoticon and emoji databases
    pub fn new() -> Self {
        // Text emoticons — order matters: longer patterns first for correct matching
        let mut emoticons: Vec<(&str, &str, EmojiCategory)> = vec![
            // Happy (longer first)
            (":-)", "😊", EmojiCategory::Happy),
            (":)", "😊", EmojiCategory::Happy),
            ("=)", "😊", EmojiCategory::Happy),
            ("(:", "😊", EmojiCategory::Happy),
            ("^_^", "😊", EmojiCategory::Happy),
            ("^^", "😊", EmojiCategory::Happy),
            // Sad
            (":-(", "😢", EmojiCategory::Sad),
            (":(", "😢", EmojiCategory::Sad),
            (":'(", "😢", EmojiCategory::Sad),
            ("T_T", "😢", EmojiCategory::Sad),
            ("T.T", "😢", EmojiCategory::Sad),
            (";_;", "😢", EmojiCategory::Sad),
            ("QQ", "😢", EmojiCategory::Sad),
            // Angry
            (">:-(", "😠", EmojiCategory::Angry),
            (">:(", "😠", EmojiCategory::Angry),
            ("D:<", "😠", EmojiCategory::Angry),
            (">.<", "😠", EmojiCategory::Angry),
            // Surprised
            (":-O", "😮", EmojiCategory::Surprised),
            (":O", "😮", EmojiCategory::Surprised),
            (":o", "😮", EmojiCategory::Surprised),
            ("O_O", "😮", EmojiCategory::Surprised),
            ("o_O", "😮", EmojiCategory::Surprised),
            ("O.O", "😮", EmojiCategory::Surprised),
            // Love
            ("<33", "❤️", EmojiCategory::Love),
            ("<3", "❤️", EmojiCategory::Love),
            // Laughing
            (":-D", "😄", EmojiCategory::Laughing),
            (":D", "😄", EmojiCategory::Laughing),
            (":')", "😂", EmojiCategory::Laughing),
            ("XD", "😂", EmojiCategory::Laughing),
            ("xD", "😂", EmojiCategory::Laughing),
            ("xd", "😂", EmojiCategory::Laughing),
            // Thinking
            (":-/", "🤔", EmojiCategory::Thinking),
            (":/", "🤔", EmojiCategory::Thinking),
            (":-\\", "🤔", EmojiCategory::Thinking),
            (":\\", "🤔", EmojiCategory::Thinking),
            // Winking
            (";-)", "😉", EmojiCategory::Winking),
            (";)", "😉", EmojiCategory::Winking),
            (";D", "😉", EmojiCategory::Winking),
            // Cool
            ("B-)", "😎", EmojiCategory::Cool),
            ("B)", "😎", EmojiCategory::Cool),
            ("8)", "😎", EmojiCategory::Cool),
            // Fearful
            ("D;", "😰", EmojiCategory::Fearful),
            ("D:", "😰", EmojiCategory::Fearful),
            (":-S", "😰", EmojiCategory::Fearful),
            (":S", "😰", EmojiCategory::Fearful),
            // Playful
            (":-P", "😜", EmojiCategory::Playful),
            (":P", "😜", EmojiCategory::Playful),
            (":-p", "😜", EmojiCategory::Playful),
            (":p", "😜", EmojiCategory::Playful),
            ("xP", "😜", EmojiCategory::Playful),
            // Neutral
            (":-|", "😐", EmojiCategory::Neutral),
            (":|", "😐", EmojiCategory::Neutral),
            ("-_-", "😐", EmojiCategory::Neutral),
            ("._.", "😐", EmojiCategory::Neutral),
        ];

        // Sort by length descending for longest-match-first
        emoticons.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Unicode emoji → category
        let emoji_categories = vec![
            // Happy
            ('😀', "😀", EmojiCategory::Happy),
            ('😃', "😃", EmojiCategory::Happy),
            ('😄', "😄", EmojiCategory::Happy),
            ('😁', "😁", EmojiCategory::Happy),
            ('😊', "😊", EmojiCategory::Happy),
            ('🙂', "🙂", EmojiCategory::Happy),
            ('😇', "😇", EmojiCategory::Happy),
            ('🤗', "🤗", EmojiCategory::Happy),
            ('🥳', "🥳", EmojiCategory::Happy),
            ('👍', "👍", EmojiCategory::Happy),
            ('✅', "✅", EmojiCategory::Happy),
            ('🎉', "🎉", EmojiCategory::Happy),
            // Sad
            ('😢', "😢", EmojiCategory::Sad),
            ('😭', "😭", EmojiCategory::Sad),
            ('😥', "😥", EmojiCategory::Sad),
            ('😿', "😿", EmojiCategory::Sad),
            ('💔', "💔", EmojiCategory::Sad),
            ('😞', "😞", EmojiCategory::Sad),
            ('😔', "😔", EmojiCategory::Sad),
            ('👎', "👎", EmojiCategory::Sad),
            // Angry
            ('😠', "😠", EmojiCategory::Angry),
            ('😡', "😡", EmojiCategory::Angry),
            ('🤬', "🤬", EmojiCategory::Angry),
            ('👿', "👿", EmojiCategory::Angry),
            ('💢', "💢", EmojiCategory::Angry),
            // Surprised
            ('😮', "😮", EmojiCategory::Surprised),
            ('😲', "😲", EmojiCategory::Surprised),
            ('🤯', "🤯", EmojiCategory::Surprised),
            ('😱', "😱", EmojiCategory::Surprised),
            ('😵', "😵", EmojiCategory::Surprised),
            // Love
            ('❤', "❤️", EmojiCategory::Love),
            ('💕', "💕", EmojiCategory::Love),
            ('💖', "💖", EmojiCategory::Love),
            ('😍', "😍", EmojiCategory::Love),
            ('🥰', "🥰", EmojiCategory::Love),
            ('💘', "💘", EmojiCategory::Love),
            ('💗', "💗", EmojiCategory::Love),
            ('💝', "💝", EmojiCategory::Love),
            ('💞', "💞", EmojiCategory::Love),
            ('😘', "😘", EmojiCategory::Love),
            // Laughing
            ('😂', "😂", EmojiCategory::Laughing),
            ('🤣', "🤣", EmojiCategory::Laughing),
            ('😆', "😆", EmojiCategory::Laughing),
            ('😹', "😹", EmojiCategory::Laughing),
            // Thinking
            ('🤔', "🤔", EmojiCategory::Thinking),
            ('🧐', "🧐", EmojiCategory::Thinking),
            ('💭', "💭", EmojiCategory::Thinking),
            // Winking
            ('😉', "😉", EmojiCategory::Winking),
            // Cool
            ('😎', "😎", EmojiCategory::Cool),
            ('🤙', "🤙", EmojiCategory::Cool),
            ('🕶', "🕶️", EmojiCategory::Cool),
            // Fearful
            ('😰', "😰", EmojiCategory::Fearful),
            ('😨', "😨", EmojiCategory::Fearful),
            ('😱', "😱", EmojiCategory::Fearful),
            ('🫣', "🫣", EmojiCategory::Fearful),
            // Playful
            ('😜', "😜", EmojiCategory::Playful),
            ('😝', "😝", EmojiCategory::Playful),
            ('😛', "😛", EmojiCategory::Playful),
            ('🤪', "🤪", EmojiCategory::Playful),
            // Neutral
            ('😐', "😐", EmojiCategory::Neutral),
            ('😑', "😑", EmojiCategory::Neutral),
            ('😶', "😶", EmojiCategory::Neutral),
            ('🫤', "🫤", EmojiCategory::Neutral),
        ];

        Self {
            emoticons,
            emoji_categories,
        }
    }

    /// Detect all emoticons and emoji in text
    pub fn detect(&self, text: &str) -> Vec<EmoticonMatch> {
        let mut matches = Vec::new();
        let mut consumed = vec![false; text.len()]; // Track consumed byte positions

        // Pass 1: Detect text emoticons (longest match first)
        for &(pattern, emoji, ref category) in &self.emoticons {
            let pattern_bytes = pattern.len();
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(pattern) {
                let abs_pos = search_from + pos;
                // Check none of these bytes are already consumed
                if (abs_pos..abs_pos + pattern_bytes).all(|i| !consumed[i]) {
                    matches.push(EmoticonMatch {
                        original: pattern.to_string(),
                        emoji: emoji.to_string(),
                        category: category.clone(),
                        sentiment_score: category.sentiment_score(),
                        byte_offset: abs_pos,
                        is_unicode: false,
                    });
                    for i in abs_pos..abs_pos + pattern_bytes {
                        consumed[i] = true;
                    }
                }
                search_from = abs_pos + pattern_bytes;
                if search_from >= text.len() {
                    break;
                }
            }
        }

        // Pass 2: Detect Unicode emoji
        for (idx, ch) in text.char_indices() {
            let char_len = ch.len_utf8();
            // Skip if already consumed by a text emoticon
            if consumed[idx] {
                continue;
            }
            if let Some(&(_, emoji_str, ref category)) =
                self.emoji_categories.iter().find(|(c, _, _)| *c == ch)
            {
                matches.push(EmoticonMatch {
                    original: ch.to_string(),
                    emoji: emoji_str.to_string(),
                    category: category.clone(),
                    sentiment_score: category.sentiment_score(),
                    byte_offset: idx,
                    is_unicode: true,
                });
                for i in idx..idx + char_len {
                    if i < consumed.len() {
                        consumed[i] = true;
                    }
                }
            }
        }

        // Sort by position
        matches.sort_by_key(|m| m.byte_offset);
        matches
    }

    /// Full analysis: detect + classify + convert + aggregate
    pub fn analyze(&self, text: &str) -> EmoticonAnalysis {
        let matches = self.detect(text);

        // Overall sentiment: average of all matches
        let overall_sentiment = if matches.is_empty() {
            0.0
        } else {
            let sum: f32 = matches.iter().map(|m| m.sentiment_score).sum();
            (sum / matches.len() as f32).clamp(-1.0, 1.0)
        };

        // Category counts
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for m in &matches {
            *category_counts.entry(m.category.to_string()).or_insert(0) += 1;
        }

        // Dominant category
        let dominant_category = category_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(cat_name, _)| {
                matches
                    .iter()
                    .find(|m| m.category.to_string() == *cat_name)
                    .map(|m| m.category.clone())
                    .unwrap_or(EmojiCategory::Neutral)
            });

        // Convert text emoticons to emoji
        let converted_text = self.convert_emoticons(text);

        EmoticonAnalysis {
            matches,
            overall_sentiment,
            dominant_category,
            converted_text,
            category_counts,
        }
    }

    /// Convert all text emoticons to Unicode emoji, preserving the rest
    pub fn convert_emoticons(&self, text: &str) -> String {
        let mut result = text.to_string();
        // Replace longest patterns first (self.emoticons is already sorted by length desc)
        for &(pattern, emoji, _) in &self.emoticons {
            result = result.replace(pattern, emoji);
        }
        result
    }

    /// Classify a single emoji or emoticon string
    pub fn classify(&self, token: &str) -> Option<EmojiCategory> {
        // Check text emoticons
        for &(pattern, _, ref category) in &self.emoticons {
            if pattern == token {
                return Some(category.clone());
            }
        }
        // Check Unicode emoji (single char)
        if let Some(ch) = token.chars().next() {
            if token.chars().count() == 1 {
                if let Some((_, _, ref category)) =
                    self.emoji_categories.iter().find(|(c, _, _)| *c == ch)
                {
                    return Some(category.clone());
                }
            }
        }
        None
    }

    /// Get sentiment score from emoticons/emoji only (ignoring words)
    pub fn sentiment_score(&self, text: &str) -> f32 {
        let matches = self.detect(text);
        if matches.is_empty() {
            return 0.0;
        }
        let sum: f32 = matches.iter().map(|m| m.sentiment_score).sum();
        (sum / matches.len() as f32).clamp(-1.0, 1.0)
    }
}

impl SentimentAnalyzer {
    /// Analyze sentiment with explicit emoticon/emoji integration
    ///
    /// Returns both the keyword-based sentiment analysis and the emoticon analysis.
    /// The sentiment score is blended: 70% keyword + 30% emoticon (when emoticons are present).
    pub fn analyze_with_emoticons(&self, text: &str) -> (SentimentAnalysis, EmoticonAnalysis) {
        let mut sentiment = self.analyze_message(text);
        let detector = EmoticonDetector::new();
        let emoticon_analysis = detector.analyze(text);

        // Blend scores when emoticons are present
        if !emoticon_analysis.matches.is_empty() {
            let blended = sentiment.score * 0.7 + emoticon_analysis.overall_sentiment * 0.3;
            sentiment.score = blended.clamp(-1.0, 1.0);
            sentiment.sentiment = Sentiment::from_score(sentiment.score);
        }

        (sentiment, emoticon_analysis)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_analysis() {
        let analyzer = SentimentAnalyzer::new();

        // Positive message
        let result = analyzer.analyze_message("This is great! Thank you so much for the help!");
        assert!(result.score > 0.0);
        assert!(matches!(
            result.sentiment,
            Sentiment::Positive | Sentiment::VeryPositive
        ));

        // Negative message
        let result = analyzer.analyze_message("This is terrible and broken. Nothing works.");
        assert!(result.score < 0.0);
        assert!(matches!(
            result.sentiment,
            Sentiment::Negative | Sentiment::VeryNegative
        ));

        // Neutral message
        let result = analyzer.analyze_message("What time is it?");
        assert!(result.score.abs() < 0.3);
    }

    #[test]
    fn test_topic_detection() {
        let detector = TopicDetector::new();

        let messages = vec![
            ChatMessage::user("How do I fix this error in my code?"),
            ChatMessage::assistant("You need to debug the function."),
            ChatMessage::user("The bug is still there after compiling."),
        ];

        let topics = detector.detect_topics(&messages);
        assert!(!topics.is_empty());
        assert!(topics
            .iter()
            .any(|t| t.name == "programming" || t.name == "error_troubleshooting"));
    }

    #[test]
    fn test_session_summary() {
        let summarizer = SessionSummarizer::new(SummaryConfig::default());

        let messages = vec![
            ChatMessage::user("How do I translate this text?"),
            ChatMessage::assistant("You can use the translation feature."),
            ChatMessage::user("Thanks! That was helpful."),
        ];

        let summary = summarizer.summarize(&messages);
        assert!(!summary.summary.is_empty());
        assert_eq!(summary.message_count, 3);
    }

    #[test]
    fn test_sentiment_negation() {
        let analyzer = SentimentAnalyzer::new();

        let good_result = analyzer.analyze_message("good");
        let not_good_result = analyzer.analyze_message("not good");

        // "not good" should score lower than "good" due to negation
        assert!(
            not_good_result.score < good_result.score,
            "\"not good\" score ({}) should be lower than \"good\" score ({})",
            not_good_result.score,
            good_result.score,
        );

        // "not good" should be negative or at most neutral (score <= 0)
        assert!(
            not_good_result.score <= 0.0,
            "\"not good\" score ({}) should be <= 0.0",
            not_good_result.score,
        );
    }

    #[test]
    fn test_sentiment_intensifiers() {
        let analyzer = SentimentAnalyzer::new();

        // The analyzer assigns different weights to different positive words.
        // "excellent" (0.9) should score higher than "good" (0.6), reflecting intensity.
        let good_result = analyzer.analyze_message("good");
        let excellent_result = analyzer.analyze_message("excellent");

        assert!(
            excellent_result.score > good_result.score,
            "\"excellent\" score ({}) should be higher than \"good\" score ({})",
            excellent_result.score,
            good_result.score,
        );

        // "perfect" (1.0) should score even higher
        let perfect_result = analyzer.analyze_message("perfect");
        assert!(
            perfect_result.score > excellent_result.score,
            "\"perfect\" score ({}) should be higher than \"excellent\" score ({})",
            perfect_result.score,
            excellent_result.score,
        );

        // Verify that the intensifier map is populated (smoke test)
        assert!(
            analyzer.intensifiers.contains_key("very"),
            "Intensifier map should contain 'very'",
        );
        assert!(
            analyzer.intensifiers.contains_key("extremely"),
            "Intensifier map should contain 'extremely'",
        );
    }

    #[test]
    fn test_topic_frequency_threshold() {
        let detector = TopicDetector::new();

        // A message with only one keyword match should NOT trigger topic detection
        // because the default min_matches is 2
        let messages = vec![ChatMessage::user("I like the ship.")];

        let topics = detector.detect_topics(&messages);
        // "ships" topic requires at least 2 keyword matches; "ship" alone is 1 match
        let ships_topic = topics.iter().find(|t| t.name == "ships");
        assert!(
            ships_topic.is_none(),
            "Topic 'ships' should not be detected with only 1 keyword match (min_matches=2), but got: {:?}",
            ships_topic,
        );

        // Now provide enough matches to cross the threshold
        let messages_with_enough = vec![ChatMessage::user(
            "I like the ship and this vehicle is great.",
        )];

        let topics_enough = detector.detect_topics(&messages_with_enough);
        let ships_topic_found = topics_enough.iter().find(|t| t.name == "ships");
        assert!(
            ships_topic_found.is_some(),
            "Topic 'ships' should be detected with 2+ keyword matches",
        );
    }

    // ========================================================================
    // New tests: Sentiment enum
    // ========================================================================

    #[test]
    fn test_sentiment_enum_scores() {
        // Each variant must return a score within [-1.0, 1.0]
        assert!((Sentiment::VeryPositive.score() - 1.0).abs() < f32::EPSILON);
        assert!((Sentiment::Positive.score() - 0.5).abs() < f32::EPSILON);
        assert!((Sentiment::Neutral.score() - 0.0).abs() < f32::EPSILON);
        assert!((Sentiment::Negative.score() - (-0.5)).abs() < f32::EPSILON);
        assert!((Sentiment::VeryNegative.score() - (-1.0)).abs() < f32::EPSILON);

        // All scores are in range
        for variant in &[
            Sentiment::VeryPositive,
            Sentiment::Positive,
            Sentiment::Neutral,
            Sentiment::Negative,
            Sentiment::VeryNegative,
        ] {
            let s = variant.score();
            assert!((-1.0..=1.0).contains(&s), "Score {} out of range for {:?}", s, variant);
        }
    }

    #[test]
    fn test_sentiment_enum_emoji() {
        // Every variant returns a non-empty emoji string
        for variant in &[
            Sentiment::VeryPositive,
            Sentiment::Positive,
            Sentiment::Neutral,
            Sentiment::Negative,
            Sentiment::VeryNegative,
        ] {
            let emoji = variant.emoji();
            assert!(!emoji.is_empty(), "Emoji should be non-empty for {:?}", variant);
        }

        // Distinct variants have distinct emojis
        assert_ne!(Sentiment::VeryPositive.emoji(), Sentiment::Neutral.emoji());
        assert_ne!(Sentiment::Positive.emoji(), Sentiment::Negative.emoji());
    }

    #[test]
    fn test_sentiment_from_score_boundaries() {
        // Exact boundary values
        assert_eq!(Sentiment::from_score(1.0), Sentiment::VeryPositive);
        assert_eq!(Sentiment::from_score(0.6), Sentiment::VeryPositive);
        assert_eq!(Sentiment::from_score(0.5), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(0.2), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(-0.19), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(-0.2), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(-0.5), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(-0.6), Sentiment::VeryNegative);
        assert_eq!(Sentiment::from_score(-1.0), Sentiment::VeryNegative);
    }

    #[test]
    fn test_sentiment_display() {
        // Display impl produces human-readable strings
        assert_eq!(format!("{}", Sentiment::VeryPositive), "Very Positive");
        assert_eq!(format!("{}", Sentiment::Positive), "Positive");
        assert_eq!(format!("{}", Sentiment::Neutral), "Neutral");
        assert_eq!(format!("{}", Sentiment::Negative), "Negative");
        assert_eq!(format!("{}", Sentiment::VeryNegative), "Very Negative");

        // Display output is non-empty for all variants
        for variant in &[
            Sentiment::VeryPositive,
            Sentiment::Positive,
            Sentiment::Neutral,
            Sentiment::Negative,
            Sentiment::VeryNegative,
        ] {
            let display = format!("{}", variant);
            assert!(!display.is_empty(), "Display should be non-empty for {:?}", variant);
        }
    }

    // ========================================================================
    // New tests: SentimentAnalyzer
    // ========================================================================

    #[test]
    fn test_sentiment_analyzer_default() {
        // Default::default() creates a valid analyzer identical to new()
        let from_default: SentimentAnalyzer = Default::default();
        let from_new = SentimentAnalyzer::new();

        // Both should be functional and produce the same results
        let text = "This is a great and wonderful day";
        let result_default = from_default.analyze_message(text);
        let result_new = from_new.analyze_message(text);

        assert!((result_default.score - result_new.score).abs() < f32::EPSILON);
        assert_eq!(result_default.sentiment, result_new.sentiment);
    }

    #[test]
    fn test_analyze_message_positive() {
        let analyzer = SentimentAnalyzer::new();

        let result = analyzer.analyze_message("This is excellent and amazing work! I love it!");
        assert!(result.score > 0.0, "Score should be positive, got {}", result.score);
        assert!(
            matches!(result.sentiment, Sentiment::Positive | Sentiment::VeryPositive),
            "Expected Positive or VeryPositive, got {:?}",
            result.sentiment,
        );
        assert!(
            !result.positive_indicators.is_empty(),
            "Should have positive indicators",
        );
    }

    #[test]
    fn test_analyze_message_negative() {
        let analyzer = SentimentAnalyzer::new();

        let result = analyzer.analyze_message("This is terrible and awful. I hate this broken thing.");
        assert!(result.score < 0.0, "Score should be negative, got {}", result.score);
        assert!(
            matches!(result.sentiment, Sentiment::Negative | Sentiment::VeryNegative),
            "Expected Negative or VeryNegative, got {:?}",
            result.sentiment,
        );
        assert!(
            !result.negative_indicators.is_empty(),
            "Should have negative indicators",
        );
    }

    #[test]
    fn test_analyze_message_neutral() {
        let analyzer = SentimentAnalyzer::new();

        let result = analyzer.analyze_message("The sky is blue today.");
        assert_eq!(result.sentiment, Sentiment::Neutral);
        assert!(
            result.score.abs() < 0.3,
            "Neutral text should have near-zero score, got {}",
            result.score,
        );
    }

    #[test]
    fn test_analyze_message_empty() {
        let analyzer = SentimentAnalyzer::new();

        // Empty string should not panic and should be neutral
        let result = analyzer.analyze_message("");
        assert_eq!(result.sentiment, Sentiment::Neutral);
        assert!((result.score - 0.0).abs() < f32::EPSILON);
        assert!(result.positive_indicators.is_empty());
        assert!(result.negative_indicators.is_empty());
    }

    #[test]
    fn test_analyze_message_confidence() {
        let analyzer = SentimentAnalyzer::new();

        // Confidence should always be between 0.0 and 1.0
        let texts = [
            "",
            "hello",
            "This is great!",
            "terrible horrible awful broken",
            "The cat sat on the mat.",
        ];

        for text in &texts {
            let result = analyzer.analyze_message(text);
            assert!(
                (0.0..=1.0).contains(&result.confidence),
                "Confidence {} out of range for text: {:?}",
                result.confidence,
                text,
            );
        }
    }

    #[test]
    fn test_analyze_conversation_empty() {
        let analyzer = SentimentAnalyzer::new();

        // Empty messages slice should not panic
        let result = analyzer.analyze_conversation(&[]);
        assert_eq!(result.overall.sentiment, Sentiment::Neutral);
        assert!((result.overall.score - 0.0).abs() < f32::EPSILON);
        assert_eq!(result.trend, SentimentTrend::Stable);
        assert!(result.message_sentiments.is_empty());
    }

    #[test]
    fn test_analyze_conversation_multi_turn() {
        let analyzer = SentimentAnalyzer::new();

        let messages = vec![
            ChatMessage::user("I'm having a problem with this error."),
            ChatMessage::assistant("Let me help you debug that issue."),
            ChatMessage::user("That fixed it! Thank you, that was perfect!"),
            ChatMessage::assistant("You're welcome! Glad it worked."),
        ];

        let result = analyzer.analyze_conversation(&messages);

        // Should have one sentiment per message
        assert_eq!(result.message_sentiments.len(), 4);

        // Overall should be valid
        assert!((-1.0..=1.0).contains(&result.overall.score));
        assert!((-1.0..=1.0).contains(&result.user_sentiment.score));

        // Trend should be one of the valid variants
        assert!(matches!(
            result.trend,
            SentimentTrend::Improving | SentimentTrend::Stable | SentimentTrend::Declining
        ));
    }

    // ========================================================================
    // New tests: TopicDetector
    // ========================================================================

    #[test]
    fn test_topic_detector_add_remove() {
        let mut detector = TopicDetector::new();

        // Add a custom topic
        detector.add_topic("custom_topic", vec!["alpha", "beta", "gamma"]);

        // Verify topic can be detected
        let messages = vec![ChatMessage::user("Testing alpha and beta parameters together.")];
        let topics = detector.detect_topics(&messages);
        assert!(
            topics.iter().any(|t| t.name == "custom_topic"),
            "Custom topic should be detected, but got topics: {:?}",
            topics.iter().map(|t| &t.name).collect::<Vec<_>>(),
        );

        // Remove the topic
        detector.remove_topic("custom_topic");

        // Topic should no longer be detected
        let topics_after = detector.detect_topics(&messages);
        assert!(
            !topics_after.iter().any(|t| t.name == "custom_topic"),
            "Removed topic should not be detected",
        );
    }

    #[test]
    fn test_topic_detector_custom_topics() {
        let mut detector = TopicDetector::new();

        detector.add_topic("machine_learning", vec!["neural", "network", "training", "model", "dataset"]);

        let messages = vec![
            ChatMessage::user("I need help training a neural network model."),
            ChatMessage::assistant("What dataset are you using for training?"),
        ];

        let topics = detector.detect_topics(&messages);
        let ml_topic = topics.iter().find(|t| t.name == "machine_learning");
        assert!(ml_topic.is_some(), "machine_learning topic should be detected");

        let ml = ml_topic.unwrap();
        assert!(ml.relevance > 0.0, "Relevance should be positive");
        assert!(!ml.keywords.is_empty(), "Should have matched keywords");
        assert!(!ml.message_indices.is_empty(), "Should have message indices");
    }

    #[test]
    fn test_get_main_topic() {
        let detector = TopicDetector::new();

        // Messages heavily about programming
        let messages = vec![
            ChatMessage::user("I have a bug in my code and the function crashes."),
            ChatMessage::assistant("Let me look at the error in your debug output."),
            ChatMessage::user("The compile error is in the test method."),
        ];

        let main_topic = detector.get_main_topic(&messages);
        assert!(main_topic.is_some(), "Should detect at least one topic");

        let main = main_topic.unwrap();
        // The main topic should have positive relevance
        assert!(main.relevance > 0.0, "Main topic should have positive relevance");

        // The main topic's relevance should be >= all other detected topics
        let all_topics = detector.detect_topics(&messages);
        for topic in &all_topics {
            assert!(
                main.relevance >= topic.relevance - f32::EPSILON,
                "Main topic relevance ({}) should be >= other topic relevance ({})",
                main.relevance,
                topic.relevance,
            );
        }
    }

    #[test]
    fn test_extract_key_terms() {
        let detector = TopicDetector::new();

        let text = "rust rust rust python python java";
        let terms = detector.extract_key_terms(text, 3);

        assert!(!terms.is_empty(), "Should extract at least one term");

        // "rust" appears 3 times and should be first
        assert_eq!(terms[0].0, "rust");
        assert_eq!(terms[0].1, 3);

        // "python" appears 2 times and should be second
        assert_eq!(terms[1].0, "python");
        assert_eq!(terms[1].1, 2);

        // Results should be sorted by frequency (descending)
        for window in terms.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Terms should be sorted by frequency descending",
            );
        }
    }

    #[test]
    fn test_extract_key_terms_empty() {
        let detector = TopicDetector::new();

        // Empty text should return empty vec
        let terms = detector.extract_key_terms("", 5);
        assert!(terms.is_empty(), "Empty text should produce no key terms");

        // Text with only stop words should also return empty
        let terms_stop = detector.extract_key_terms("the a an is are", 5);
        assert!(
            terms_stop.is_empty(),
            "Text with only stop words should produce no key terms",
        );
    }

    // ========================================================================
    // New tests: SummaryConfig and SessionSummarizer
    // ========================================================================

    #[test]
    fn test_session_summarizer_config() {
        let config = SummaryConfig::default();

        // Default config should have sensible values
        assert!(config.enabled, "Default config should be enabled");
        assert!(
            config.trigger_message_count > 0,
            "trigger_message_count should be positive",
        );
        assert!(
            config.max_summary_tokens > 0,
            "max_summary_tokens should be positive",
        );
        assert!(config.include_topics, "Default should include topics");
        assert!(config.include_sentiment, "Default should include sentiment");

        // Check the specific default values
        assert_eq!(config.trigger_message_count, 10);
        assert_eq!(config.max_summary_tokens, 500);
    }

    #[test]
    fn test_session_summarizer_empty() {
        let summarizer = SessionSummarizer::new(SummaryConfig::default());

        // Summarize empty messages should not panic
        let summary = summarizer.summarize(&[]);
        assert_eq!(summary.message_count, 0);
        assert!(!summary.summary.is_empty(), "Summary text should not be empty even for empty input");
        assert_eq!(summary.sentiment, Sentiment::Neutral);
        assert!(summary.key_points.is_empty());
        assert!(summary.topics.is_empty());
        assert!(summary.user_questions.is_empty());
        assert!(summary.solutions_provided.is_empty());
    }

    #[test]
    fn test_should_summarize_threshold() {
        let config = SummaryConfig {
            enabled: true,
            trigger_message_count: 5,
            max_summary_tokens: 500,
            include_topics: true,
            include_sentiment: true,
        };
        let summarizer = SessionSummarizer::new(config);

        // Below threshold
        assert!(!summarizer.should_summarize(0), "0 messages: should not summarize");
        assert!(!summarizer.should_summarize(1), "1 message: should not summarize");
        assert!(!summarizer.should_summarize(4), "4 messages: should not summarize");

        // At and above threshold
        assert!(summarizer.should_summarize(5), "5 messages: should summarize");
        assert!(summarizer.should_summarize(10), "10 messages: should summarize");
        assert!(summarizer.should_summarize(100), "100 messages: should summarize");

        // Disabled config should never trigger
        let disabled_config = SummaryConfig {
            enabled: false,
            trigger_message_count: 5,
            max_summary_tokens: 500,
            include_topics: true,
            include_sentiment: true,
        };
        let disabled_summarizer = SessionSummarizer::new(disabled_config);
        assert!(
            !disabled_summarizer.should_summarize(100),
            "Disabled summarizer should never trigger",
        );
    }

    // ========================================================================
    // Emoticon & Emoji Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_simple_smiley() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("hello :)");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original, ":)");
        assert_eq!(matches[0].category, EmojiCategory::Happy);
        assert!(!matches[0].is_unicode);
    }

    #[test]
    fn test_detect_sad_emoticon() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("oh no :(");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, EmojiCategory::Sad);
        assert_eq!(matches[0].emoji, "😢");
    }

    #[test]
    fn test_detect_love_heart() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("I love you <3");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, EmojiCategory::Love);
        assert_eq!(matches[0].emoji, "❤️");
    }

    #[test]
    fn test_detect_laughing() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("that was funny XD");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, EmojiCategory::Laughing);
    }

    #[test]
    fn test_detect_unicode_emoji() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("great job 😊");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, EmojiCategory::Happy);
        assert!(matches[0].is_unicode);
    }

    #[test]
    fn test_detect_unicode_sad() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("so sad 😢");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, EmojiCategory::Sad);
        assert!(matches[0].is_unicode);
    }

    #[test]
    fn test_detect_multiple() {
        let detector = EmoticonDetector::new();
        let matches = detector.detect("I'm happy :) and excited <3");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].category, EmojiCategory::Happy);
        assert_eq!(matches[1].category, EmojiCategory::Love);
    }

    #[test]
    fn test_convert_emoticons() {
        let detector = EmoticonDetector::new();
        let result = detector.convert_emoticons(":) hello :(");
        assert_eq!(result, "😊 hello 😢");
    }

    #[test]
    fn test_classify_known() {
        let detector = EmoticonDetector::new();
        assert_eq!(detector.classify(":)"), Some(EmojiCategory::Happy));
        assert_eq!(detector.classify(":("), Some(EmojiCategory::Sad));
        assert_eq!(detector.classify("<3"), Some(EmojiCategory::Love));
    }

    #[test]
    fn test_classify_unknown() {
        let detector = EmoticonDetector::new();
        assert_eq!(detector.classify("???"), None);
        assert_eq!(detector.classify("hello"), None);
    }

    #[test]
    fn test_sentiment_positive_emoticons() {
        let detector = EmoticonDetector::new();
        let score = detector.sentiment_score(":) :D <3");
        assert!(score > 0.5, "Positive emoticons should give positive score, got {}", score);
    }

    #[test]
    fn test_sentiment_negative_emoticons() {
        let detector = EmoticonDetector::new();
        let score = detector.sentiment_score(":( >:( D:");
        assert!(score < -0.5, "Negative emoticons should give negative score, got {}", score);
    }

    #[test]
    fn test_sentiment_mixed() {
        let detector = EmoticonDetector::new();
        let score = detector.sentiment_score(":) :(");
        assert!(score.abs() < 0.3, "Mixed emoticons should be near-neutral, got {}", score);
    }

    #[test]
    fn test_analyze_full() {
        let detector = EmoticonDetector::new();
        let analysis = detector.analyze("Great :) I love it <3 but also :(");
        assert_eq!(analysis.matches.len(), 3);
        assert!(analysis.category_counts.contains_key("Happy"));
        assert!(analysis.category_counts.contains_key("Love"));
        assert!(analysis.category_counts.contains_key("Sad"));
        assert!(!analysis.converted_text.contains(":)"));
        assert!(analysis.converted_text.contains("😊"));
    }

    #[test]
    fn test_dominant_category() {
        let detector = EmoticonDetector::new();
        let analysis = detector.analyze(":) :D =) :(");
        assert_eq!(
            analysis.dominant_category,
            Some(EmojiCategory::Happy),
            "3 happy vs 1 sad → dominant should be Happy"
        );
    }

    #[test]
    fn test_no_emoticons() {
        let detector = EmoticonDetector::new();
        let analysis = detector.analyze("Just plain text here.");
        assert!(analysis.matches.is_empty());
        assert_eq!(analysis.overall_sentiment, 0.0);
        assert!(analysis.dominant_category.is_none());
    }

    #[test]
    fn test_longest_match_priority() {
        let detector = EmoticonDetector::new();
        // :-( should be matched as one emoticon, not :- + (
        let matches = detector.detect(":-(");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original, ":-(");
        assert_eq!(matches[0].category, EmojiCategory::Sad);
    }

    #[test]
    fn test_analyzer_with_emoticons() {
        let analyzer = SentimentAnalyzer::new();
        let (sentiment, emoticon_analysis) =
            analyzer.analyze_with_emoticons("This is great! :) <3");
        assert!(sentiment.score > 0.0, "Blended score should be positive");
        assert!(!emoticon_analysis.matches.is_empty());
        assert!(emoticon_analysis.overall_sentiment > 0.0);
    }
}
