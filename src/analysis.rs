//! Conversation analysis: sentiment, topics, auto-summarization

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::messages::ChatMessage;

// ============================================================================
// Sentiment Analysis
// ============================================================================

/// Sentiment of a message or conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
            ("great", 0.8), ("excellent", 0.9), ("amazing", 0.9), ("wonderful", 0.85),
            ("fantastic", 0.9), ("perfect", 1.0), ("good", 0.6), ("nice", 0.5),
            ("helpful", 0.7), ("thanks", 0.6), ("thank", 0.6), ("love", 0.85),
            ("awesome", 0.85), ("brilliant", 0.9), ("outstanding", 0.9),
            ("happy", 0.7), ("pleased", 0.7), ("satisfied", 0.7), ("enjoy", 0.7),
            ("beautiful", 0.75), ("impressive", 0.8), ("useful", 0.65),
            ("clear", 0.5), ("easy", 0.5), ("fast", 0.5), ("efficient", 0.6),
            ("works", 0.4), ("working", 0.4), ("solved", 0.7), ("fixed", 0.7),
            ("genial", 0.8), ("excelente", 0.9), ("perfecto", 1.0), ("gracias", 0.6),
            ("bueno", 0.6), ("bien", 0.5), ("increíble", 0.9), ("maravilloso", 0.85),
        ] {
            positive.insert(word, weight);
        }

        // Negative words with weights
        for (word, weight) in [
            ("bad", -0.7), ("terrible", -0.9), ("awful", -0.9), ("horrible", -0.9),
            ("poor", -0.6), ("wrong", -0.6), ("error", -0.5), ("fail", -0.7),
            ("failed", -0.7), ("broken", -0.8), ("bug", -0.5), ("issue", -0.4),
            ("problem", -0.5), ("difficult", -0.4), ("hard", -0.3), ("confusing", -0.5),
            ("confused", -0.4), ("frustrated", -0.7), ("annoying", -0.6), ("annoyed", -0.6),
            ("disappointing", -0.7), ("disappointed", -0.7), ("useless", -0.8),
            ("slow", -0.4), ("crash", -0.8), ("crashes", -0.8), ("stuck", -0.5),
            ("hate", -0.9), ("worst", -1.0), ("never", -0.3), ("can't", -0.3),
            ("cannot", -0.3), ("doesn't", -0.3), ("doesn't work", -0.7),
            ("malo", -0.7), ("terrible", -0.9), ("error", -0.5), ("falla", -0.7),
            ("problema", -0.5), ("difícil", -0.4), ("confuso", -0.5),
        ] {
            negative.insert(word, weight);
        }

        let mut intensifiers = HashMap::new();
        for (word, mult) in [
            ("very", 1.5), ("really", 1.4), ("extremely", 1.8), ("incredibly", 1.7),
            ("absolutely", 1.6), ("completely", 1.5), ("totally", 1.5),
            ("quite", 1.2), ("fairly", 1.1), ("somewhat", 0.8), ("slightly", 0.6),
            ("much", 1.3), ("so", 1.4), ("too", 1.3),
            ("muy", 1.5), ("realmente", 1.4), ("extremadamente", 1.8),
        ] {
            intensifiers.insert(word, mult);
        }

        let negators = vec![
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
            "shouldn't", "isn't", "aren't", "wasn't", "weren't",
            "no", "nunca", "ni", "nadie", "nada",
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
                (1..=3).any(|j| {
                    i >= j && self.negators.contains(&words[i - j])
                })
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
            let first_half: f32 = user_scores[..user_scores.len()/2].iter().sum::<f32>()
                / (user_scores.len()/2) as f32;
            let second_half: f32 = user_scores[user_scores.len()/2..].iter().sum::<f32>()
                / (user_scores.len() - user_scores.len()/2) as f32;

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
        topics.insert("programming".to_string(), vec![
            "code", "function", "variable", "bug", "error", "compile", "runtime",
            "debug", "test", "class", "method", "api", "library", "framework",
            "código", "función", "variable", "error", "compilar",
        ]);

        topics.insert("help_request".to_string(), vec![
            "help", "how", "why", "what", "can you", "could you", "please",
            "explain", "show", "tell", "ayuda", "cómo", "por qué", "qué",
        ]);

        topics.insert("error_troubleshooting".to_string(), vec![
            "error", "crash", "fail", "broken", "issue", "problem", "bug",
            "not working", "doesn't work", "won't", "can't",
            "error", "falla", "problema", "no funciona",
        ]);

        topics.insert("configuration".to_string(), vec![
            "config", "setting", "setup", "install", "configure", "option",
            "preference", "enable", "disable", "path", "file",
            "configuración", "ajuste", "instalar", "opción",
        ]);

        topics.insert("performance".to_string(), vec![
            "slow", "fast", "speed", "performance", "optimize", "memory",
            "cpu", "lag", "freeze", "efficient",
            "lento", "rápido", "rendimiento", "optimizar", "memoria",
        ]);

        topics.insert("feature_request".to_string(), vec![
            "would be nice", "could you add", "feature", "suggestion", "idea",
            "want", "need", "wish", "hope", "request",
            "sería bueno", "podrías añadir", "característica", "sugerencia",
        ]);

        topics.insert("gratitude".to_string(), vec![
            "thank", "thanks", "appreciate", "helpful", "great", "awesome",
            "perfect", "exactly", "solved",
            "gracias", "agradezco", "útil", "genial", "perfecto",
        ]);

        // Star Citizen specific topics
        topics.insert("ships".to_string(), vec![
            "ship", "vehicle", "fighter", "bomber", "cargo", "mining", "exploration",
            "nave", "vehículo", "caza", "bombardero", "carga", "minería",
        ]);

        topics.insert("localization".to_string(), vec![
            "translation", "translate", "language", "localization", "text",
            "traducción", "traducir", "idioma", "localización", "texto",
        ]);

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
                    let entry = topic_matches.entry(topic_name.clone())
                        .or_insert((0, Vec::new(), Vec::new()));
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
        let mut topics: Vec<Topic> = topic_matches.into_iter()
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
        topics.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));

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
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while",
            "el", "la", "los", "las", "un", "una", "de", "en", "con", "por",
            "para", "que", "es", "son", "está", "están", "como", "más", "pero",
            "this", "that", "these", "those", "it", "its", "i", "you", "he",
            "she", "we", "they", "my", "your", "his", "her", "our", "their",
        ].iter().copied().collect();

        for word in text.to_lowercase()
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
        let user_questions: Vec<String> = messages.iter()
            .filter(|m| m.role == "user")
            .filter(|m| m.content.contains('?') || m.content.to_lowercase().starts_with("how")
                || m.content.to_lowercase().starts_with("what")
                || m.content.to_lowercase().starts_with("why")
                || m.content.to_lowercase().starts_with("cómo")
                || m.content.to_lowercase().starts_with("qué")
                || m.content.to_lowercase().starts_with("por qué"))
            .map(|m| {
                // Take first sentence or first 100 chars
                let content = &m.content;
                content.split(|c| c == '?' || c == '.')
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
        let topic_names: Vec<String> = topics.iter()
            .take(3)
            .map(|t| t.name.clone())
            .collect();

        // Analyze sentiment
        let sentiment_analysis = self.sentiment_analyzer.analyze_conversation(messages);

        // Extract key terms for key points
        let all_text: String = messages.iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let key_terms = self.topic_detector.extract_key_terms(&all_text, 10);
        let key_points: Vec<String> = key_terms.iter()
            .take(5)
            .map(|(term, count)| format!("{} (mentioned {} times)", term, count))
            .collect();

        // Generate summary text
        let summary = self.generate_summary_text(messages, &topic_names, &sentiment_analysis);

        // Extract solutions (assistant messages that seem conclusive)
        let solutions: Vec<String> = messages.iter()
            .filter(|m| m.role == "assistant")
            .filter(|m| {
                let lower = m.content.to_lowercase();
                lower.contains("you can") || lower.contains("to do this")
                    || lower.contains("the solution") || lower.contains("here's how")
                    || lower.contains("puedes") || lower.contains("la solución")
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
        assert!(matches!(result.sentiment, Sentiment::Positive | Sentiment::VeryPositive));

        // Negative message
        let result = analyzer.analyze_message("This is terrible and broken. Nothing works.");
        assert!(result.score < 0.0);
        assert!(matches!(result.sentiment, Sentiment::Negative | Sentiment::VeryNegative));

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
        assert!(topics.iter().any(|t| t.name == "programming" || t.name == "error_troubleshooting"));
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
}
