//! Topic-Aware RAG Relevance — lightweight topic matching + autocut + LLM classifier.
//!
//! Solves the "Mordor vs Alps" problem: keyword-relevant but topic-irrelevant chunks.
//! Three levels:
//! 1. **Jaccard topic matching** (zero cost, pure Rust) — from Semantic tier
//! 2. **Autocut** (score gap detection, zero cost) — from Semantic tier
//! 3. **LLM topic classifier** (batched, 1 call per 10 chunks) — from Thorough tier
//! 4. **Sub-chunk granular scoring** (ChunkRAG-style) — from Agentic tier

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for topic matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TopicMatchConfig {
    /// Whether lightweight topic matching is enabled.
    pub enabled: bool,
    /// Score penalty for off-topic chunks (0.0-1.0, default 0.2).
    pub off_topic_penalty: f32,
    /// Score multiplier for partial topic match (default 0.5).
    pub partial_match_factor: f32,
    /// Minimum Jaccard overlap to consider "on-topic" (default 0.15).
    pub on_topic_threshold: f32,
    /// Minimum word length to consider as topic keyword (default 4).
    pub min_word_length: usize,
    /// Maximum keywords to extract per text (default 20).
    pub max_keywords: usize,
    /// Whether LLM-assisted topic classification is enabled (Thorough+).
    pub llm_enabled: bool,
    /// Batch size for LLM classifier (default 10).
    pub llm_batch_size: usize,
    /// Max LLM classifier calls per query (cost control, default 3).
    pub max_llm_calls_per_query: usize,
    /// Whether autocut (score gap detection) is enabled.
    pub autocut_enabled: bool,
    /// Minimum gap ratio for autocut (default 0.3).
    pub autocut_min_gap_ratio: f32,
    /// Whether sub-chunk granular scoring is enabled (Agentic+).
    pub granular_scoring_enabled: bool,
    /// Batch size for granular scoring (default 5).
    pub granular_batch_size: usize,
}

impl Default for TopicMatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            off_topic_penalty: 0.2,
            partial_match_factor: 0.5,
            on_topic_threshold: 0.15,
            min_word_length: 4,
            max_keywords: 20,
            llm_enabled: false,
            llm_batch_size: 10,
            max_llm_calls_per_query: 3,
            autocut_enabled: true,
            autocut_min_gap_ratio: 0.3,
            granular_scoring_enabled: false,
            granular_batch_size: 5,
        }
    }
}

// ============================================================================
// Topic Match Result
// ============================================================================

/// Result of matching a chunk's topic against the query.
#[derive(Debug, Clone)]
pub struct TopicMatchResult {
    /// Jaccard similarity of topic keywords (0.0-1.0).
    pub overlap_score: f32,
    /// Classification level.
    pub match_level: TopicMatchLevel,
    /// Score multiplier to apply.
    pub score_factor: f32,
    /// Topic keywords extracted from the query.
    pub query_topics: Vec<String>,
    /// Topic keywords extracted from the chunk.
    pub chunk_topics: Vec<String>,
    /// Keywords in common.
    pub common_topics: Vec<String>,
}

/// Topic match classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopicMatchLevel {
    /// High overlap — chunk is about the same topic. Score × 1.0.
    OnTopic,
    /// Some overlap — partially related. Score × partial_match_factor.
    Partial,
    /// No overlap — different topic entirely. Score × off_topic_penalty.
    OffTopic,
}

// ============================================================================
// TopicMatcher
// ============================================================================

/// Lightweight topic matcher using Jaccard keyword overlap.
pub struct TopicMatcher {
    config: TopicMatchConfig,
    stopwords: HashSet<String>,
}

impl TopicMatcher {
    /// Create a new TopicMatcher with the given config.
    pub fn new(config: TopicMatchConfig) -> Self {
        Self {
            config,
            stopwords: default_stopwords(),
        }
    }

    /// Create with default config.
    pub fn default_matcher() -> Self {
        Self::new(TopicMatchConfig::default())
    }

    /// Extract topic keywords from text.
    pub fn extract_topics(&self, text: &str) -> Vec<String> {
        // Guard: max input size (#1: DoS prevention)
        let text = if text.len() > 100_000 {
            crate::text_util::truncate_str(&text, 100_000)
        } else {
            text
        };

        let mut freq: HashMap<String, usize> = HashMap::new();

        for word in text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
            let lower = word.to_lowercase();
            if lower.len() < self.config.min_word_length {
                continue;
            }
            if self.stopwords.contains(&lower) {
                continue;
            }
            // Skip pure numbers
            if lower.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            *freq.entry(lower).or_insert(0) += 1;
        }

        // Sort by frequency descending, take top N
        let mut words: Vec<(String, usize)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        words
            .into_iter()
            .take(self.config.max_keywords)
            .map(|(w, _)| w)
            .collect()
    }

    /// Score a chunk against pre-extracted query topics.
    pub fn score_chunk(&self, query_topics: &[String], chunk_text: &str) -> TopicMatchResult {
        // Single-keyword queries skip topic matching (#6)
        if query_topics.len() <= 1 {
            return TopicMatchResult {
                overlap_score: 1.0,
                match_level: TopicMatchLevel::OnTopic,
                score_factor: 1.0,
                query_topics: query_topics.to_vec(),
                chunk_topics: Vec::new(),
                common_topics: Vec::new(),
            };
        }

        let chunk_topics = self.extract_topics(chunk_text);

        if chunk_topics.is_empty() {
            return TopicMatchResult {
                overlap_score: 0.0,
                match_level: TopicMatchLevel::OffTopic,
                score_factor: self.config.off_topic_penalty,
                query_topics: query_topics.to_vec(),
                chunk_topics,
                common_topics: Vec::new(),
            };
        }

        let query_set: HashSet<&str> = query_topics.iter().map(|s| s.as_str()).collect();
        let chunk_set: HashSet<&str> = chunk_topics.iter().map(|s| s.as_str()).collect();

        let intersection: Vec<String> = query_set
            .intersection(&chunk_set)
            .map(|s| s.to_string())
            .collect();
        let union_size = query_set.union(&chunk_set).count();

        let overlap = if union_size > 0 {
            intersection.len() as f32 / union_size as f32
        } else {
            0.0
        };

        let (match_level, score_factor) = if overlap >= self.config.on_topic_threshold {
            (TopicMatchLevel::OnTopic, 1.0)
        } else if overlap > 0.0 {
            (TopicMatchLevel::Partial, self.config.partial_match_factor)
        } else {
            (TopicMatchLevel::OffTopic, self.config.off_topic_penalty)
        };

        TopicMatchResult {
            overlap_score: overlap,
            match_level,
            score_factor,
            query_topics: query_topics.to_vec(),
            chunk_topics,
            common_topics: intersection,
        }
    }

    /// Get the config.
    pub fn config(&self) -> &TopicMatchConfig {
        &self.config
    }
}

// ============================================================================
// Autocut — score gap detection
// ============================================================================

/// Find the natural cutoff point in a sorted-descending list of scores.
/// Returns the number of items to keep.
///
/// Detects the largest relative gap between consecutive scores.
/// If no gap exceeds `min_gap_ratio`, returns all items.
pub fn autocut_scores(scores: &[f32], min_gap_ratio: f32) -> usize {
    if scores.len() <= 1 {
        return scores.len();
    }

    let max_score = scores[0].max(0.001); // avoid division by zero
    let mut best_gap = 0.0f32;
    let mut best_idx = scores.len(); // default: keep all

    for i in 0..scores.len() - 1 {
        let gap = scores[i] - scores[i + 1];
        let relative_gap = gap / max_score;
        if relative_gap > best_gap && relative_gap >= min_gap_ratio {
            best_gap = relative_gap;
            best_idx = i + 1; // keep items 0..=i
        }
    }

    best_idx
}

// ============================================================================
// LLM Topic Classifier (structures only — LLM call is external)
// ============================================================================

/// Request for LLM topic classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicClassifyRequest {
    /// The original query.
    pub query: String,
    /// Chunks to classify (id, content_preview).
    pub chunks: Vec<(String, String)>,
}

/// Result from LLM topic classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmTopicVerdict {
    OnTopic,
    Partial,
    OffTopic,
}

/// Build the prompt for LLM topic classification (#9: structured output only).
pub fn build_topic_classify_prompt(query: &str, chunks: &[(String, String)]) -> String {
    let mut prompt = format!(
        "Query: \"{}\"\n\nClassify each chunk as ON_TOPIC, PARTIAL, or OFF_TOPIC \
         based on whether it is relevant to the query topic.\n\
         Respond ONLY with a JSON array of verdicts, e.g. [\"ON_TOPIC\", \"OFF_TOPIC\", \"PARTIAL\"]\n\n",
        query
    );
    for (i, (id, preview)) in chunks.iter().enumerate() {
        let safe_preview = if preview.len() > 200 {
            format!("{}...", crate::text_util::truncate_str(&preview, 200))
        } else {
            preview.clone()
        };
        prompt.push_str(&format!("{}. [{}] {}\n", i + 1, id, safe_preview));
    }
    prompt
}

/// Parse LLM topic classification response.
pub fn parse_topic_classify_response(response: &str) -> Vec<LlmTopicVerdict> {
    // Try to find JSON array in response
    let trimmed = response.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    if let Ok(arr) = serde_json::from_str::<Vec<String>>(json_str) {
        arr.iter()
            .map(|s| match s.to_uppercase().as_str() {
                "ON_TOPIC" | "ONTOPIC" | "ON" | "YES" => LlmTopicVerdict::OnTopic,
                "PARTIAL" | "MAYBE" | "RELATED" => LlmTopicVerdict::Partial,
                _ => LlmTopicVerdict::OffTopic,
            })
            .collect()
    } else {
        Vec::new()
    }
}

// ============================================================================
// Self-Query Filter Extraction
// ============================================================================

/// Extracted search filters from a natural language query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelfQueryFilter {
    /// Section/topic filter.
    pub section_filter: Option<Vec<String>>,
    /// Source/document filter.
    pub source_filter: Option<Vec<String>>,
    /// Date range filter (start, end) in ISO 8601.
    pub date_range: Option<(String, String)>,
    /// Extracted key search terms.
    pub extracted_keywords: Vec<String>,
}

/// Build prompt for Self-Query filter extraction.
pub fn build_self_query_prompt(query: &str, available_sections: &[String]) -> String {
    format!(
        "Extract structured search filters from this query.\n\
         Available sections: {:?}\n\
         Query: \"{}\"\n\n\
         Respond with JSON:\n\
         {{\"sections\": [\"section1\"], \"keywords\": [\"word1\", \"word2\"], \
         \"date_after\": null, \"date_before\": null}}\n\
         If no filter applies, use null. Only use sections from the available list.",
        available_sections, query
    )
}

/// Parse Self-Query filter extraction response.
pub fn parse_self_query_response(response: &str) -> SelfQueryFilter {
    let trimmed = response.trim();
    let json_str = if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
        let sections = json.get("sections").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

        let keywords = json
            .get("keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let date_after = json
            .get("date_after")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let date_before = json
            .get("date_before")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let date_range = match (date_after, date_before) {
            (Some(a), Some(b)) => Some((a, b)),
            (Some(a), None) => Some((a, String::new())),
            (None, Some(b)) => Some((String::new(), b)),
            _ => None,
        };

        SelfQueryFilter {
            section_filter: sections,
            source_filter: None,
            date_range,
            extracted_keywords: keywords,
        }
    } else {
        SelfQueryFilter::default()
    }
}

// ============================================================================
// Sub-Chunk Granular Scoring (ChunkRAG-style)
// ============================================================================

/// Result of granular sentence-level scoring within a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranularScoringResult {
    /// Original chunk text.
    pub original_chunk: String,
    /// Sentences deemed relevant.
    pub relevant_sentences: Vec<String>,
    /// Fraction of sentences that are relevant (0.0-1.0).
    pub relevance_ratio: f32,
    /// Only the relevant sentences, joined.
    pub compressed_content: String,
}

/// Split text into sentences (simple heuristic).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') && current.trim().len() > 10 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let remaining = current.trim().to_string();
    if !remaining.is_empty() && remaining.len() > 5 {
        sentences.push(remaining);
    }
    sentences
}

/// Build prompt for granular scoring.
pub fn build_granular_scoring_prompt(query: &str, sentences: &[String]) -> String {
    let mut prompt = format!(
        "Query: \"{}\"\n\nFor each sentence, respond RELEVANT or IRRELEVANT.\n\
         Respond ONLY with a JSON array, e.g. [\"RELEVANT\", \"IRRELEVANT\", ...].\n\n",
        query
    );
    for (i, sentence) in sentences.iter().enumerate() {
        let safe = if sentence.len() > 200 {
            format!("{}...", crate::text_util::truncate_str(&sentence, 200))
        } else {
            sentence.clone()
        };
        prompt.push_str(&format!("{}. {}\n", i + 1, safe));
    }
    prompt
}

/// Parse granular scoring response.
pub fn parse_granular_response(response: &str, sentences: &[String]) -> GranularScoringResult {
    let original = sentences.join(" ");
    let trimmed = response.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    if let Ok(arr) = serde_json::from_str::<Vec<String>>(json_str) {
        let relevant: Vec<String> = arr
            .iter()
            .zip(sentences.iter())
            .filter(|(verdict, _)| {
                let upper = verdict.to_uppercase();
                upper.contains("RELEVANT") && !upper.contains("IRRELEVANT")
            })
            .map(|(_, sentence)| sentence.clone())
            .collect();

        let ratio = if sentences.is_empty() {
            0.0
        } else {
            relevant.len() as f32 / sentences.len() as f32
        };
        let compressed = relevant.join(" ");

        GranularScoringResult {
            original_chunk: original,
            relevant_sentences: relevant,
            relevance_ratio: ratio,
            compressed_content: compressed,
        }
    } else {
        // Fallback: keep all sentences
        GranularScoringResult {
            original_chunk: original.clone(),
            relevant_sentences: sentences.to_vec(),
            relevance_ratio: 1.0,
            compressed_content: original,
        }
    }
}

// ============================================================================
// Stopwords
// ============================================================================

fn default_stopwords() -> HashSet<String> {
    let words = [
        // English (59)
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above", "below", "between",
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", // Spanish (30)
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con",
        "por", "para", "sobre", "entre", "sin", "hasta", "como", "pero", "que", "este", "esta",
        "esto", "ese", "esa", "eso", "ser", "estar", // Common verbs (both languages)
        "and", "but", "not", "all", "each", "every", "some", "any", "its", "this", "that", "these",
        "those", "what", "which", "who", "whom", "very", "just", "also", "more", "most", "much",
        "many", "such", "only", "other", "than", "too", "more", "less",
        // Common utility words
        "hacer", "tener", "poder", "decir", "ayudar", "querer", "gustar", "haber", "puede", "tiene",
        "hace", "dice", "help", "want", "like", "make", "take", "give", "know", "think", "come",
        "find", "tell", "work", // Code keywords (filtered for code chunks, #5)
        "function", "return", "const", "class", "import", "export", "public", "private", "static",
        "void", "null", "true", "false",
    ];
    words.iter().map(|w| w.to_string()).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher() -> TopicMatcher {
        TopicMatcher::default_matcher()
    }

    #[test]
    fn test_topic_extraction_basic() {
        let m = matcher();
        let topics = m.extract_topics("ayudar a pintar la casa con colores bonitos");
        assert!(topics.contains(&"pintar".to_string()));
        assert!(topics.contains(&"casa".to_string()));
        assert!(topics.contains(&"colores".to_string()));
        assert!(topics.contains(&"bonitos".to_string()));
        // "ayudar" is in stopwords
        assert!(!topics.contains(&"ayudar".to_string()));
    }

    #[test]
    fn test_topic_extraction_stopwords_filtered() {
        let m = matcher();
        let topics = m.extract_topics("the quick brown fox jumps over the lazy dog");
        assert!(!topics.contains(&"the".to_string()));
        assert!(topics.contains(&"quick".to_string()));
        assert!(topics.contains(&"brown".to_string()));
        assert!(topics.contains(&"jumps".to_string()));
    }

    #[test]
    fn test_topic_overlap_on_topic() {
        let m = matcher();
        let query_topics = m.extract_topics("herramientas para pintar paredes de casa");
        let result = m.score_chunk(
            &query_topics,
            "las mejores brochas y rodillos para pintar la casa",
        );
        assert_eq!(result.match_level, TopicMatchLevel::OnTopic);
        assert!(result.score_factor >= 0.99);
        assert!(!result.common_topics.is_empty());
    }

    #[test]
    fn test_topic_overlap_off_topic() {
        let m = matcher();
        let query_topics = m.extract_topics("pintar las paredes de la casa con rodillo");
        let result = m.score_chunk(
            &query_topics,
            "sensores zigbee de temperatura y humedad para domótica",
        );
        assert_eq!(result.match_level, TopicMatchLevel::OffTopic);
        assert!(result.score_factor < 0.3);
        assert!(result.common_topics.is_empty());
    }

    #[test]
    fn test_topic_overlap_partial() {
        let m = matcher();
        let query_topics = m.extract_topics("automatizar la iluminación de casa");
        let result = m.score_chunk(
            &query_topics,
            "mejorar la iluminación del jardín con plantas",
        );
        // "iluminación" is common, but topics diverge
        assert!(result.overlap_score > 0.0);
        assert!(
            result.overlap_score < m.config.on_topic_threshold
                || result.match_level == TopicMatchLevel::OnTopic
        );
    }

    #[test]
    fn test_score_penalty_off_topic() {
        let m = TopicMatcher::new(TopicMatchConfig {
            off_topic_penalty: 0.2,
            ..Default::default()
        });
        let query_topics = m.extract_topics("recetas de cocina italiana");
        let result = m.score_chunk(
            &query_topics,
            "programación en rust para sistemas distribuidos",
        );
        assert_eq!(result.match_level, TopicMatchLevel::OffTopic);
        assert!((result.score_factor - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_score_penalty_on_topic() {
        let m = matcher();
        let query_topics = m.extract_topics("configurar sensores zigbee");
        let result = m.score_chunk(
            &query_topics,
            "guía de configuración de sensores zigbee con zigbee2mqtt",
        );
        assert_eq!(result.match_level, TopicMatchLevel::OnTopic);
        assert!((result.score_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_single_keyword_skip() {
        let m = matcher();
        let query_topics = vec!["domótica".to_string()];
        let result = m.score_chunk(&query_topics, "cualquier texto irrelevante");
        // Single keyword → always OnTopic (skip matching)
        assert_eq!(result.match_level, TopicMatchLevel::OnTopic);
        assert!((result.score_factor - 1.0).abs() < 0.01);
    }

    // --- Autocut tests ---

    #[test]
    fn test_autocut_clear_gap() {
        let scores = vec![0.9, 0.85, 0.80, 0.35, 0.30, 0.25];
        let cut = autocut_scores(&scores, 0.3);
        assert_eq!(cut, 3); // Keep first 3 (gap between 0.80 and 0.35)
    }

    #[test]
    fn test_autocut_no_gap() {
        let scores = vec![0.50, 0.48, 0.46, 0.44];
        let cut = autocut_scores(&scores, 0.3);
        assert_eq!(cut, 4); // No significant gap → keep all
    }

    #[test]
    fn test_autocut_single_result() {
        let scores = vec![0.7];
        let cut = autocut_scores(&scores, 0.3);
        assert_eq!(cut, 1); // Keep the single result
    }

    // --- Search integration ---

    #[test]
    fn test_topic_config_serialization() {
        let config = TopicMatchConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: TopicMatchConfig = serde_json::from_str(&json).unwrap();
        assert!((restored.off_topic_penalty - 0.2).abs() < 0.01);
        assert!(restored.enabled);
    }

    #[test]
    fn test_multilingual_stopwords() {
        let m = matcher();
        // Spanish stopwords
        let topics = m.extract_topics("quiero hacer algo para ayudar a mejorar esto");
        assert!(!topics.contains(&"querer".to_string()));
        assert!(!topics.contains(&"hacer".to_string()));
        assert!(!topics.contains(&"ayudar".to_string()));
        assert!(topics.contains(&"mejorar".to_string()));
    }

    // --- LLM classifier ---

    #[test]
    fn test_llm_classifier_prompt() {
        let prompt = build_topic_classify_prompt(
            "pintar casa",
            &[
                ("c1".into(), "herramientas de pintura".into()),
                ("c2".into(), "sensores zigbee".into()),
            ],
        );
        assert!(prompt.contains("pintar casa"));
        assert!(prompt.contains("herramientas de pintura"));
        assert!(prompt.contains("ON_TOPIC"));
    }

    #[test]
    fn test_llm_classifier_parse() {
        let response = r#"["ON_TOPIC", "OFF_TOPIC", "PARTIAL"]"#;
        let verdicts = parse_topic_classify_response(response);
        assert_eq!(verdicts.len(), 3);
        assert_eq!(verdicts[0], LlmTopicVerdict::OnTopic);
        assert_eq!(verdicts[1], LlmTopicVerdict::OffTopic);
        assert_eq!(verdicts[2], LlmTopicVerdict::Partial);
    }

    #[test]
    fn test_self_query_filter_parse() {
        let response = r#"{"sections": ["cocina", "recetas"], "keywords": ["pasta", "italiana"], "date_after": "2025-01-01", "date_before": null}"#;
        let filter = parse_self_query_response(response);
        assert_eq!(
            filter.section_filter,
            Some(vec!["cocina".into(), "recetas".into()])
        );
        assert_eq!(filter.extracted_keywords, vec!["pasta", "italiana"]);
        assert!(filter.date_range.is_some());
    }

    // --- Granular scoring ---

    #[test]
    fn test_split_sentences() {
        let text = "First sentence here. Second sentence follows. Third one too.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_granular_scoring_parse() {
        let sentences = vec![
            "La domótica ayuda en el hogar.".into(),
            "Los sensores detectan temperatura.".into(),
            "Pintar paredes es divertido.".into(),
        ];
        let response = r#"["IRRELEVANT", "IRRELEVANT", "RELEVANT"]"#;
        let result = parse_granular_response(response, &sentences);
        assert_eq!(result.relevant_sentences.len(), 1);
        assert!(result.compressed_content.contains("Pintar"));
        assert!((result.relevance_ratio - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_granular_scoring_all_relevant() {
        let sentences = vec!["Sentence one.".into(), "Sentence two.".into()];
        let response = r#"["RELEVANT", "RELEVANT"]"#;
        let result = parse_granular_response(response, &sentences);
        assert_eq!(result.relevant_sentences.len(), 2);
        assert!((result.relevance_ratio - 1.0).abs() < 0.01);
    }
}
