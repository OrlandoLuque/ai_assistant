//! Memory consolidation: clusters episodes into reusable procedures.
//! Also includes enhanced consolidation pipeline with semantic fact extraction.

use serde::{Deserialize, Serialize};

use super::episodic::Episode;
use super::procedural::Procedure;
use super::helpers::keyword_overlap;

// ============================================================
// Basic Consolidation
// ============================================================

/// Configuration and logic for consolidating episodic memories into procedures.
pub struct MemoryConsolidator {
    /// Minimum number of episodes required to consider forming a procedure.
    pub min_episodes_for_procedure: usize,
    /// Tag/keyword overlap threshold (0.0-1.0) to consider two episodes similar.
    pub similarity_threshold: f64,
    /// Minimum cluster size to actually generate a procedure from a cluster.
    pub min_cluster_size: usize,
}

/// Result of a consolidation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub procedures_created: Vec<Procedure>,
    pub episodes_clustered: usize,
    pub clusters_found: usize,
}

impl MemoryConsolidator {
    /// Create a consolidator with sensible defaults.
    pub fn new() -> Self {
        Self {
            min_episodes_for_procedure: 3,
            similarity_threshold: 0.3,
            min_cluster_size: 2,
        }
    }

    /// Cluster the given episodes by content similarity (shared tags + keyword
    /// overlap) and generate procedures from sufficiently large clusters.
    pub fn consolidate(&self, episodes: &[Episode]) -> ConsolidationResult {
        if episodes.len() < self.min_episodes_for_procedure {
            return ConsolidationResult {
                procedures_created: Vec::new(),
                episodes_clustered: 0,
                clusters_found: 0,
            };
        }

        // Simple single-pass greedy clustering
        let mut assigned: Vec<bool> = vec![false; episodes.len()];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..episodes.len() {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..episodes.len() {
                if assigned[j] {
                    continue;
                }
                let sim = self.episode_similarity(&episodes[i], &episodes[j]);
                if sim >= self.similarity_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            if cluster.len() >= self.min_cluster_size {
                clusters.push(cluster);
            }
        }

        // Build procedures from qualifying clusters
        let mut procedures = Vec::new();
        let mut episodes_clustered = 0usize;

        for cluster in &clusters {
            episodes_clustered += cluster.len();

            // Collect shared tags
            let first_tags: std::collections::HashSet<&str> = episodes[cluster[0]]
                .tags
                .iter()
                .map(|t| t.as_str())
                .collect();
            let shared_tags: Vec<String> = first_tags
                .iter()
                .filter(|tag| {
                    cluster
                        .iter()
                        .all(|&idx| episodes[idx].tags.iter().any(|t| t == **tag))
                })
                .map(|s| s.to_string())
                .collect();

            // Build procedure steps from episode contents
            let steps: Vec<String> = cluster
                .iter()
                .map(|&idx| episodes[idx].content.clone())
                .collect();

            let created_from: Vec<String> =
                cluster.iter().map(|&idx| episodes[idx].id.clone()).collect();

            let name = if shared_tags.is_empty() {
                format!("procedure_{}", uuid::Uuid::new_v4())
            } else {
                format!("procedure_{}", shared_tags.join("_"))
            };

            let condition = if shared_tags.is_empty() {
                "general".to_string()
            } else {
                shared_tags.join(", ")
            };

            procedures.push(Procedure {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                condition,
                steps,
                success_count: 0,
                failure_count: 0,
                confidence: 0.5, // neutral starting confidence
                created_from,
                tags: shared_tags,
            });
        }

        ConsolidationResult {
            procedures_created: procedures,
            episodes_clustered,
            clusters_found: clusters.len(),
        }
    }

    /// Compute similarity between two episodes using tag overlap and keyword
    /// overlap in content/context fields.
    fn episode_similarity(&self, a: &Episode, b: &Episode) -> f64 {
        // Tag Jaccard
        let tags_a: std::collections::HashSet<&str> =
            a.tags.iter().map(|t| t.as_str()).collect();
        let tags_b: std::collections::HashSet<&str> =
            b.tags.iter().map(|t| t.as_str()).collect();
        let tag_sim = if tags_a.is_empty() && tags_b.is_empty() {
            0.0
        } else {
            let inter = tags_a.intersection(&tags_b).count();
            let union = tags_a.union(&tags_b).count();
            if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            }
        };

        // Content keyword overlap
        let content_sim = keyword_overlap(&a.content, &b.content);

        // Context keyword overlap
        let context_sim = keyword_overlap(&a.context, &b.context);

        // Weighted combination
        0.4 * tag_sim + 0.35 * content_sim + 0.25 * context_sim
    }
}

// ============================================================
// Enhanced Consolidation Pipeline
// ============================================================

/// A semantic fact extracted from episodic memories — a subject-predicate-object triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source_episodes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_confirmed: chrono::DateTime<chrono::Utc>,
}

/// Trait for extracting semantic facts from episodes.
pub trait FactExtractor {
    /// Extract semantic facts from a set of episodes.
    fn extract(&self, episodes: &[Episode]) -> Vec<SemanticFact>;
    /// Name of this extractor for diagnostics.
    fn name(&self) -> &str;
}

/// Extracts facts using keyword pattern matching (regex-like substring matching).
///
/// Patterns are (subject_pattern, predicate_pattern, object_pattern) tuples.
/// For each episode, the extractor scans the content and context for patterns
/// like "X prefers Y", "X is Y", "X uses Y", etc.
pub struct PatternFactExtractor {
    /// Each tuple is (subject_pattern, predicate_keyword, object_pattern).
    /// The extractor looks for `<word(s)> <predicate_keyword> <word(s)>` in text.
    patterns: Vec<(String, String, String)>,
}

impl PatternFactExtractor {
    /// Create an empty pattern extractor.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Create a pattern extractor pre-loaded with common patterns.
    pub fn with_default_patterns() -> Self {
        let patterns = vec![
            (r"(\w+)".to_string(), "prefers".to_string(), r"(\w+)".to_string()),
            (r"(\w+)".to_string(), "is".to_string(), r"(\w+)".to_string()),
            (r"(\w+)".to_string(), "uses".to_string(), r"(\w+)".to_string()),
            (r"(\w+)".to_string(), "likes".to_string(), r"(\w+)".to_string()),
            (r"(\w+)".to_string(), "works with".to_string(), r"(\w+)".to_string()),
        ];
        Self { patterns }
    }

    /// Add a custom pattern.
    pub fn add_pattern(&mut self, subject: &str, predicate: &str, object: &str) {
        self.patterns.push((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
    }

    /// Try to extract a fact from a single line of text using the given predicate keyword.
    fn extract_from_text(
        &self,
        text: &str,
        predicate_keyword: &str,
        episode_id: &str,
    ) -> Option<SemanticFact> {
        let text_lower = text.to_lowercase();
        let pred_lower = predicate_keyword.to_lowercase();

        if let Some(pred_pos) = text_lower.find(&pred_lower) {
            let before = text[..pred_pos].trim();
            let after = text[pred_pos + predicate_keyword.len()..].trim();

            // Extract subject: last word(s) before predicate
            let subject = before
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_string();

            // Extract object: first word(s) after predicate
            let object = after
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();

            if !subject.is_empty() && !object.is_empty() {
                let now = chrono::Utc::now();
                return Some(SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    subject,
                    predicate: predicate_keyword.to_string(),
                    object,
                    confidence: 0.7,
                    source_episodes: vec![episode_id.to_string()],
                    created_at: now,
                    last_confirmed: now,
                });
            }
        }
        None
    }
}

impl FactExtractor for PatternFactExtractor {
    fn extract(&self, episodes: &[Episode]) -> Vec<SemanticFact> {
        let mut facts = Vec::new();
        for episode in episodes {
            for (_, predicate, _) in &self.patterns {
                // Scan content
                if let Some(fact) = self.extract_from_text(&episode.content, predicate, &episode.id)
                {
                    facts.push(fact);
                }
                // Scan context
                if let Some(fact) = self.extract_from_text(&episode.context, predicate, &episode.id)
                {
                    facts.push(fact);
                }
            }
        }
        facts
    }

    fn name(&self) -> &str {
        "PatternFactExtractor"
    }
}

/// Extracts facts using heuristic NLP (sentence splitting, keyword extraction).
///
/// Simulates an LLM-based approach by analyzing sentence structure to find
/// subject-predicate-object triples. In production this would call an actual LLM.
pub struct LlmFactExtractor {
    /// Minimum sentence length (in words) to attempt extraction.
    min_sentence_words: usize,
}

impl LlmFactExtractor {
    /// Create a new LLM fact extractor with defaults.
    pub fn new() -> Self {
        Self {
            min_sentence_words: 3,
        }
    }

    /// Split text into sentences (by period, exclamation, question mark).
    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let sentence = text[start..=i].trim();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = i + c.len_utf8();
            }
        }
        // Remainder (no trailing punctuation)
        let remainder = text[start..].trim();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
        sentences
    }

    /// Try to extract a triple from a sentence using simple heuristics.
    /// Looks for patterns: Subject Verb Object where Verb is a known linking/action verb.
    fn extract_triple(sentence: &str, episode_id: &str) -> Option<SemanticFact> {
        let linking_verbs = [
            "is", "are", "was", "were", "uses", "prefers", "likes", "needs",
            "requires", "provides", "supports", "handles", "creates", "runs",
            "works", "depends",
        ];

        let words: Vec<&str> = sentence
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.len() < 3 {
            return None;
        }

        // Find the first linking verb
        for (i, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();
            if linking_verbs.contains(&word_lower.as_str()) && i > 0 && i < words.len() - 1 {
                let subject = words[..i].join(" ");
                let predicate = word_lower;
                let object = words[i + 1..].join(" ");

                if !subject.is_empty() && !object.is_empty() {
                    let now = chrono::Utc::now();
                    return Some(SemanticFact {
                        id: uuid::Uuid::new_v4().to_string(),
                        subject,
                        predicate,
                        object,
                        confidence: 0.6,
                        source_episodes: vec![episode_id.to_string()],
                        created_at: now,
                        last_confirmed: now,
                    });
                }
            }
        }
        None
    }
}

impl FactExtractor for LlmFactExtractor {
    fn extract(&self, episodes: &[Episode]) -> Vec<SemanticFact> {
        let mut facts = Vec::new();
        for episode in episodes {
            // Process content
            for sentence in Self::split_sentences(&episode.content) {
                let word_count = sentence.split_whitespace().count();
                if word_count >= self.min_sentence_words {
                    if let Some(fact) = Self::extract_triple(sentence, &episode.id) {
                        facts.push(fact);
                    }
                }
            }
            // Process context
            for sentence in Self::split_sentences(&episode.context) {
                let word_count = sentence.split_whitespace().count();
                if word_count >= self.min_sentence_words {
                    if let Some(fact) = Self::extract_triple(sentence, &episode.id) {
                        facts.push(fact);
                    }
                }
            }
        }
        facts
    }

    fn name(&self) -> &str {
        "LlmFactExtractor"
    }
}

/// Store for semantic facts with deduplication and querying.
pub struct FactStore {
    facts: Vec<SemanticFact>,
}

impl FactStore {
    /// Create an empty fact store.
    pub fn new() -> Self {
        Self { facts: Vec::new() }
    }

    /// Add a fact. Returns `true` if the fact is new, `false` if it was merged
    /// with an existing fact that has the same subject+predicate+object.
    pub fn add_fact(&mut self, fact: SemanticFact) -> bool {
        // Check for duplicate (same subject+predicate+object, case-insensitive)
        let existing_idx = self.facts.iter().position(|f| {
            f.subject.to_lowercase() == fact.subject.to_lowercase()
                && f.predicate.to_lowercase() == fact.predicate.to_lowercase()
                && f.object.to_lowercase() == fact.object.to_lowercase()
        });

        if let Some(idx) = existing_idx {
            // Merge: increase confidence, add source episodes
            let existing = &mut self.facts[idx];
            existing.confidence = (existing.confidence + fact.confidence * 0.5).min(1.0);
            existing.last_confirmed = chrono::Utc::now();
            for src in &fact.source_episodes {
                if !existing.source_episodes.contains(src) {
                    existing.source_episodes.push(src.clone());
                }
            }
            false
        } else {
            self.facts.push(fact);
            true
        }
    }

    /// Find all facts with the given subject (case-insensitive).
    pub fn find_by_subject(&self, subject: &str) -> Vec<&SemanticFact> {
        let subject_lower = subject.to_lowercase();
        self.facts
            .iter()
            .filter(|f| f.subject.to_lowercase() == subject_lower)
            .collect()
    }

    /// Find all facts with the given predicate (case-insensitive).
    pub fn find_by_predicate(&self, predicate: &str) -> Vec<&SemanticFact> {
        let pred_lower = predicate.to_lowercase();
        self.facts
            .iter()
            .filter(|f| f.predicate.to_lowercase() == pred_lower)
            .collect()
    }

    /// Get all facts.
    pub fn get_all(&self) -> &[SemanticFact] {
        &self.facts
    }

    /// Increase confidence and add a source episode to an existing fact.
    pub fn merge_confidence(
        &mut self,
        existing_id: &str,
        new_confidence: f64,
        source_episode: &str,
    ) {
        if let Some(fact) = self.facts.iter_mut().find(|f| f.id == existing_id) {
            fact.confidence = (fact.confidence + new_confidence * 0.5).min(1.0);
            fact.last_confirmed = chrono::Utc::now();
            if !fact.source_episodes.contains(&source_episode.to_string()) {
                fact.source_episodes.push(source_episode.to_string());
            }
        }
    }

    /// Remove facts with confidence below the threshold. Returns the count removed.
    pub fn remove_low_confidence(&mut self, threshold: f64) -> usize {
        let before = self.facts.len();
        self.facts.retain(|f| f.confidence >= threshold);
        before - self.facts.len()
    }

    /// Number of stored facts.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// Schedule for automatic consolidation.
#[derive(Debug, Clone)]
pub enum ConsolidationSchedule {
    /// Only consolidate when explicitly called.
    OnDemand,
    /// Consolidate every N episodes added.
    EveryNEpisodes(usize),
    /// Consolidate on a periodic interval (seconds).
    Periodic { interval_secs: u64 },
}

/// Result of a consolidation pipeline run.
#[derive(Debug, Clone)]
pub struct ConsolidationPipelineResult {
    /// Total facts extracted across all extractors.
    pub facts_extracted: usize,
    /// Number of genuinely new facts added.
    pub facts_new: usize,
    /// Number of facts merged with existing ones.
    pub facts_merged: usize,
    /// Duration of the consolidation in milliseconds.
    pub duration_ms: u64,
}

/// Orchestrates fact extraction and storage using multiple extractors.
pub struct EnhancedConsolidator {
    extractors: Vec<Box<dyn FactExtractor>>,
    fact_store: FactStore,
    schedule: ConsolidationSchedule,
    episodes_since_last: usize,
}

impl EnhancedConsolidator {
    /// Create a new enhanced consolidator with the given schedule.
    pub fn new(schedule: ConsolidationSchedule) -> Self {
        Self {
            extractors: Vec::new(),
            fact_store: FactStore::new(),
            schedule,
            episodes_since_last: 0,
        }
    }

    /// Add an extractor to the pipeline.
    pub fn add_extractor(&mut self, extractor: Box<dyn FactExtractor>) -> &mut Self {
        self.extractors.push(extractor);
        self
    }

    /// Check if consolidation should run based on the schedule.
    pub fn should_consolidate(&self) -> bool {
        match &self.schedule {
            ConsolidationSchedule::OnDemand => false,
            ConsolidationSchedule::EveryNEpisodes(n) => self.episodes_since_last >= *n,
            ConsolidationSchedule::Periodic { .. } => {
                // In a real system this would check elapsed time.
                // For testing purposes we return false; consolidation is triggered manually.
                false
            }
        }
    }

    /// Run all extractors on the given episodes and store results.
    pub fn consolidate(&mut self, episodes: &[Episode]) -> ConsolidationPipelineResult {
        let start = std::time::Instant::now();
        let mut total_extracted = 0usize;
        let mut total_new = 0usize;
        let mut total_merged = 0usize;

        for extractor in &self.extractors {
            let facts = extractor.extract(episodes);
            total_extracted += facts.len();
            for fact in facts {
                if self.fact_store.add_fact(fact) {
                    total_new += 1;
                } else {
                    total_merged += 1;
                }
            }
        }

        self.episodes_since_last = 0;

        ConsolidationPipelineResult {
            facts_extracted: total_extracted,
            facts_new: total_new,
            facts_merged: total_merged,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Get a reference to the internal fact store.
    pub fn get_facts(&self) -> &FactStore {
        &self.fact_store
    }

    /// Notify the consolidator that an episode was added. Increments the
    /// internal counter used by `EveryNEpisodes` scheduling.
    pub fn notify_episode_added(&mut self) {
        self.episodes_since_last += 1;
    }
}
