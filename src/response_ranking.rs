//! Response ranking
//!
//! Rank multiple response candidates to select the best one.

use std::collections::HashMap;

/// Response candidate
#[derive(Debug, Clone)]
pub struct ResponseCandidate {
    pub id: String,
    pub content: String,
    pub model: String,
    pub generation_time_ms: u64,
    pub token_count: usize,
    pub metadata: HashMap<String, String>,
}

impl ResponseCandidate {
    pub fn new(content: &str, model: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            model: model.to_string(),
            generation_time_ms: 0,
            token_count: content.split_whitespace().count(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_time(mut self, ms: u64) -> Self {
        self.generation_time_ms = ms;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Ranking criteria
#[derive(Debug, Clone, Copy)]
pub struct RankingCriteria {
    /// Weight for relevance (0-1)
    pub relevance_weight: f64,
    /// Weight for coherence (0-1)
    pub coherence_weight: f64,
    /// Weight for completeness (0-1)
    pub completeness_weight: f64,
    /// Weight for conciseness (0-1)
    pub conciseness_weight: f64,
    /// Weight for safety (0-1)
    pub safety_weight: f64,
    /// Preferred response length (0 = any)
    pub preferred_length: usize,
}

impl Default for RankingCriteria {
    fn default() -> Self {
        Self {
            relevance_weight: 0.3,
            coherence_weight: 0.2,
            completeness_weight: 0.2,
            conciseness_weight: 0.15,
            safety_weight: 0.15,
            preferred_length: 0,
        }
    }
}

/// Score breakdown for a response
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    pub relevance: f64,
    pub coherence: f64,
    pub completeness: f64,
    pub conciseness: f64,
    pub safety: f64,
    pub overall: f64,
}

/// Ranked response
#[derive(Debug, Clone)]
pub struct RankedResponse {
    pub candidate: ResponseCandidate,
    pub rank: usize,
    pub score: f64,
    pub breakdown: ScoreBreakdown,
}

/// Response ranker
pub struct ResponseRanker {
    criteria: RankingCriteria,
    unsafe_patterns: Vec<String>,
    quality_keywords: Vec<String>,
}

impl ResponseRanker {
    pub fn new(criteria: RankingCriteria) -> Self {
        Self {
            criteria,
            unsafe_patterns: vec![
                "i cannot".to_string(),
                "i'm unable".to_string(),
                "as an ai".to_string(),
                "i don't have access".to_string(),
            ],
            quality_keywords: vec![
                "because".to_string(),
                "therefore".to_string(),
                "for example".to_string(),
                "specifically".to_string(),
                "in conclusion".to_string(),
            ],
        }
    }

    /// Rank multiple responses
    pub fn rank(&self, query: &str, candidates: Vec<ResponseCandidate>) -> Vec<RankedResponse> {
        let mut ranked: Vec<RankedResponse> = candidates.into_iter()
            .map(|c| self.score_candidate(query, c))
            .collect();

        // Sort by score descending
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Assign ranks
        for (i, r) in ranked.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        ranked
    }

    /// Select the best response
    pub fn select_best(&self, query: &str, candidates: Vec<ResponseCandidate>) -> Option<RankedResponse> {
        self.rank(query, candidates).into_iter().next()
    }

    fn score_candidate(&self, query: &str, candidate: ResponseCandidate) -> RankedResponse {
        let relevance = self.score_relevance(query, &candidate.content);
        let coherence = self.score_coherence(&candidate.content);
        let completeness = self.score_completeness(query, &candidate.content);
        let conciseness = self.score_conciseness(&candidate.content);
        let safety = self.score_safety(&candidate.content);

        let overall =
            relevance * self.criteria.relevance_weight +
            coherence * self.criteria.coherence_weight +
            completeness * self.criteria.completeness_weight +
            conciseness * self.criteria.conciseness_weight +
            safety * self.criteria.safety_weight;

        RankedResponse {
            candidate,
            rank: 0,
            score: overall,
            breakdown: ScoreBreakdown {
                relevance,
                coherence,
                completeness,
                conciseness,
                safety,
                overall,
            },
        }
    }

    fn score_relevance(&self, query: &str, response: &str) -> f64 {
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let response_lower = response.to_lowercase();
        let matches = query_words.iter()
            .filter(|w| response_lower.contains(*w))
            .count();

        if query_words.is_empty() {
            return 0.5;
        }

        (matches as f64 / query_words.len() as f64).min(1.0)
    }

    fn score_coherence(&self, response: &str) -> f64 {
        let mut score: f64 = 0.5;

        // Check for quality indicators
        let lower = response.to_lowercase();
        for keyword in &self.quality_keywords {
            if lower.contains(keyword) {
                score += 0.1;
            }
        }

        // Check sentence structure
        let sentences: Vec<_> = response.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.len() >= 2 {
            score += 0.1;
        }

        // Penalize very short or very repetitive responses
        if response.len() < 20 {
            score -= 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    fn score_completeness(&self, query: &str, response: &str) -> f64 {
        let mut score: f64 = 0.5;

        // Check if response addresses question type
        let query_lower = query.to_lowercase();
        let response_lower = response.to_lowercase();

        if query_lower.contains("how") && (response_lower.contains("step") || response_lower.contains("first")) {
            score += 0.2;
        }

        if query_lower.contains("why") && (response_lower.contains("because") || response_lower.contains("reason")) {
            score += 0.2;
        }

        if query_lower.contains("what") && response.len() > 50 {
            score += 0.1;
        }

        // Check for examples
        if response_lower.contains("example") || response_lower.contains("for instance") {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    fn score_conciseness(&self, response: &str) -> f64 {
        let word_count = response.split_whitespace().count();

        if self.criteria.preferred_length > 0 {
            // Score based on distance from preferred length
            let diff = (word_count as i64 - self.criteria.preferred_length as i64).abs();
            let tolerance = self.criteria.preferred_length / 4;

            if diff as usize <= tolerance {
                return 1.0;
            } else {
                return (1.0 - (diff as f64 / (self.criteria.preferred_length as f64 * 2.0))).max(0.2);
            }
        }

        // Default scoring: prefer 50-200 words
        if word_count < 20 {
            0.3
        } else if word_count < 50 {
            0.6
        } else if word_count <= 200 {
            1.0
        } else if word_count <= 500 {
            0.7
        } else {
            0.4
        }
    }

    fn score_safety(&self, response: &str) -> f64 {
        let lower = response.to_lowercase();

        // Check for refusal patterns (not unsafe, but less helpful)
        for pattern in &self.unsafe_patterns {
            if lower.contains(pattern) {
                return 0.3;
            }
        }

        1.0
    }
}

impl Default for ResponseRanker {
    fn default() -> Self {
        Self::new(RankingCriteria::default())
    }
}

/// Builder for ranking criteria
pub struct RankingCriteriaBuilder {
    criteria: RankingCriteria,
}

impl RankingCriteriaBuilder {
    pub fn new() -> Self {
        Self {
            criteria: RankingCriteria::default(),
        }
    }

    pub fn relevance_weight(mut self, weight: f64) -> Self {
        self.criteria.relevance_weight = weight;
        self
    }

    pub fn coherence_weight(mut self, weight: f64) -> Self {
        self.criteria.coherence_weight = weight;
        self
    }

    pub fn completeness_weight(mut self, weight: f64) -> Self {
        self.criteria.completeness_weight = weight;
        self
    }

    pub fn conciseness_weight(mut self, weight: f64) -> Self {
        self.criteria.conciseness_weight = weight;
        self
    }

    pub fn safety_weight(mut self, weight: f64) -> Self {
        self.criteria.safety_weight = weight;
        self
    }

    pub fn preferred_length(mut self, length: usize) -> Self {
        self.criteria.preferred_length = length;
        self
    }

    pub fn build(self) -> ResponseRanker {
        ResponseRanker::new(self.criteria)
    }
}

impl Default for RankingCriteriaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranking() {
        let ranker = ResponseRanker::default();

        let candidates = vec![
            ResponseCandidate::new("Short", "model1"),
            ResponseCandidate::new(
                "This is a more complete response that addresses the question with examples and reasoning.",
                "model2"
            ),
            ResponseCandidate::new("I cannot help with that.", "model3"),
        ];

        let ranked = ranker.rank("Tell me about AI", candidates);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].rank, 1);
        // The longer, more complete response should rank higher
        assert!(ranked[0].candidate.content.len() > 20);
    }

    #[test]
    fn test_select_best() {
        let ranker = ResponseRanker::default();

        let candidates = vec![
            ResponseCandidate::new("Good answer with details.", "model1"),
            ResponseCandidate::new("Bad", "model2"),
        ];

        let best = ranker.select_best("Question?", candidates).unwrap();
        assert!(best.candidate.content.contains("details"));
    }

    #[test]
    fn test_relevance_scoring() {
        let ranker = ResponseRanker::default();

        let relevant = ResponseCandidate::new(
            "Python is a programming language known for its simplicity.",
            "model"
        );
        let irrelevant = ResponseCandidate::new(
            "The weather is nice today.",
            "model"
        );

        let query = "Tell me about Python programming";
        let ranked = ranker.rank(query, vec![relevant, irrelevant]);

        assert!(ranked[0].breakdown.relevance > ranked[1].breakdown.relevance);
    }
}
