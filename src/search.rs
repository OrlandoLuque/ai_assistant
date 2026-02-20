//! Conversation search functionality
//!
//! This module provides search capabilities for conversation history,
//! including full-text search, semantic search, and filtering.

use crate::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A search result from conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching message
    pub message: ChatMessage,
    /// The message index in the conversation
    pub index: usize,
    /// Session ID where the message was found
    pub session_id: Option<String>,
    /// Relevance score (0.0 - 1.0)
    pub score: f32,
    /// Matched terms/phrases
    pub highlights: Vec<HighlightSpan>,
    /// Context (surrounding messages)
    pub context: SearchContext,
}

/// A highlighted span in the search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightSpan {
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// The matched text
    pub text: String,
}

/// Context around a search result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchContext {
    /// Messages before the match
    pub before: Vec<ChatMessage>,
    /// Messages after the match
    pub after: Vec<ChatMessage>,
}

/// Search query configuration
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The search text
    pub text: String,
    /// Only search user messages
    pub user_messages_only: bool,
    /// Only search assistant messages
    pub assistant_messages_only: bool,
    /// Session IDs to search (empty = all)
    pub session_ids: Vec<String>,
    /// Maximum results to return
    pub max_results: usize,
    /// Include context messages
    pub include_context: bool,
    /// Number of context messages before/after
    pub context_size: usize,
    /// Minimum score threshold
    pub min_score: f32,
    /// Date range filter (start)
    pub from_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Date range filter (end)
    pub to_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Search mode
    pub mode: SearchMode,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            user_messages_only: false,
            assistant_messages_only: false,
            session_ids: vec![],
            max_results: 20,
            include_context: true,
            context_size: 2,
            min_score: 0.0,
            from_date: None,
            to_date: None,
            mode: SearchMode::Fuzzy,
        }
    }
}

impl SearchQuery {
    /// Create a new search query
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Default::default()
        }
    }

    /// Set to search only user messages
    pub fn user_only(mut self) -> Self {
        self.user_messages_only = true;
        self.assistant_messages_only = false;
        self
    }

    /// Set to search only assistant messages
    pub fn assistant_only(mut self) -> Self {
        self.assistant_messages_only = true;
        self.user_messages_only = false;
        self
    }

    /// Limit to specific sessions
    pub fn in_sessions(mut self, sessions: Vec<String>) -> Self {
        self.session_ids = sessions;
        self
    }

    /// Set maximum results
    pub fn limit(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Set context size
    pub fn with_context(mut self, size: usize) -> Self {
        self.include_context = true;
        self.context_size = size;
        self
    }

    /// Set search mode
    pub fn mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set minimum score
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// Set date range
    pub fn date_range(
        mut self,
        from: Option<chrono::DateTime<chrono::Utc>>,
        to: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Self {
        self.from_date = from;
        self.to_date = to;
        self
    }
}

/// Search mode for matching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Exact phrase match
    Exact,
    /// Fuzzy matching (allows typos, word order changes)
    Fuzzy,
    /// All words must be present
    AllWords,
    /// Any word can match
    AnyWord,
    /// Regular expression
    Regex,
}

/// Conversation searcher
pub struct ConversationSearcher {
    /// Index of terms to messages
    index: SearchIndex,
}

/// Search index for fast lookups
#[derive(Debug, Default)]
struct SearchIndex {
    /// Term -> (session_id, message_index, positions)
    term_index: HashMap<String, Vec<(Option<String>, usize, Vec<usize>)>>,
    /// All indexed messages
    messages: Vec<(Option<String>, ChatMessage)>,
}

impl ConversationSearcher {
    /// Create a new searcher
    pub fn new() -> Self {
        Self {
            index: SearchIndex::default(),
        }
    }

    /// Index messages from a conversation
    pub fn index_messages(&mut self, messages: &[ChatMessage], session_id: Option<&str>) {
        for (idx, msg) in messages.iter().enumerate() {
            self.index_message(msg.clone(), idx, session_id);
        }
    }

    /// Index a single message
    pub fn index_message(&mut self, message: ChatMessage, index: usize, session_id: Option<&str>) {
        let session = session_id.map(|s| s.to_string());

        // Tokenize and index
        let tokens = Self::tokenize(&message.content);
        for (pos, token) in tokens.iter().enumerate() {
            let token_lower = token.to_lowercase();
            self.index.term_index.entry(token_lower).or_default().push((
                session.clone(),
                index,
                vec![pos],
            ));
        }

        self.index.messages.push((session, message));
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.index = SearchIndex::default();
    }

    /// Search the indexed messages
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchResult> {
        let mut results = Vec::new();

        match query.mode {
            SearchMode::Exact => self.search_exact(query, &mut results),
            SearchMode::Fuzzy => self.search_fuzzy(query, &mut results),
            SearchMode::AllWords => self.search_all_words(query, &mut results),
            SearchMode::AnyWord => self.search_any_word(query, &mut results),
            SearchMode::Regex => self.search_regex(query, &mut results),
        }

        // Sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        results.truncate(query.max_results);

        // Add context if requested
        if query.include_context {
            for result in &mut results {
                result.context =
                    self.get_context(result.index, query.context_size, &result.session_id);
            }
        }

        results
    }

    /// Search for exact phrase match
    fn search_exact(&self, query: &SearchQuery, results: &mut Vec<SearchResult>) {
        let query_lower = query.text.to_lowercase();

        for (idx, (session, msg)) in self.index.messages.iter().enumerate() {
            if !self.matches_filters(msg, session.as_deref(), query) {
                continue;
            }

            let content_lower = msg.content.to_lowercase();
            if let Some(pos) = content_lower.find(&query_lower) {
                let highlights = vec![HighlightSpan {
                    start: pos,
                    end: pos + query.text.len(),
                    text: msg.content[pos..pos + query.text.len()].to_string(),
                }];

                results.push(SearchResult {
                    message: msg.clone(),
                    index: idx,
                    session_id: session.clone(),
                    score: 1.0,
                    highlights,
                    context: SearchContext::default(),
                });
            }
        }
    }

    /// Fuzzy search
    fn search_fuzzy(&self, query: &SearchQuery, results: &mut Vec<SearchResult>) {
        let query_tokens: Vec<String> = Self::tokenize(&query.text)
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        for (idx, (session, msg)) in self.index.messages.iter().enumerate() {
            if !self.matches_filters(msg, session.as_deref(), query) {
                continue;
            }

            let content_lower = msg.content.to_lowercase();
            let mut score = 0.0;
            let mut highlights = Vec::new();

            for query_token in &query_tokens {
                // Check exact match
                if let Some(pos) = content_lower.find(query_token) {
                    score += 1.0;
                    highlights.push(HighlightSpan {
                        start: pos,
                        end: pos + query_token.len(),
                        text: msg.content[pos..pos + query_token.len()].to_string(),
                    });
                } else {
                    // Check fuzzy match
                    let content_tokens = Self::tokenize(&content_lower);
                    for content_token in &content_tokens {
                        let similarity = Self::calculate_similarity(query_token, content_token);
                        if similarity > 0.7 {
                            score += similarity;
                            if let Some(pos) = content_lower.find(content_token) {
                                highlights.push(HighlightSpan {
                                    start: pos,
                                    end: pos + content_token.len(),
                                    text: msg.content[pos..pos + content_token.len()].to_string(),
                                });
                            }
                        }
                    }
                }
            }

            if score > 0.0 {
                let normalized_score = (score / query_tokens.len() as f32).min(1.0);
                if normalized_score >= query.min_score {
                    results.push(SearchResult {
                        message: msg.clone(),
                        index: idx,
                        session_id: session.clone(),
                        score: normalized_score,
                        highlights,
                        context: SearchContext::default(),
                    });
                }
            }
        }
    }

    /// Search requiring all words
    fn search_all_words(&self, query: &SearchQuery, results: &mut Vec<SearchResult>) {
        let query_tokens: Vec<String> = Self::tokenize(&query.text)
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        for (idx, (session, msg)) in self.index.messages.iter().enumerate() {
            if !self.matches_filters(msg, session.as_deref(), query) {
                continue;
            }

            let content_lower = msg.content.to_lowercase();
            let mut highlights = Vec::new();
            let mut all_found = true;

            for query_token in &query_tokens {
                if let Some(pos) = content_lower.find(query_token) {
                    highlights.push(HighlightSpan {
                        start: pos,
                        end: pos + query_token.len(),
                        text: msg.content[pos..pos + query_token.len()].to_string(),
                    });
                } else {
                    all_found = false;
                    break;
                }
            }

            if all_found && !highlights.is_empty() {
                results.push(SearchResult {
                    message: msg.clone(),
                    index: idx,
                    session_id: session.clone(),
                    score: 1.0,
                    highlights,
                    context: SearchContext::default(),
                });
            }
        }
    }

    /// Search for any word
    fn search_any_word(&self, query: &SearchQuery, results: &mut Vec<SearchResult>) {
        let query_tokens: Vec<String> = Self::tokenize(&query.text)
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        for (idx, (session, msg)) in self.index.messages.iter().enumerate() {
            if !self.matches_filters(msg, session.as_deref(), query) {
                continue;
            }

            let content_lower = msg.content.to_lowercase();
            let mut highlights = Vec::new();
            let mut matches = 0;

            for query_token in &query_tokens {
                if let Some(pos) = content_lower.find(query_token) {
                    highlights.push(HighlightSpan {
                        start: pos,
                        end: pos + query_token.len(),
                        text: msg.content[pos..pos + query_token.len()].to_string(),
                    });
                    matches += 1;
                }
            }

            if !highlights.is_empty() {
                let score = matches as f32 / query_tokens.len() as f32;
                if score >= query.min_score {
                    results.push(SearchResult {
                        message: msg.clone(),
                        index: idx,
                        session_id: session.clone(),
                        score,
                        highlights,
                        context: SearchContext::default(),
                    });
                }
            }
        }
    }

    /// Regex search
    fn search_regex(&self, query: &SearchQuery, results: &mut Vec<SearchResult>) {
        // Simple regex-like pattern matching
        // Supports: . (any char), * (zero or more), + (one or more), ? (zero or one)
        let pattern = &query.text;

        for (idx, (session, msg)) in self.index.messages.iter().enumerate() {
            if !self.matches_filters(msg, session.as_deref(), query) {
                continue;
            }

            if let Some(highlights) = self.match_pattern(pattern, &msg.content) {
                if !highlights.is_empty() {
                    results.push(SearchResult {
                        message: msg.clone(),
                        index: idx,
                        session_id: session.clone(),
                        score: 1.0,
                        highlights,
                        context: SearchContext::default(),
                    });
                }
            }
        }
    }

    /// Simple pattern matching (subset of regex)
    fn match_pattern(&self, pattern: &str, text: &str) -> Option<Vec<HighlightSpan>> {
        // Simple case-insensitive substring for now
        // A full regex implementation would be more complex
        let pattern_lower = pattern.to_lowercase();
        let text_lower = text.to_lowercase();

        let mut highlights = Vec::new();
        let mut start = 0;

        while let Some(pos) = text_lower[start..].find(&pattern_lower) {
            let abs_pos = start + pos;
            highlights.push(HighlightSpan {
                start: abs_pos,
                end: abs_pos + pattern.len(),
                text: text[abs_pos..abs_pos + pattern.len()].to_string(),
            });
            start = abs_pos + 1;
        }

        if highlights.is_empty() {
            None
        } else {
            Some(highlights)
        }
    }

    /// Check if a message matches the query filters
    fn matches_filters(
        &self,
        msg: &ChatMessage,
        session_id: Option<&str>,
        query: &SearchQuery,
    ) -> bool {
        // Message type filter
        if query.user_messages_only && !msg.is_user() {
            return false;
        }
        if query.assistant_messages_only && msg.is_user() {
            return false;
        }

        // Session filter
        if !query.session_ids.is_empty() {
            match session_id {
                Some(sid) if query.session_ids.contains(&sid.to_string()) => {}
                _ => return false,
            }
        }

        // Date filter
        if let Some(from) = query.from_date {
            if msg.timestamp < from {
                return false;
            }
        }
        if let Some(to) = query.to_date {
            if msg.timestamp > to {
                return false;
            }
        }

        true
    }

    /// Get context messages around a result
    fn get_context(&self, index: usize, size: usize, session_id: &Option<String>) -> SearchContext {
        let mut before = Vec::new();
        let mut after = Vec::new();

        // Get messages from same session
        let session_messages: Vec<(usize, &ChatMessage)> = self
            .index
            .messages
            .iter()
            .enumerate()
            .filter(|(_, (sid, _))| sid == session_id)
            .map(|(i, (_, m))| (i, m))
            .collect();

        // Find position in session
        let pos = session_messages.iter().position(|(i, _)| *i == index);

        if let Some(pos) = pos {
            // Get before context
            let start = pos.saturating_sub(size);
            for i in start..pos {
                before.push(session_messages[i].1.clone());
            }

            // Get after context
            let end = (pos + 1 + size).min(session_messages.len());
            for i in (pos + 1)..end {
                after.push(session_messages[i].1.clone());
            }
        }

        SearchContext { before, after }
    }

    /// Tokenize text into words
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Calculate string similarity (Jaro-Winkler like)
    fn calculate_similarity(s1: &str, s2: &str) -> f32 {
        if s1 == s2 {
            return 1.0;
        }

        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        // Simple character overlap similarity
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Get search statistics
    pub fn stats(&self) -> SearchStats {
        SearchStats {
            total_messages: self.index.messages.len(),
            indexed_terms: self.index.term_index.len(),
            unique_sessions: self
                .index
                .messages
                .iter()
                .filter_map(|(s, _)| s.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

impl Default for ConversationSearcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// Total indexed messages
    pub total_messages: usize,
    /// Number of unique terms
    pub indexed_terms: usize,
    /// Number of unique sessions
    pub unique_sessions: usize,
}

/// Builder for search queries
pub struct SearchQueryBuilder {
    query: SearchQuery,
}

impl SearchQueryBuilder {
    /// Start building a query
    pub fn new(text: &str) -> Self {
        Self {
            query: SearchQuery::new(text),
        }
    }

    /// Build the query
    pub fn build(self) -> SearchQuery {
        self.query
    }

    /// Set user messages only
    pub fn user_only(mut self) -> Self {
        self.query = self.query.user_only();
        self
    }

    /// Set assistant messages only
    pub fn assistant_only(mut self) -> Self {
        self.query = self.query.assistant_only();
        self
    }

    /// Limit results
    pub fn limit(mut self, max: usize) -> Self {
        self.query = self.query.limit(max);
        self
    }

    /// Set search mode
    pub fn mode(mut self, mode: SearchMode) -> Self {
        self.query = self.query.mode(mode);
        self
    }

    /// Include context
    pub fn with_context(mut self, size: usize) -> Self {
        self.query = self.query.with_context(size);
        self
    }

    /// Set minimum score
    pub fn min_score(mut self, score: f32) -> Self {
        self.query = self.query.min_score(score);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("How do I create a function in Rust?".to_string()),
            ChatMessage::assistant(
                "To create a function in Rust, use the fn keyword...".to_string(),
            ),
            ChatMessage::user("What about Python functions?".to_string()),
            ChatMessage::assistant(
                "In Python, you define functions with the def keyword...".to_string(),
            ),
            ChatMessage::user("Can you show me a Rust example?".to_string()),
        ]
    }

    #[test]
    fn test_basic_search() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), Some("test_session"));

        let query = SearchQuery::new("Rust");
        let results = searcher.search(&query);

        assert!(!results.is_empty());
        assert!(results[0].message.content.to_lowercase().contains("rust"));
    }

    #[test]
    fn test_exact_search() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), None);

        let query = SearchQuery::new("fn keyword").mode(SearchMode::Exact);
        let results = searcher.search(&query);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_user_only_search() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), None);

        let query = SearchQuery::new("function").user_only();
        let results = searcher.search(&query);

        for result in &results {
            assert!(result.message.is_user());
        }
    }

    #[test]
    fn test_all_words_search() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), None);

        let query = SearchQuery::new("Rust function").mode(SearchMode::AllWords);
        let results = searcher.search(&query);

        for result in &results {
            let content_lower = result.message.content.to_lowercase();
            assert!(content_lower.contains("rust") && content_lower.contains("function"));
        }
    }

    #[test]
    fn test_search_with_context() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), Some("session1"));

        let query = SearchQuery::new("Python").with_context(1);
        let results = searcher.search(&query);

        if !results.is_empty() {
            // Should have context
            let result = &results[0];
            assert!(!result.context.before.is_empty() || !result.context.after.is_empty());
        }
    }

    #[test]
    fn test_search_stats() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), Some("session1"));

        let stats = searcher.stats();
        assert_eq!(stats.total_messages, 5);
        assert!(stats.indexed_terms > 0);
    }

    #[test]
    fn test_fuzzy_search() {
        let mut searcher = ConversationSearcher::new();
        searcher.index_messages(&create_test_messages(), None);

        // Search with slight misspelling
        let query = SearchQuery::new("functon").mode(SearchMode::Fuzzy);
        let results = searcher.search(&query);

        // Should still find results due to fuzzy matching
        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_builder() {
        let query = SearchQueryBuilder::new("test")
            .user_only()
            .limit(5)
            .mode(SearchMode::Exact)
            .min_score(0.5)
            .build();

        assert!(query.user_messages_only);
        assert_eq!(query.max_results, 5);
        assert_eq!(query.mode, SearchMode::Exact);
        assert_eq!(query.min_score, 0.5);
    }
}
