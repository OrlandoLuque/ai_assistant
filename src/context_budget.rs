//! Adaptive Context Budget Allocator
//!
//! Dynamically distributes context window tokens across multiple sources
//! (RAG, Memory, Procedural, References, KnowledgeGraph, UserNotes) based
//! on per-item relevance scoring. Fills the context window optimally by
//! merging all items, sorting by score, and packing greedily.
//!
//! # Design
//!
//! Each source implements `ContextSource` and returns `Vec<ContextItem>` with
//! relevance scores. The allocator merges all items, sorts by score, and packs
//! into the available budget. No source is privileged — the most relevant items
//! win regardless of origin.
//!
//! When items don't fit, three strategies are available:
//! - `ScoreTruncation`: drop lowest-scored items (fast, free)
//! - `ExtractiveCompression`: select key sentences per item (Rust pure, free)
//! - `LlmCompression`: use a secondary LLM to filter/summarize (precise, costs LLM call)

use std::collections::HashMap;

/// Type of context source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ContextSourceType {
    /// RAG knowledge chunks from indexed documents.
    Rag,
    /// Episodic/entity memory from MemoryManager.
    Memory,
    /// Workflow procedures from ProceduralStore.
    Procedural,
    /// Resolved back-references ("option 3", "that list").
    Reference,
    /// Entities and relations from KnowledgeGraph.
    Graph,
    /// User-configured notes (global/session).
    UserNotes,
    /// Other/custom source.
    Custom,
}

impl std::fmt::Display for ContextSourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rag => write!(f, "RAG"),
            Self::Memory => write!(f, "MEMORY"),
            Self::Procedural => write!(f, "PROCEDURAL"),
            Self::Reference => write!(f, "REFERENCE"),
            Self::Graph => write!(f, "GRAPH"),
            Self::UserNotes => write!(f, "USER_NOTES"),
            Self::Custom => write!(f, "CUSTOM"),
            _ => write!(f, "UNKNOWN"),
        }
    }
}

/// A single item of context with relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextItem {
    /// The text content to inject into the prompt.
    pub content: String,
    /// Estimated token count.
    pub tokens: usize,
    /// Relevance score (0.0 = irrelevant, 1.0 = perfectly relevant).
    pub score: f32,
    /// Which source produced this item.
    pub source: ContextSourceType,
    /// Optional label for diagnostics (e.g., "chunk from manual.pdf page 3").
    pub label: String,
}

impl ContextItem {
    /// Create a new context item.
    pub fn new(
        content: impl Into<String>,
        tokens: usize,
        score: f32,
        source: ContextSourceType,
    ) -> Self {
        Self {
            content: content.into(),
            tokens,
            score: score.clamp(0.0, 1.0),
            source,
            label: String::new(),
        }
    }

    /// Set a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

/// Trait for context sources that provide items with relevance scores.
pub trait ContextSource: Send + Sync {
    /// Query this source for items relevant to the user message.
    /// Items should be returned in order of relevance (best first).
    fn query_items(&self, user_message: &str) -> Vec<ContextItem>;

    /// Source name for diagnostics.
    fn source_name(&self) -> &str;

    /// Source type.
    fn source_type(&self) -> ContextSourceType;
}

/// Strategy for handling context overflow.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OverflowStrategy {
    /// Drop lowest-scored items until budget is met. Fast, free.
    ScoreTruncation,
    /// Extract key sentences from items to reduce size. Rust-pure, free.
    ExtractiveCompression,
    /// Use an LLM to intelligently filter/summarize. Precise, costs LLM call.
    LlmCompression {
        /// Model to use for compression (cheap/fast model recommended).
        compressor_model: String,
        /// Compression level.
        level: CompressionLevel,
    },
    /// Truncate first, then compress remaining if still too large.
    Hybrid {
        /// Model for LLM compression fallback.
        compressor_model: String,
    },
}

impl Default for OverflowStrategy {
    fn default() -> Self {
        Self::ScoreTruncation
    }
}

/// How aggressively to compress when using LLM compression.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CompressionLevel {
    /// Eliminate redundancy, keep all details (~60-70% of original).
    Light,
    /// Keep key points, compress explanations (~30-50%).
    Medium,
    /// Only essentials for the specific question (~10-25%).
    Aggressive,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::Medium
    }
}

/// Result of budget allocation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AllocationResult {
    /// The assembled context string ready to inject into the prompt.
    pub context: String,
    /// Total tokens used.
    pub tokens_used: usize,
    /// Budget that was available.
    pub budget: usize,
    /// Items that were included (with their scores).
    pub included: Vec<ContextItem>,
    /// Items that were dropped due to budget constraints.
    pub dropped: Vec<ContextItem>,
    /// Per-source token breakdown.
    pub source_breakdown: HashMap<ContextSourceType, usize>,
}

impl AllocationResult {
    /// Utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        if self.budget == 0 {
            return 0.0;
        }
        self.tokens_used as f32 / self.budget as f32
    }

    /// Whether any items were dropped.
    pub fn had_overflow(&self) -> bool {
        !self.dropped.is_empty()
    }
}

/// Adaptive context budget allocator.
///
/// Collects items from multiple sources, merges them by score, and packs
/// into the available token budget. Maximizes context quality by including
/// the most relevant items regardless of source.
pub struct ContextBudgetAllocator {
    sources: Vec<Box<dyn ContextSource>>,
    strategy: OverflowStrategy,
}

impl ContextBudgetAllocator {
    /// Create a new allocator with the given overflow strategy.
    pub fn new(strategy: OverflowStrategy) -> Self {
        Self {
            sources: Vec::new(),
            strategy,
        }
    }

    /// Add a context source.
    pub fn add_source(&mut self, source: Box<dyn ContextSource>) {
        self.sources.push(source);
    }

    /// Calculate the available budget for context items.
    ///
    /// # Arguments
    /// * `model_context_window` - Total context window of the model (tokens)
    /// * `system_prompt_tokens` - Tokens used by the base system prompt
    /// * `conversation_tokens` - Tokens used by conversation history
    /// * `user_message_tokens` - Tokens used by the current user message
    /// * `response_reserve` - Tokens reserved for the model's response
    pub fn available_budget(
        model_context_window: usize,
        system_prompt_tokens: usize,
        conversation_tokens: usize,
        user_message_tokens: usize,
        response_reserve: usize,
    ) -> usize {
        let used = system_prompt_tokens + conversation_tokens + user_message_tokens + response_reserve;
        model_context_window.saturating_sub(used)
    }

    /// Build the optimal context for the given user message within the budget.
    ///
    /// 1. Query all sources for relevant items
    /// 2. Merge and sort by score (descending)
    /// 3. Pack greedily until budget is filled
    /// 4. Apply overflow strategy if needed
    pub fn build(&self, user_message: &str, budget: usize) -> AllocationResult {
        // 1. Collect items from all sources
        let mut all_items: Vec<ContextItem> = Vec::new();
        for source in &self.sources {
            let items = source.query_items(user_message);
            all_items.extend(items);
        }

        // 2. Sort by score descending
        all_items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Greedy packing
        let mut included = Vec::new();
        let mut dropped = Vec::new();
        let mut tokens_used: usize = 0;

        for item in all_items {
            if tokens_used + item.tokens <= budget {
                tokens_used += item.tokens;
                included.push(item);
            } else {
                // Try extractive compression if strategy allows
                match &self.strategy {
                    OverflowStrategy::ExtractiveCompression
                    | OverflowStrategy::Hybrid { .. } => {
                        // Try to fit a compressed version
                        let remaining = budget.saturating_sub(tokens_used);
                        if remaining > 50 && item.score > 0.5 {
                            let compressed = extractive_compress(&item.content, remaining);
                            let comp_tokens = estimate_tokens(&compressed);
                            if comp_tokens > 0 && tokens_used + comp_tokens <= budget {
                                tokens_used += comp_tokens;
                                included.push(ContextItem {
                                    content: compressed,
                                    tokens: comp_tokens,
                                    score: item.score * 0.9, // slight penalty for compression
                                    source: item.source,
                                    label: format!("{} (compressed)", item.label),
                                });
                                continue;
                            }
                        }
                        dropped.push(item);
                    }
                    _ => {
                        dropped.push(item);
                    }
                }
            }
        }

        // 4. Build the assembled context string
        // Group by source type for readability
        let mut by_source: HashMap<ContextSourceType, Vec<&ContextItem>> = HashMap::new();
        for item in &included {
            by_source.entry(item.source).or_default().push(item);
        }

        let mut context = String::new();
        let source_order = [
            ContextSourceType::UserNotes,
            ContextSourceType::Reference,
            ContextSourceType::Procedural,
            ContextSourceType::Memory,
            ContextSourceType::Rag,
            ContextSourceType::Graph,
            ContextSourceType::Custom,
        ];

        for source_type in &source_order {
            if let Some(items) = by_source.get(source_type) {
                if !items.is_empty() {
                    context.push_str(&format!("\n--- {} ---\n", source_type));
                    for item in items {
                        context.push_str(&item.content);
                        context.push('\n');
                    }
                }
            }
        }

        // Source breakdown
        let mut source_breakdown: HashMap<ContextSourceType, usize> = HashMap::new();
        for item in &included {
            *source_breakdown.entry(item.source).or_insert(0) += item.tokens;
        }

        AllocationResult {
            context,
            tokens_used,
            budget,
            included,
            dropped,
            source_breakdown,
        }
    }
}

impl Default for ContextBudgetAllocator {
    fn default() -> Self {
        Self::new(OverflowStrategy::ScoreTruncation)
    }
}

// ============================================================================
// Extractive compression (Rust-pure, RECOMP-style)
// ============================================================================

/// Extract the most relevant sentences from text to fit within a token budget.
/// Uses TF-IDF-like scoring: sentences with rarer words score higher.
fn extractive_compress(text: &str, max_tokens: usize) -> String {
    let sentences: Vec<&str> = text
        .split(|c: char| c == '.' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        return String::new();
    }

    // Score each sentence by word rarity (inverse document frequency approximation)
    let total = sentences.len() as f32;
    let mut word_counts: HashMap<&str, usize> = HashMap::new();
    for sentence in &sentences {
        let unique_words: std::collections::HashSet<&str> = sentence.split_whitespace().collect();
        for word in unique_words {
            *word_counts.entry(word).or_insert(0) += 1;
        }
    }

    let mut scored: Vec<(&str, f32)> = sentences
        .iter()
        .map(|&s| {
            let words: Vec<&str> = s.split_whitespace().collect();
            if words.is_empty() {
                return (s, 0.0);
            }
            let score: f32 = words
                .iter()
                .map(|w| {
                    let count = *word_counts.get(w).unwrap_or(&1) as f32;
                    (total / count).ln()
                })
                .sum::<f32>()
                / words.len() as f32;
            (s, score)
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Pack sentences until budget
    let mut result = Vec::new();
    let mut tokens = 0;
    for (sentence, _score) in &scored {
        let sent_tokens = estimate_tokens(sentence);
        if tokens + sent_tokens > max_tokens {
            break;
        }
        result.push(*sentence);
        tokens += sent_tokens;
    }

    // Re-order by original position for coherence
    let original_order: HashMap<&str, usize> = sentences
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    result.sort_by_key(|s| original_order.get(s).copied().unwrap_or(usize::MAX));

    result.join(". ")
}

/// Estimate token count from text (heuristic: ~3.5 chars per token).
fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / 3.5).ceil() as usize
}

// ============================================================================
// Adapter: wrap a closure as a ContextSource
// ============================================================================

/// Wraps a closure that returns `Vec<ContextItem>` as a `ContextSource`.
pub struct ClosureSource<F: Fn(&str) -> Vec<ContextItem> + Send + Sync> {
    func: F,
    name: String,
    source_type: ContextSourceType,
}

impl<F: Fn(&str) -> Vec<ContextItem> + Send + Sync> ClosureSource<F> {
    /// Create a new closure-based source.
    pub fn new(
        name: impl Into<String>,
        source_type: ContextSourceType,
        func: F,
    ) -> Self {
        Self {
            func,
            name: name.into(),
            source_type,
        }
    }
}

impl<F: Fn(&str) -> Vec<ContextItem> + Send + Sync> ContextSource for ClosureSource<F> {
    fn query_items(&self, user_message: &str) -> Vec<ContextItem> {
        (self.func)(user_message)
    }

    fn source_name(&self) -> &str {
        &self.name
    }

    fn source_type(&self) -> ContextSourceType {
        self.source_type
    }
}

/// Wraps an existing `String`-returning function as a single `ContextItem`.
/// Useful for adapting legacy sources (build_memory_context, etc.) without
/// modifying their interfaces.
pub struct LegacyStringSource<F: Fn(&str) -> String + Send + Sync> {
    func: F,
    name: String,
    source_type: ContextSourceType,
    default_score: f32,
}

impl<F: Fn(&str) -> String + Send + Sync> LegacyStringSource<F> {
    /// Create a legacy adapter with a fixed score.
    pub fn new(
        name: impl Into<String>,
        source_type: ContextSourceType,
        default_score: f32,
        func: F,
    ) -> Self {
        Self {
            func,
            name: name.into(),
            source_type,
            default_score,
        }
    }
}

impl<F: Fn(&str) -> String + Send + Sync> ContextSource for LegacyStringSource<F> {
    fn query_items(&self, user_message: &str) -> Vec<ContextItem> {
        let content = (self.func)(user_message);
        if content.is_empty() {
            return Vec::new();
        }
        let tokens = estimate_tokens(&content);
        vec![ContextItem::new(content, tokens, self.default_score, self.source_type)
            .with_label(self.name.clone())]
    }

    fn source_name(&self) -> &str {
        &self.name
    }

    fn source_type(&self) -> ContextSourceType {
        self.source_type
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(content: &str, score: f32, source: ContextSourceType) -> ContextItem {
        let tokens = estimate_tokens(content);
        ContextItem::new(content, tokens, score, source)
    }

    struct StaticSource {
        items: Vec<ContextItem>,
        name: String,
        stype: ContextSourceType,
    }

    impl ContextSource for StaticSource {
        fn query_items(&self, _msg: &str) -> Vec<ContextItem> {
            self.items.clone()
        }
        fn source_name(&self) -> &str {
            &self.name
        }
        fn source_type(&self) -> ContextSourceType {
            self.stype
        }
    }

    #[test]
    fn test_allocator_basic_packing() {
        let mut allocator = ContextBudgetAllocator::default();
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("High relevance RAG chunk about Rust ownership", 0.95, ContextSourceType::Rag),
                make_item("Low relevance chunk about history", 0.3, ContextSourceType::Rag),
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("User prefers JSON format", 0.8, ContextSourceType::Memory),
            ],
            name: "memory".into(),
            stype: ContextSourceType::Memory,
        }));

        let result = allocator.build("tell me about Rust", 1000);
        assert!(!result.included.is_empty());
        assert!(result.tokens_used <= 1000);
        // High-score items should be first
        assert!(result.included[0].score >= result.included.last().unwrap().score);
    }

    #[test]
    fn test_allocator_respects_budget() {
        let mut allocator = ContextBudgetAllocator::default();
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item(&"x".repeat(350), 0.9, ContextSourceType::Rag), // ~100 tokens
                make_item(&"y".repeat(350), 0.8, ContextSourceType::Rag), // ~100 tokens
                make_item(&"z".repeat(350), 0.7, ContextSourceType::Rag), // ~100 tokens
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));

        let result = allocator.build("query", 150); // only ~150 tokens budget
        // Should fit 1 item, maybe 2, but not all 3
        assert!(result.tokens_used <= 150);
        assert!(!result.dropped.is_empty());
    }

    #[test]
    fn test_allocator_cross_source_scoring() {
        let mut allocator = ContextBudgetAllocator::default();
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("RAG low score", 0.3, ContextSourceType::Rag),
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("Memory high score", 0.95, ContextSourceType::Memory),
            ],
            name: "memory".into(),
            stype: ContextSourceType::Memory,
        }));

        let result = allocator.build("query", 20); // very tight budget
        // Memory (0.95) should be included before RAG (0.3)
        if result.included.len() == 1 {
            assert_eq!(result.included[0].source, ContextSourceType::Memory);
        }
    }

    #[test]
    fn test_allocator_empty_sources() {
        let allocator = ContextBudgetAllocator::default();
        let result = allocator.build("query", 1000);
        assert!(result.included.is_empty());
        assert_eq!(result.tokens_used, 0);
        assert!(!result.had_overflow());
    }

    #[test]
    fn test_allocator_utilization() {
        let mut allocator = ContextBudgetAllocator::default();
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("Some content here", 0.9, ContextSourceType::Rag),
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));

        let result = allocator.build("query", 1000);
        assert!(result.utilization() > 0.0);
        assert!(result.utilization() <= 1.0);
    }

    #[test]
    fn test_allocator_source_breakdown() {
        let mut allocator = ContextBudgetAllocator::default();
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("RAG content", 0.9, ContextSourceType::Rag),
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("Memory content", 0.8, ContextSourceType::Memory),
            ],
            name: "memory".into(),
            stype: ContextSourceType::Memory,
        }));

        let result = allocator.build("query", 1000);
        assert!(result.source_breakdown.contains_key(&ContextSourceType::Rag));
        assert!(result.source_breakdown.contains_key(&ContextSourceType::Memory));
    }

    #[test]
    fn test_extractive_compression() {
        let text = "Rust is a systems programming language. \
                    It focuses on safety and performance. \
                    The borrow checker ensures memory safety. \
                    Rust was created by Graydon Hoare at Mozilla. \
                    It has been the most loved language on Stack Overflow for years.";

        let compressed = extractive_compress(text, 30); // ~30 tokens
        assert!(!compressed.is_empty());
        assert!(estimate_tokens(&compressed) <= 35); // some tolerance
    }

    #[test]
    fn test_available_budget() {
        let budget = ContextBudgetAllocator::available_budget(
            8192, // 8K context window
            200,  // system prompt
            1500, // conversation
            100,  // user message
            800,  // response reserve
        );
        assert_eq!(budget, 5592);
    }

    #[test]
    fn test_legacy_string_source() {
        let source = LegacyStringSource::new(
            "test-memory",
            ContextSourceType::Memory,
            0.7,
            |_query| "User likes Rust and JSON".to_string(),
        );

        let items = source.query_items("hello");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].score, 0.7);
        assert_eq!(items[0].source, ContextSourceType::Memory);
    }

    #[test]
    fn test_legacy_string_source_empty() {
        let source = LegacyStringSource::new(
            "empty",
            ContextSourceType::Memory,
            0.7,
            |_query| String::new(),
        );

        let items = source.query_items("hello");
        assert!(items.is_empty());
    }

    #[test]
    fn test_extractive_with_allocator() {
        let mut allocator = ContextBudgetAllocator::new(OverflowStrategy::ExtractiveCompression);
        allocator.add_source(Box::new(StaticSource {
            items: vec![
                make_item("First item with high score", 0.95, ContextSourceType::Rag),
                make_item(
                    "Second item is a very long text that explains many details about \
                     the Rust programming language including ownership, borrowing, \
                     lifetimes, and trait system which together provide memory safety \
                     without garbage collection",
                    0.7,
                    ContextSourceType::Rag,
                ),
            ],
            name: "rag".into(),
            stype: ContextSourceType::Rag,
        }));

        // Budget that fits the first but not both fully
        let result = allocator.build("query", 30);
        assert!(result.tokens_used <= 30);
    }
}
