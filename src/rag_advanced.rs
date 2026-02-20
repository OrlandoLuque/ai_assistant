//! Advanced RAG features: intelligent chunking, deduplication, re-ranking

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ============================================================================
// Intelligent Chunking
// ============================================================================

/// Strategy for splitting documents into chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed size chunks (character count)
    FixedSize,
    /// Split by sentences
    Sentence,
    /// Split by paragraphs
    Paragraph,
    /// Split by markdown sections (headers)
    MarkdownSection,
    /// Semantic boundaries (tries to keep related content together)
    Semantic,
    /// Hybrid: uses multiple strategies and picks best
    Adaptive,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Configuration for intelligent chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Chunking strategy
    pub strategy: ChunkingStrategy,
    /// Target chunk size in tokens (approximate)
    pub target_tokens: usize,
    /// Minimum chunk size
    pub min_tokens: usize,
    /// Maximum chunk size
    pub max_tokens: usize,
    /// Overlap between chunks (in tokens)
    pub overlap_tokens: usize,
    /// Preserve markdown structure
    pub preserve_markdown: bool,
    /// Preserve code blocks
    pub preserve_code_blocks: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Adaptive,
            target_tokens: 300,
            min_tokens: 50,
            max_tokens: 500,
            overlap_tokens: 30,
            preserve_markdown: true,
            preserve_code_blocks: true,
        }
    }
}

/// A chunk with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartChunk {
    /// Chunk content
    pub content: String,
    /// Estimated token count
    pub tokens: usize,
    /// Start position in original document
    pub start_offset: usize,
    /// End position in original document
    pub end_offset: usize,
    /// Chunk index
    pub index: usize,
    /// Section header (if applicable)
    pub section: Option<String>,
    /// Is code block
    pub is_code: bool,
    /// Detected language (for code)
    pub language: Option<String>,
}

/// Type alias for an embedding function callback.
///
/// Takes a text string and returns a vector of f32 embeddings.
pub type EmbeddingFn = Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// Intelligent document chunker
pub struct SmartChunker {
    config: ChunkingConfig,
    /// Optional embedding function for semantic chunking.
    embedding_fn: Option<EmbeddingFn>,
}

impl SmartChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            embedding_fn: None,
        }
    }

    /// Set an embedding function for embedding-based semantic chunking.
    ///
    /// When set, the `Semantic` strategy uses cosine similarity of embeddings
    /// to detect topic boundaries rather than keyword overlap (Jaccard).
    pub fn with_embedding_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        self.embedding_fn = Some(Arc::new(f));
        self
    }

    /// Estimate tokens in text (rough: ~4 chars per token)
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }

    /// Chunk a document
    pub fn chunk(&self, document: &str) -> Vec<SmartChunk> {
        match self.config.strategy {
            ChunkingStrategy::FixedSize => self.chunk_fixed_size(document),
            ChunkingStrategy::Sentence => self.chunk_by_sentences(document),
            ChunkingStrategy::Paragraph => self.chunk_by_paragraphs(document),
            ChunkingStrategy::MarkdownSection => self.chunk_by_markdown(document),
            ChunkingStrategy::Semantic => self.chunk_semantic(document),
            ChunkingStrategy::Adaptive => self.chunk_adaptive(document),
        }
    }

    /// Fixed size chunking
    fn chunk_fixed_size(&self, document: &str) -> Vec<SmartChunk> {
        let target_chars = self.config.target_tokens * 4;
        let overlap_chars = self.config.overlap_tokens * 4;

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < document.len() {
            let end = (start + target_chars).min(document.len());

            // Try to break at word boundary
            let actual_end = if end < document.len() {
                document[start..end]
                    .rfind(char::is_whitespace)
                    .map(|p| start + p)
                    .unwrap_or(end)
            } else {
                end
            };

            let content = document[start..actual_end].trim().to_string();
            if !content.is_empty() {
                chunks.push(SmartChunk {
                    content: content.clone(),
                    tokens: Self::estimate_tokens(&content),
                    start_offset: start,
                    end_offset: actual_end,
                    index,
                    section: None,
                    is_code: false,
                    language: None,
                });
                index += 1;
            }

            start = actual_end.saturating_sub(overlap_chars);
            if start >= actual_end {
                start = actual_end;
            }
        }

        chunks
    }

    /// Chunk by sentences
    fn chunk_by_sentences(&self, document: &str) -> Vec<SmartChunk> {
        let sentences: Vec<&str> = document
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        let mut index = 0;

        for sentence in sentences {
            let trimmed = sentence.trim();
            let sentence_tokens = Self::estimate_tokens(trimmed);

            if Self::estimate_tokens(&current) + sentence_tokens > self.config.max_tokens
                && !current.is_empty()
            {
                // Save current chunk
                let content = current.trim().to_string();
                chunks.push(SmartChunk {
                    content: content.clone(),
                    tokens: Self::estimate_tokens(&content),
                    start_offset: current_start,
                    end_offset: current_start + current.len(),
                    index,
                    section: None,
                    is_code: false,
                    language: None,
                });
                index += 1;
                current_start += current.len();
                current.clear();
            }

            current.push_str(trimmed);
            current.push_str(". ");
        }

        // Add remaining content
        if !current.trim().is_empty() {
            let content = current.trim().to_string();
            chunks.push(SmartChunk {
                content: content.clone(),
                tokens: Self::estimate_tokens(&content),
                start_offset: current_start,
                end_offset: document.len(),
                index,
                section: None,
                is_code: false,
                language: None,
            });
        }

        chunks
    }

    /// Chunk by paragraphs
    fn chunk_by_paragraphs(&self, document: &str) -> Vec<SmartChunk> {
        let paragraphs: Vec<&str> = document
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        let mut index = 0;

        for para in paragraphs {
            let para_tokens = Self::estimate_tokens(para);

            // If paragraph alone is too large, split it
            if para_tokens > self.config.max_tokens {
                // Save current first
                if !current.trim().is_empty() {
                    let content = current.trim().to_string();
                    chunks.push(SmartChunk {
                        content: content.clone(),
                        tokens: Self::estimate_tokens(&content),
                        start_offset: current_start,
                        end_offset: current_start + current.len(),
                        index,
                        section: None,
                        is_code: false,
                        language: None,
                    });
                    index += 1;
                    current_start += current.len();
                    current.clear();
                }

                // Split large paragraph
                let sub_chunks = self.chunk_by_sentences(para);
                for sub in sub_chunks {
                    chunks.push(SmartChunk {
                        index,
                        start_offset: current_start + sub.start_offset,
                        end_offset: current_start + sub.end_offset,
                        ..sub
                    });
                    index += 1;
                }
                continue;
            }

            if Self::estimate_tokens(&current) + para_tokens > self.config.max_tokens
                && !current.is_empty()
            {
                let content = current.trim().to_string();
                chunks.push(SmartChunk {
                    content: content.clone(),
                    tokens: Self::estimate_tokens(&content),
                    start_offset: current_start,
                    end_offset: current_start + current.len(),
                    index,
                    section: None,
                    is_code: false,
                    language: None,
                });
                index += 1;
                current_start += current.len();
                current.clear();
            }

            current.push_str(para);
            current.push_str("\n\n");
        }

        if !current.trim().is_empty() {
            let content = current.trim().to_string();
            chunks.push(SmartChunk {
                content: content.clone(),
                tokens: Self::estimate_tokens(&content),
                start_offset: current_start,
                end_offset: document.len(),
                index,
                section: None,
                is_code: false,
                language: None,
            });
        }

        chunks
    }

    /// Chunk by markdown sections
    fn chunk_by_markdown(&self, document: &str) -> Vec<SmartChunk> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_section: Option<String> = None;
        let mut current_start = 0;
        let mut index = 0;
        let mut in_code_block = false;
        let mut code_language: Option<String> = None;

        for line in document.lines() {
            // Track code blocks
            if line.starts_with("```") {
                if !in_code_block {
                    in_code_block = true;
                    code_language = line.strip_prefix("```").map(|s| s.trim().to_string());
                } else {
                    in_code_block = false;
                }
                current.push_str(line);
                current.push('\n');
                continue;
            }

            // Check for headers (only outside code blocks)
            if !in_code_block && line.starts_with('#') {
                // Save previous section
                if !current.trim().is_empty()
                    && Self::estimate_tokens(&current) >= self.config.min_tokens
                {
                    let content = current.trim().to_string();
                    let is_code = content.contains("```");
                    chunks.push(SmartChunk {
                        content: content.clone(),
                        tokens: Self::estimate_tokens(&content),
                        start_offset: current_start,
                        end_offset: current_start + current.len(),
                        index,
                        section: current_section.clone(),
                        is_code,
                        language: if is_code { code_language.clone() } else { None },
                    });
                    index += 1;
                    current_start += current.len();
                    current.clear();
                }

                // Extract section name
                current_section = Some(line.trim_start_matches('#').trim().to_string());
            }

            current.push_str(line);
            current.push('\n');

            // Check if current chunk is getting too large
            if Self::estimate_tokens(&current) > self.config.max_tokens && !in_code_block {
                let content = current.trim().to_string();
                let is_code = content.contains("```");
                chunks.push(SmartChunk {
                    content: content.clone(),
                    tokens: Self::estimate_tokens(&content),
                    start_offset: current_start,
                    end_offset: current_start + current.len(),
                    index,
                    section: current_section.clone(),
                    is_code,
                    language: if is_code { code_language.clone() } else { None },
                });
                index += 1;
                current_start += current.len();
                current.clear();
            }
        }

        // Add remaining content
        if !current.trim().is_empty() {
            let content = current.trim().to_string();
            let is_code = content.contains("```");
            chunks.push(SmartChunk {
                content: content.clone(),
                tokens: Self::estimate_tokens(&content),
                start_offset: current_start,
                end_offset: document.len(),
                index,
                section: current_section,
                is_code,
                language: if is_code { code_language } else { None },
            });
        }

        chunks
    }

    /// Semantic chunking (keeps related content together)
    fn chunk_semantic(&self, document: &str) -> Vec<SmartChunk> {
        // If we have an embedding function, use embedding-based semantic chunking.
        if let Some(ref embed_fn) = self.embedding_fn {
            return self.chunk_semantic_embedding(document, embed_fn);
        }

        // Fallback: keyword-based (Jaccard) semantic chunking.
        self.chunk_semantic_keyword(document)
    }

    /// Embedding-based semantic chunking.
    ///
    /// Algorithm:
    /// 1. Split into sentences
    /// 2. Embed each sentence
    /// 3. Compute cosine similarity between consecutive sentences
    /// 4. Split where similarity drops below threshold (0.5)
    fn chunk_semantic_embedding(&self, document: &str, embed_fn: &EmbeddingFn) -> Vec<SmartChunk> {
        let sentences = Self::split_sentences(document);
        if sentences.is_empty() {
            return Vec::new();
        }
        if sentences.len() == 1 {
            return vec![SmartChunk {
                content: sentences[0].to_string(),
                tokens: Self::estimate_tokens(sentences[0]),
                start_offset: 0,
                end_offset: document.len(),
                index: 0,
                section: None,
                is_code: false,
                language: None,
            }];
        }

        // Embed all sentences
        let embeddings: Vec<Vec<f32>> = sentences.iter().map(|s| embed_fn(s)).collect();

        // Compute consecutive cosine similarities
        let mut similarities = Vec::new();
        for i in 0..embeddings.len() - 1 {
            similarities.push(Self::cosine_similarity(&embeddings[i], &embeddings[i + 1]));
        }

        // Build chunks by splitting at low-similarity boundaries
        let threshold = 0.5;
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        let mut index = 0;

        for (i, sentence) in sentences.iter().enumerate() {
            current.push_str(sentence);
            current.push(' ');

            let at_boundary = if i < similarities.len() {
                similarities[i] < threshold
            } else {
                false
            };

            let too_large = Self::estimate_tokens(&current) > self.config.target_tokens;

            if (at_boundary || too_large)
                && Self::estimate_tokens(&current) >= self.config.min_tokens
            {
                let content = current.trim().to_string();
                if !content.is_empty() {
                    chunks.push(SmartChunk {
                        content: content.clone(),
                        tokens: Self::estimate_tokens(&content),
                        start_offset: current_start,
                        end_offset: current_start + current.len(),
                        index,
                        section: None,
                        is_code: false,
                        language: None,
                    });
                    index += 1;
                    current_start += current.len();
                    current.clear();
                }
            }
        }

        // Final chunk
        if !current.trim().is_empty() {
            let content = current.trim().to_string();
            chunks.push(SmartChunk {
                content: content.clone(),
                tokens: Self::estimate_tokens(&content),
                start_offset: current_start,
                end_offset: document.len(),
                index,
                section: None,
                is_code: false,
                language: None,
            });
        }

        chunks
    }

    /// Keyword-based (Jaccard) semantic chunking — the original fallback.
    fn chunk_semantic_keyword(&self, document: &str) -> Vec<SmartChunk> {
        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = document.split("\n\n").collect();

        let mut current = String::new();
        let mut current_start = 0;
        let mut index = 0;
        let mut prev_keywords: HashSet<String> = HashSet::new();

        for para in paragraphs {
            if para.trim().is_empty() {
                continue;
            }

            let para_keywords = self.extract_keywords(para);
            let similarity = self.keyword_similarity(&prev_keywords, &para_keywords);

            // Start new chunk if:
            // 1. Current is too large, or
            // 2. Topic shift detected (low similarity), or
            // 3. Structural boundary (code block, list start, etc.)
            let is_structural_boundary = para.starts_with("```")
                || para.starts_with("- ")
                || para.starts_with("* ")
                || para.starts_with("1. ");

            let should_split = Self::estimate_tokens(&current) > self.config.target_tokens
                || (similarity < 0.2
                    && !current.is_empty()
                    && Self::estimate_tokens(&current) > self.config.min_tokens)
                || (is_structural_boundary && !current.is_empty());

            if should_split {
                let content = current.trim().to_string();
                if !content.is_empty() {
                    chunks.push(SmartChunk {
                        content: content.clone(),
                        tokens: Self::estimate_tokens(&content),
                        start_offset: current_start,
                        end_offset: current_start + current.len(),
                        index,
                        section: None,
                        is_code: content.contains("```"),
                        language: None,
                    });
                    index += 1;
                    current_start += current.len();
                    current.clear();
                    prev_keywords.clear();
                }
            }

            current.push_str(para);
            current.push_str("\n\n");
            prev_keywords.extend(para_keywords);
        }

        if !current.trim().is_empty() {
            let content = current.trim().to_string();
            chunks.push(SmartChunk {
                content: content.clone(),
                tokens: Self::estimate_tokens(&content),
                start_offset: current_start,
                end_offset: document.len(),
                index,
                section: None,
                is_code: content.contains("```"),
                language: None,
            });
        }

        chunks
    }

    /// Split text into sentences (at '.', '!', '?').
    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let end = i + c.len_utf8();
                let sentence = text[start..end].trim();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = end;
            }
        }
        // Add trailing text
        let remainder = text[start..].trim();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
        sentences
    }

    /// Cosine similarity between two f32 vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Adaptive chunking (chooses best strategy based on content)
    fn chunk_adaptive(&self, document: &str) -> Vec<SmartChunk> {
        // Detect document type
        let has_markdown_headers = document.lines().any(|l| l.starts_with('#'));
        let has_code_blocks = document.contains("```");
        let paragraph_count = document.split("\n\n").count();

        if has_markdown_headers {
            self.chunk_by_markdown(document)
        } else if has_code_blocks || paragraph_count < 3 {
            self.chunk_semantic(document)
        } else if paragraph_count > 10 {
            self.chunk_by_paragraphs(document)
        } else {
            self.chunk_by_sentences(document)
        }
    }

    fn extract_keywords(&self, text: &str) -> HashSet<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3)
            .map(|s| s.to_string())
            .collect()
    }

    fn keyword_similarity(&self, set1: &HashSet<String>, set2: &HashSet<String>) -> f32 {
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }
        let intersection = set1.intersection(set2).count();
        let union = set1.union(set2).count();
        intersection as f32 / union as f32
    }
}

// ============================================================================
// Chunk Deduplication
// ============================================================================

/// Result of deduplication
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    /// Number of chunks before deduplication
    pub original_count: usize,
    /// Number of chunks after deduplication
    pub deduplicated_count: usize,
    /// Duplicate pairs found (indices)
    pub duplicates_found: Vec<(usize, usize)>,
}

/// Deduplicator for removing similar chunks
pub struct ChunkDeduplicator {
    /// Similarity threshold (0.0 to 1.0)
    threshold: f32,
}

impl Default for ChunkDeduplicator {
    fn default() -> Self {
        Self::new(0.85)
    }
}

impl ChunkDeduplicator {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Calculate similarity between two texts using Jaccard similarity
    pub fn text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();

        let words1: HashSet<&str> = text1_lower.split_whitespace().collect();
        let words2: HashSet<&str> = text2_lower.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f32 / union as f32
    }

    /// Calculate n-gram based similarity (more accurate for longer texts)
    pub fn ngram_similarity(&self, text1: &str, text2: &str, n: usize) -> f32 {
        let ngrams1: HashSet<String> = self.extract_ngrams(text1, n);
        let ngrams2: HashSet<String> = self.extract_ngrams(text2, n);

        if ngrams1.is_empty() && ngrams2.is_empty() {
            return 1.0;
        }
        if ngrams1.is_empty() || ngrams2.is_empty() {
            return 0.0;
        }

        let intersection = ngrams1.intersection(&ngrams2).count();
        let union = ngrams1.union(&ngrams2).count();

        intersection as f32 / union as f32
    }

    fn extract_ngrams(&self, text: &str, n: usize) -> HashSet<String> {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        if words.len() < n {
            return HashSet::new();
        }

        words.windows(n).map(|w| w.join(" ")).collect()
    }

    /// Deduplicate a list of chunks
    pub fn deduplicate(&self, chunks: Vec<SmartChunk>) -> (Vec<SmartChunk>, DeduplicationResult) {
        let original_count = chunks.len();
        let mut duplicates_found = Vec::new();
        let mut keep_indices: HashSet<usize> = (0..chunks.len()).collect();

        // Compare each pair
        for i in 0..chunks.len() {
            if !keep_indices.contains(&i) {
                continue;
            }

            for j in (i + 1)..chunks.len() {
                if !keep_indices.contains(&j) {
                    continue;
                }

                let sim = self.ngram_similarity(&chunks[i].content, &chunks[j].content, 3);
                if sim >= self.threshold {
                    // Keep the longer one (more content)
                    if chunks[i].content.len() >= chunks[j].content.len() {
                        keep_indices.remove(&j);
                    } else {
                        keep_indices.remove(&i);
                    }
                    duplicates_found.push((i, j));
                }
            }
        }

        let deduplicated: Vec<SmartChunk> = chunks
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep_indices.contains(i))
            .map(|(_, c)| c)
            .collect();

        let result = DeduplicationResult {
            original_count,
            deduplicated_count: deduplicated.len(),
            duplicates_found,
        };

        (deduplicated, result)
    }
}

// ============================================================================
// Re-ranking
// ============================================================================

/// Configuration for re-ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Weight for BM25/keyword score
    pub keyword_weight: f32,
    /// Weight for recency
    pub recency_weight: f32,
    /// Weight for source priority
    pub priority_weight: f32,
    /// Weight for section relevance
    pub section_weight: f32,
    /// Boost for exact phrase matches
    pub exact_match_boost: f32,
    /// Penalty for very long chunks
    pub length_penalty: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            keyword_weight: 0.4,
            recency_weight: 0.1,
            priority_weight: 0.2,
            section_weight: 0.15,
            exact_match_boost: 0.15,
            length_penalty: 0.1,
        }
    }
}

/// A chunk with re-ranking score
#[derive(Debug, Clone)]
pub struct RankedChunk<T: Clone> {
    pub chunk: T,
    pub original_score: f32,
    pub reranked_score: f32,
    pub score_breakdown: HashMap<String, f32>,
}

/// Re-ranker for improving search results
pub struct ChunkReranker {
    config: RerankConfig,
}

impl ChunkReranker {
    pub fn new(config: RerankConfig) -> Self {
        Self { config }
    }

    /// Re-rank chunks based on query relevance
    pub fn rerank<T: Clone + AsRef<str>>(
        &self,
        query: &str,
        chunks: Vec<(T, f32)>, // (chunk, original_score)
        get_metadata: impl Fn(&T) -> ChunkMetadata,
    ) -> Vec<RankedChunk<T>> {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

        let mut ranked: Vec<RankedChunk<T>> = chunks
            .into_iter()
            .map(|(chunk, original_score)| {
                let content = chunk.as_ref();
                let content_lower = content.to_lowercase();
                let metadata = get_metadata(&chunk);
                let mut breakdown = HashMap::new();

                // 1. Keyword score (normalized original score)
                let keyword_score = original_score.min(1.0);
                breakdown.insert("keyword".to_string(), keyword_score);

                // 2. Exact phrase match boost
                let exact_boost = if content_lower.contains(&query_lower) {
                    self.config.exact_match_boost
                } else {
                    0.0
                };
                breakdown.insert("exact_match".to_string(), exact_boost);

                // 3. Priority score
                let priority_score = (metadata.priority as f32 / 10.0).min(1.0);
                breakdown.insert("priority".to_string(), priority_score);

                // 4. Section relevance (if query matches section title)
                let section_score = metadata
                    .section
                    .map(|s| {
                        let s_lower = s.to_lowercase();
                        let matches = query_words.iter().filter(|w| s_lower.contains(*w)).count();
                        (matches as f32 / query_words.len().max(1) as f32).min(1.0)
                    })
                    .unwrap_or(0.0);
                breakdown.insert("section".to_string(), section_score);

                // 5. Length penalty (very long chunks may be less focused)
                let token_estimate = content.len() / 4;
                let length_penalty = if token_estimate > 400 {
                    self.config.length_penalty * ((token_estimate - 400) as f32 / 400.0).min(1.0)
                } else {
                    0.0
                };
                breakdown.insert("length_penalty".to_string(), -length_penalty);

                // Calculate final score
                let reranked_score = keyword_score * self.config.keyword_weight
                    + exact_boost
                    + priority_score * self.config.priority_weight
                    + section_score * self.config.section_weight
                    - length_penalty;

                RankedChunk {
                    chunk,
                    original_score,
                    reranked_score: reranked_score.max(0.0),
                    score_breakdown: breakdown,
                }
            })
            .collect();

        // Sort by reranked score (descending)
        ranked.sort_by(|a, b| {
            b.reranked_score
                .partial_cmp(&a.reranked_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
    }

    /// Simple re-ranking for string chunks
    pub fn rerank_simple(
        &self,
        query: &str,
        chunks: Vec<(String, f32)>,
    ) -> Vec<RankedChunk<String>> {
        self.rerank(query, chunks, |_| ChunkMetadata::default())
    }
}

/// Metadata for re-ranking
#[derive(Debug, Clone, Default)]
pub struct ChunkMetadata {
    pub priority: i32,
    pub section: Option<String>,
    pub source: Option<String>,
    pub indexed_at: Option<chrono::DateTime<chrono::Utc>>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_chunker() {
        let config = ChunkingConfig {
            target_tokens: 50,
            max_tokens: 100,
            ..Default::default()
        };
        let chunker = SmartChunker::new(config);

        let document =
            "# Introduction\n\nThis is the first paragraph.\n\n## Details\n\nThis is more content.";
        let chunks = chunker.chunk(document);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_deduplicator() {
        let dedup = ChunkDeduplicator::new(0.8);

        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "The quick brown fox jumps over a lazy dog";
        let text3 = "Something completely different here";

        let sim12 = dedup.text_similarity(text1, text2);
        let sim13 = dedup.text_similarity(text1, text3);

        assert!(sim12 > 0.7); // Very similar
        assert!(sim13 < 0.3); // Not similar
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((SmartChunker::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(SmartChunker::cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((SmartChunker::cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. How are you? I am fine!";
        let sentences = SmartChunker::split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("Hello"));
        assert!(sentences[1].contains("How"));
        assert!(sentences[2].contains("fine"));
    }

    #[test]
    fn test_semantic_chunking_with_embeddings() {
        // Mock embedding: map each word to a distinct dimension
        let embed_fn = |text: &str| -> Vec<f32> {
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut vec = vec![0.0f32; 10];
            for word in words {
                let idx = match word.to_lowercase().chars().next().unwrap_or('a') {
                    'a'..='c' => 0,
                    'd'..='f' => 1,
                    'g'..='i' => 2,
                    'j'..='l' => 3,
                    'm'..='o' => 4,
                    'p'..='r' => 5,
                    's'..='u' => 6,
                    'v'..='x' => 7,
                    _ => 8,
                };
                vec[idx] += 1.0;
            }
            vec
        };

        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Semantic,
            target_tokens: 50,
            min_tokens: 5,
            max_tokens: 100,
            ..Default::default()
        };
        let chunker = SmartChunker::new(config).with_embedding_fn(embed_fn);

        let document =
            "Apple banana cherry. Date elderberry fig. Grape hibiscus iris. Jackfruit kiwi lemon.";
        let chunks = chunker.chunk(document);
        assert!(!chunks.is_empty());
        // All text should be accounted for
        let all_text: String = chunks
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("Apple"));
        assert!(all_text.contains("lemon"));
    }

    #[test]
    fn test_semantic_chunking_fallback_without_embeddings() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Semantic,
            target_tokens: 20,
            min_tokens: 5,
            max_tokens: 50,
            ..Default::default()
        };
        let chunker = SmartChunker::new(config);

        let document = "Cats are great pets.\n\nDogs are loyal companions.\n\nFish need aquariums.\n\nBirds can fly high.";
        let chunks = chunker.chunk(document);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_embedding_boundary_detection() {
        // Embeddings that are similar for first 3 sentences, then very different
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let embed_fn = move |_text: &str| -> Vec<f32> {
            let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < 3 {
                // Similar embeddings (topic A)
                vec![1.0, 0.0, 0.0]
            } else {
                // Very different (topic B)
                vec![0.0, 0.0, 1.0]
            }
        };

        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Semantic,
            target_tokens: 200,
            min_tokens: 1,
            max_tokens: 400,
            ..Default::default()
        };
        let chunker = SmartChunker::new(config).with_embedding_fn(embed_fn);

        let document = "Sent one. Sent two. Sent three. Different topic. Another different.";
        let chunks = chunker.chunk(document);
        // Should have at least 2 chunks because of the topic shift
        assert!(
            chunks.len() >= 2,
            "Expected >=2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(SmartChunker::cosine_similarity(&[], &[]), 0.0);
        assert_eq!(SmartChunker::cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let nonzero = vec![1.0, 2.0, 3.0];
        assert_eq!(SmartChunker::cosine_similarity(&zero, &nonzero), 0.0);
    }

    #[test]
    fn test_reranker() {
        let reranker = ChunkReranker::new(RerankConfig::default());

        let chunks = vec![
            ("This is about dogs and cats".to_string(), 0.5),
            ("Dogs are great pets".to_string(), 0.6),
            ("Weather is nice today".to_string(), 0.3),
        ];

        let ranked = reranker.rerank_simple("tell me about dogs", chunks);

        assert!(!ranked.is_empty());
        // "Dogs are great pets" should rank higher
        assert!(ranked[0].chunk.contains("dogs") || ranked[0].chunk.contains("Dogs"));
    }
}
