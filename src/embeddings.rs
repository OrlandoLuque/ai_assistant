//! Local embeddings for semantic search (without external API calls)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// TF-IDF Based Embeddings
// ============================================================================

/// Configuration for local embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimensionality of the embedding vectors
    pub dimensions: usize,
    /// Minimum word frequency to include in vocabulary
    pub min_word_freq: usize,
    /// Maximum vocabulary size
    pub max_vocab_size: usize,
    /// Use subword tokenization
    pub use_subwords: bool,
    /// N-gram range for features
    pub ngram_range: (usize, usize),
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimensions: 256,
            min_word_freq: 2,
            max_vocab_size: 10000,
            use_subwords: true,
            ngram_range: (1, 2),
        }
    }
}

/// Local embedding model using TF-IDF with hashing trick
pub struct LocalEmbedder {
    config: EmbeddingConfig,
    /// Document frequency for IDF calculation
    doc_freq: HashMap<String, usize>,
    /// Total documents seen
    total_docs: usize,
    /// Vocabulary (word -> index)
    vocab: HashMap<String, usize>,
}

impl LocalEmbedder {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            doc_freq: HashMap::new(),
            total_docs: 0,
            vocab: HashMap::new(),
        }
    }

    /// Train the embedder on a corpus
    pub fn train(&mut self, documents: &[&str]) {
        self.total_docs = documents.len();

        // Build document frequency
        for doc in documents {
            let tokens = self.tokenize(doc);
            let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();

            for token in unique_tokens {
                *self.doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Build vocabulary from most frequent terms
        let mut freq_vec: Vec<_> = self.doc_freq.iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(a.1));

        self.vocab.clear();
        for (i, (word, freq)) in freq_vec.iter().enumerate() {
            if i >= self.config.max_vocab_size {
                break;
            }
            if **freq >= self.config.min_word_freq {
                self.vocab
                    .insert(word.to_string(), i % self.config.dimensions);
            }
        }
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.config.dimensions];
        let tokens = self.tokenize(text);
        let token_count = tokens.len() as f32;

        if token_count == 0.0 {
            return embedding;
        }

        // Count term frequencies
        let mut tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        // Calculate TF-IDF and hash to dimensions
        for (token, count) in tf {
            let term_freq = count as f32 / token_count;
            let idf = self.calculate_idf(&token);
            let tfidf = term_freq * idf;

            // Use hashing trick for out-of-vocabulary words
            let idx = self.get_index(&token);
            embedding[idx] += tfidf;
        }

        // L2 normalize
        self.normalize(&mut embedding);

        embedding
    }

    /// Generate embeddings for multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
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

    /// Find most similar texts to a query
    pub fn find_similar(
        &self,
        query: &str,
        candidates: &[(&str, Vec<f32>)],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let query_embedding = self.embed(query);

        let mut similarities: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, (_, emb))| (i, Self::cosine_similarity(&query_embedding, emb)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let text_lower = text.to_lowercase();

        // Word tokenization
        let words: Vec<&str> = text_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .collect();

        // Add unigrams
        if self.config.ngram_range.0 <= 1 {
            tokens.extend(words.iter().map(|w| w.to_string()));
        }

        // Add bigrams
        if self.config.ngram_range.1 >= 2 && words.len() >= 2 {
            for window in words.windows(2) {
                tokens.push(format!("{}_{}", window[0], window[1]));
            }
        }

        // Add character n-grams (subwords)
        if self.config.use_subwords {
            for word in &words {
                if word.len() >= 3 {
                    // Character trigrams
                    let chars: Vec<char> = word.chars().collect();
                    for i in 0..chars.len().saturating_sub(2) {
                        let ngram: String = chars[i..i + 3].iter().collect();
                        tokens.push(format!("#{}", ngram));
                    }
                }
            }
        }

        tokens
    }

    fn calculate_idf(&self, token: &str) -> f32 {
        let doc_freq = self.doc_freq.get(token).copied().unwrap_or(0) as f32;
        let total = self.total_docs.max(1) as f32;

        // Smooth IDF
        ((total + 1.0) / (doc_freq + 1.0)).ln() + 1.0
    }

    fn get_index(&self, token: &str) -> usize {
        // Check vocabulary first
        if let Some(&idx) = self.vocab.get(token) {
            return idx;
        }

        // Use hash for OOV words
        let hash = Self::fnv1a_hash(token);
        hash % self.config.dimensions
    }

    fn fnv1a_hash(s: &str) -> usize {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;

        let mut hash = FNV_OFFSET;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash as usize
    }

    fn normalize(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }
}

// ============================================================================
// Semantic Search Index
// ============================================================================

/// An indexed document with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDocument {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Pre-computed embedding
    pub embedding: Vec<f32>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Semantic search index
pub struct SemanticIndex {
    embedder: LocalEmbedder,
    documents: Vec<IndexedDocument>,
}

impl SemanticIndex {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            embedder: LocalEmbedder::new(config),
            documents: Vec::new(),
        }
    }

    /// Build index from documents (trains embedder and indexes all documents)
    pub fn build(&mut self, documents: Vec<(String, String, HashMap<String, String>)>) {
        // Train embedder on corpus
        let texts: Vec<&str> = documents.iter().map(|(_, c, _)| c.as_str()).collect();
        self.embedder.train(&texts);

        // Index documents
        self.documents = documents
            .into_iter()
            .map(|(id, content, metadata)| {
                let embedding = self.embedder.embed(&content);
                IndexedDocument {
                    id,
                    content,
                    embedding,
                    metadata,
                }
            })
            .collect();
    }

    /// Add a single document to the index
    pub fn add_document(&mut self, id: String, content: String, metadata: HashMap<String, String>) {
        let embedding = self.embedder.embed(&content);
        self.documents.push(IndexedDocument {
            id,
            content,
            embedding,
            metadata,
        });
    }

    /// Remove a document by ID
    pub fn remove_document(&mut self, id: &str) -> bool {
        let len_before = self.documents.len();
        self.documents.retain(|d| d.id != id);
        self.documents.len() != len_before
    }

    /// Search for similar documents
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_embedding = self.embedder.embed(query);

        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .map(|doc| {
                let similarity = LocalEmbedder::cosine_similarity(&query_embedding, &doc.embedding);
                SearchResult {
                    id: doc.id.clone(),
                    content: doc.content.clone(),
                    score: similarity,
                    metadata: doc.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Search with minimum score threshold
    pub fn search_with_threshold(
        &self,
        query: &str,
        top_k: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        self.search(query, top_k)
            .into_iter()
            .filter(|r| r.score >= min_score)
            .collect()
    }

    /// Get document count
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Get a document by ID
    pub fn get_document(&self, id: &str) -> Option<&IndexedDocument> {
        self.documents.iter().find(|d| d.id == id)
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.documents.clear();
    }

    /// Export index to serializable format
    pub fn export(&self) -> Vec<IndexedDocument> {
        self.documents.clone()
    }

    /// Import documents (requires re-training embedder for best results)
    pub fn import(&mut self, documents: Vec<IndexedDocument>) {
        self.documents = documents;
    }
}

/// Result from semantic search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// Hybrid Search (combining BM25 and semantic)
// ============================================================================

/// Configuration for hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight for keyword/BM25 score (0.0 to 1.0)
    pub keyword_weight: f32,
    /// Weight for semantic score (0.0 to 1.0)
    pub semantic_weight: f32,
    /// Minimum semantic score to consider
    pub min_semantic_score: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            keyword_weight: 0.5,
            semantic_weight: 0.5,
            min_semantic_score: 0.1,
        }
    }
}

/// Hybrid search combining keyword and semantic search
pub struct HybridSearcher {
    config: HybridSearchConfig,
    semantic_index: SemanticIndex,
}

impl HybridSearcher {
    pub fn new(config: HybridSearchConfig, embedding_config: EmbeddingConfig) -> Self {
        Self {
            config,
            semantic_index: SemanticIndex::new(embedding_config),
        }
    }

    /// Build the semantic index
    pub fn build_index(&mut self, documents: Vec<(String, String, HashMap<String, String>)>) {
        self.semantic_index.build(documents);
    }

    /// Perform hybrid search
    ///
    /// Takes keyword search results (id, keyword_score) and combines with semantic search
    pub fn search(
        &self,
        query: &str,
        keyword_results: &[(String, f32)],
        top_k: usize,
    ) -> Vec<HybridSearchResult> {
        // Get semantic results
        let semantic_results = self.semantic_index.search(query, top_k * 2);
        let semantic_map: HashMap<_, _> = semantic_results
            .iter()
            .map(|r| (r.id.clone(), r.score))
            .collect();

        // Combine scores
        let mut combined: HashMap<String, HybridSearchResult> = HashMap::new();

        // Add keyword results
        for (id, keyword_score) in keyword_results {
            let semantic_score = semantic_map.get(id).copied().unwrap_or(0.0);
            let combined_score = keyword_score * self.config.keyword_weight
                + semantic_score * self.config.semantic_weight;

            combined.insert(
                id.clone(),
                HybridSearchResult {
                    id: id.clone(),
                    keyword_score: *keyword_score,
                    semantic_score,
                    combined_score,
                    content: self
                        .semantic_index
                        .get_document(id)
                        .map(|d| d.content.clone())
                        .unwrap_or_default(),
                },
            );
        }

        // Add semantic-only results (not in keyword results)
        for result in semantic_results {
            if !combined.contains_key(&result.id) && result.score >= self.config.min_semantic_score
            {
                combined.insert(
                    result.id.clone(),
                    HybridSearchResult {
                        id: result.id,
                        keyword_score: 0.0,
                        semantic_score: result.score,
                        combined_score: result.score * self.config.semantic_weight,
                        content: result.content,
                    },
                );
            }
        }

        // Sort by combined score
        let mut results: Vec<_> = combined.into_values().collect();
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }
}

/// Result from hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: String,
    pub keyword_score: f32,
    pub semantic_score: f32,
    pub combined_score: f32,
    pub content: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_embedder() {
        let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());

        let docs = vec![
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps above a sleepy canine",
            "Hello world this is a test",
        ];

        embedder.train(&docs);

        let emb1 = embedder.embed(docs[0]);
        let emb2 = embedder.embed(docs[1]);
        let emb3 = embedder.embed(docs[2]);

        // Similar documents should have higher similarity
        let sim12 = LocalEmbedder::cosine_similarity(&emb1, &emb2);
        let sim13 = LocalEmbedder::cosine_similarity(&emb1, &emb3);

        assert!(sim12 > sim13, "Similar docs should score higher");
    }

    #[test]
    fn test_semantic_index() {
        let mut index = SemanticIndex::new(EmbeddingConfig::default());

        let docs = vec![
            (
                "doc1".to_string(),
                "Rust programming language".to_string(),
                HashMap::new(),
            ),
            (
                "doc2".to_string(),
                "Python programming language".to_string(),
                HashMap::new(),
            ),
            (
                "doc3".to_string(),
                "Cooking recipes for dinner".to_string(),
                HashMap::new(),
            ),
        ];

        index.build(docs);

        let results = index.search("rust code", 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1"); // Rust doc should be first
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((LocalEmbedder::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((LocalEmbedder::cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_normalization() {
        let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());

        let docs = vec![
            "Rust is a systems programming language",
            "Python is great for data science",
            "JavaScript runs in the browser",
        ];
        embedder.train(&docs);

        let embedding = embedder.embed("Rust programming language features");

        // Compute L2 norm
        let l2_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        // After normalization the L2 norm should be approximately 1.0
        assert!(
            (l2_norm - 1.0).abs() < 0.001,
            "L2 norm should be ~1.0 after normalization, got {}",
            l2_norm,
        );
    }

    #[test]
    fn test_batch_embed() {
        let mut embedder = LocalEmbedder::new(EmbeddingConfig::default());

        let docs = vec![
            "Machine learning algorithms",
            "Deep neural networks",
            "Natural language processing",
        ];
        embedder.train(&docs);

        let texts: Vec<&str> = vec![
            "Machine learning is interesting",
            "Neural networks are powerful",
            "NLP is a subfield of AI",
        ];

        let embeddings = embedder.embed_batch(&texts);

        // Should produce exactly 3 embeddings
        assert_eq!(
            embeddings.len(),
            3,
            "embed_batch should return one embedding per input text",
        );

        // Each embedding should have the configured dimensionality
        let expected_dim = EmbeddingConfig::default().dimensions;
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(
                emb.len(),
                expected_dim,
                "Embedding {} should have dimension {}, got {}",
                i,
                expected_dim,
                emb.len(),
            );
        }
    }
}
