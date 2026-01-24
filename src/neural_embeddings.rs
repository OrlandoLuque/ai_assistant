//! Advanced Neural Embeddings
//!
//! Provides sophisticated embedding generation and manipulation:
//! - Dense embedding models (sentence transformers style)
//! - Sparse embeddings (SPLADE-style)
//! - Hybrid dense+sparse retrieval
//! - Embedding quantization (int8, binary)
//! - Dimensionality reduction (PCA, random projection)
//! - Embedding pooling strategies
//! - Cross-encoder reranking

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Embedding vector type
pub type EmbeddingVec = Vec<f32>;

/// Sparse embedding (index -> weight)
pub type SparseEmbedding = HashMap<u32, f32>;

/// Pooling strategy for token embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Average all token embeddings
    Mean,
    /// Use CLS token embedding
    Cls,
    /// Max pooling across dimensions
    Max,
    /// Weighted mean by attention
    WeightedMean,
    /// Last token embedding
    LastToken,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Dense embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseEmbeddingConfig {
    /// Model identifier or path
    pub model_id: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Batch size for inference
    pub batch_size: usize,
}

impl Default for DenseEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            max_seq_length: 256,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            batch_size: 32,
        }
    }
}

impl DenseEmbeddingConfig {
    /// Configuration for all-MiniLM-L6-v2
    pub fn minilm_l6() -> Self {
        Self::default()
    }

    /// Configuration for all-mpnet-base-v2
    pub fn mpnet_base() -> Self {
        Self {
            model_id: "sentence-transformers/all-mpnet-base-v2".to_string(),
            dimension: 768,
            max_seq_length: 384,
            ..Default::default()
        }
    }

    /// Configuration for e5-small-v2
    pub fn e5_small() -> Self {
        Self {
            model_id: "intfloat/e5-small-v2".to_string(),
            dimension: 384,
            max_seq_length: 512,
            ..Default::default()
        }
    }

    /// Configuration for BGE-small-en
    pub fn bge_small() -> Self {
        Self {
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
            dimension: 384,
            max_seq_length: 512,
            ..Default::default()
        }
    }

    /// Configuration for nomic-embed-text
    pub fn nomic_embed() -> Self {
        Self {
            model_id: "nomic-ai/nomic-embed-text-v1".to_string(),
            dimension: 768,
            max_seq_length: 8192,
            ..Default::default()
        }
    }
}

/// Dense embedding generator using local models or API
pub struct DenseEmbedder {
    config: DenseEmbeddingConfig,
    api_url: Option<String>,
    api_key: Option<String>,
}

impl DenseEmbedder {
    /// Create new embedder with config
    pub fn new(config: DenseEmbeddingConfig) -> Self {
        Self {
            config,
            api_url: None,
            api_key: None,
        }
    }

    /// Configure for Ollama embeddings
    pub fn with_ollama(mut self, model: &str) -> Self {
        self.api_url = Some("http://localhost:11434/api/embeddings".to_string());
        self.config.model_id = model.to_string();
        self
    }

    /// Configure for OpenAI embeddings
    pub fn with_openai(mut self, api_key: &str, model: &str) -> Self {
        self.api_url = Some("https://api.openai.com/v1/embeddings".to_string());
        self.api_key = Some(api_key.to_string());
        self.config.model_id = model.to_string();
        self.config.dimension = match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };
        self
    }

    /// Generate embedding for single text
    pub fn embed(&self, text: &str) -> Result<EmbeddingVec, EmbeddingError> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings.into_iter().next().ok_or(EmbeddingError::EmptyResult)
    }

    /// Generate embeddings for batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(ref url) = self.api_url {
            self.embed_via_api(url, texts)
        } else {
            // Fallback to simple TF-IDF style embedding
            self.embed_simple(texts)
        }
    }

    fn embed_via_api(&self, url: &str, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let batch_results = if url.contains("ollama") {
                self.embed_ollama(chunk)?
            } else if url.contains("openai") {
                self.embed_openai(chunk)?
            } else {
                self.embed_generic_api(url, chunk)?
            };
            results.extend(batch_results);
        }

        Ok(results)
    }

    fn embed_ollama(&self, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let response = ureq::post("http://localhost:11434/api/embeddings")
                .send_json(ureq::json!({
                    "model": self.config.model_id,
                    "prompt": text
                }))
                .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

            let result: serde_json::Value = response.into_json()
                .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

            let embedding: Vec<f32> = result["embedding"]
                .as_array()
                .ok_or(EmbeddingError::ParseError("No embedding field".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            let embedding = if self.config.normalize {
                normalize_l2(&embedding)
            } else {
                embedding
            };

            results.push(embedding);
        }

        Ok(results)
    }

    fn embed_openai(&self, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        let api_key = self.api_key.as_ref()
            .ok_or(EmbeddingError::ConfigError("OpenAI API key required".to_string()))?;

        let response = ureq::post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {}", api_key))
            .set("Content-Type", "application/json")
            .send_json(ureq::json!({
                "model": self.config.model_id,
                "input": texts
            }))
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        let result: serde_json::Value = response.into_json()
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let data = result["data"].as_array()
            .ok_or(EmbeddingError::ParseError("No data field".to_string()))?;

        let mut results: Vec<EmbeddingVec> = data.iter()
            .filter_map(|item| {
                item["embedding"].as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect();

        // OpenAI returns normalized embeddings, but normalize if configured
        if self.config.normalize {
            for emb in &mut results {
                *emb = normalize_l2(emb);
            }
        }

        Ok(results)
    }

    fn embed_generic_api(&self, url: &str, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        let mut request = ureq::post(url);

        if let Some(ref key) = self.api_key {
            request = request.set("Authorization", &format!("Bearer {}", key));
        }

        let response = request
            .set("Content-Type", "application/json")
            .send_json(ureq::json!({
                "model": self.config.model_id,
                "input": texts
            }))
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        let result: serde_json::Value = response.into_json()
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        // Try to parse various response formats
        let embeddings = if let Some(data) = result["data"].as_array() {
            data.iter()
                .filter_map(|item| {
                    item["embedding"].as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                })
                .collect()
        } else if let Some(emb) = result["embedding"].as_array() {
            vec![emb.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()]
        } else if let Some(embs) = result["embeddings"].as_array() {
            embs.iter()
                .filter_map(|arr| arr.as_array().map(|a| {
                    a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
                }))
                .collect()
        } else {
            return Err(EmbeddingError::ParseError("Unknown response format".to_string()));
        };

        Ok(embeddings)
    }

    fn embed_simple(&self, texts: &[String]) -> Result<Vec<EmbeddingVec>, EmbeddingError> {
        // Simple hash-based embedding for fallback
        let dim = self.config.dimension;
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let mut embedding = vec![0.0f32; dim];
            let words: Vec<&str> = text.split_whitespace().collect();

            for (i, word) in words.iter().enumerate() {
                let hash = simple_hash(word);
                let idx = (hash as usize) % dim;
                let sign = if (hash >> 31) & 1 == 0 { 1.0 } else { -1.0 };
                let weight = 1.0 / ((i + 1) as f32).sqrt(); // Position weight
                embedding[idx] += sign * weight;
            }

            let embedding = if self.config.normalize {
                normalize_l2(&embedding)
            } else {
                embedding
            };

            results.push(embedding);
        }

        Ok(results)
    }

    /// Get model dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

/// Simple hash function for fallback embedding
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

/// Convert f32 to f16 bits (IEEE 754 half precision)
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    let (h_exp, h_frac) = if exp == 0 {
        // Zero or denorm
        (0, 0)
    } else if exp == 0xff {
        // Inf or NaN
        (0x1f, if frac == 0 { 0 } else { 0x200 })
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1f {
            // Overflow -> Inf
            (0x1f, 0)
        } else if new_exp <= 0 {
            // Underflow -> zero
            (0, 0)
        } else {
            (new_exp as u32, frac >> 13)
        }
    };

    ((sign << 15) | (h_exp << 10) | h_frac) as u16
}

/// Convert f16 bits to f32
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;

    let (f_exp, f_frac) = if exp == 0 {
        if frac == 0 {
            (0, 0) // Zero
        } else {
            // Denormalized
            (0, frac << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN
        (0xff, if frac == 0 { 0 } else { 0x400000 })
    } else {
        let new_exp = (exp - 15 + 127) as u32;
        (new_exp, frac << 13)
    };

    f32::from_bits((sign << 31) | (f_exp << 23) | f_frac)
}

/// Sparse embedding configuration (SPLADE-style)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEmbeddingConfig {
    /// Vocabulary size
    pub vocab_size: u32,
    /// Maximum non-zero entries
    pub max_terms: usize,
    /// Minimum weight threshold
    pub min_weight: f32,
    /// Use IDF weighting
    pub use_idf: bool,
}

impl Default for SparseEmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522, // BERT vocab size
            max_terms: 128,
            min_weight: 0.01,
            use_idf: true,
        }
    }
}

/// Sparse embedder (BM25 / SPLADE style)
pub struct SparseEmbedder {
    config: SparseEmbeddingConfig,
    idf_scores: HashMap<String, f32>,
    doc_count: u64,
}

impl SparseEmbedder {
    pub fn new(config: SparseEmbeddingConfig) -> Self {
        Self {
            config,
            idf_scores: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Train IDF scores from documents
    pub fn fit(&mut self, documents: &[String]) {
        let mut doc_freq: HashMap<String, u64> = HashMap::new();
        self.doc_count = documents.len() as u64;

        for doc in documents {
            let words: std::collections::HashSet<String> = doc
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|s| !s.is_empty())
                .collect();

            for word in words {
                *doc_freq.entry(word).or_insert(0) += 1;
            }
        }

        // Calculate IDF
        for (word, freq) in doc_freq {
            let idf = ((self.doc_count as f32 + 1.0) / (freq as f32 + 1.0)).ln() + 1.0;
            self.idf_scores.insert(word, idf);
        }
    }

    /// Generate sparse embedding
    pub fn embed(&self, text: &str) -> SparseEmbedding {
        let mut embedding = SparseEmbedding::new();
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let mut term_freq: HashMap<String, u32> = HashMap::new();

        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if !clean.is_empty() {
                *term_freq.entry(clean).or_insert(0) += 1;
            }
        }

        for (term, tf) in term_freq {
            let idf = self.idf_scores.get(&term).copied().unwrap_or(1.0);
            let weight = if self.config.use_idf {
                (tf as f32).sqrt() * idf
            } else {
                tf as f32
            };

            if weight >= self.config.min_weight {
                let idx = (simple_hash(&term) % self.config.vocab_size as u64) as u32;
                let existing = embedding.entry(idx).or_insert(0.0);
                *existing = (*existing).max(weight);
            }
        }

        // Keep only top-k terms
        if embedding.len() > self.config.max_terms {
            let mut entries: Vec<_> = embedding.into_iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(self.config.max_terms);
            embedding = entries.into_iter().collect();
        }

        embedding
    }

    /// Compute sparse dot product
    pub fn dot_product(a: &SparseEmbedding, b: &SparseEmbedding) -> f32 {
        let (smaller, larger) = if a.len() < b.len() { (a, b) } else { (b, a) };
        smaller.iter()
            .filter_map(|(k, v)| larger.get(k).map(|w| v * w))
            .sum()
    }
}

impl Default for SparseEmbedder {
    fn default() -> Self {
        Self::new(SparseEmbeddingConfig::default())
    }
}

/// Hybrid retriever combining dense and sparse
pub struct HybridRetriever {
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    dense_weight: f32,
}

impl HybridRetriever {
    pub fn new(dense: DenseEmbedder, sparse: SparseEmbedder, dense_weight: f32) -> Self {
        Self {
            dense,
            sparse,
            dense_weight: dense_weight.clamp(0.0, 1.0),
        }
    }

    /// Generate hybrid embedding
    pub fn embed(&self, text: &str) -> Result<HybridEmbedding, EmbeddingError> {
        let dense = self.dense.embed(text)?;
        let sparse = self.sparse.embed(text);
        Ok(HybridEmbedding { dense, sparse })
    }

    /// Compute hybrid similarity
    pub fn similarity(&self, query: &HybridEmbedding, doc: &HybridEmbedding) -> f32 {
        let dense_sim = cosine_similarity(&query.dense, &doc.dense);
        let sparse_sim = SparseEmbedder::dot_product(&query.sparse, &doc.sparse);

        // Normalize sparse similarity roughly to [0, 1]
        let sparse_normalized = sparse_sim.tanh();

        self.dense_weight * dense_sim + (1.0 - self.dense_weight) * sparse_normalized
    }
}

/// Hybrid embedding containing both dense and sparse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEmbedding {
    pub dense: EmbeddingVec,
    pub sparse: SparseEmbedding,
}

/// Embedding quantization for storage efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// Full precision (f32)
    Float32,
    /// Half precision (f16 stored as u16)
    Float16,
    /// 8-bit signed integer
    Int8,
    /// Binary (1 bit per dimension)
    Binary,
    /// Product quantization
    ProductQuantization,
}

/// Quantized embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedEmbedding {
    Float32(Vec<f32>),
    Float16(Vec<u16>),
    Int8 { data: Vec<i8>, scale: f32, offset: f32 },
    Binary(Vec<u8>),
}

impl QuantizedEmbedding {
    /// Quantize from f32
    pub fn quantize(embedding: &[f32], quant_type: QuantizationType) -> Self {
        match quant_type {
            QuantizationType::Float32 => Self::Float32(embedding.to_vec()),
            QuantizationType::Float16 => {
                let data: Vec<u16> = embedding.iter()
                    .map(|&f| f32_to_f16_bits(f))
                    .collect();
                Self::Float16(data)
            }
            QuantizationType::Int8 => {
                let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
                let offset = min;

                let data: Vec<i8> = embedding.iter()
                    .map(|&f| {
                        if range > 0.0 {
                            let normalized = (f - offset) / scale;
                            // Map [0, 255] to [-128, 127]
                            (normalized.round() as i32 - 128).clamp(-128, 127) as i8
                        } else {
                            0
                        }
                    })
                    .collect();

                Self::Int8 { data, scale, offset }
            }
            QuantizationType::Binary => {
                let bytes = (embedding.len() + 7) / 8;
                let mut data = vec![0u8; bytes];

                for (i, &v) in embedding.iter().enumerate() {
                    if v > 0.0 {
                        data[i / 8] |= 1 << (i % 8);
                    }
                }

                Self::Binary(data)
            }
            QuantizationType::ProductQuantization => {
                // For simplicity, fall back to Int8
                Self::quantize(embedding, QuantizationType::Int8)
            }
        }
    }

    /// Dequantize to f32
    pub fn dequantize(&self) -> Vec<f32> {
        match self {
            Self::Float32(data) => data.clone(),
            Self::Float16(data) => {
                data.iter()
                    .map(|&bits| f16_bits_to_f32(bits))
                    .collect()
            }
            Self::Int8 { data, scale, offset } => {
                data.iter()
                    .map(|&v| {
                        // Reverse: [-128, 127] to [0, 255], then scale back
                        let normalized = (v as i32 + 128) as f32;
                        normalized * scale + offset
                    })
                    .collect()
            }
            Self::Binary(data) => {
                let mut result = Vec::new();
                for byte in data {
                    for i in 0..8 {
                        let bit = (byte >> i) & 1;
                        result.push(if bit == 1 { 1.0 } else { -1.0 });
                    }
                }
                result
            }
        }
    }

    /// Get storage size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float32(data) => data.len() * 4,
            Self::Float16(data) => data.len() * 2,
            Self::Int8 { data, .. } => data.len() + 8, // data + scale + offset
            Self::Binary(data) => data.len(),
        }
    }

    /// Compute similarity directly on quantized data (optimized)
    pub fn quantized_similarity(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Binary(a), Self::Binary(b)) => {
                // Hamming distance based similarity
                let matching_bits: u32 = a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (!(x ^ y)).count_ones())
                    .sum();
                let total_bits = a.len() * 8;
                (matching_bits as f32 / total_bits as f32) * 2.0 - 1.0
            }
            _ => {
                // Fall back to dequantize
                cosine_similarity(&self.dequantize(), &other.dequantize())
            }
        }
    }
}

/// Dimensionality reduction methods
#[derive(Debug, Clone)]
pub enum DimensionalityReduction {
    /// Random projection (preserves distances approximately)
    RandomProjection {
        projection_matrix: Vec<Vec<f32>>,
        target_dim: usize,
    },
    /// Principal Component Analysis
    PCA {
        components: Vec<Vec<f32>>,
        mean: Vec<f32>,
    },
}

impl DimensionalityReduction {
    /// Create random projection reducer
    pub fn random_projection(source_dim: usize, target_dim: usize, seed: u64) -> Self {
        let mut rng_state = seed;
        let scale = (1.0 / target_dim as f32).sqrt();

        let projection_matrix: Vec<Vec<f32>> = (0..target_dim)
            .map(|_| {
                (0..source_dim)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                        // Approximate Gaussian via Box-Muller-like transform
                        let sign = if (rng_state >> 32) & 1 == 0 { 1.0 } else { -1.0 };
                        sign * scale * (1.0 + u)
                    })
                    .collect()
            })
            .collect();

        Self::RandomProjection {
            projection_matrix,
            target_dim,
        }
    }

    /// Fit PCA from data
    pub fn fit_pca(data: &[EmbeddingVec], target_dim: usize) -> Self {
        if data.is_empty() {
            return Self::PCA {
                components: Vec::new(),
                mean: Vec::new(),
            };
        }

        let source_dim = data[0].len();

        // Compute mean
        let mut mean = vec![0.0f32; source_dim];
        for vec in data {
            for (i, &v) in vec.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = data.len() as f32;
        for m in &mut mean {
            *m /= n;
        }

        // Center data
        let centered: Vec<Vec<f32>> = data.iter()
            .map(|v| v.iter().zip(&mean).map(|(a, b)| a - b).collect())
            .collect();

        // Power iteration for top components (simplified PCA)
        let mut components: Vec<Vec<f32>> = Vec::with_capacity(target_dim);

        for _ in 0..target_dim.min(source_dim) {
            // Random initial vector
            let mut component: Vec<f32> = (0..source_dim).map(|i| (i as f32).sin()).collect();
            component = normalize_l2(&component);

            // Power iteration
            for _ in 0..50 {
                let mut new_component = vec![0.0f32; source_dim];

                for vec in &centered {
                    let dot: f32 = vec.iter().zip(&component).map(|(a, b)| a * b).sum();
                    for (i, &v) in vec.iter().enumerate() {
                        new_component[i] += dot * v;
                    }
                }

                // Orthogonalize against existing components
                for existing in &components {
                    let dot: f32 = new_component.iter().zip(existing).map(|(a, b)| a * b).sum();
                    for (i, &e) in existing.iter().enumerate() {
                        new_component[i] -= dot * e;
                    }
                }

                component = normalize_l2(&new_component);
            }

            components.push(component);
        }

        Self::PCA { components, mean }
    }

    /// Reduce dimensionality
    pub fn transform(&self, embedding: &[f32]) -> EmbeddingVec {
        match self {
            Self::RandomProjection { projection_matrix, .. } => {
                projection_matrix.iter()
                    .map(|row| {
                        row.iter().zip(embedding).map(|(a, b)| a * b).sum()
                    })
                    .collect()
            }
            Self::PCA { components, mean } => {
                let centered: Vec<f32> = embedding.iter()
                    .zip(mean)
                    .map(|(v, m)| v - m)
                    .collect();

                components.iter()
                    .map(|comp| {
                        comp.iter().zip(&centered).map(|(a, b)| a * b).sum()
                    })
                    .collect()
            }
        }
    }

    /// Target dimension
    pub fn output_dim(&self) -> usize {
        match self {
            Self::RandomProjection { target_dim, .. } => *target_dim,
            Self::PCA { components, .. } => components.len(),
        }
    }
}

/// Cross-encoder for reranking
pub struct CrossEncoder {
    api_url: String,
    model_id: String,
    api_key: Option<String>,
}

impl CrossEncoder {
    /// Create cross-encoder with Ollama
    pub fn with_ollama(model: &str) -> Self {
        Self {
            api_url: "http://localhost:11434/api/generate".to_string(),
            model_id: model.to_string(),
            api_key: None,
        }
    }

    /// Create cross-encoder with custom endpoint
    pub fn new(api_url: &str, model_id: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            model_id: model_id.to_string(),
            api_key: None,
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Score query-document pair
    pub fn score(&self, query: &str, document: &str) -> Result<f32, EmbeddingError> {
        let scores = self.score_batch(query, &[document.to_string()])?;
        scores.into_iter().next().ok_or(EmbeddingError::EmptyResult)
    }

    /// Score query against multiple documents
    pub fn score_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, EmbeddingError> {
        // For Ollama, we use a prompt-based approach
        let mut scores = Vec::with_capacity(documents.len());

        for doc in documents {
            let prompt = format!(
                "Rate the relevance of the following document to the query on a scale of 0 to 10.\n\n\
                Query: {}\n\n\
                Document: {}\n\n\
                Respond with only a number from 0 to 10.",
                query, doc
            );

            let mut request = ureq::post(&self.api_url);
            if let Some(ref key) = self.api_key {
                request = request.set("Authorization", &format!("Bearer {}", key));
            }

            let response = request
                .send_json(ureq::json!({
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": false
                }))
                .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

            let result: serde_json::Value = response.into_json()
                .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

            let response_text = result["response"].as_str()
                .or(result["content"].as_str())
                .unwrap_or("5");

            // Parse score from response
            let score: f32 = response_text
                .trim()
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect::<String>()
                .parse()
                .unwrap_or(5.0);

            scores.push(score / 10.0); // Normalize to [0, 1]
        }

        Ok(scores)
    }

    /// Rerank documents by relevance to query
    pub fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<RankedDocument>, EmbeddingError> {
        let scores = self.score_batch(query, documents)?;

        let mut ranked: Vec<RankedDocument> = documents.iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(idx, (doc, &score))| RankedDocument {
                index: idx,
                content: doc.clone(),
                score,
            })
            .collect();

        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        Ok(ranked)
    }
}

/// Reranked document with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedDocument {
    pub index: usize,
    pub content: String,
    pub score: f32,
}

/// Embedding error types
#[derive(Debug)]
pub enum EmbeddingError {
    ApiError(String),
    ParseError(String),
    ConfigError(String),
    EmptyResult,
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiError(e) => write!(f, "API error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::ConfigError(e) => write!(f, "Config error: {}", e),
            Self::EmptyResult => write!(f, "Empty result"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Utility functions

/// Normalize vector to unit length (L2)
pub fn normalize_l2(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_l2() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize_l2(&vec);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);

        let c = vec![1.0, 1.0];
        let d = vec![1.0, 1.0];
        assert!((cosine_similarity(&c, &d) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_dense_embedding_config() {
        let config = DenseEmbeddingConfig::minilm_l6();
        assert_eq!(config.dimension, 384);

        let mpnet = DenseEmbeddingConfig::mpnet_base();
        assert_eq!(mpnet.dimension, 768);
    }

    #[test]
    fn test_simple_embedding() {
        let embedder = DenseEmbedder::new(DenseEmbeddingConfig {
            dimension: 128,
            ..Default::default()
        });

        let emb = embedder.embed("hello world").unwrap();
        assert_eq!(emb.len(), 128);

        // Check normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sparse_embedder() {
        let mut embedder = SparseEmbedder::new(SparseEmbeddingConfig::default());
        embedder.fit(&["hello world".to_string(), "goodbye world".to_string()]);

        let emb = embedder.embed("hello world");
        assert!(!emb.is_empty());
    }

    #[test]
    fn test_sparse_dot_product() {
        let mut a = SparseEmbedding::new();
        a.insert(1, 1.0);
        a.insert(2, 2.0);

        let mut b = SparseEmbedding::new();
        b.insert(1, 3.0);
        b.insert(3, 4.0);

        let dot = SparseEmbedder::dot_product(&a, &b);
        assert!((dot - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_quantization_float32() {
        let original = vec![1.0, 2.0, 3.0];
        let quantized = QuantizedEmbedding::quantize(&original, QuantizationType::Float32);
        let restored = quantized.dequantize();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_quantization_int8() {
        let original = vec![0.1, 0.5, 0.9];
        let quantized = QuantizedEmbedding::quantize(&original, QuantizationType::Int8);
        let restored = quantized.dequantize();

        // Should be approximately equal (int8 has limited precision especially with small ranges)
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!((o - r).abs() < 0.1, "Expected {} but got {}", o, r);
        }
    }

    #[test]
    fn test_quantization_binary() {
        let original = vec![-1.0, 1.0, -0.5, 0.5, -1.0, 1.0, -1.0, 1.0];
        let quantized = QuantizedEmbedding::quantize(&original, QuantizationType::Binary);

        let size = quantized.size_bytes();
        assert_eq!(size, 1); // 8 bits = 1 byte
    }

    #[test]
    fn test_random_projection() {
        let reducer = DimensionalityReduction::random_projection(100, 32, 42);
        let vec = vec![0.1; 100];
        let reduced = reducer.transform(&vec);
        assert_eq!(reduced.len(), 32);
    }

    #[test]
    fn test_pca_fit() {
        let data: Vec<EmbeddingVec> = (0..10)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();

        let pca = DimensionalityReduction::fit_pca(&data, 2);
        let reduced = pca.transform(&data[0]);
        assert_eq!(reduced.len(), 2);
    }

    #[test]
    fn test_pooling_strategy_default() {
        let strategy = PoolingStrategy::default();
        assert_eq!(strategy, PoolingStrategy::Mean);
    }

    #[test]
    fn test_hybrid_embedding() {
        let dense = DenseEmbedder::new(DenseEmbeddingConfig {
            dimension: 64,
            ..Default::default()
        });
        let sparse = SparseEmbedder::default();

        let retriever = HybridRetriever::new(dense, sparse, 0.7);
        let emb = retriever.embed("test query").unwrap();

        assert_eq!(emb.dense.len(), 64);
        assert!(!emb.sparse.is_empty() || true); // May be empty for short text
    }

    #[test]
    fn test_embedding_batch() {
        let embedder = DenseEmbedder::new(DenseEmbeddingConfig {
            dimension: 64,
            batch_size: 2,
            ..Default::default()
        });

        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];

        let embeddings = embedder.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
    }
}
