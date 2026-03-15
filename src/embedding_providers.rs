//! Embedding providers for generating text embeddings.
//!
//! Provides a unified `EmbeddingProvider` trait with implementations for:
//! - Local TF-IDF (wraps `LocalEmbedder`)
//! - Ollama (local LLM server)
//! - OpenAI (cloud API)
//! - HuggingFace (Inference API)

use anyhow::{Context, Result};
use std::time::Duration;

use crate::embeddings::{EmbeddingConfig, LocalEmbedder};

// ============================================================================
// EmbeddingProvider Trait
// ============================================================================

/// Trait for embedding providers — both local and remote API-based.
///
/// # Stability
///
/// New methods may be added to this trait in minor versions with default
/// implementations. Required methods will only change in major versions.
pub trait EmbeddingProvider: Send + Sync {
    /// Provider name
    fn name(&self) -> &str;
    /// Vector dimensions
    fn dimensions(&self) -> usize;
    /// Maximum input tokens per text
    fn max_tokens(&self) -> usize;
    /// Embed multiple texts at once (batch)
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embed a single text (default: delegates to batch embed)
    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding result"))
    }
}

// ============================================================================
// LocalTfIdfEmbedding
// ============================================================================

/// Local TF-IDF embedding provider wrapping `LocalEmbedder`.
pub struct LocalTfIdfEmbedding {
    embedder: LocalEmbedder,
    config: EmbeddingConfig,
}

impl LocalTfIdfEmbedding {
    /// Create with default `EmbeddingConfig`.
    pub fn new() -> Self {
        let config = EmbeddingConfig::default();
        Self {
            embedder: LocalEmbedder::new(config.clone()),
            config,
        }
    }

    /// Create with a custom `EmbeddingConfig`.
    pub fn with_config(config: EmbeddingConfig) -> Self {
        Self {
            embedder: LocalEmbedder::new(config.clone()),
            config,
        }
    }

    /// Train the embedder on a corpus of documents.
    pub fn train(&mut self, documents: &[&str]) {
        self.embedder.train(documents);
    }
}

impl EmbeddingProvider for LocalTfIdfEmbedding {
    fn name(&self) -> &str {
        "local-tfidf"
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn max_tokens(&self) -> usize {
        8192
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(self.embedder.embed_batch(texts))
    }
}

// ============================================================================
// OllamaEmbeddings
// ============================================================================

/// Embedding provider using the Ollama `/api/embed` endpoint.
pub struct OllamaEmbeddings {
    base_url: String,
    model: String,
    dimensions: usize,
}

impl OllamaEmbeddings {
    /// Create a new Ollama embedding provider.
    pub fn new(base_url: &str, model: &str) -> Self {
        // Infer dimensions based on known models, default to 768
        let dimensions = match model {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            "all-minilm" => 384,
            _ => 768,
        };
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    /// Create with default settings (localhost:11434, nomic-embed-text, 768 dims).
    pub fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
        }
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl EmbeddingProvider for OllamaEmbeddings {
    fn name(&self) -> &str {
        "ollama"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn max_tokens(&self) -> usize {
        8192
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);

        let input: Vec<&str> = texts.to_vec();
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });

        let response = ureq::post(&url)
            .timeout(Duration::from_secs(30))
            .send_json(&body)
            .context("Ollama embed API request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse Ollama embed response")?;

        let embeddings = json
            .get("embeddings")
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow::anyhow!("Ollama response missing 'embeddings' field"))?;

        let mut result = Vec::with_capacity(embeddings.len());
        for emb in embeddings {
            let vec = emb
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid embedding format in Ollama response"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            result.push(vec);
        }

        Ok(result)
    }
}

// ============================================================================
// OpenAIEmbeddings
// ============================================================================

/// Embedding provider using OpenAI's `/v1/embeddings` endpoint.
///
/// Also compatible with Azure OpenAI and other OpenAI-compatible APIs
/// via `with_base_url`.
#[derive(Debug)]
pub struct OpenAIEmbeddings {
    api_key: String,
    base_url: String,
    model: String,
    dimensions: usize,
}

impl OpenAIEmbeddings {
    /// Create a new OpenAI embedding provider with defaults.
    ///
    /// Defaults: base_url="https://api.openai.com", model="text-embedding-3-small", dims=1536.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com".to_string(),
            model: "text-embedding-3-small".to_string(),
            dimensions: 1536,
        }
    }

    /// Create from the `OPENAI_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key =
            std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY environment variable not set")?;
        Ok(Self::new(&api_key))
    }

    /// Change the model and dimensions.
    pub fn with_model(mut self, model: &str, dimensions: usize) -> Self {
        self.model = model.to_string();
        self.dimensions = dimensions;
        self
    }

    /// Change the base URL (for Azure OpenAI or compatible APIs).
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl EmbeddingProvider for OpenAIEmbeddings {
    fn name(&self) -> &str {
        "openai"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn max_tokens(&self) -> usize {
        8191
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let input: Vec<&str> = texts.to_vec();
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(30))
            .send_json(&body)
            .context("OpenAI embeddings API request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse OpenAI embeddings response")?;

        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow::anyhow!("OpenAI response missing 'data' field"))?;

        // OpenAI returns embeddings in order, each with an "embedding" field
        let mut result = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    anyhow::anyhow!("Invalid embedding format in OpenAI response")
                })?;

            let vec: Vec<f32> = embedding
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            result.push(vec);
        }

        Ok(result)
    }
}

// ============================================================================
// HuggingFaceEmbeddings
// ============================================================================

/// Embedding provider using the HuggingFace Inference API.
pub struct HuggingFaceEmbeddings {
    api_key: String,
    model: String,
    dimensions: usize,
}

impl HuggingFaceEmbeddings {
    /// Create a new HuggingFace embedding provider.
    pub fn new(api_key: &str, model: &str, dimensions: usize) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    /// Create from environment variable (`HF_API_KEY` or `HUGGING_FACE_HUB_TOKEN`).
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("HF_API_KEY")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .context(
                "Neither HF_API_KEY nor HUGGING_FACE_HUB_TOKEN environment variable is set",
            )?;
        Ok(Self::default_model(&api_key))
    }

    /// Create with the default model (sentence-transformers/all-MiniLM-L6-v2, 384 dims).
    pub fn default_model(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
        }
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl EmbeddingProvider for HuggingFaceEmbeddings {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn max_tokens(&self) -> usize {
        512
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/{}",
            self.model
        );

        let inputs: Vec<&str> = texts.to_vec();
        let body = serde_json::json!({
            "inputs": inputs,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(30))
            .send_json(&body)
            .context("HuggingFace Inference API request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse HuggingFace response")?;

        // HF returns [[f32...], ...] for batch input
        let outer = json
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("HuggingFace response is not an array"))?;

        let mut result = Vec::with_capacity(outer.len());
        for item in outer {
            let vec = item
                .as_array()
                .ok_or_else(|| {
                    anyhow::anyhow!("Invalid embedding format in HuggingFace response")
                })?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            result.push(vec);
        }

        Ok(result)
    }
}

// ============================================================================
// Factory Function
// ============================================================================

/// Create an embedding provider by name.
///
/// Supported names:
/// - `"local"` or `"tfidf"` — Local TF-IDF embeddings
/// - `"ollama"` — Ollama (default: localhost:11434, nomic-embed-text)
/// - `"openai"` — OpenAI (requires `OPENAI_API_KEY` env var)
/// - `"huggingface"` or `"hf"` — HuggingFace (requires `HF_API_KEY` or `HUGGING_FACE_HUB_TOKEN`)
pub fn create_embedding_provider(name: &str) -> Result<Box<dyn EmbeddingProvider>> {
    match name {
        "local" | "tfidf" => Ok(Box::new(LocalTfIdfEmbedding::new())),
        "ollama" => Ok(Box::new(OllamaEmbeddings::default())),
        "openai" => {
            OpenAIEmbeddings::from_env().map(|e| Box::new(e) as Box<dyn EmbeddingProvider>)
        }
        "huggingface" | "hf" => {
            HuggingFaceEmbeddings::from_env().map(|e| Box::new(e) as Box<dyn EmbeddingProvider>)
        }
        _ => anyhow::bail!("Unknown embedding provider: {}", name),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_tfidf_creation() {
        let provider = LocalTfIdfEmbedding::new();
        assert_eq!(provider.name(), "local-tfidf");
        assert_eq!(provider.dimensions(), 256);
        assert_eq!(provider.max_tokens(), 8192);
    }

    #[test]
    fn test_local_tfidf_embed() {
        let mut provider = LocalTfIdfEmbedding::new();
        let docs = vec![
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps above a sleepy canine",
            "Hello world this is a test document",
        ];
        provider.train(&docs);

        let result = provider.embed(&["The quick brown fox"]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 256);
    }

    #[test]
    fn test_local_tfidf_embed_batch() {
        let mut provider = LocalTfIdfEmbedding::new();
        let docs = vec![
            "Machine learning algorithms are powerful",
            "Deep neural networks transform data",
            "Natural language processing understands text",
        ];
        provider.train(&docs);

        let texts: Vec<&str> = vec![
            "machine learning models",
            "neural network architecture",
            "text processing pipeline",
        ];
        let result = provider.embed(&texts).unwrap();
        assert_eq!(result.len(), 3);
        for emb in &result {
            assert_eq!(emb.len(), 256);
        }
    }

    #[test]
    fn test_local_tfidf_embed_single() {
        let mut provider = LocalTfIdfEmbedding::new();
        let docs = vec![
            "Rust is a systems programming language",
            "Python is great for scripting",
        ];
        provider.train(&docs);

        let result = provider.embed_single("Rust programming").unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_local_tfidf_with_config() {
        let config = EmbeddingConfig {
            dimensions: 512,
            min_word_freq: 1,
            max_vocab_size: 5000,
            use_subwords: false,
            ngram_range: (1, 1),
        };
        let provider = LocalTfIdfEmbedding::with_config(config);
        assert_eq!(provider.dimensions(), 512);
        assert_eq!(provider.name(), "local-tfidf");
    }

    #[test]
    fn test_ollama_creation() {
        let provider = OllamaEmbeddings::new("http://myhost:1234", "mxbai-embed-large");
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.dimensions(), 1024);
        assert_eq!(provider.base_url(), "http://myhost:1234");
        assert_eq!(provider.model(), "mxbai-embed-large");
    }

    #[test]
    fn test_ollama_default() {
        let provider = OllamaEmbeddings::default();
        assert_eq!(provider.base_url(), "http://localhost:11434");
        assert_eq!(provider.model(), "nomic-embed-text");
        assert_eq!(provider.dimensions(), 768);
        assert_eq!(provider.max_tokens(), 8192);
    }

    #[test]
    fn test_openai_creation() {
        let provider = OpenAIEmbeddings::new("test-key-123");
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), "text-embedding-3-small");
        assert_eq!(provider.dimensions(), 1536);
        assert_eq!(provider.base_url(), "https://api.openai.com");
        assert_eq!(provider.max_tokens(), 8191);
    }

    #[test]
    fn test_openai_with_model() {
        let provider = OpenAIEmbeddings::new("test-key")
            .with_model("text-embedding-3-large", 3072);
        assert_eq!(provider.model(), "text-embedding-3-large");
        assert_eq!(provider.dimensions(), 3072);
    }

    #[test]
    fn test_openai_with_base_url() {
        let provider = OpenAIEmbeddings::new("test-key")
            .with_base_url("https://my-azure-endpoint.openai.azure.com");
        assert_eq!(
            provider.base_url(),
            "https://my-azure-endpoint.openai.azure.com"
        );
    }

    #[test]
    fn test_openai_from_env_missing() {
        // Temporarily clear the env var if it happens to be set
        let original = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OPENAI_API_KEY");

        let result = OpenAIEmbeddings::from_env();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("OPENAI_API_KEY")
        );

        // Restore if it was set
        if let Some(key) = original {
            std::env::set_var("OPENAI_API_KEY", key);
        }
    }

    #[test]
    fn test_huggingface_creation() {
        let provider = HuggingFaceEmbeddings::new("hf-key-123", "custom/model", 768);
        assert_eq!(provider.name(), "huggingface");
        assert_eq!(provider.model(), "custom/model");
        assert_eq!(provider.dimensions(), 768);
        assert_eq!(provider.max_tokens(), 512);
    }

    #[test]
    fn test_huggingface_default_model() {
        let provider = HuggingFaceEmbeddings::default_model("hf-key");
        assert_eq!(provider.model(), "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(provider.dimensions(), 384);
        assert_eq!(provider.name(), "huggingface");
    }

    #[test]
    fn test_create_embedding_provider_local() {
        let provider = create_embedding_provider("local").unwrap();
        assert_eq!(provider.name(), "local-tfidf");
        assert_eq!(provider.dimensions(), 256);

        let provider2 = create_embedding_provider("tfidf").unwrap();
        assert_eq!(provider2.name(), "local-tfidf");
    }

    #[test]
    fn test_create_embedding_provider_unknown() {
        let result = create_embedding_provider("nonexistent");
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("Unknown embedding provider"),
            "Expected 'Unknown embedding provider' in error, got: {}",
            err_msg
        );
    }
}
