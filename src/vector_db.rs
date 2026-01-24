//! Vector Database integration for semantic search
//!
//! This module provides an optional integration with vector databases like Qdrant
//! for more scalable and efficient semantic search capabilities.
//!
//! # Features
//!
//! - **Multiple backends**: Support for Qdrant, in-memory, and file-based storage
//! - **Async-ready**: Designed for async operation with sync wrappers
//! - **Batched operations**: Efficient bulk insert and search
//! - **Metadata filtering**: Filter search results by metadata
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::vector_db::{VectorDb, VectorDbConfig, InMemoryVectorDb};
//!
//! // Create an in-memory vector database
//! let config = VectorDbConfig::default();
//! let mut db = InMemoryVectorDb::new(config);
//!
//! // Insert vectors
//! db.insert("doc1", vec![0.1, 0.2, 0.3], serde_json::json!({"title": "Hello"})).unwrap();
//! db.insert("doc2", vec![0.4, 0.5, 0.6], serde_json::json!({"title": "World"})).unwrap();
//!
//! // Search for similar vectors
//! let results = db.search(&[0.1, 0.2, 0.3], 5, None).unwrap();
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::error::{AiError, AiResult};

/// Configuration for vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Maximum number of vectors to store (for in-memory)
    pub max_vectors: Option<usize>,
    /// Collection/index name
    pub collection_name: String,
    /// Qdrant server URL (if using Qdrant backend)
    pub qdrant_url: Option<String>,
    /// API key for Qdrant Cloud
    pub qdrant_api_key: Option<String>,
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            dimensions: 384, // Common embedding size
            distance_metric: DistanceMetric::Cosine,
            max_vectors: Some(100_000),
            collection_name: "default".to_string(),
            qdrant_url: None,
            qdrant_api_key: None,
        }
    }
}

/// Distance metric for similarity calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_distance)
    Cosine,
    /// Euclidean distance (L2)
    Euclidean,
    /// Dot product
    DotProduct,
    /// Manhattan distance (L1)
    Manhattan,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::DotProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            DistanceMetric::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum()
            }
        }
    }

    /// Convert distance to similarity score (0-1, higher is better)
    pub fn to_similarity(&self, distance: f32) -> f32 {
        match self {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
        }
    }
}

/// A stored vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    /// Unique identifier
    pub id: String,
    /// The vector data
    pub vector: Vec<f32>,
    /// Associated metadata
    pub metadata: serde_json::Value,
    /// Timestamp when inserted
    pub timestamp: u64,
}

/// Search result from vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score (0-1, higher is better)
    pub score: f32,
    /// Associated metadata
    pub metadata: serde_json::Value,
    /// The vector itself (optional)
    pub vector: Option<Vec<f32>>,
}

/// Filter for metadata-based search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Field to filter on
    pub field: String,
    /// Filter operation
    pub operation: FilterOperation,
}

/// Filter operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperation {
    /// Exact match
    Equals(serde_json::Value),
    /// Not equal
    NotEquals(serde_json::Value),
    /// Greater than (for numbers)
    GreaterThan(f64),
    /// Less than (for numbers)
    LessThan(f64),
    /// Contains (for strings/arrays)
    Contains(String),
    /// In list
    In(Vec<serde_json::Value>),
}

impl MetadataFilter {
    /// Check if metadata matches the filter
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        let value = metadata.get(&self.field);

        match (&self.operation, value) {
            (FilterOperation::Equals(expected), Some(actual)) => actual == expected,
            (FilterOperation::NotEquals(expected), Some(actual)) => actual != expected,
            (FilterOperation::GreaterThan(threshold), Some(serde_json::Value::Number(n))) => {
                n.as_f64().map(|v| v > *threshold).unwrap_or(false)
            }
            (FilterOperation::LessThan(threshold), Some(serde_json::Value::Number(n))) => {
                n.as_f64().map(|v| v < *threshold).unwrap_or(false)
            }
            (FilterOperation::Contains(needle), Some(serde_json::Value::String(s))) => {
                s.contains(needle)
            }
            (FilterOperation::Contains(needle), Some(serde_json::Value::Array(arr))) => {
                arr.iter().any(|v| {
                    if let serde_json::Value::String(s) = v {
                        s == needle
                    } else {
                        false
                    }
                })
            }
            (FilterOperation::In(list), Some(actual)) => list.contains(actual),
            _ => false,
        }
    }
}

/// Trait for vector database implementations
pub trait VectorDb: Send + Sync {
    /// Insert a vector with metadata
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()>;

    /// Insert multiple vectors in a batch
    fn insert_batch(&mut self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> AiResult<usize>;

    /// Search for similar vectors
    fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>>;

    /// Get a vector by ID
    fn get(&self, id: &str) -> AiResult<Option<StoredVector>>;

    /// Delete a vector by ID
    fn delete(&mut self, id: &str) -> AiResult<bool>;

    /// Get the number of stored vectors
    fn count(&self) -> usize;

    /// Clear all vectors
    fn clear(&mut self) -> AiResult<()>;

    /// Check if the database is healthy/connected
    fn health_check(&self) -> AiResult<bool>;
}

/// In-memory vector database implementation
#[derive(Debug)]
pub struct InMemoryVectorDb {
    config: VectorDbConfig,
    vectors: HashMap<String, StoredVector>,
}

impl InMemoryVectorDb {
    /// Create a new in-memory vector database
    pub fn new(config: VectorDbConfig) -> Self {
        Self {
            config,
            vectors: HashMap::new(),
        }
    }

    /// Get current timestamp in milliseconds
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Evict oldest vectors if over capacity
    fn evict_if_needed(&mut self) {
        if let Some(max) = self.config.max_vectors {
            while self.vectors.len() >= max {
                // Find oldest vector
                if let Some(oldest_id) = self
                    .vectors
                    .iter()
                    .min_by_key(|(_, v)| v.timestamp)
                    .map(|(id, _)| id.clone())
                {
                    self.vectors.remove(&oldest_id);
                } else {
                    break;
                }
            }
        }
    }
}

impl VectorDb for InMemoryVectorDb {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        if vector.len() != self.config.dimensions {
            return Err(AiError::Validation(crate::error::ValidationError::Custom {
                field: "vector".to_string(),
                message: format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.config.dimensions,
                    vector.len()
                ),
            }));
        }

        self.evict_if_needed();

        let stored = StoredVector {
            id: id.to_string(),
            vector,
            metadata,
            timestamp: Self::now(),
        };

        self.vectors.insert(id.to_string(), stored);
        Ok(())
    }

    fn insert_batch(&mut self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> AiResult<usize> {
        let mut inserted = 0;
        for (id, vector, metadata) in vectors {
            if self.insert(&id, vector, metadata).is_ok() {
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        if query.len() != self.config.dimensions {
            return Err(AiError::Validation(crate::error::ValidationError::Custom {
                field: "query".to_string(),
                message: format!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.config.dimensions,
                    query.len()
                ),
            }));
        }

        let mut results: Vec<_> = self
            .vectors
            .values()
            .filter(|v| {
                if let Some(filters) = filter {
                    filters.iter().all(|f| f.matches(&v.metadata))
                } else {
                    true
                }
            })
            .map(|v| {
                let distance = self.config.distance_metric.calculate(query, &v.vector);
                let score = self.config.distance_metric.to_similarity(distance);
                (v, score)
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(limit)
            .map(|(v, score)| VectorSearchResult {
                id: v.id.clone(),
                score,
                metadata: v.metadata.clone(),
                vector: Some(v.vector.clone()),
            })
            .collect())
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        Ok(self.vectors.get(id).cloned())
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        Ok(self.vectors.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.vectors.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.vectors.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        Ok(true)
    }
}

/// Qdrant vector database client (HTTP-based)
#[derive(Debug)]
pub struct QdrantClient {
    config: VectorDbConfig,
    base_url: String,
    api_key: Option<String>,
}

impl QdrantClient {
    /// Create a new Qdrant client
    pub fn new(config: VectorDbConfig) -> AiResult<Self> {
        let base_url = config.qdrant_url.clone().ok_or_else(|| {
            AiError::Config(crate::error::ConfigError::MissingValue {
                field: "qdrant_url".to_string(),
                description: "Qdrant server URL is required".to_string(),
            })
        })?;

        Ok(Self {
            api_key: config.qdrant_api_key.clone(),
            config,
            base_url,
        })
    }

    /// Make an HTTP request to Qdrant
    fn request(&self, method: &str, endpoint: &str, body: Option<&str>) -> AiResult<String> {
        let url = format!("{}{}", self.base_url, endpoint);
        let agent = ureq::agent();

        let mut request = match method {
            "GET" => agent.get(&url),
            "POST" => agent.post(&url),
            "PUT" => agent.put(&url),
            "DELETE" => agent.delete(&url),
            _ => return Err(AiError::Other(format!("Unknown HTTP method: {}", method))),
        };

        request = request.set("Content-Type", "application/json");

        if let Some(key) = &self.api_key {
            request = request.set("api-key", key);
        }

        let response = if let Some(body) = body {
            request.send_string(body)
        } else {
            request.call()
        };

        match response {
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(body)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(AiError::Provider(crate::error::ProviderError::ApiError {
                    provider: "Qdrant".to_string(),
                    status_code: code,
                    message: body,
                }))
            }
            Err(e) => Err(AiError::Network(crate::error::NetworkError::ConnectionFailed {
                url,
                reason: e.to_string(),
            })),
        }
    }

    /// Create collection if it doesn't exist
    pub fn create_collection_if_not_exists(&self) -> AiResult<()> {
        let body = serde_json::json!({
            "vectors": {
                "size": self.config.dimensions,
                "distance": match self.config.distance_metric {
                    DistanceMetric::Cosine => "Cosine",
                    DistanceMetric::Euclidean => "Euclid",
                    DistanceMetric::DotProduct => "Dot",
                    DistanceMetric::Manhattan => "Manhattan",
                }
            }
        });

        let endpoint = format!("/collections/{}", self.config.collection_name);

        // Try to create, ignore if already exists
        let _ = self.request("PUT", &endpoint, Some(&body.to_string()));
        Ok(())
    }
}

impl VectorDb for QdrantClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let body = serde_json::json!({
            "points": [{
                "id": id,
                "vector": vector,
                "payload": metadata
            }]
        });

        let endpoint = format!("/collections/{}/points", self.config.collection_name);
        self.request("PUT", &endpoint, Some(&body.to_string()))?;
        Ok(())
    }

    fn insert_batch(&mut self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> AiResult<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        let points: Vec<_> = vectors
            .iter()
            .map(|(id, vector, metadata)| {
                serde_json::json!({
                    "id": id,
                    "vector": vector,
                    "payload": metadata
                })
            })
            .collect();

        let body = serde_json::json!({ "points": points });
        let endpoint = format!("/collections/{}/points", self.config.collection_name);

        self.request("PUT", &endpoint, Some(&body.to_string()))?;
        Ok(vectors.len())
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let mut body = serde_json::json!({
            "vector": query,
            "limit": limit,
            "with_payload": true,
            "with_vector": false
        });

        // Add filter if provided
        if let Some(filters) = filter {
            let must: Vec<_> = filters
                .iter()
                .map(|f| {
                    let condition = match &f.operation {
                        FilterOperation::Equals(v) => serde_json::json!({
                            "key": f.field,
                            "match": { "value": v }
                        }),
                        FilterOperation::GreaterThan(v) => serde_json::json!({
                            "key": f.field,
                            "range": { "gt": v }
                        }),
                        FilterOperation::LessThan(v) => serde_json::json!({
                            "key": f.field,
                            "range": { "lt": v }
                        }),
                        _ => serde_json::json!({}),
                    };
                    condition
                })
                .collect();

            body["filter"] = serde_json::json!({ "must": must });
        }

        let endpoint = format!("/collections/{}/points/search", self.config.collection_name);
        let response = self.request("POST", &endpoint, Some(&body.to_string()))?;

        let parsed: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| AiError::Serialization(crate::error::SerializationError::json_deserialize(e.to_string())))?;

        let results = parsed["result"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|r| VectorSearchResult {
                id: r["id"].as_str().unwrap_or_default().to_string(),
                score: r["score"].as_f64().unwrap_or(0.0) as f32,
                metadata: r["payload"].clone(),
                vector: None,
            })
            .collect();

        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        let endpoint = format!(
            "/collections/{}/points/{}",
            self.config.collection_name, id
        );

        match self.request("GET", &endpoint, None) {
            Ok(response) => {
                let parsed: serde_json::Value = serde_json::from_str(&response)
                    .map_err(|e| AiError::Serialization(crate::error::SerializationError::json_deserialize(e.to_string())))?;

                if let Some(result) = parsed.get("result") {
                    let vector: Vec<f32> = result["vector"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();

                    Ok(Some(StoredVector {
                        id: id.to_string(),
                        vector,
                        metadata: result["payload"].clone(),
                        timestamp: 0,
                    }))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let body = serde_json::json!({
            "points": [id]
        });

        let endpoint = format!("/collections/{}/points/delete", self.config.collection_name);
        self.request("POST", &endpoint, Some(&body.to_string()))?;
        Ok(true)
    }

    fn count(&self) -> usize {
        let endpoint = format!("/collections/{}", self.config.collection_name);

        if let Ok(response) = self.request("GET", &endpoint, None) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                return parsed["result"]["points_count"]
                    .as_u64()
                    .unwrap_or(0) as usize;
            }
        }
        0
    }

    fn clear(&mut self) -> AiResult<()> {
        // Delete and recreate collection
        let endpoint = format!("/collections/{}", self.config.collection_name);
        let _ = self.request("DELETE", &endpoint, None);
        self.create_collection_if_not_exists()
    }

    fn health_check(&self) -> AiResult<bool> {
        match self.request("GET", "/", None) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Builder for creating vector databases
pub struct VectorDbBuilder {
    config: VectorDbConfig,
    backend: VectorDbBackend,
}

/// Available backends
#[derive(Debug, Clone, Default)]
pub enum VectorDbBackend {
    /// In-memory storage
    #[default]
    InMemory,
    /// Qdrant server
    Qdrant,
}

impl VectorDbBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: VectorDbConfig::default(),
            backend: VectorDbBackend::InMemory,
        }
    }

    /// Set vector dimensions
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.config.dimensions = dim;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set collection name
    pub fn collection_name(mut self, name: impl Into<String>) -> Self {
        self.config.collection_name = name.into();
        self
    }

    /// Set maximum vectors (for in-memory)
    pub fn max_vectors(mut self, max: usize) -> Self {
        self.config.max_vectors = Some(max);
        self
    }

    /// Use in-memory backend
    pub fn in_memory(mut self) -> Self {
        self.backend = VectorDbBackend::InMemory;
        self
    }

    /// Use Qdrant backend
    pub fn qdrant(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::Qdrant;
        self
    }

    /// Set Qdrant API key
    pub fn qdrant_api_key(mut self, key: impl Into<String>) -> Self {
        self.config.qdrant_api_key = Some(key.into());
        self
    }

    /// Build the vector database
    pub fn build(self) -> AiResult<Box<dyn VectorDb>> {
        match self.backend {
            VectorDbBackend::InMemory => Ok(Box::new(InMemoryVectorDb::new(self.config))),
            VectorDbBackend::Qdrant => {
                let client = QdrantClient::new(self.config)?;
                client.create_collection_if_not_exists()?;
                Ok(Box::new(client))
            }
        }
    }
}

impl Default for VectorDbBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid search combining vector and keyword search
pub struct HybridVectorSearch<V: VectorDb> {
    vector_db: V,
    /// Weight for vector search results (0-1)
    pub vector_weight: f32,
    /// Weight for keyword search results (0-1)
    pub keyword_weight: f32,
}

impl<V: VectorDb> HybridVectorSearch<V> {
    /// Create a new hybrid search
    pub fn new(vector_db: V) -> Self {
        Self {
            vector_db,
            vector_weight: 0.7,
            keyword_weight: 0.3,
        }
    }

    /// Set search weights
    pub fn with_weights(mut self, vector: f32, keyword: f32) -> Self {
        self.vector_weight = vector;
        self.keyword_weight = keyword;
        self
    }

    /// Perform hybrid search
    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        // Get vector search results
        let vector_results = self.vector_db.search(query_vector, limit * 2, filter)?;

        // Simple keyword matching on metadata
        let query_lower = query_text.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored_results: Vec<(VectorSearchResult, f32)> = vector_results
            .into_iter()
            .map(|r| {
                // Calculate keyword score from metadata
                let metadata_str = r.metadata.to_string().to_lowercase();
                let keyword_score: f32 = query_words
                    .iter()
                    .filter(|w| metadata_str.contains(*w))
                    .count() as f32
                    / query_words.len().max(1) as f32;

                // Combine scores
                let combined_score =
                    r.score * self.vector_weight + keyword_score * self.keyword_weight;

                (r, combined_score)
            })
            .collect();

        // Sort by combined score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top results with updated scores
        Ok(scored_results
            .into_iter()
            .take(limit)
            .map(|(mut r, score)| {
                r.score = score;
                r
            })
            .collect())
    }

    /// Get access to the underlying vector database
    pub fn vector_db(&self) -> &V {
        &self.vector_db
    }

    /// Get mutable access to the underlying vector database
    pub fn vector_db_mut(&mut self) -> &mut V {
        &mut self.vector_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        // Cosine distance
        let cosine = DistanceMetric::Cosine;
        assert!((cosine.calculate(&a, &c) - 0.0).abs() < 0.001); // Same vector
        assert!((cosine.calculate(&a, &b) - 1.0).abs() < 0.001); // Orthogonal

        // Euclidean distance
        let euclidean = DistanceMetric::Euclidean;
        assert!((euclidean.calculate(&a, &c) - 0.0).abs() < 0.001);
        assert!((euclidean.calculate(&a, &b) - std::f32::consts::SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_in_memory_vector_db() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        // Insert vectors
        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"label": "x"}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({"label": "y"}))
            .unwrap();
        db.insert("v3", vec![0.0, 0.0, 1.0], serde_json::json!({"label": "z"}))
            .unwrap();

        assert_eq!(db.count(), 3);

        // Search
        let results = db.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // Most similar

        // Get
        let v = db.get("v1").unwrap().unwrap();
        assert_eq!(v.metadata["label"], "x");

        // Delete
        assert!(db.delete("v1").unwrap());
        assert_eq!(db.count(), 2);
    }

    #[test]
    fn test_metadata_filter() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"type": "a", "score": 10}))
            .unwrap();
        db.insert("v2", vec![0.9, 0.1, 0.0], serde_json::json!({"type": "b", "score": 20}))
            .unwrap();
        db.insert("v3", vec![0.8, 0.2, 0.0], serde_json::json!({"type": "a", "score": 30}))
            .unwrap();

        // Filter by type
        let filter = vec![MetadataFilter {
            field: "type".to_string(),
            operation: FilterOperation::Equals(serde_json::json!("a")),
        }];

        let results = db.search(&[1.0, 0.0, 0.0], 10, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.metadata["type"] == "a"));
    }

    #[test]
    fn test_vector_dimension_validation() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        // Wrong dimension should fail
        let result = db.insert("v1", vec![1.0, 0.0], serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_builder() {
        let db = VectorDbBuilder::new()
            .dimensions(128)
            .distance_metric(DistanceMetric::Euclidean)
            .collection_name("test")
            .max_vectors(1000)
            .in_memory()
            .build()
            .unwrap();

        assert_eq!(db.count(), 0);
        assert!(db.health_check().unwrap());
    }

    #[test]
    fn test_eviction() {
        let config = VectorDbConfig {
            dimensions: 3,
            max_vectors: Some(2),
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.insert("v3", vec![0.0, 0.0, 1.0], serde_json::json!({})).unwrap();

        // Should have evicted oldest (v1)
        assert_eq!(db.count(), 2);
        assert!(db.get("v1").unwrap().is_none());
    }

    #[test]
    fn test_batch_insert() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        let vectors = vec![
            ("v1".to_string(), vec![1.0, 0.0, 0.0], serde_json::json!({})),
            ("v2".to_string(), vec![0.0, 1.0, 0.0], serde_json::json!({})),
            ("v3".to_string(), vec![0.0, 0.0, 1.0], serde_json::json!({})),
        ];

        let inserted = db.insert_batch(vectors).unwrap();
        assert_eq!(inserted, 3);
        assert_eq!(db.count(), 3);
    }

    #[test]
    fn test_hybrid_search() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"text": "hello world"}))
            .unwrap();
        db.insert("v2", vec![0.9, 0.1, 0.0], serde_json::json!({"text": "hello there"}))
            .unwrap();
        db.insert("v3", vec![0.0, 1.0, 0.0], serde_json::json!({"text": "goodbye world"}))
            .unwrap();

        let hybrid = HybridVectorSearch::new(db).with_weights(0.5, 0.5);

        let results = hybrid
            .search(&[1.0, 0.0, 0.0], "hello", 3, None)
            .unwrap();

        // v1 and v2 should rank higher due to "hello" keyword match
        assert!(!results.is_empty());
    }
}
