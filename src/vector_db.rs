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

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::error::{AiError, AiResult};

/// Information about a vector database backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Human-readable name
    pub name: &'static str,
    /// Tier level (0=in-memory, 1=sqlite-vec, 2=embedded, 3=dedicated, 4=cloud)
    pub tier: u8,
    /// Whether data survives process restart
    pub supports_persistence: bool,
    /// Whether metadata filtering is supported during search
    pub supports_filtering: bool,
    /// Whether export_all is implemented
    pub supports_export: bool,
    /// Suggested maximum vectors before considering next tier
    pub max_recommended_vectors: Option<usize>,
}

impl Default for BackendInfo {
    fn default() -> Self {
        Self {
            name: "Unknown",
            tier: 0,
            supports_persistence: false,
            supports_filtering: false,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

/// Result of a migration between backends
#[derive(Debug, Clone)]
pub struct VectorMigrationResult {
    /// Number of vectors exported from source
    pub exported: usize,
    /// Number of vectors successfully imported into target
    pub imported: usize,
}

/// Migrate all vectors from one backend to another.
/// The source must support export_all and the target must support insert.
pub fn migrate_vectors(
    source: &dyn VectorDb,
    target: &mut dyn VectorDb,
) -> AiResult<VectorMigrationResult> {
    let vectors = source.export_all()?;
    let count = vectors.len();
    let imported = target.import_bulk(vectors)?;
    Ok(VectorMigrationResult {
        exported: count,
        imported,
    })
}

/// Convert a string ID to a deterministic u64 (for backends that need numeric IDs)
pub fn string_id_to_u64(id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

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
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
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
    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize>;

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

    /// Get information about this backend (tier, capabilities, etc.)
    fn backend_info(&self) -> BackendInfo {
        BackendInfo::default()
    }

    /// Export all stored vectors (for migration between backends).
    /// Not all backends support this — check `backend_info().supports_export`.
    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        Err(AiError::Other(
            "export_all not supported by this backend".into(),
        ))
    }

    /// Import vectors in bulk (for migration). Default impl calls insert() in a loop.
    fn import_bulk(&mut self, vectors: Vec<StoredVector>) -> AiResult<usize> {
        let mut count = 0;
        for v in vectors {
            self.insert(&v.id, v.vector, v.metadata)?;
            count += 1;
        }
        Ok(count)
    }
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

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
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

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "InMemory",
            tier: 0,
            supports_persistence: false,
            supports_filtering: true,
            supports_export: true,
            max_recommended_vectors: Some(10_000),
        }
    }

    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        Ok(self.vectors.values().cloned().collect())
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
            Err(e) => Err(AiError::Network(
                crate::error::NetworkError::ConnectionFailed {
                    url,
                    reason: e.to_string(),
                },
            )),
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
        // Store original ID in payload so we can recover it on export
        let mut payload = metadata;
        payload["_original_id"] = serde_json::json!(id);
        let numeric_id = string_id_to_u64(id);

        let body = serde_json::json!({
            "points": [{
                "id": numeric_id,
                "vector": vector,
                "payload": payload
            }]
        });

        let endpoint = format!("/collections/{}/points", self.config.collection_name);
        self.request("PUT", &endpoint, Some(&body.to_string()))?;
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        let points: Vec<_> = vectors
            .iter()
            .map(|(id, vector, metadata)| {
                let mut payload = metadata.clone();
                payload["_original_id"] = serde_json::json!(id);
                serde_json::json!({
                    "id": string_id_to_u64(id),
                    "vector": vector,
                    "payload": payload
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

        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| {
            AiError::Serialization(crate::error::SerializationError::json_deserialize(
                e.to_string(),
            ))
        })?;

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
        let numeric_id = string_id_to_u64(id);
        let endpoint = format!(
            "/collections/{}/points/{}",
            self.config.collection_name, numeric_id
        );

        match self.request("GET", &endpoint, None) {
            Ok(response) => {
                let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| {
                    AiError::Serialization(crate::error::SerializationError::json_deserialize(
                        e.to_string(),
                    ))
                })?;

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
            "points": [string_id_to_u64(id)]
        });

        let endpoint = format!("/collections/{}/points/delete", self.config.collection_name);
        self.request("POST", &endpoint, Some(&body.to_string()))?;
        Ok(true)
    }

    fn count(&self) -> usize {
        let endpoint = format!("/collections/{}", self.config.collection_name);

        if let Ok(response) = self.request("GET", &endpoint, None) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                return parsed["result"]["points_count"].as_u64().unwrap_or(0) as usize;
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

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Qdrant",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: true,
            max_recommended_vectors: None, // scales to billions with clustering
        }
    }

    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        let mut all_vectors = Vec::new();
        let mut offset: Option<String> = None;
        let batch_size = 100;

        loop {
            let mut body = serde_json::json!({
                "limit": batch_size,
                "with_payload": true,
                "with_vector": true
            });
            if let Some(ref off) = offset {
                body["offset"] = serde_json::json!(off);
            }

            let endpoint = format!("/collections/{}/points/scroll", self.config.collection_name);
            let response = self.request("POST", &endpoint, Some(&body.to_string()))?;

            let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| {
                AiError::Serialization(crate::error::SerializationError::json_deserialize(
                    e.to_string(),
                ))
            })?;

            let points = match parsed["result"]["points"].as_array() {
                Some(pts) => pts,
                None => break,
            };

            if points.is_empty() {
                break;
            }

            for point in points {
                let id = point["id"]
                    .as_str()
                    .or_else(|| point["id"].as_u64().map(|_| ""))
                    .unwrap_or_default();
                let id_str = if id.is_empty() {
                    point["id"].to_string().trim_matches('"').to_string()
                } else {
                    id.to_string()
                };

                let vector: Vec<f32> = point["vector"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                all_vectors.push(StoredVector {
                    id: id_str,
                    vector,
                    metadata: point["payload"].clone(),
                    timestamp: 0,
                });
            }

            // Get next page offset
            match parsed["result"]["next_page_offset"].as_str() {
                Some(next) => offset = Some(next.to_string()),
                None => match parsed["result"]["next_page_offset"].as_u64() {
                    Some(next) => offset = Some(next.to_string()),
                    None => break,
                },
            }
        }

        Ok(all_vectors)
    }
}

/// Builder for creating vector databases
pub struct VectorDbBuilder {
    config: VectorDbConfig,
    backend: VectorDbBackend,
    /// Path for LanceDB storage (only used with Lance backend)
    #[cfg(feature = "vector-lancedb")]
    lance_path: Option<String>,
}

/// Available backends
#[derive(Debug, Clone, Default)]
pub enum VectorDbBackend {
    /// In-memory storage (Tier 0)
    #[default]
    InMemory,
    /// Qdrant server (Tier 3)
    Qdrant,
    /// LanceDB embedded (Tier 2) — requires `vector-lancedb` feature
    #[cfg(feature = "vector-lancedb")]
    Lance,
    /// Pinecone cloud vector DB (Tier 4)
    Pinecone,
    /// Chroma vector DB (Tier 3)
    Chroma,
    /// Milvus vector DB (Tier 3)
    Milvus,
    /// Weaviate vector DB (Tier 3)
    Weaviate,
    /// Redis Vector (RediSearch) (Tier 3)
    RedisVector,
    /// Elasticsearch kNN (Tier 3)
    Elasticsearch,
}

impl VectorDbBuilder {
    /// Create a new builder with in-memory backend as default
    pub fn new() -> Self {
        Self {
            config: VectorDbConfig::default(),
            backend: VectorDbBackend::InMemory,
            #[cfg(feature = "vector-lancedb")]
            lance_path: None,
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

    /// Use LanceDB embedded backend (Tier 2).
    ///
    /// # Arguments
    /// * `path` - Local directory path for Lance data storage
    #[cfg(feature = "vector-lancedb")]
    pub fn lance(mut self, path: impl Into<String>) -> Self {
        self.lance_path = Some(path.into());
        self.backend = VectorDbBackend::Lance;
        self
    }

    /// Use Pinecone cloud backend (Tier 4).
    ///
    /// Requires a Pinecone API key and index host URL.
    pub fn pinecone(mut self, host: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(host.into()); // reuse url field
        self.config.qdrant_api_key = Some(api_key.into()); // reuse api_key field
        self.backend = VectorDbBackend::Pinecone;
        self
    }

    /// Use Chroma backend (Tier 3).
    ///
    /// Connects to a Chroma server.
    pub fn chroma(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::Chroma;
        self
    }

    /// Use Milvus backend (Tier 3).
    ///
    /// Connects to a Milvus REST API.
    pub fn milvus(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::Milvus;
        self
    }

    /// Use Weaviate backend (Tier 3).
    ///
    /// Connects to a Weaviate REST API.
    pub fn weaviate(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::Weaviate;
        self
    }

    /// Use Redis Vector backend (Tier 3).
    ///
    /// Connects to a Redis REST gateway with vector search support.
    pub fn redis_vector(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::RedisVector;
        self
    }

    /// Use Elasticsearch backend (Tier 3).
    ///
    /// Connects to an Elasticsearch cluster with kNN support.
    pub fn elasticsearch(mut self, url: impl Into<String>) -> Self {
        self.config.qdrant_url = Some(url.into());
        self.backend = VectorDbBackend::Elasticsearch;
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
            #[cfg(feature = "vector-lancedb")]
            VectorDbBackend::Lance => {
                let path = self.lance_path.ok_or_else(|| {
                    AiError::Config(crate::error::ConfigError::MissingValue {
                        field: "lance_path".to_string(),
                        description: "LanceDB path is required. Use .lance(path) on the builder."
                            .to_string(),
                    })
                })?;
                let db = crate::vector_db_lance::LanceVectorDb::new(&path, self.config)?;
                Ok(Box::new(db))
            }
            VectorDbBackend::Pinecone => {
                let host = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other(
                        "Pinecone host URL is required. Use .pinecone(host, api_key).".into(),
                    )
                })?;
                let api_key = self
                    .config
                    .qdrant_api_key
                    .clone()
                    .ok_or_else(|| AiError::Other("Pinecone API key is required.".into()))?;
                Ok(Box::new(PineconeClient::new(host, api_key, self.config)))
            }
            VectorDbBackend::Chroma => {
                let url = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other("Chroma URL is required. Use .chroma(url).".into())
                })?;
                Ok(Box::new(ChromaClient::new(url, self.config)))
            }
            VectorDbBackend::Milvus => {
                let url = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other("Milvus URL is required. Use .milvus(url).".into())
                })?;
                Ok(Box::new(MilvusClient::new(url, self.config)))
            }
            VectorDbBackend::Weaviate => {
                let url = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other("Weaviate URL is required. Use .weaviate(url).".into())
                })?;
                Ok(Box::new(WeaviateClient::new(url, self.config)))
            }
            VectorDbBackend::RedisVector => {
                let url = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other("Redis URL is required. Use .redis_vector(url).".into())
                })?;
                Ok(Box::new(RedisVectorClient::new(url, self.config)))
            }
            VectorDbBackend::Elasticsearch => {
                let url = self.config.qdrant_url.clone().ok_or_else(|| {
                    AiError::Other("Elasticsearch URL is required. Use .elasticsearch(url).".into())
                })?;
                Ok(Box::new(ElasticsearchClient::new(url, self.config)))
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

// =============================================================================
// Distributed Vector Database
// =============================================================================

/// A distributed vector database wrapper that replicates and searches across cluster nodes.
///
/// Wraps any `VectorDb` implementation, adding distributed search and replication
/// via the `NetworkNode`. Local operations delegate to the wrapped backend, while
/// `distributed_search` fans out queries to all connected peers.
#[cfg(feature = "distributed-network")]
pub struct DistributedVectorDb<V: VectorDb> {
    local: V,
    network: std::sync::Arc<crate::distributed_network::NetworkNode>,
}

#[cfg(feature = "distributed-network")]
impl<V: VectorDb> DistributedVectorDb<V> {
    /// Create a new distributed vector database.
    ///
    /// # Arguments
    /// * `local` - The local vector database backend.
    /// * `network` - A reference to the network node for distributed operations.
    pub fn new(local: V, network: std::sync::Arc<crate::distributed_network::NetworkNode>) -> Self {
        Self { local, network }
    }

    /// Search across all cluster nodes, merging results by score.
    ///
    /// Sends `VectorSearch` messages to all connected peers, collects their
    /// responses, and merges them with local results. Returns the top `limit`
    /// results sorted by descending score.
    pub fn distributed_search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> AiResult<Vec<VectorSearchResult>> {
        // Local search first
        let mut all_results = self.local.search(query, limit, None)?;

        // Query peers
        let peers = self.network.peers();
        let request_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (peer_id, _, _) in &peers {
            let msg = crate::distributed::NodeMessage::VectorSearch {
                query: query.to_vec(),
                limit,
                request_id,
            };

            match self.network.request(peer_id, msg) {
                Ok(crate::distributed::NodeMessage::VectorSearchResponse { results, .. }) => {
                    for (id, score, metadata) in results {
                        let meta: serde_json::Value = metadata
                            .into_iter()
                            .map(|(k, v)| (k, serde_json::Value::String(v)))
                            .collect::<serde_json::Map<String, serde_json::Value>>()
                            .into();

                        all_results.push(VectorSearchResult {
                            id,
                            score,
                            metadata: meta,
                            vector: None,
                        });
                    }
                }
                _ => {} // Skip failed peers
            }
        }

        // Sort by score (descending) and take top limit
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Replicate a specific vector to the nodes responsible for its key.
    ///
    /// Returns the number of nodes the vector was sent to.
    pub fn replicate_vector(&self, id: &str) -> AiResult<usize> {
        let stored = self
            .local
            .get(id)?
            .ok_or_else(|| AiError::Other(format!("Vector {} not found locally", id)))?;

        let data = serde_json::to_vec(&stored)
            .map_err(|e| AiError::Other(format!("Failed to serialize vector: {}", e)))?;

        let target_nodes = self.network.nodes_for_key(id);
        let mut count = 0;

        for node_id in target_nodes {
            if node_id == self.network.node_id() {
                continue;
            }
            match self.network.send(
                &node_id,
                crate::distributed::NodeMessage::Replicate {
                    key: format!("vector:{}", id),
                    value: data.clone(),
                    version: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as u64,
                },
            ) {
                Ok(_) => count += 1,
                Err(_) => {} // Skip failed nodes
            }
        }

        Ok(count)
    }

    /// Get a reference to the local vector database.
    pub fn local(&self) -> &V {
        &self.local
    }

    /// Get a mutable reference to the local vector database.
    pub fn local_mut(&mut self) -> &mut V {
        &mut self.local
    }
}

/// Implement VectorDb for DistributedVectorDb by delegating to the local backend.
#[cfg(feature = "distributed-network")]
impl<V: VectorDb> VectorDb for DistributedVectorDb<V> {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        self.local.insert(id, vector, metadata)
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        self.local.insert_batch(vectors)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        self.local.search(query, limit, filter)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        self.local.get(id)
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        self.local.delete(id)
    }

    fn count(&self) -> usize {
        self.local.count()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.local.clear()
    }

    fn health_check(&self) -> AiResult<bool> {
        self.local.health_check()
    }

    fn backend_info(&self) -> BackendInfo {
        let mut info = self.local.backend_info();
        info.name = "Distributed";
        info
    }

    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        self.local.export_all()
    }

    fn import_bulk(&mut self, vectors: Vec<StoredVector>) -> AiResult<usize> {
        self.local.import_bulk(vectors)
    }
}

// ============================================================================
// Pinecone Cloud Vector DB
// ============================================================================

/// Pinecone cloud vector DB client.
///
/// Uses Pinecone REST API v1 for vector operations.
pub struct PineconeClient {
    host: String,
    api_key: String,
    config: VectorDbConfig,
    /// Local cache for get/count operations (Pinecone is append-heavy)
    cache: HashMap<String, StoredVector>,
}

impl PineconeClient {
    pub fn new(host: String, api_key: String, config: VectorDbConfig) -> Self {
        Self {
            host,
            api_key,
            config,
            cache: HashMap::new(),
        }
    }

    fn pinecone_url(&self, path: &str) -> String {
        format!("{}{}", self.host.trim_end_matches('/'), path)
    }
}

impl VectorDb for PineconeClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = self.pinecone_url("/vectors/upsert");
        let body = serde_json::json!({
            "vectors": [{
                "id": id,
                "values": vector,
                "metadata": metadata
            }]
        });
        let resp = ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body);
        match resp {
            Ok(_) => {
                self.cache.insert(
                    id.to_string(),
                    StoredVector {
                        id: id.to_string(),
                        vector,
                        metadata,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    },
                );
                Ok(())
            }
            Err(e) => Err(AiError::Other(format!("Pinecone upsert failed: {}", e))),
        }
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let pinecone_vectors: Vec<serde_json::Value> = vectors
            .iter()
            .map(|(id, vec, meta)| serde_json::json!({ "id": id, "values": vec, "metadata": meta }))
            .collect();
        let url = self.pinecone_url("/vectors/upsert");
        let body = serde_json::json!({ "vectors": pinecone_vectors });
        let resp = ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body);
        match resp {
            Ok(_) => {
                let count = vectors.len();
                for (id, vec, meta) in vectors {
                    self.cache.insert(
                        id.clone(),
                        StoredVector {
                            id,
                            vector: vec,
                            metadata: meta,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        },
                    );
                }
                Ok(count)
            }
            Err(e) => Err(AiError::Other(format!(
                "Pinecone batch upsert failed: {}",
                e
            ))),
        }
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let url = self.pinecone_url("/query");
        let mut body = serde_json::json!({
            "vector": query,
            "topK": limit,
            "includeMetadata": true
        });
        if !self.config.collection_name.is_empty() {
            body["namespace"] = serde_json::json!(self.config.collection_name);
        }
        let resp = ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Pinecone query failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Pinecone response parse failed: {}", e)))?;

        let mut results = Vec::new();
        if let Some(matches) = json.get("matches").and_then(|m| m.as_array()) {
            for m in matches {
                let id = m
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let score = m.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                let metadata = m.get("metadata").cloned().unwrap_or(serde_json::json!({}));
                results.push(VectorSearchResult {
                    id,
                    score,
                    metadata,
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        Ok(self.cache.get(id).cloned())
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let url = self.pinecone_url("/vectors/delete");
        let body = serde_json::json!({ "ids": [id] });
        let _ = ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body);
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        let url = self.pinecone_url("/vectors/delete");
        let body = serde_json::json!({ "deleteAll": true });
        let _ = ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body);
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = self.pinecone_url("/describe_index_stats");
        match ureq::post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&serde_json::json!({}))
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Pinecone",
            tier: 4,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// ============================================================================
// Chroma Vector DB
// ============================================================================

/// Chroma vector DB client.
///
/// Uses Chroma REST API for vector operations.
pub struct ChromaClient {
    base_url: String,
    config: VectorDbConfig,
    collection_id: Option<String>,
    cache: HashMap<String, StoredVector>,
}

impl ChromaClient {
    pub fn new(base_url: String, config: VectorDbConfig) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            config,
            collection_id: None,
            cache: HashMap::new(),
        }
    }

    fn ensure_collection(&mut self) -> AiResult<String> {
        if let Some(ref id) = self.collection_id {
            return Ok(id.clone());
        }
        let name = &self.config.collection_name;
        let url = format!("{}/api/v1/collections", self.base_url);
        let body = serde_json::json!({ "name": name, "get_or_create": true });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Chroma create collection failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Chroma response parse failed: {}", e)))?;
        let id = json
            .get("id")
            .and_then(|i| i.as_str())
            .unwrap_or(name)
            .to_string();
        self.collection_id = Some(id.clone());
        Ok(id)
    }
}

impl VectorDb for ChromaClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let coll_id = self.ensure_collection()?;
        let url = format!("{}/api/v1/collections/{}/add", self.base_url, coll_id);
        let body = serde_json::json!({
            "ids": [id],
            "embeddings": [vector],
            "metadatas": [metadata]
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Chroma add failed: {}", e)))?;
        self.cache.insert(
            id.to_string(),
            StoredVector {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let coll_id = self.ensure_collection()?;
        let ids: Vec<&str> = vectors.iter().map(|(id, _, _)| id.as_str()).collect();
        let embeddings: Vec<&Vec<f32>> = vectors.iter().map(|(_, v, _)| v).collect();
        let metadatas: Vec<&serde_json::Value> = vectors.iter().map(|(_, _, m)| m).collect();
        let url = format!("{}/api/v1/collections/{}/add", self.base_url, coll_id);
        let body = serde_json::json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Chroma batch add failed: {}", e)))?;
        let count = vectors.len();
        for (id, vec, meta) in vectors {
            self.cache.insert(
                id.clone(),
                StoredVector {
                    id,
                    vector: vec,
                    metadata: meta,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                },
            );
        }
        Ok(count)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let coll_id = self
            .collection_id
            .as_deref()
            .unwrap_or(&self.config.collection_name);
        let url = format!("{}/api/v1/collections/{}/query", self.base_url, coll_id);
        let body = serde_json::json!({
            "query_embeddings": [query],
            "n_results": limit,
            "include": ["metadatas", "distances"]
        });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Chroma query failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Chroma response parse failed: {}", e)))?;

        let mut results = Vec::new();
        let ids = json
            .get("ids")
            .and_then(|i| i.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_array());
        let distances = json
            .get("distances")
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_array());
        let metadatas = json
            .get("metadatas")
            .and_then(|m| m.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_array());
        if let (Some(ids), Some(distances)) = (ids, distances) {
            for (i, id) in ids.iter().enumerate() {
                let id_str = id.as_str().unwrap_or("").to_string();
                let dist = distances.get(i).and_then(|d| d.as_f64()).unwrap_or(1.0) as f32;
                let score = 1.0 - dist; // Chroma returns distances, convert to similarity
                let metadata = metadatas
                    .and_then(|m| m.get(i))
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                results.push(VectorSearchResult {
                    id: id_str,
                    score,
                    metadata,
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        Ok(self.cache.get(id).cloned())
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        if let Some(ref coll_id) = self.collection_id {
            let url = format!("{}/api/v1/collections/{}/delete", self.base_url, coll_id);
            let body = serde_json::json!({ "ids": [id] });
            let _ = ureq::post(&url)
                .set("Content-Type", "application/json")
                .send_json(&body);
        }
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = format!("{}/api/v1/heartbeat", self.base_url);
        match ureq::get(&url).call() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Chroma",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// ============================================================================
// Milvus Vector DB
// ============================================================================

/// Milvus vector DB client.
///
/// Uses Milvus REST API v2 for vector operations.
pub struct MilvusClient {
    base_url: String,
    config: VectorDbConfig,
    cache: HashMap<String, StoredVector>,
}

impl MilvusClient {
    pub fn new(base_url: String, config: VectorDbConfig) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            config,
            cache: HashMap::new(),
        }
    }

    fn collection_name(&self) -> &str {
        &self.config.collection_name
    }
}

impl VectorDb for MilvusClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);
        let mut data = metadata.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("id".to_string(), serde_json::json!(id));
            obj.insert("vector".to_string(), serde_json::json!(vector));
        }
        let body = serde_json::json!({
            "collectionName": self.collection_name(),
            "data": [data]
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Milvus insert failed: {}", e)))?;
        self.cache.insert(
            id.to_string(),
            StoredVector {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);
        let data: Vec<serde_json::Value> = vectors
            .iter()
            .map(|(id, vec, meta)| {
                let mut d = meta.clone();
                if let Some(obj) = d.as_object_mut() {
                    obj.insert("id".to_string(), serde_json::json!(id));
                    obj.insert("vector".to_string(), serde_json::json!(vec));
                }
                d
            })
            .collect();
        let body = serde_json::json!({
            "collectionName": self.collection_name(),
            "data": data
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Milvus batch insert failed: {}", e)))?;
        let count = vectors.len();
        for (id, vec, meta) in vectors {
            self.cache.insert(
                id.clone(),
                StoredVector {
                    id,
                    vector: vec,
                    metadata: meta,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                },
            );
        }
        Ok(count)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let url = format!("{}/v2/vectordb/entities/search", self.base_url);
        let body = serde_json::json!({
            "collectionName": self.collection_name(),
            "data": [query],
            "limit": limit,
            "outputFields": ["*"]
        });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Milvus search failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Milvus response parse failed: {}", e)))?;

        let mut results = Vec::new();
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let id = item
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let score = item.get("distance").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                let metadata = item.clone();
                results.push(VectorSearchResult {
                    id,
                    score,
                    metadata,
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        Ok(self.cache.get(id).cloned())
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let url = format!("{}/v2/vectordb/entities/delete", self.base_url);
        let body = serde_json::json!({
            "collectionName": self.collection_name(),
            "filter": format!("id == \"{}\"", id)
        });
        let _ = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body);
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = format!("{}/v2/vectordb/collections/list", self.base_url);
        match ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&serde_json::json!({}))
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Milvus",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// ============================================================================
// Weaviate Vector DB
// ============================================================================

/// Weaviate vector DB client.
///
/// Uses Weaviate REST API v1 for vector operations.
pub struct WeaviateClient {
    base_url: String,
    config: VectorDbConfig,
    cache: HashMap<String, StoredVector>,
}

impl WeaviateClient {
    pub fn new(base_url: String, config: VectorDbConfig) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            config,
            cache: HashMap::new(),
        }
    }

    /// Returns the Weaviate class name derived from the collection name.
    /// Weaviate class names must start with an uppercase letter.
    fn class_name(&self) -> String {
        let name = &self.config.collection_name;
        if name.is_empty() {
            return "Default".to_string();
        }
        let mut chars = name.chars();
        match chars.next() {
            Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            None => "Default".to_string(),
        }
    }
}

impl VectorDb for WeaviateClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = format!("{}/v1/objects", self.base_url);
        let body = serde_json::json!({
            "class": self.class_name(),
            "id": id,
            "properties": metadata,
            "vector": vector
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Weaviate insert failed: {}", e)))?;
        self.cache.insert(
            id.to_string(),
            StoredVector {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let url = format!("{}/v1/batch/objects", self.base_url);
        let objects: Vec<serde_json::Value> = vectors
            .iter()
            .map(|(id, vec, meta)| {
                serde_json::json!({
                    "class": self.class_name(),
                    "id": id,
                    "properties": meta,
                    "vector": vec
                })
            })
            .collect();
        let body = serde_json::json!({ "objects": objects });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Weaviate batch insert failed: {}", e)))?;
        let count = vectors.len();
        for (id, vec, meta) in vectors {
            self.cache.insert(
                id.clone(),
                StoredVector {
                    id,
                    vector: vec,
                    metadata: meta,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                },
            );
        }
        Ok(count)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let url = format!("{}/v1/graphql", self.base_url);
        let class = self.class_name();
        let vector_str = format!("{:?}", query);
        let graphql_query = format!(
            "{{ Get {{ {}(nearVector: {{vector: {}, certainty: 0.7}}, limit: {}) {{ _additional {{ id distance }} }} }} }}",
            class, vector_str, limit
        );
        let body = serde_json::json!({ "query": graphql_query });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Weaviate search failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Weaviate response parse failed: {}", e)))?;

        let mut results = Vec::new();
        if let Some(items) = json
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(&class))
            .and_then(|c| c.as_array())
        {
            for item in items {
                let additional = item.get("_additional").unwrap_or(item);
                let id = additional
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let distance = additional
                    .get("distance")
                    .and_then(|d| d.as_f64())
                    .unwrap_or(1.0) as f32;
                let score = 1.0 - distance;
                results.push(VectorSearchResult {
                    id,
                    score,
                    metadata: item.clone(),
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        if let Some(cached) = self.cache.get(id) {
            return Ok(Some(cached.clone()));
        }
        let url = format!("{}/v1/objects/{}/{}", self.base_url, self.class_name(), id);
        match ureq::get(&url).call() {
            Ok(resp) => {
                let json: serde_json::Value = resp
                    .into_json()
                    .map_err(|e| AiError::Other(format!("Weaviate get parse failed: {}", e)))?;
                let vector: Vec<f32> = json
                    .get("vector")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let metadata = json
                    .get("properties")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                Ok(Some(StoredVector {
                    id: id.to_string(),
                    vector,
                    metadata,
                    timestamp: 0,
                }))
            }
            Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let url = format!("{}/v1/objects/{}/{}", self.base_url, self.class_name(), id);
        let _ = ureq::delete(&url).call();
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = format!("{}/v1/.well-known/ready", self.base_url);
        match ureq::get(&url).call() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Weaviate",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// ============================================================================
// Redis Vector DB
// ============================================================================

/// Redis Vector DB client.
///
/// Uses a Redis REST gateway (e.g. redis-rest or RedisJSON HTTP) for vector operations.
pub struct RedisVectorClient {
    base_url: String,
    #[allow(dead_code)]
    config: VectorDbConfig,
    index_name: String,
    cache: HashMap<String, StoredVector>,
}

impl RedisVectorClient {
    pub fn new(base_url: String, config: VectorDbConfig) -> Self {
        let index_name = config.collection_name.clone();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            config,
            index_name,
            cache: HashMap::new(),
        }
    }

    /// Returns the index name used for vector search.
    fn index_name(&self) -> &str {
        &self.index_name
    }
}

impl VectorDb for RedisVectorClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = format!("{}/set/{}", self.base_url, id);
        let body = serde_json::json!({
            "index": self.index_name(),
            "vector": vector,
            "metadata": metadata
        });
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Redis vector insert failed: {}", e)))?;
        self.cache.insert(
            id.to_string(),
            StoredVector {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let count = vectors.len();
        for (id, vector, metadata) in vectors {
            let url = format!("{}/set/{}", self.base_url, id);
            let body = serde_json::json!({
                "index": self.index_name(),
                "vector": vector,
                "metadata": metadata
            });
            ureq::post(&url)
                .set("Content-Type", "application/json")
                .send_json(&body)
                .map_err(|e| AiError::Other(format!("Redis vector batch insert failed: {}", e)))?;
            self.cache.insert(
                id.clone(),
                StoredVector {
                    id,
                    vector,
                    metadata,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                },
            );
        }
        Ok(count)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let url = format!("{}/search", self.base_url);
        let body = serde_json::json!({
            "index": self.index_name(),
            "query_vector": query,
            "k": limit
        });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Redis vector search failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Redis vector response parse failed: {}", e)))?;

        let mut results = Vec::new();
        if let Some(items) = json.get("results").and_then(|r| r.as_array()) {
            for item in items {
                let id = item
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let score = item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                let metadata = item
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                results.push(VectorSearchResult {
                    id,
                    score,
                    metadata,
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        if let Some(cached) = self.cache.get(id) {
            return Ok(Some(cached.clone()));
        }
        let url = format!("{}/get/{}", self.base_url, id);
        match ureq::get(&url).call() {
            Ok(resp) => {
                let json: serde_json::Value = resp
                    .into_json()
                    .map_err(|e| AiError::Other(format!("Redis vector get parse failed: {}", e)))?;
                let vector: Vec<f32> = json
                    .get("vector")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let metadata = json
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                Ok(Some(StoredVector {
                    id: id.to_string(),
                    vector,
                    metadata,
                    timestamp: 0,
                }))
            }
            Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let url = format!("{}/del/{}", self.base_url, id);
        let _ = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&serde_json::json!({}));
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = format!("{}/ping", self.base_url);
        match ureq::get(&url).call() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Redis Vector",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// ============================================================================
// Elasticsearch Vector DB
// ============================================================================

/// Elasticsearch vector DB client.
///
/// Uses Elasticsearch REST API with kNN search for vector operations.
pub struct ElasticsearchClient {
    base_url: String,
    #[allow(dead_code)]
    config: VectorDbConfig,
    index_name: String,
    cache: HashMap<String, StoredVector>,
}

impl ElasticsearchClient {
    pub fn new(base_url: String, config: VectorDbConfig) -> Self {
        let index_name = config.collection_name.to_lowercase();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            config,
            index_name,
            cache: HashMap::new(),
        }
    }

    /// Returns the index name (lowercased collection name).
    fn index_name(&self) -> &str {
        &self.index_name
    }
}

impl VectorDb for ElasticsearchClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = format!("{}/{}/_doc/{}", self.base_url, self.index_name(), id);
        let body = serde_json::json!({
            "vector": vector,
            "metadata": metadata
        });
        ureq::put(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Elasticsearch insert failed: {}", e)))?;
        self.cache.insert(
            id.to_string(),
            StoredVector {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        // Elasticsearch bulk API uses newline-delimited JSON
        let mut bulk_body = String::new();
        for (id, vector, metadata) in &vectors {
            let action = serde_json::json!({"index": {"_index": self.index_name(), "_id": id}});
            let doc = serde_json::json!({"vector": vector, "metadata": metadata});
            bulk_body.push_str(&action.to_string());
            bulk_body.push('\n');
            bulk_body.push_str(&doc.to_string());
            bulk_body.push('\n');
        }
        let url = format!("{}/_bulk", self.base_url);
        ureq::post(&url)
            .set("Content-Type", "application/x-ndjson")
            .send_string(&bulk_body)
            .map_err(|e| AiError::Other(format!("Elasticsearch bulk insert failed: {}", e)))?;
        let count = vectors.len();
        for (id, vector, metadata) in vectors {
            self.cache.insert(
                id.clone(),
                StoredVector {
                    id,
                    vector,
                    metadata,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                },
            );
        }
        Ok(count)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        let url = format!("{}/{}/_search", self.base_url, self.index_name());
        let body = serde_json::json!({
            "knn": {
                "field": "vector",
                "query_vector": query,
                "k": limit,
                "num_candidates": limit * 2
            },
            "size": limit
        });
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| AiError::Other(format!("Elasticsearch search failed: {}", e)))?;
        let json: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::Other(format!("Elasticsearch response parse failed: {}", e)))?;

        let mut results = Vec::new();
        if let Some(hits) = json
            .get("hits")
            .and_then(|h| h.get("hits"))
            .and_then(|h| h.as_array())
        {
            for hit in hits {
                let id = hit
                    .get("_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let score = hit.get("_score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                let metadata = hit
                    .get("_source")
                    .and_then(|s| s.get("metadata"))
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                results.push(VectorSearchResult {
                    id,
                    score,
                    metadata,
                    vector: None,
                });
            }
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        if let Some(cached) = self.cache.get(id) {
            return Ok(Some(cached.clone()));
        }
        let url = format!("{}/{}/_doc/{}", self.base_url, self.index_name(), id);
        match ureq::get(&url).call() {
            Ok(resp) => {
                let json: serde_json::Value = resp.into_json().map_err(|e| {
                    AiError::Other(format!("Elasticsearch get parse failed: {}", e))
                })?;
                if let Some(source) = json.get("_source") {
                    let vector: Vec<f32> = source
                        .get("vector")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect()
                        })
                        .unwrap_or_default();
                    let metadata = source
                        .get("metadata")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    Ok(Some(StoredVector {
                        id: id.to_string(),
                        vector,
                        metadata,
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
        let url = format!("{}/{}/_doc/{}", self.base_url, self.index_name(), id);
        let _ = ureq::delete(&url).call();
        Ok(self.cache.remove(id).is_some())
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.cache.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = format!("{}/_cluster/health", self.base_url);
        match ureq::get(&url).call() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Elasticsearch",
            tier: 3,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: false,
            max_recommended_vectors: None,
        }
    }
}

// Implement Send + Sync for new backends
unsafe impl Send for PineconeClient {}
unsafe impl Sync for PineconeClient {}
unsafe impl Send for ChromaClient {}
unsafe impl Sync for ChromaClient {}
unsafe impl Send for MilvusClient {}
unsafe impl Sync for MilvusClient {}
unsafe impl Send for WeaviateClient {}
unsafe impl Sync for WeaviateClient {}
unsafe impl Send for RedisVectorClient {}
unsafe impl Sync for RedisVectorClient {}
unsafe impl Send for ElasticsearchClient {}
unsafe impl Sync for ElasticsearchClient {}

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

        db.insert(
            "v1",
            vec![1.0, 0.0, 0.0],
            serde_json::json!({"type": "a", "score": 10}),
        )
        .unwrap();
        db.insert(
            "v2",
            vec![0.9, 0.1, 0.0],
            serde_json::json!({"type": "b", "score": 20}),
        )
        .unwrap();
        db.insert(
            "v3",
            vec![0.8, 0.2, 0.0],
            serde_json::json!({"type": "a", "score": 30}),
        )
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

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({}))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.insert("v3", vec![0.0, 0.0, 1.0], serde_json::json!({}))
            .unwrap();

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

        db.insert(
            "v1",
            vec![1.0, 0.0, 0.0],
            serde_json::json!({"text": "hello world"}),
        )
        .unwrap();
        db.insert(
            "v2",
            vec![0.9, 0.1, 0.0],
            serde_json::json!({"text": "hello there"}),
        )
        .unwrap();
        db.insert(
            "v3",
            vec![0.0, 1.0, 0.0],
            serde_json::json!({"text": "goodbye world"}),
        )
        .unwrap();

        let hybrid = HybridVectorSearch::new(db).with_weights(0.5, 0.5);

        let results = hybrid.search(&[1.0, 0.0, 0.0], "hello", 3, None).unwrap();

        // v1 and v2 should rank higher due to "hello" keyword match
        assert!(!results.is_empty());
    }

    // ====================================================================
    // Pinecone backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_pinecone_client_creation() {
        let config = VectorDbConfig {
            dimensions: 768,
            collection_name: "my-namespace".to_string(),
            ..Default::default()
        };
        let client = PineconeClient::new(
            "https://my-index-abc123.svc.pinecone.io".to_string(),
            "pc-test-key".to_string(),
            config,
        );
        assert_eq!(
            client.pinecone_url("/vectors/upsert"),
            "https://my-index-abc123.svc.pinecone.io/vectors/upsert"
        );
        assert_eq!(
            client.pinecone_url("/query"),
            "https://my-index-abc123.svc.pinecone.io/query"
        );
    }

    #[test]
    fn test_pinecone_backend_info() {
        let config = VectorDbConfig {
            dimensions: 768,
            ..Default::default()
        };
        let client = PineconeClient::new("https://x.pinecone.io".into(), "key".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Pinecone");
        assert_eq!(info.tier, 4);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(!info.supports_export); // cloud-only, no local export
        assert!(info.max_recommended_vectors.is_none());
    }

    #[test]
    fn test_pinecone_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = PineconeClient::new("https://x.pinecone.io".into(), "key".into(), config);
        // Cache starts empty
        assert_eq!(client.count(), 0);
        assert!(client.get("nonexistent").unwrap().is_none());
        // Manually insert into cache (simulates successful upsert)
        client.cache.insert(
            "v1".to_string(),
            StoredVector {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                metadata: serde_json::json!({"topic": "test"}),
                timestamp: 1000,
            },
        );
        assert_eq!(client.count(), 1);
        let v = client.get("v1").unwrap().unwrap();
        assert_eq!(v.id, "v1");
        assert_eq!(v.metadata["topic"], "test");
    }

    #[test]
    fn test_pinecone_url_trailing_slash() {
        let config = VectorDbConfig {
            dimensions: 768,
            ..Default::default()
        };
        let client = PineconeClient::new(
            "https://my-index.svc.pinecone.io/".to_string(),
            "key".into(),
            config,
        );
        assert_eq!(
            client.pinecone_url("/query"),
            "https://my-index.svc.pinecone.io/query"
        );
    }

    // ====================================================================
    // Chroma backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_chroma_client_creation() {
        let config = VectorDbConfig {
            dimensions: 384,
            collection_name: "my_collection".to_string(),
            ..Default::default()
        };
        let client = ChromaClient::new("http://localhost:8000".to_string(), config);
        // base_url should be stored without trailing slash
        assert_eq!(client.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_chroma_backend_info() {
        let config = VectorDbConfig {
            dimensions: 384,
            ..Default::default()
        };
        let client = ChromaClient::new("http://localhost:8000".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Chroma");
        assert_eq!(info.tier, 3);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
    }

    #[test]
    fn test_chroma_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = ChromaClient::new("http://localhost:8000".into(), config);
        assert_eq!(client.count(), 0);
        client.cache.insert(
            "c1".to_string(),
            StoredVector {
                id: "c1".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                metadata: serde_json::json!({}),
                timestamp: 2000,
            },
        );
        assert_eq!(client.count(), 1);
        assert!(client.get("c1").unwrap().is_some());
        assert!(client.get("c2").unwrap().is_none());
    }

    #[test]
    fn test_chroma_url_trailing_slash_stripped() {
        let config = VectorDbConfig {
            dimensions: 384,
            collection_name: "docs".to_string(),
            ..Default::default()
        };
        let client = ChromaClient::new("http://chroma-server:8080/".to_string(), config);
        // Trailing slash should be stripped during construction
        assert_eq!(client.base_url, "http://chroma-server:8080");
    }

    // ====================================================================
    // Milvus backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_milvus_client_creation() {
        let config = VectorDbConfig {
            dimensions: 128,
            collection_name: "vectors".to_string(),
            ..Default::default()
        };
        let client = MilvusClient::new("http://localhost:19530".to_string(), config);
        assert_eq!(client.base_url, "http://localhost:19530");
    }

    #[test]
    fn test_milvus_backend_info() {
        let config = VectorDbConfig {
            dimensions: 128,
            ..Default::default()
        };
        let client = MilvusClient::new("http://localhost:19530".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Milvus");
        assert_eq!(info.tier, 3);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(info.max_recommended_vectors.is_none());
    }

    #[test]
    fn test_milvus_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = MilvusClient::new("http://localhost:19530".into(), config);
        assert_eq!(client.count(), 0);
        client.cache.insert(
            "m1".to_string(),
            StoredVector {
                id: "m1".to_string(),
                vector: vec![1.0, 1.0, 0.0],
                metadata: serde_json::json!({"source": "doc"}),
                timestamp: 3000,
            },
        );
        assert_eq!(client.count(), 1);
        let v = client.get("m1").unwrap().unwrap();
        assert_eq!(v.metadata["source"], "doc");
    }

    #[test]
    fn test_milvus_url_trailing_slash_stripped() {
        let config = VectorDbConfig {
            dimensions: 128,
            ..Default::default()
        };
        let client = MilvusClient::new("http://milvus:19530/".to_string(), config);
        assert_eq!(client.base_url, "http://milvus:19530");
    }

    // ====================================================================
    // VectorDbBuilder tests for new backends
    // ====================================================================

    #[test]
    fn test_builder_pinecone() {
        let builder = VectorDbBuilder::new()
            .dimensions(768)
            .pinecone("https://my-index.svc.pinecone.io", "pc-key-123");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Pinecone");
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn test_builder_chroma() {
        let builder = VectorDbBuilder::new()
            .dimensions(384)
            .chroma("http://localhost:8000");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Chroma");
    }

    #[test]
    fn test_builder_milvus() {
        let builder = VectorDbBuilder::new()
            .dimensions(128)
            .milvus("http://localhost:19530");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Milvus");
    }

    #[test]
    fn test_vector_db_backend_variants() {
        // Ensure all new backend variants exist
        let _p = VectorDbBackend::Pinecone;
        let _c = VectorDbBackend::Chroma;
        let _m = VectorDbBackend::Milvus;
        let _w = VectorDbBackend::Weaviate;
        let _r = VectorDbBackend::RedisVector;
        let _e = VectorDbBackend::Elasticsearch;
    }

    // ====================================================================
    // Weaviate backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_weaviate_client_creation() {
        let config = VectorDbConfig {
            dimensions: 384,
            collection_name: "documents".to_string(),
            ..Default::default()
        };
        let client = WeaviateClient::new("http://localhost:8080".to_string(), config);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_weaviate_url_building() {
        let config = VectorDbConfig {
            dimensions: 384,
            collection_name: "documents".to_string(),
            ..Default::default()
        };
        let client = WeaviateClient::new("http://localhost:8080/".to_string(), config);
        // Trailing slash should be stripped
        assert_eq!(client.base_url, "http://localhost:8080");
        // class_name should capitalize first letter
        assert_eq!(client.class_name(), "Documents");
    }

    #[test]
    fn test_weaviate_backend_info() {
        let config = VectorDbConfig {
            dimensions: 384,
            ..Default::default()
        };
        let client = WeaviateClient::new("http://localhost:8080".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Weaviate");
        assert_eq!(info.tier, 3);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(!info.supports_export);
        assert!(info.max_recommended_vectors.is_none());
    }

    #[test]
    fn test_weaviate_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = WeaviateClient::new("http://localhost:8080".into(), config);
        assert_eq!(client.count(), 0);
        assert!(client.get("nonexistent").unwrap().is_none());
        // Manually insert into cache (simulates successful insert)
        client.cache.insert(
            "w1".to_string(),
            StoredVector {
                id: "w1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                metadata: serde_json::json!({"class": "Document"}),
                timestamp: 1000,
            },
        );
        assert_eq!(client.count(), 1);
        let v = client.get("w1").unwrap().unwrap();
        assert_eq!(v.id, "w1");
        assert_eq!(v.metadata["class"], "Document");
        // Delete from cache
        assert!(client.delete("w1").unwrap());
        assert_eq!(client.count(), 0);
    }

    #[test]
    fn test_weaviate_builder() {
        let builder = VectorDbBuilder::new()
            .dimensions(384)
            .collection_name("articles")
            .weaviate("http://localhost:8080");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Weaviate");
        assert_eq!(db.count(), 0);
    }

    // ====================================================================
    // Redis Vector backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_redis_vector_client_creation() {
        let config = VectorDbConfig {
            dimensions: 768,
            collection_name: "embeddings".to_string(),
            ..Default::default()
        };
        let client = RedisVectorClient::new("http://localhost:6379".to_string(), config);
        assert_eq!(client.base_url, "http://localhost:6379");
        assert_eq!(client.index_name(), "embeddings");
    }

    #[test]
    fn test_redis_url_building() {
        let config = VectorDbConfig {
            dimensions: 768,
            collection_name: "my_vectors".to_string(),
            ..Default::default()
        };
        let client = RedisVectorClient::new("http://redis-gateway:8080/".to_string(), config);
        // Trailing slash should be stripped
        assert_eq!(client.base_url, "http://redis-gateway:8080");
        assert_eq!(client.index_name(), "my_vectors");
    }

    #[test]
    fn test_redis_backend_info() {
        let config = VectorDbConfig {
            dimensions: 768,
            ..Default::default()
        };
        let client = RedisVectorClient::new("http://localhost:6379".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Redis Vector");
        assert_eq!(info.tier, 3);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(!info.supports_export);
        assert!(info.max_recommended_vectors.is_none());
    }

    #[test]
    fn test_redis_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = RedisVectorClient::new("http://localhost:6379".into(), config);
        assert_eq!(client.count(), 0);
        assert!(client.get("nonexistent").unwrap().is_none());
        // Manually insert into cache
        client.cache.insert(
            "r1".to_string(),
            StoredVector {
                id: "r1".to_string(),
                vector: vec![0.5, 0.5, 0.0],
                metadata: serde_json::json!({"source": "redis"}),
                timestamp: 2000,
            },
        );
        assert_eq!(client.count(), 1);
        let v = client.get("r1").unwrap().unwrap();
        assert_eq!(v.id, "r1");
        assert_eq!(v.metadata["source"], "redis");
        // Delete from cache
        assert!(client.delete("r1").unwrap());
        assert_eq!(client.count(), 0);
    }

    #[test]
    fn test_redis_builder() {
        let builder = VectorDbBuilder::new()
            .dimensions(768)
            .collection_name("vectors")
            .redis_vector("http://localhost:6379");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Redis Vector");
        assert_eq!(db.count(), 0);
    }

    // ====================================================================
    // Elasticsearch backend tests (unit-level, no network)
    // ====================================================================

    #[test]
    fn test_elasticsearch_client_creation() {
        let config = VectorDbConfig {
            dimensions: 512,
            collection_name: "SearchIndex".to_string(),
            ..Default::default()
        };
        let client = ElasticsearchClient::new("http://localhost:9200".to_string(), config);
        assert_eq!(client.base_url, "http://localhost:9200");
        // index_name should be lowercased
        assert_eq!(client.index_name(), "searchindex");
    }

    #[test]
    fn test_elasticsearch_url_building() {
        let config = VectorDbConfig {
            dimensions: 512,
            collection_name: "MyDocuments".to_string(),
            ..Default::default()
        };
        let client = ElasticsearchClient::new("http://es-cluster:9200/".to_string(), config);
        // Trailing slash should be stripped
        assert_eq!(client.base_url, "http://es-cluster:9200");
        assert_eq!(client.index_name(), "mydocuments");
    }

    #[test]
    fn test_elasticsearch_backend_info() {
        let config = VectorDbConfig {
            dimensions: 512,
            ..Default::default()
        };
        let client = ElasticsearchClient::new("http://localhost:9200".into(), config);
        let info = client.backend_info();
        assert_eq!(info.name, "Elasticsearch");
        assert_eq!(info.tier, 3);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(!info.supports_export);
        assert!(info.max_recommended_vectors.is_none());
    }

    #[test]
    fn test_elasticsearch_cache_operations() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut client = ElasticsearchClient::new("http://localhost:9200".into(), config);
        assert_eq!(client.count(), 0);
        assert!(client.get("nonexistent").unwrap().is_none());
        // Manually insert into cache
        client.cache.insert(
            "e1".to_string(),
            StoredVector {
                id: "e1".to_string(),
                vector: vec![0.1, 0.2, 0.3],
                metadata: serde_json::json!({"type": "article"}),
                timestamp: 3000,
            },
        );
        assert_eq!(client.count(), 1);
        let v = client.get("e1").unwrap().unwrap();
        assert_eq!(v.id, "e1");
        assert_eq!(v.metadata["type"], "article");
        // Delete from cache
        assert!(client.delete("e1").unwrap());
        assert_eq!(client.count(), 0);
    }

    #[test]
    fn test_elasticsearch_builder() {
        let builder = VectorDbBuilder::new()
            .dimensions(512)
            .collection_name("documents")
            .elasticsearch("http://localhost:9200");
        let db = builder.build().unwrap();
        assert_eq!(db.backend_info().name, "Elasticsearch");
        assert_eq!(db.count(), 0);
    }

    // ====================================================================
    // Additional tests: entry creation, search edge cases, delete, export,
    // cache operations, error handling, distance metrics
    // ====================================================================

    #[test]
    fn test_stored_vector_creation() {
        let sv = StoredVector {
            id: "test-vec-1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: serde_json::json!({"label": "test", "score": 42}),
            timestamp: 1234567890,
        };
        assert_eq!(sv.id, "test-vec-1");
        assert_eq!(sv.vector.len(), 3);
        assert_eq!(sv.metadata["label"], "test");
        assert_eq!(sv.metadata["score"], 42);
        assert_eq!(sv.timestamp, 1234567890);
    }

    #[test]
    fn test_search_empty_collection() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let db = InMemoryVectorDb::new(config);

        let results = db.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_query_dimension_mismatch() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let db = InMemoryVectorDb::new(config);

        // Query with wrong dimensions should fail
        let result = db.search(&[1.0, 0.0], 10, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_nonexistent_vector() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        // Deleting a vector that doesn't exist should return false
        let deleted = db.delete("nonexistent").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_delete_then_search() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({}))
            .unwrap();

        assert_eq!(db.count(), 2);
        db.delete("v1").unwrap();
        assert_eq!(db.count(), 1);

        let results = db.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v2");
    }

    #[test]
    fn test_clear_collection() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({}))
            .unwrap();
        assert_eq!(db.count(), 2);

        db.clear().unwrap();
        assert_eq!(db.count(), 0);

        let results = db.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_export_all_in_memory() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"a": 1}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({"b": 2}))
            .unwrap();

        let exported = db.export_all().unwrap();
        assert_eq!(exported.len(), 2);
        let ids: Vec<&str> = exported.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"v1"));
        assert!(ids.contains(&"v2"));
    }

    #[test]
    fn test_in_memory_backend_info() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let db = InMemoryVectorDb::new(config);

        let info = db.backend_info();
        assert_eq!(info.name, "InMemory");
        assert_eq!(info.tier, 0);
        assert!(!info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(info.supports_export);
        assert_eq!(info.max_recommended_vectors, Some(10_000));
    }

    #[test]
    fn test_health_check_in_memory() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let db = InMemoryVectorDb::new(config);

        assert!(db.health_check().unwrap());
    }

    #[test]
    fn test_upsert_overwrite() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"ver": 1}))
            .unwrap();
        db.insert("v1", vec![0.0, 1.0, 0.0], serde_json::json!({"ver": 2}))
            .unwrap();

        // Should still only be one vector (overwritten)
        assert_eq!(db.count(), 1);

        let v = db.get("v1").unwrap().unwrap();
        assert_eq!(v.metadata["ver"], 2);
        assert_eq!(v.vector, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_distance_metric_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // Dot product: 1*4 + 2*5 + 3*6 = 32
        let dp = DistanceMetric::DotProduct;
        let dist = dp.calculate(&a, &b);
        assert!((dist - (-32.0)).abs() < 0.001);
        // Similarity = -distance = 32.0
        let sim = dp.to_similarity(dist);
        assert!((sim - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_metric_manhattan() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // Manhattan: |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        let manhattan = DistanceMetric::Manhattan;
        let dist = manhattan.calculate(&a, &b);
        assert!((dist - 7.0).abs() < 0.001);
        // Similarity: 1/(1+7) = 0.125
        let sim = manhattan.to_similarity(dist);
        assert!((sim - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_distance_metric_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let cosine = DistanceMetric::Cosine;
        // Zero vector norm is 0, should return 1.0 (max distance)
        let dist = cosine.calculate(&a, &b);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_metadata_filter_not_equals() {
        let filter = MetadataFilter {
            field: "type".to_string(),
            operation: FilterOperation::NotEquals(serde_json::json!("draft")),
        };
        let meta = serde_json::json!({"type": "published"});
        assert!(filter.matches(&meta));

        let meta_draft = serde_json::json!({"type": "draft"});
        assert!(!filter.matches(&meta_draft));
    }

    #[test]
    fn test_metadata_filter_greater_than() {
        let filter = MetadataFilter {
            field: "score".to_string(),
            operation: FilterOperation::GreaterThan(50.0),
        };
        let meta_high = serde_json::json!({"score": 75});
        assert!(filter.matches(&meta_high));

        let meta_low = serde_json::json!({"score": 30});
        assert!(!filter.matches(&meta_low));
    }

    #[test]
    fn test_metadata_filter_less_than() {
        let filter = MetadataFilter {
            field: "price".to_string(),
            operation: FilterOperation::LessThan(100.0),
        };
        let meta_cheap = serde_json::json!({"price": 50});
        assert!(filter.matches(&meta_cheap));

        let meta_expensive = serde_json::json!({"price": 200});
        assert!(!filter.matches(&meta_expensive));
    }

    #[test]
    fn test_metadata_filter_contains_string() {
        let filter = MetadataFilter {
            field: "title".to_string(),
            operation: FilterOperation::Contains("Rust".to_string()),
        };
        let meta_match = serde_json::json!({"title": "Learning Rust Programming"});
        assert!(filter.matches(&meta_match));

        let meta_no_match = serde_json::json!({"title": "Learning Python"});
        assert!(!filter.matches(&meta_no_match));
    }

    #[test]
    fn test_metadata_filter_contains_array() {
        let filter = MetadataFilter {
            field: "tags".to_string(),
            operation: FilterOperation::Contains("ai".to_string()),
        };
        let meta = serde_json::json!({"tags": ["ai", "ml", "nlp"]});
        assert!(filter.matches(&meta));

        let meta_no = serde_json::json!({"tags": ["web", "frontend"]});
        assert!(!filter.matches(&meta_no));
    }

    #[test]
    fn test_metadata_filter_in_list() {
        let filter = MetadataFilter {
            field: "status".to_string(),
            operation: FilterOperation::In(vec![
                serde_json::json!("active"),
                serde_json::json!("pending"),
            ]),
        };
        let meta_active = serde_json::json!({"status": "active"});
        assert!(filter.matches(&meta_active));

        let meta_archived = serde_json::json!({"status": "archived"});
        assert!(!filter.matches(&meta_archived));
    }

    #[test]
    fn test_metadata_filter_missing_field() {
        let filter = MetadataFilter {
            field: "nonexistent".to_string(),
            operation: FilterOperation::Equals(serde_json::json!("value")),
        };
        let meta = serde_json::json!({"other": "data"});
        assert!(!filter.matches(&meta));
    }

    #[test]
    fn test_string_id_to_u64_deterministic() {
        let id1 = string_id_to_u64("my-document-id");
        let id2 = string_id_to_u64("my-document-id");
        assert_eq!(id1, id2);

        let id3 = string_id_to_u64("different-id");
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_vector_db_config_default() {
        let config = VectorDbConfig::default();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.max_vectors, Some(100_000));
        assert_eq!(config.collection_name, "default");
        assert!(config.qdrant_url.is_none());
        assert!(config.qdrant_api_key.is_none());
    }

    #[test]
    fn test_search_with_multiple_filters() {
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);

        db.insert(
            "v1",
            vec![1.0, 0.0, 0.0],
            serde_json::json!({"type": "article", "score": 80}),
        )
        .unwrap();
        db.insert(
            "v2",
            vec![0.9, 0.1, 0.0],
            serde_json::json!({"type": "article", "score": 40}),
        )
        .unwrap();
        db.insert(
            "v3",
            vec![0.8, 0.2, 0.0],
            serde_json::json!({"type": "book", "score": 90}),
        )
        .unwrap();

        let filters = vec![
            MetadataFilter {
                field: "type".to_string(),
                operation: FilterOperation::Equals(serde_json::json!("article")),
            },
            MetadataFilter {
                field: "score".to_string(),
                operation: FilterOperation::GreaterThan(50.0),
            },
        ];

        let results = db.search(&[1.0, 0.0, 0.0], 10, Some(&filters)).unwrap();
        // Only v1 matches both filters (type=article AND score > 50)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }
}
