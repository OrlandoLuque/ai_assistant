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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

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
    Ok(VectorMigrationResult { exported: count, imported })
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

    /// Get information about this backend (tier, capabilities, etc.)
    fn backend_info(&self) -> BackendInfo {
        BackendInfo::default()
    }

    /// Export all stored vectors (for migration between backends).
    /// Not all backends support this — check `backend_info().supports_export`.
    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        Err(AiError::Other("export_all not supported by this backend".into()))
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

    fn insert_batch(&mut self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> AiResult<usize> {
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
        let numeric_id = string_id_to_u64(id);
        let endpoint = format!(
            "/collections/{}/points/{}",
            self.config.collection_name, numeric_id
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

            let parsed: serde_json::Value = serde_json::from_str(&response)
                .map_err(|e| AiError::Serialization(crate::error::SerializationError::json_deserialize(e.to_string())))?;

            let points = match parsed["result"]["points"].as_array() {
                Some(pts) => pts,
                None => break,
            };

            if points.is_empty() {
                break;
            }

            for point in points {
                let id = point["id"].as_str()
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
                        description: "LanceDB path is required. Use .lance(path) on the builder.".to_string(),
                    })
                })?;
                let db = crate::vector_db_lance::LanceVectorDb::new(&path, self.config)?;
                Ok(Box::new(db))
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
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Replicate a specific vector to the nodes responsible for its key.
    ///
    /// Returns the number of nodes the vector was sent to.
    pub fn replicate_vector(&self, id: &str) -> AiResult<usize> {
        let stored = self.local.get(id)?
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

    fn insert_batch(&mut self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> AiResult<usize> {
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
