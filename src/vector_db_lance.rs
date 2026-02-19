//! LanceDB vector database backend (Tier 2 — embedded, persistent)
//!
//! This module provides an embedded vector database using LanceDB, which stores data
//! on local disk in Lance columnar format. It supports ANN (Approximate Nearest Neighbor)
//! search via IVF-PQ indexes and scales to hundreds of millions of vectors.
//!
//! LanceDB is the recommended upgrade from in-memory storage for datasets
//! between 10K and 10M vectors.
//!
//! # Requirements
//!
//! Enable the `vector-lancedb` feature flag:
//! ```toml
//! ai_assistant = { features = ["vector-lancedb"] }
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::vector_db::{VectorDb, VectorDbConfig};
//! use ai_assistant::vector_db_lance::LanceVectorDb;
//!
//! let config = VectorDbConfig {
//!     dimensions: 384,
//!     collection_name: "my_vectors".to_string(),
//!     ..Default::default()
//! };
//! let mut db = LanceVectorDb::new("./data/lance", config).unwrap();
//! db.insert("doc1", vec![0.1; 384], serde_json::json!({"title": "Hello"})).unwrap();
//! let results = db.search(&vec![0.1; 384], 5, None).unwrap();
//! ```

use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::runtime::Runtime;

use crate::error::{AiError, AiResult};
use crate::vector_db::{
    BackendInfo, MetadataFilter, StoredVector, VectorDb, VectorDbConfig, VectorSearchResult,
};

/// LanceDB embedded vector database backend.
///
/// Wraps the async LanceDB API in synchronous calls using a dedicated tokio runtime.
/// Data is stored in Lance columnar format on local disk at the configured path.
pub struct LanceVectorDb {
    /// LanceDB database connection
    db: lancedb::Connection,
    /// Active table handle (None if not yet created)
    table: Option<lancedb::Table>,
    /// Vector DB configuration
    config: VectorDbConfig,
    /// Arrow schema for the table
    schema: Arc<Schema>,
    /// Tokio runtime for async -> sync bridge
    rt: Runtime,
    /// Table name (from config.collection_name)
    table_name: String,
}

impl std::fmt::Debug for LanceVectorDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceVectorDb")
            .field("table_name", &self.table_name)
            .field("config", &self.config)
            .finish()
    }
}

/// Build the Arrow schema for the vectors table.
///
/// Schema: id (Utf8), vector (FixedSizeList<Float32>), metadata (Utf8/JSON), timestamp (UInt64)
fn build_schema(dimensions: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions as i32,
            ),
            true,
        ),
        Field::new("metadata", DataType::Utf8, true),
        Field::new("timestamp", DataType::UInt64, false),
    ]))
}

/// Build an Arrow RecordBatch from vectors data.
///
/// Each parameter is a parallel Vec — all must have the same length.
fn build_record_batch(
    schema: &Arc<Schema>,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    metadatas: Vec<String>,
    timestamps: Vec<u64>,
    dim: usize,
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let id_array = Arc::new(StringArray::from(ids));
    let metadata_array = Arc::new(StringArray::from(metadatas));
    let timestamp_array = Arc::new(UInt64Array::from(timestamps));

    let vector_array = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .into_iter()
            .map(|v| Some(v.into_iter().map(Some).collect::<Vec<_>>())),
        dim as i32,
    ));

    RecordBatch::try_new(
        schema.clone(),
        vec![id_array, vector_array, metadata_array, timestamp_array],
    )
}

/// Wrap a RecordBatch into a RecordBatchIterator that implements IntoArrow.
fn into_arrow(batch: RecordBatch, schema: Arc<Schema>) -> Box<RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>> {
    Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema,
    ))
}

/// Get current timestamp in milliseconds since UNIX epoch.
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Extract StoredVector records from Arrow RecordBatch results.
fn extract_stored_vectors(batches: &[RecordBatch]) -> Vec<StoredVector> {
    let mut results = Vec::new();
    for batch in batches {
        let ids = match batch.column_by_name("id") {
            Some(col) => match col.as_any().downcast_ref::<StringArray>() {
                Some(arr) => arr,
                None => continue,
            },
            None => continue,
        };
        let metadatas = batch
            .column_by_name("metadata")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
        let vectors = batch
            .column_by_name("vector")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

        for i in 0..batch.num_rows() {
            let id = ids.value(i).to_string();
            let metadata_str = metadatas.map(|m| m.value(i)).unwrap_or("{}");
            let metadata: serde_json::Value =
                serde_json::from_str(metadata_str).unwrap_or(serde_json::Value::Null);
            let timestamp = timestamps.map(|t| t.value(i)).unwrap_or(0);
            let vector = vectors
                .map(|vecs| {
                    let arr = vecs.value(i);
                    let floats = arr.as_any().downcast_ref::<Float32Array>().expect("Arrow array must be Float32Array");
                    (0..floats.len()).map(|j| floats.value(j)).collect()
                })
                .unwrap_or_default();

            results.push(StoredVector {
                id,
                vector,
                metadata,
                timestamp,
            });
        }
    }
    results
}

/// Extract search results from Arrow RecordBatch (includes _distance column).
fn extract_search_results(batches: &[RecordBatch]) -> Vec<VectorSearchResult> {
    let mut results = Vec::new();
    for batch in batches {
        let ids = match batch.column_by_name("id") {
            Some(col) => match col.as_any().downcast_ref::<StringArray>() {
                Some(arr) => arr,
                None => continue,
            },
            None => continue,
        };
        let metadatas = batch
            .column_by_name("metadata")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        let vectors = batch
            .column_by_name("vector")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

        for i in 0..batch.num_rows() {
            let id = ids.value(i).to_string();
            let metadata_str = metadatas.map(|m| m.value(i)).unwrap_or("{}");
            let metadata: serde_json::Value =
                serde_json::from_str(metadata_str).unwrap_or(serde_json::Value::Null);
            let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
            // Convert distance to similarity (1.0 - distance for cosine, 1/(1+d) for L2)
            let score = 1.0 - distance.min(1.0);
            let vector = vectors.map(|vecs| {
                let arr = vecs.value(i);
                let floats = arr.as_any().downcast_ref::<Float32Array>().expect("Arrow array must be Float32Array");
                (0..floats.len()).map(|j| floats.value(j)).collect()
            });

            results.push(VectorSearchResult {
                id,
                score,
                metadata,
                vector,
            });
        }
    }
    results
}

/// Convert AiError-compatible wrapper for lancedb::Error
fn lance_err(e: lancedb::Error) -> AiError {
    AiError::Other(format!("LanceDB error: {}", e))
}

/// Convert AiError-compatible wrapper for arrow errors
fn arrow_err(e: arrow_schema::ArrowError) -> AiError {
    AiError::Other(format!("Arrow error: {}", e))
}

impl LanceVectorDb {
    /// Create a new LanceDB vector database at the given path.
    ///
    /// The path is a directory where Lance data files will be stored.
    /// If the directory doesn't exist, it will be created.
    /// If a table with `config.collection_name` already exists, it will be opened.
    ///
    /// # Arguments
    /// * `path` - Local directory path for Lance data storage
    /// * `config` - Vector DB configuration (dimensions, collection_name, etc.)
    ///
    /// # Errors
    /// Returns an error if the tokio runtime cannot be created or LanceDB fails to connect.
    pub fn new(path: &str, config: VectorDbConfig) -> AiResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| AiError::Other(format!("Failed to create tokio runtime: {}", e)))?;
        let schema = build_schema(config.dimensions);
        let table_name = config.collection_name.clone();

        let db = rt
            .block_on(async { lancedb::connect(path).execute().await })
            .map_err(lance_err)?;

        // Try to open existing table, fallback to create empty
        let table = rt.block_on(async {
            match db.open_table(&table_name).execute().await {
                Ok(t) => Ok(t),
                Err(_) => db
                    .create_empty_table(&table_name, schema.clone())
                    .execute()
                    .await,
            }
        }).map_err(lance_err)?;

        Ok(Self {
            db,
            table: Some(table),
            config,
            schema,
            rt,
            table_name,
        })
    }

    /// Get a reference to the active table, or error if not initialized.
    fn table(&self) -> AiResult<&lancedb::Table> {
        self.table
            .as_ref()
            .ok_or_else(|| AiError::Other("LanceDB table not initialized".into()))
    }

    /// Escape a string for use in SQL predicates (prevent SQL injection).
    fn escape_sql(s: &str) -> String {
        s.replace('\'', "''")
    }
}

impl VectorDb for LanceVectorDb {
    fn insert(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> AiResult<()> {
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

        let table = self.table()?;
        let metadata_str = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());

        let batch = build_record_batch(
            &self.schema,
            vec![id.to_string()],
            vec![vector],
            vec![metadata_str],
            vec![now_millis()],
            self.config.dimensions,
        )
        .map_err(arrow_err)?;

        // Delete existing entry with same ID to support upsert semantics
        let predicate = format!("id = '{}'", Self::escape_sql(id));
        let arrow_data = into_arrow(batch, self.schema.clone());
        self.rt
            .block_on(async {
                let _ = table.delete(&predicate).await;
                table.add(arrow_data).execute().await
            })
            .map_err(lance_err)?;

        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        let table = self.table()?;
        let count = vectors.len();

        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count);
        let mut metas = Vec::with_capacity(count);
        let mut timestamps = Vec::with_capacity(count);
        let ts = now_millis();

        for (id, vector, metadata) in &vectors {
            if vector.len() != self.config.dimensions {
                return Err(AiError::Validation(crate::error::ValidationError::Custom {
                    field: "vector".to_string(),
                    message: format!(
                        "Vector dimension mismatch for '{}': expected {}, got {}",
                        id,
                        self.config.dimensions,
                        vector.len()
                    ),
                }));
            }
            ids.push(id.clone());
            vecs.push(vector.clone());
            metas.push(serde_json::to_string(metadata).unwrap_or_else(|_| "{}".to_string()));
            timestamps.push(ts);
        }

        let batch =
            build_record_batch(&self.schema, ids, vecs, metas, timestamps, self.config.dimensions)
                .map_err(arrow_err)?;

        let arrow_data = into_arrow(batch, self.schema.clone());
        self.rt
            .block_on(async { table.add(arrow_data).execute().await })
            .map_err(lance_err)?;

        Ok(count)
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

        let table = self.table()?;

        self.rt.block_on(async {
            let query_vec: Vec<f32> = query.to_vec();
            let mut builder = table
                .vector_search(query_vec)
                .map_err(lance_err)?;

            builder = builder.limit(limit);

            // Apply metadata filters as SQL WHERE clause
            if let Some(filters) = filter {
                let conditions: Vec<String> = filters
                    .iter()
                    .filter_map(|f| metadata_filter_to_sql(f))
                    .collect();
                if !conditions.is_empty() {
                    builder = builder.only_if(conditions.join(" AND "));
                }
            }

            let batches: Vec<RecordBatch> = builder
                .execute()
                .await
                .map_err(lance_err)?
                .try_collect::<Vec<_>>()
                .await
                .map_err(lance_err)?;

            Ok(extract_search_results(&batches))
        })
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        let table = self.table()?;
        let predicate = format!("id = '{}'", Self::escape_sql(id));

        self.rt.block_on(async {
            let batches: Vec<RecordBatch> = table
                .query()
                .only_if(predicate)
                .limit(1)
                .execute()
                .await
                .map_err(lance_err)?
                .try_collect::<Vec<_>>()
                .await
                .map_err(lance_err)?;

            let vectors = extract_stored_vectors(&batches);
            Ok(vectors.into_iter().next())
        })
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        let table = self.table()?;
        let predicate = format!("id = '{}'", Self::escape_sql(id));

        // Check if exists first
        let exists = self.get(id)?.is_some();
        if !exists {
            return Ok(false);
        }

        self.rt
            .block_on(async { table.delete(&predicate).await })
            .map_err(lance_err)?;

        Ok(true)
    }

    fn count(&self) -> usize {
        let table = match self.table() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        self.rt
            .block_on(async { table.count_rows(None).await })
            .unwrap_or(0)
    }

    fn clear(&mut self) -> AiResult<()> {
        // Drop and recreate the table for a clean clear
        let table_name = self.table_name.clone();
        let schema = self.schema.clone();
        let new_table = self.rt
            .block_on(async {
                self.db.drop_table(&table_name, &[]).await.map_err(lance_err)?;
                self.db
                    .create_empty_table(&table_name, schema)
                    .execute()
                    .await
                    .map_err(lance_err)
            })?;
        self.table = Some(new_table);
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        // Verify we can list tables (connection alive)
        match self.rt.block_on(async {
            self.db.table_names().execute().await
        }) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "LanceDB",
            tier: 2,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: true,
            max_recommended_vectors: Some(10_000_000),
        }
    }

    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        let table = self.table()?;

        self.rt.block_on(async {
            let batches: Vec<RecordBatch> = table
                .query()
                .execute()
                .await
                .map_err(lance_err)?
                .try_collect::<Vec<_>>()
                .await
                .map_err(lance_err)?;

            Ok(extract_stored_vectors(&batches))
        })
    }

    fn import_bulk(&mut self, vectors: Vec<StoredVector>) -> AiResult<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        let table = self.table()?;
        let count = vectors.len();

        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count);
        let mut metas = Vec::with_capacity(count);
        let mut timestamps = Vec::with_capacity(count);

        for v in vectors {
            ids.push(v.id);
            vecs.push(v.vector);
            metas.push(serde_json::to_string(&v.metadata).unwrap_or_else(|_| "{}".to_string()));
            timestamps.push(v.timestamp);
        }

        let batch =
            build_record_batch(&self.schema, ids, vecs, metas, timestamps, self.config.dimensions)
                .map_err(arrow_err)?;

        let arrow_data = into_arrow(batch, self.schema.clone());
        self.rt
            .block_on(async { table.add(arrow_data).execute().await })
            .map_err(lance_err)?;

        Ok(count)
    }
}

/// Convert a MetadataFilter into a SQL WHERE clause fragment.
///
/// Since metadata is stored as a JSON string column, filters operate on the
/// serialized string. For proper JSON field access, the metadata would need
/// to be stored as structured columns — this is a reasonable trade-off for
/// the generic VectorDb trait.
fn metadata_filter_to_sql(filter: &MetadataFilter) -> Option<String> {
    use crate::vector_db::FilterOperation;
    match &filter.operation {
        FilterOperation::Equals(v) => {
            let val_str = match v {
                serde_json::Value::String(s) => format!("\"{}\"", s),
                other => other.to_string(),
            };
            // Search for "field": value pattern in the JSON string
            Some(format!(
                "metadata LIKE '%\"{}\":{}'%'",
                filter.field, val_str
            ))
        }
        FilterOperation::Contains(needle) => {
            Some(format!("metadata LIKE '%{}%'", needle.replace('\'', "''")))
        }
        _ => None, // Other filter types not easily mapped to string-based SQL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_lance_db(dimensions: usize) -> LanceVectorDb {
        let dir = tempfile::tempdir().unwrap();
        let config = VectorDbConfig {
            dimensions,
            collection_name: "test_vectors".to_string(),
            max_vectors: None,
            ..Default::default()
        };
        LanceVectorDb::new(dir.path().to_str().unwrap(), config).unwrap()
    }

    #[test]
    fn test_lance_insert_and_get() {
        let mut db = temp_lance_db(3);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"label": "x"}))
            .unwrap();

        let v = db.get("v1").unwrap().unwrap();
        assert_eq!(v.id, "v1");
        assert_eq!(v.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(v.metadata["label"], "x");
    }

    #[test]
    fn test_lance_insert_replaces_existing() {
        let mut db = temp_lance_db(3);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"v": 1}))
            .unwrap();
        db.insert("v1", vec![0.0, 1.0, 0.0], serde_json::json!({"v": 2}))
            .unwrap();

        assert_eq!(db.count(), 1);
        let v = db.get("v1").unwrap().unwrap();
        assert_eq!(v.metadata["v"], 2);
    }

    #[test]
    fn test_lance_search() {
        let mut db = temp_lance_db(3);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"label": "x"}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({"label": "y"}))
            .unwrap();
        db.insert("v3", vec![0.9, 0.1, 0.0], serde_json::json!({"label": "near_x"}))
            .unwrap();

        let results = db.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        // The most similar to [1,0,0] should be v1 or v3
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_lance_delete() {
        let mut db = temp_lance_db(3);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        assert_eq!(db.count(), 1);

        assert!(db.delete("v1").unwrap());
        assert_eq!(db.count(), 0);
        assert!(!db.delete("v1").unwrap()); // Already deleted
    }

    #[test]
    fn test_lance_batch_insert() {
        let mut db = temp_lance_db(3);

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
    fn test_lance_clear() {
        let mut db = temp_lance_db(3);

        db.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({}))
            .unwrap();
        assert_eq!(db.count(), 2);

        db.clear().unwrap();
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn test_lance_dimension_validation() {
        let mut db = temp_lance_db(3);

        let result = db.insert("v1", vec![1.0, 0.0], serde_json::json!({}));
        assert!(result.is_err());

        let result = db.search(&[1.0, 0.0], 5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lance_export_import() {
        let mut db1 = temp_lance_db(3);

        db1.insert("v1", vec![1.0, 0.0, 0.0], serde_json::json!({"a": 1}))
            .unwrap();
        db1.insert("v2", vec![0.0, 1.0, 0.0], serde_json::json!({"a": 2}))
            .unwrap();

        let exported = db1.export_all().unwrap();
        assert_eq!(exported.len(), 2);

        let mut db2 = temp_lance_db(3);
        let imported = db2.import_bulk(exported).unwrap();
        assert_eq!(imported, 2);
        assert_eq!(db2.count(), 2);

        let v = db2.get("v1").unwrap().unwrap();
        assert_eq!(v.metadata["a"], 1);
    }

    #[test]
    fn test_lance_backend_info() {
        let db = temp_lance_db(3);
        let info = db.backend_info();

        assert_eq!(info.name, "LanceDB");
        assert_eq!(info.tier, 2);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(info.supports_export);
    }

    #[test]
    fn test_lance_health_check() {
        let db = temp_lance_db(3);
        assert!(db.health_check().unwrap());
    }

    #[test]
    fn test_lance_empty_operations() {
        let db = temp_lance_db(3);

        assert_eq!(db.count(), 0);
        assert!(db.get("nonexistent").unwrap().is_none());
        assert!(db.export_all().unwrap().is_empty());

        let results = db.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_lance_batch_insert_dimension_mismatch() {
        let mut db = temp_lance_db(3);

        let vectors = vec![
            ("v1".to_string(), vec![1.0, 0.0, 0.0], serde_json::json!({})),
            ("v2".to_string(), vec![0.0, 1.0], serde_json::json!({})), // Wrong dimension
        ];

        let result = db.insert_batch(vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_lance_import_empty() {
        let mut db = temp_lance_db(3);
        let imported = db.import_bulk(vec![]).unwrap();
        assert_eq!(imported, 0);
    }

    #[test]
    fn test_lance_migration_roundtrip() {
        use crate::vector_db::{InMemoryVectorDb, migrate_vectors};

        // Create in-memory DB with data
        let config = VectorDbConfig {
            dimensions: 3,
            ..Default::default()
        };
        let mut mem_db = InMemoryVectorDb::new(config);
        mem_db
            .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({"from": "memory"}))
            .unwrap();
        mem_db
            .insert("b", vec![0.0, 1.0, 0.0], serde_json::json!({"from": "memory"}))
            .unwrap();

        // Migrate to Lance
        let mut lance_db = temp_lance_db(3);
        let result = migrate_vectors(&mem_db, &mut lance_db).unwrap();
        assert_eq!(result.exported, 2);
        assert_eq!(result.imported, 2);
        assert_eq!(lance_db.count(), 2);

        let v = lance_db.get("a").unwrap().unwrap();
        assert_eq!(v.metadata["from"], "memory");
    }
}
