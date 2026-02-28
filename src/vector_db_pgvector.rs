//! PostgreSQL pgvector backend for vector storage.
//!
//! Provides a [`VectorDb`] implementation backed by PostgreSQL with the pgvector extension.
//! Requires the `vector-pgvector` feature flag and a PostgreSQL server with pgvector installed.
//!
//! ## SQL schema
//!
//! ```sql
//! CREATE EXTENSION IF NOT EXISTS vector;
//! CREATE TABLE IF NOT EXISTS {table} (
//!     id TEXT PRIMARY KEY,
//!     vector vector({dim}),
//!     metadata JSONB NOT NULL DEFAULT '{}',
//!     timestamp BIGINT NOT NULL DEFAULT 0
//! );
//! ```

use crate::error::{AiError, AiResult};
use crate::vector_db::{BackendInfo, MetadataFilter, StoredVector, VectorDb, VectorSearchResult};

/// Configuration for pgvector connection.
#[derive(Debug, Clone)]
pub struct PgVectorConfig {
    /// PostgreSQL connection string (e.g., "host=localhost dbname=vectors user=postgres")
    pub connection_string: String,
    /// Table name for vector storage
    pub table_name: String,
    /// Vector dimensions
    pub dimensions: usize,
}

impl Default for PgVectorConfig {
    fn default() -> Self {
        Self {
            connection_string: "host=localhost dbname=vectors user=postgres".to_string(),
            table_name: "embeddings".to_string(),
            dimensions: 384,
        }
    }
}

/// pgvector-backed vector database.
///
/// Uses PostgreSQL with the pgvector extension for similarity search.
/// Supports L2 distance (`<->`) and cosine distance (`<=>`) operators.
pub struct PgVectorDb {
    config: PgVectorConfig,
    /// Local count cache (refreshed on operations)
    count_cache: usize,
}

impl PgVectorDb {
    pub fn new(config: PgVectorConfig) -> Self {
        Self {
            config,
            count_cache: 0,
        }
    }

    /// Generate the CREATE TABLE SQL for this configuration.
    pub fn create_table_sql(&self) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS {} (\
                id TEXT PRIMARY KEY, \
                embedding vector({}), \
                metadata JSONB NOT NULL DEFAULT '{{}}', \
                created_at BIGINT NOT NULL DEFAULT 0\
            )",
            self.config.table_name, self.config.dimensions
        )
    }

    /// Generate the CREATE EXTENSION SQL.
    pub fn create_extension_sql(&self) -> String {
        "CREATE EXTENSION IF NOT EXISTS vector".to_string()
    }

    /// Generate an INSERT/upsert SQL statement.
    pub fn upsert_sql(&self) -> String {
        format!(
            "INSERT INTO {} (id, embedding, metadata, created_at) \
             VALUES ($1, $2::vector, $3::jsonb, $4) \
             ON CONFLICT (id) DO UPDATE SET \
                embedding = EXCLUDED.embedding, \
                metadata = EXCLUDED.metadata, \
                created_at = EXCLUDED.created_at",
            self.config.table_name
        )
    }

    /// Generate a similarity search SQL using cosine distance.
    pub fn search_sql(&self) -> String {
        format!(
            "SELECT id, 1 - (embedding <=> $1::vector) AS score, metadata \
             FROM {} \
             ORDER BY embedding <=> $1::vector \
             LIMIT $2",
            self.config.table_name
        )
    }

    /// Generate a search SQL with L2 distance.
    pub fn search_l2_sql(&self) -> String {
        format!(
            "SELECT id, 1.0 / (1.0 + (embedding <-> $1::vector)) AS score, metadata \
             FROM {} \
             ORDER BY embedding <-> $1::vector \
             LIMIT $2",
            self.config.table_name
        )
    }

    /// Generate a GET by ID SQL.
    pub fn get_sql(&self) -> String {
        format!(
            "SELECT id, embedding::text, metadata, created_at FROM {} WHERE id = $1",
            self.config.table_name
        )
    }

    /// Generate a DELETE SQL.
    pub fn delete_sql(&self) -> String {
        format!("DELETE FROM {} WHERE id = $1", self.config.table_name)
    }

    /// Generate a COUNT SQL.
    pub fn count_sql(&self) -> String {
        format!("SELECT COUNT(*) FROM {}", self.config.table_name)
    }

    /// Generate a TRUNCATE SQL.
    pub fn clear_sql(&self) -> String {
        format!("TRUNCATE TABLE {}", self.config.table_name)
    }

    /// Generate a CREATE INDEX SQL for HNSW (cosine distance).
    pub fn create_index_sql(&self) -> String {
        format!(
            "CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} \
             USING hnsw (embedding vector_cosine_ops)",
            table = self.config.table_name
        )
    }

    /// Format a vector as a pgvector string literal: '[1.0,2.0,3.0]'
    pub fn format_vector(vector: &[f32]) -> String {
        let inner: String = vector
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("[{}]", inner)
    }

    /// Parse a pgvector string literal back to Vec<f32>: '[1.0,2.0,3.0]' -> vec![1.0, 2.0, 3.0]
    pub fn parse_vector(s: &str) -> Vec<f32> {
        let trimmed = s.trim_start_matches('[').trim_end_matches(']');
        if trimmed.is_empty() {
            return Vec::new();
        }
        trimmed
            .split(',')
            .filter_map(|v| v.trim().parse::<f32>().ok())
            .collect()
    }
}

// VectorDb trait implementation — returns errors for operations requiring a live database connection.
// This module's primary value is SQL generation (create_table_sql, upsert_sql, search_sql, etc.)
// which users execute via their own postgres connection (ureq REST, postgres crate, sqlx, etc.).
// The VectorDb trait impl is provided for interface compatibility with VectorDbBuilder.

impl VectorDb for PgVectorDb {
    fn insert(
        &mut self,
        _id: &str,
        _vector: Vec<f32>,
        _metadata: serde_json::Value,
    ) -> AiResult<()> {
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let _ = vectors.len();
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn search(
        &self,
        _query: &[f32],
        _limit: usize,
        _filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn get(&self, _id: &str) -> AiResult<Option<StoredVector>> {
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn delete(&mut self, _id: &str) -> AiResult<bool> {
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn count(&self) -> usize {
        self.count_cache
    }

    fn clear(&mut self) -> AiResult<()> {
        Err(AiError::Other("pgvector: no live connection. Use PgVectorDb SQL generation methods with your own postgres client.".into()))
    }

    fn health_check(&self) -> AiResult<bool> {
        Ok(false) // Not connected
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "pgvector",
            tier: 4,
            supports_persistence: true,
            supports_filtering: true,
            supports_export: true,
            max_recommended_vectors: None, // PostgreSQL scales well
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgvector_config_default() {
        let config = PgVectorConfig::default();
        assert_eq!(config.table_name, "embeddings");
        assert_eq!(config.dimensions, 384);
        assert!(config.connection_string.contains("localhost"));
    }

    #[test]
    fn test_create_table_sql() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "my_vectors".to_string(),
            dimensions: 768,
            ..Default::default()
        });
        let sql = db.create_table_sql();
        assert!(sql.contains("my_vectors"));
        assert!(sql.contains("vector(768)"));
        assert!(sql.contains("PRIMARY KEY"));
        assert!(sql.contains("JSONB"));
    }

    #[test]
    fn test_create_extension_sql() {
        let db = PgVectorDb::new(PgVectorConfig::default());
        assert_eq!(
            db.create_extension_sql(),
            "CREATE EXTENSION IF NOT EXISTS vector"
        );
    }

    #[test]
    fn test_upsert_sql() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "embeddings".to_string(),
            ..Default::default()
        });
        let sql = db.upsert_sql();
        assert!(sql.contains("INSERT INTO embeddings"));
        assert!(sql.contains("ON CONFLICT (id) DO UPDATE"));
        assert!(sql.contains("$1"));
        assert!(sql.contains("$2::vector"));
    }

    #[test]
    fn test_search_sql_cosine() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "vecs".to_string(),
            ..Default::default()
        });
        let sql = db.search_sql();
        assert!(sql.contains("<=>")); // cosine distance operator
        assert!(sql.contains("LIMIT $2"));
        assert!(sql.contains("AS score"));
    }

    #[test]
    fn test_search_l2_sql() {
        let db = PgVectorDb::new(PgVectorConfig::default());
        let sql = db.search_l2_sql();
        assert!(sql.contains("<->")); // L2 distance operator
    }

    #[test]
    fn test_create_index_sql() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "my_table".to_string(),
            ..Default::default()
        });
        let sql = db.create_index_sql();
        assert!(sql.contains("CREATE INDEX IF NOT EXISTS"));
        assert!(sql.contains("USING hnsw"));
        assert!(sql.contains("vector_cosine_ops"));
        assert!(sql.contains("my_table"));
    }

    #[test]
    fn test_format_vector() {
        assert_eq!(PgVectorDb::format_vector(&[1.0, 2.5, 3.0]), "[1,2.5,3]");
        assert_eq!(PgVectorDb::format_vector(&[]), "[]");
        assert_eq!(PgVectorDb::format_vector(&[0.0]), "[0]");
    }

    #[test]
    fn test_parse_vector() {
        assert_eq!(
            PgVectorDb::parse_vector("[1.0,2.5,3.0]"),
            vec![1.0, 2.5, 3.0]
        );
        assert_eq!(PgVectorDb::parse_vector("[]"), Vec::<f32>::new());
        assert_eq!(PgVectorDb::parse_vector("[0]"), vec![0.0]);
    }

    #[test]
    fn test_format_parse_roundtrip() {
        let original = vec![1.5, -2.3, 0.0, 4.7];
        let formatted = PgVectorDb::format_vector(&original);
        let parsed = PgVectorDb::parse_vector(&formatted);
        assert_eq!(parsed.len(), original.len());
        for (a, b) in original.iter().zip(parsed.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_backend_info() {
        let db = PgVectorDb::new(PgVectorConfig::default());
        let info = db.backend_info();
        assert_eq!(info.name, "pgvector");
        assert_eq!(info.tier, 4);
        assert!(info.supports_persistence);
        assert!(info.supports_filtering);
        assert!(info.supports_export);
    }

    #[test]
    fn test_health_check_not_connected() {
        let db = PgVectorDb::new(PgVectorConfig::default());
        // Without a real connection, health check returns false
        assert_eq!(db.health_check().unwrap(), false);
    }

    #[test]
    fn test_count_starts_at_zero() {
        let db = PgVectorDb::new(PgVectorConfig::default());
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn test_delete_sql() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "docs".to_string(),
            ..Default::default()
        });
        let sql = db.delete_sql();
        assert!(sql.contains("DELETE FROM docs"));
        assert!(sql.contains("$1"));
    }

    #[test]
    fn test_count_sql() {
        let db = PgVectorDb::new(PgVectorConfig {
            table_name: "embeddings".to_string(),
            ..Default::default()
        });
        let sql = db.count_sql();
        assert!(sql.contains("SELECT COUNT(*)"));
        assert!(sql.contains("embeddings"));
    }
}
