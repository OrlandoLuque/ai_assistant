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
use crate::http_client::{HttpClient, HttpMethod, HttpRequest, HttpResponse, UreqClient};

/// El error que los 38 sitios de este modulo escribian a mano.
///
/// `what` es el mensaje que ya tenian ("Chroma add failed"), y ahora cubre las
/// TRES cosas que pueden ir mal en vez de una: serializar el cuerpo, el
/// transporte y el no-2xx. Antes solo habia una forma de fallar **porque
/// `send_json(..)` metia el no-2xx dentro del error de transporte** --- y sin
/// cuerpo, asi que el motivo que el backend daba no llegaba a ninguna parte.
fn failed(what: &str, e: impl std::fmt::Display) -> AiError {
    AiError::Other(format!("{}: {}", what, e))
}

/// Manda una peticion por el puerto y **exige un 2xx**, con el cuerpo del
/// rechazo en el mensaje.
///
/// Detras del puerto un no-2xx vuelve como `Ok`, asi que cada sitio que antes
/// se apoyaba en el `?` de `send_json` necesita decir esto en voz alta. Lo que
/// se gana al decirlo: el mensaje pasa de «http status: 400» a llevar el texto
/// con el que Qdrant o Weaviate explican QUE rechazaron.
fn send_ok(request: &HttpRequest, what: &str) -> AiResult<HttpResponse> {
    let response = UreqClient.send(request).map_err(|e| failed(what, e))?;
    response
        .error_for_status()
        .map_err(|e| failed(what, e))
        .map(|()| response)
}

/// Un `POST` de JSON --- la forma de 24 de los 38 sitios --- exigiendo un 2xx.
///
/// `headers` son las cabeceras propias del backend (la `Api-Key` de Pinecone);
/// el `Content-Type` lo pone el puerto.
fn post_json_ok(
    url: &str,
    body: &serde_json::Value,
    headers: &[(&str, &str)],
    what: &str,
) -> AiResult<HttpResponse> {
    let mut request = HttpRequest::post_json(url, body).map_err(|e| failed(what, e))?;
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    send_ok(&request, what)
}

/// Si un backend contesta 2xx a un sondeo, sin leerle el cuerpo.
///
/// Hay nueve implementaciones de `health_check` en este modulo, y **seis
/// hablan HTTP**: cuatro sondean con un `GET` y dos con un `POST` vacio (las
/// otras tres son la de Qdrant, que va por su `request`, y las dos locales, que
/// no tienen a quien preguntar). Las seis decian
/// `{ Ok(_) => Ok(true), Err(_) => Ok(false) }`, y ahi el `Ok(_)` SI valia:
/// `ureq` metia el no-2xx en el `Err`. Detras del puerto el `Ok(_)` casaria
/// tambien con un 500, y un backend que contesta «error interno» pasaria por
/// sano. La comprobacion es la misma de siempre, dicha donde se ve.
fn probe_is_healthy(request: &HttpRequest) -> bool {
    UreqClient.send(request).is_ok_and(|resp| resp.is_success())
}

/// Que dijo el **servidor** al borrar.
///
/// Las tres situaciones que un `bool` no podia distinguir (N158).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeleteConfirmation {
    /// El servidor **confirmo** el borrado.
    ///
    /// No es una afirmacion de que existiera: un borrado es idempotente en
    /// varios backends --- Pinecone contesta 200 con el id presente o ausente
    /// --- asi que ahi la ausencia tambien llega como `Deleted`. Lo que esta
    /// variante garantiza es que **el servidor hablo y acepto**.
    Deleted,
    /// El servidor dijo que **no estaba** (un 404).
    ///
    /// No todos los backends pueden decir esto; ver
    /// [`DeleteConfirmation::Deleted`].
    NotFound,
    /// **No hubo confirmacion**: ni se pudo preguntar, o el servidor rechazo.
    Unconfirmed {
        /// Por que: un error de transporte, o el estado y el cuerpo del rechazo.
        reason: String,
    },
}

/// Que paso de verdad al borrar un vector.
///
/// Existe porque `delete() -> AiResult<bool>` **no podia decir la verdad** en
/// los seis backends remotos: descartaban la respuesta del servidor y
/// contestaban por la cache local, asi que con el backend caido o contestando
/// 401 devolvian `Ok(true)` y el vector seguia ahi --- reapareciendo al
/// siguiente arranque, con la cache ya vacia (N158).
///
/// # Por que son DOS campos y no tres variantes
///
/// La primera version de esto era un enum de tres variantes, con
/// `LocalOnly { removed_locally }`. Lo tumbo un test que ya existia
/// --- `test_weaviate_cache_operations` ---, y el motivo merece quedar escrito:
/// **lo que dice el servidor y lo que paso en la copia local son ortogonales**.
/// Un servidor puede contestar «no esta» mientras la cache local si lo tenia,
/// y en el enum ese estado se perdia: `NotFound` implicaba «no se quito nada»,
/// que era falso.
///
/// Con los dos campos separados, `delete()` sigue contestando exactamente lo
/// que contestaba --- por la copia local --- y la pregunta nueva, «¿lo confirmo
/// el servidor?», se contesta aparte sin contaminar la vieja.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteOutcome {
    /// Que dijo el servidor.
    pub server: DeleteConfirmation,
    /// Si habia copia local y se quito.
    pub removed_locally: bool,
}

impl DeleteOutcome {
    /// Si **algo** dejo de estar donde estaba: la copia local, o el dato en el
    /// servidor.
    ///
    /// Es el valor que `delete()` devuelve, y la parte `removed_locally` es la
    /// que lo mantiene igual a lo que devolvia antes de N158.
    pub fn removed_something(&self) -> bool {
        self.removed_locally || self.server == DeleteConfirmation::Deleted
    }

    /// Si el **servidor** se pronuncio, diga que si o que no.
    ///
    /// Esta es la pregunta que un booleano no podia contestar, y la que
    /// distingue «borrado» de «creo que borrado».
    pub fn confirmed_by_server(&self) -> bool {
        !matches!(self.server, DeleteConfirmation::Unconfirmed { .. })
    }

    /// El motivo, cuando no hubo confirmacion.
    pub fn unconfirmed_reason(&self) -> Option<&str> {
        match &self.server {
            DeleteConfirmation::Unconfirmed { reason } => Some(reason.as_str()),
            _ => None,
        }
    }
}

/// Borra en el servidor y en la cache, y devuelve las dos mitades: que dijo el
/// servidor y que paso en local.
///
/// **Un solo sitio para la decision.** Los seis backends remotos tenian la
/// misma forma copiada --- descartar la respuesta y contestar por la cache ---
/// y lo que cambia entre ellos es solo la peticion, que entra como cierre.
/// Seis copias de una regla son seis sitios donde arreglarla y uno donde
/// olvidarse.
///
/// La traduccion de estado a veredicto es uniforme a proposito:
///
/// | Lo que paso | `server` |
/// |---|---|
/// | error de transporte | `Unconfirmed` con el error |
/// | 404 | `NotFound` --- el servidor contesto que no esta |
/// | 2xx | `Deleted` |
/// | otro no-2xx | `Unconfirmed` con el estado y el cuerpo |
fn delete_remote_then_cache(
    cache: &mut HashMap<String, StoredVector>,
    id: &str,
    send: impl FnOnce() -> Result<HttpResponse, anyhow::Error>,
) -> DeleteOutcome {
    let server = match send() {
        Ok(response) if response.is_success() => DeleteConfirmation::Deleted,
        Ok(response) if response.status == 404 => DeleteConfirmation::NotFound,
        // El cuerpo es donde el backend explica el rechazo, asi que viaja en el
        // motivo --- es la diferencia entre «no se confirmo» y poder arreglarlo.
        Ok(response) => DeleteConfirmation::Unconfirmed {
            reason: format!(
                "HTTP {}{}",
                response.status,
                match response.text() {
                    Ok(body) if !body.trim().is_empty() => format!(": {}", body.trim()),
                    _ => String::new(),
                }
            ),
        },
        Err(e) => DeleteConfirmation::Unconfirmed {
            reason: format!("{:#}", e),
        },
    };
    DeleteOutcome {
        server,
        removed_locally: cache.remove(id).is_some(),
    }
}

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
#[non_exhaustive]
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

with_setters!(VectorDbConfig {
    with_dimensions -> dimensions: usize,
    with_distance_metric -> distance_metric: DistanceMetric,
    with_max_vectors -> max_vectors: Option<usize>,
    with_collection_name -> collection_name: String,
    with_qdrant_url -> qdrant_url: Option<String>,
    with_qdrant_api_key -> qdrant_api_key: Option<String>,
});

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
#[non_exhaustive]
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
#[non_exhaustive]
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
///
/// # Stability
///
/// New methods may be added to this trait in minor versions with default
/// implementations. Required methods will only change in major versions.
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
    ///
    /// `true` significa «algo dejo de estar donde estaba». Para saber si el
    /// **servidor** lo confirmo --- que es otra pregunta --- usa
    /// [`VectorDb::delete_detailed`].
    fn delete(&mut self, id: &str) -> AiResult<bool>;

    /// Igual que [`VectorDb::delete`], diciendo **cual** de las tres
    /// situaciones ocurrio. Ver [`DeleteOutcome`].
    ///
    /// El defecto es correcto para una implementacion **sin servidor remoto**
    /// --- en memoria, HNSW, LanceDB, pgvector ---: ahi lo que dice `delete()`
    /// es la verdad completa, porque no hay nadie mas a quien preguntar y
    /// `LocalOnly` no puede ocurrir.
    ///
    /// **Cuidado al sobrescribirlo:** este defecto llama a `delete()`, asi que
    /// una implementacion cuyo `delete()` delegue AQUI sin sobrescribir esto
    /// entra en recursion infinita. Los seis backends remotos lo evitan
    /// poniendo el trabajo en un metodo propio al que llaman los dos; es la
    /// unica forma de que el compilador no pueda avisar y por eso queda escrito.
    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        let removed = self.delete(id)?;
        Ok(DeleteOutcome {
            server: if removed {
                DeleteConfirmation::Deleted
            } else {
                DeleteConfirmation::NotFound
            },
            removed_locally: removed,
        })
    }

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

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::VectorDbInMemory,
            self.vectors.len(),
        );

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

        // Esta funcion era un puerto HTTP escrito a mano, y de los buenos: ya
        // construia una `http::Request` para que el metodo no fuera un `match`
        // de tipos que no unifican (en ureq 3 `get` devuelve
        // `RequestBuilder<WithoutBody>` y `post` `RequestBuilder<WithBody>`), y
        // ya pedia `http_status_as_error(false)` para quedarse el cuerpo del
        // rechazo. Detras del puerto es lo mismo sin la maquinaria --- y de
        // paso entra en el pool de conexiones, porque construia un `Agent`
        // nuevo en cada peticion.
        let method = HttpMethod::from_name(method).ok_or_else(|| {
            AiError::Other(format!("metodo HTTP desconocido para {}: {}", url, method))
        })?;

        let mut request = HttpRequest::new(method, &url).header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            request = request.header("api-key", key);
        }
        if let Some(body) = body {
            request = request.body(body.as_bytes().to_vec());
        }

        match UreqClient.send(&request) {
            Ok(resp) if resp.is_success() => Ok(resp.text().unwrap_or_default()),
            Ok(resp) => Err(AiError::Provider(crate::error::ProviderError::ApiError {
                provider: "Qdrant".to_string(),
                status_code: resp.status,
                message: resp.text().unwrap_or_default(),
            })),
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
#[derive(Debug)]
pub struct VectorDbBuilder {
    config: VectorDbConfig,
    backend: VectorDbBackend,
    /// Path for LanceDB storage (only used with Lance backend)
    #[cfg(feature = "vector-lancedb")]
    lance_path: Option<String>,
}

/// Available backends
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
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
#[derive(Debug)]
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
impl<V: VectorDb + std::fmt::Debug> std::fmt::Debug for DistributedVectorDb<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedVectorDb")
            .field("local", &self.local)
            .field("network", &"<Arc<NetworkNode>>")
            .finish()
    }
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
#[derive(Debug)]
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

impl PineconeClient {
    /// El trabajo de borrar, que `delete` y `delete_detailed` comparten.
    ///
    /// Esta en un metodo PROPIO y no en uno de los dos, a proposito: si
    /// `delete` delegara en `delete_detailed` y este no estuviera sobrescrito,
    /// el defecto del trait llamaria a `delete` otra vez --- recursion infinita
    /// que el compilador no puede ver. Un metodo propio lo hace imposible, y
    /// ademas las seis copias de la decision se quedan en
    /// `delete_remote_then_cache` (N158).
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let url = self.pinecone_url("/vectors/delete");
        let body = serde_json::json!({ "ids": [id] });
        let api_key = self.api_key.clone();
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::post_json(&url, &body)?.header("Api-Key", &api_key))
        })
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
        // El `Ok(_)` de antes valia porque `send_json` metia el no-2xx en el
        // `Err`. Detras del puerto hay que pedir el 2xx: si no, un 401 de
        // Pinecone caeria por aqui y el vector se cacharia como guardado.
        let resp = post_json_ok(
            &url,
            &body,
            &[("Api-Key", &self.api_key)],
            "Pinecone upsert failed",
        );
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
        // El `Ok(_)` de antes valia porque `send_json` metia el no-2xx en el
        // `Err`. Detras del puerto hay que pedir el 2xx: si no, un 401 de
        // Pinecone caeria por aqui y el vector se cacharia como guardado.
        let resp = post_json_ok(
            &url,
            &body,
            &[("Api-Key", &self.api_key)],
            "Pinecone upsert failed",
        );
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
        let resp = post_json_ok(
            &url,
            &body,
            &[("Api-Key", &self.api_key)],
            "Pinecone query failed",
        )?;
        let json: serde_json::Value = resp
            .json()
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
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    /// Vacia el indice.
    ///
    /// La firma es `AiResult<()>`, o sea que **puede** informar del fallo, y
    /// hasta N158 descartaba la respuesta: con la clave caducada vaciaba la
    /// cache, devolvia `Ok(())` y dejaba el indice entero en Pinecone. Un
    /// `clear()` que miente es peor que un `delete()` que miente, porque el
    /// llamante da por hecho que no queda nada.
    ///
    /// Ahora **propaga**. La cache se vacia igual --- si el servidor rechazo, lo
    /// que hay en memoria tampoco es de fiar --- pero el error llega.
    fn clear(&mut self) -> AiResult<()> {
        let url = self.pinecone_url("/vectors/delete");
        let body = serde_json::json!({ "deleteAll": true });
        let result = post_json_ok(
            &url,
            &body,
            &[("Api-Key", &self.api_key)],
            "Pinecone clear failed",
        );
        self.cache.clear();
        result.map(|_| ())
    }

    fn health_check(&self) -> AiResult<bool> {
        let url = self.pinecone_url("/describe_index_stats");
        let request = HttpRequest::post_json(&url, &serde_json::json!({}))
            .map_err(|e| failed("Pinecone health check failed", e))?
            .header("Api-Key", &self.api_key);
        Ok(probe_is_healthy(&request))
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
#[derive(Debug)]
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
        let resp = post_json_ok(&url, &body, &[], "Chroma create collection failed")?;
        let json: serde_json::Value = resp
            .json()
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

impl ChromaClient {
    /// Ver [`PineconeClient::delete_outcome`] para por que es un metodo propio.
    ///
    /// Sin coleccion resuelta no hay a quien preguntar, asi que es `LocalOnly`
    /// con el motivo dicho --- y no un `Deleted` que seria mentira.
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let Some(coll_id) = self.collection_id.clone() else {
            return DeleteOutcome {
                server: DeleteConfirmation::Unconfirmed {
                    reason: "no Chroma collection resolved yet".to_string(),
                },
                removed_locally: self.cache.remove(id).is_some(),
            };
        };
        let url = format!("{}/api/v1/collections/{}/delete", self.base_url, coll_id);
        let body = serde_json::json!({ "ids": [id] });
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::post_json(&url, &body)?)
        })
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
        post_json_ok(&url, &body, &[], "Chroma add failed")?;
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
        post_json_ok(&url, &body, &[], "Chroma batch add failed")?;
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
        let resp = post_json_ok(&url, &body, &[], "Chroma query failed")?;
        let json: serde_json::Value = resp
            .json()
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
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
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
        Ok(probe_is_healthy(&HttpRequest::get(&url)))
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
#[derive(Debug)]
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

impl MilvusClient {
    /// Ver [`PineconeClient::delete_outcome`] para por que es un metodo propio.
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let url = format!("{}/v2/vectordb/entities/delete", self.base_url);
        let body = serde_json::json!({
            "collectionName": self.collection_name(),
            "filter": format!("id == \"{}\"", id)
        });
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::post_json(&url, &body)?)
        })
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
        post_json_ok(&url, &body, &[], "Milvus insert failed")?;
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
        post_json_ok(&url, &body, &[], "Milvus batch insert failed")?;
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
        let resp = post_json_ok(&url, &body, &[], "Milvus search failed")?;
        let json: serde_json::Value = resp
            .json()
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
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
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
        let request = HttpRequest::post_json(&url, &serde_json::json!({}))
            .map_err(|e| failed("Milvus health check failed", e))?;
        Ok(probe_is_healthy(&request))
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
#[derive(Debug)]
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

impl WeaviateClient {
    /// Ver [`PineconeClient::delete_outcome`] para por que es un metodo propio.
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let url = format!("{}/v1/objects/{}/{}", self.base_url, self.class_name(), id);
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::new(HttpMethod::Delete, &url))
        })
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
        post_json_ok(&url, &body, &[], "Weaviate insert failed")?;
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
        post_json_ok(&url, &body, &[], "Weaviate batch insert failed")?;
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
        let resp = post_json_ok(&url, &body, &[], "Weaviate search failed")?;
        let json: serde_json::Value = resp
            .json()
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
        // `is_success()` y no solo `Ok(..)`: hoy un 404 llega como `Err` y cae
        // en `Ok(None)` --- «no esta», que es la respuesta correcta. Detras del
        // puerto el 404 es un `Ok`, y como el parseo de abajo esta hecho de
        // `unwrap_or_default()`, leeria el cuerpo del error y devolveria
        // `Some(StoredVector { vector: [], metadata: {} })`: un vector
        // INVENTADO, con embedding vacio, por el que una busqueda puntuaria.
        match UreqClient.send(&HttpRequest::get(&url)) {
            Ok(resp) if resp.is_success() => {
                let json: serde_json::Value = resp
                    .json()
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
            // El no-2xx va por aqui, junto al fallo de transporte: las dos
            // cosas significan «no hay vector que devolver».
            Ok(_) | Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
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
        Ok(probe_is_healthy(&HttpRequest::get(&url)))
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
#[derive(Debug)]
pub struct RedisVectorClient {
    base_url: String,
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

    /// Returns the vector DB configuration.
    pub fn config(&self) -> &VectorDbConfig {
        &self.config
    }

    /// Returns the index name used for vector search.
    fn index_name(&self) -> &str {
        &self.index_name
    }
}

impl RedisVectorClient {
    /// Ver [`PineconeClient::delete_outcome`] para por que es un metodo propio.
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let url = format!("{}/del/{}", self.base_url, id);
        let body = serde_json::json!({});
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::post_json(&url, &body)?)
        })
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
        post_json_ok(&url, &body, &[], "Redis vector insert failed")?;
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
            post_json_ok(&url, &body, &[], "Redis vector batch insert failed")?;
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
        let resp = post_json_ok(&url, &body, &[], "Redis vector search failed")?;
        let json: serde_json::Value = resp
            .json()
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
        // `is_success()` y no solo `Ok(..)`: hoy un 404 llega como `Err` y cae
        // en `Ok(None)` --- «no esta», que es la respuesta correcta. Detras del
        // puerto el 404 es un `Ok`, y como el parseo de abajo esta hecho de
        // `unwrap_or_default()`, leeria el cuerpo del error y devolveria
        // `Some(StoredVector { vector: [], metadata: {} })`: un vector
        // INVENTADO, con embedding vacio, por el que una busqueda puntuaria.
        match UreqClient.send(&HttpRequest::get(&url)) {
            Ok(resp) if resp.is_success() => {
                let json: serde_json::Value = resp
                    .json()
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
            // El no-2xx va por aqui, junto al fallo de transporte: las dos
            // cosas significan «no hay vector que devolver».
            Ok(_) | Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
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
        Ok(probe_is_healthy(&HttpRequest::get(&url)))
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
#[derive(Debug)]
pub struct ElasticsearchClient {
    base_url: String,
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

    /// Returns the vector DB configuration.
    pub fn config(&self) -> &VectorDbConfig {
        &self.config
    }

    /// Returns the index name (lowercased collection name).
    fn index_name(&self) -> &str {
        &self.index_name
    }
}

impl ElasticsearchClient {
    /// Ver [`PineconeClient::delete_outcome`] para por que es un metodo propio.
    ///
    /// Elasticsearch **si** distingue: contesta 404 cuando el documento no
    /// esta, asi que aqui `NotFound` es una respuesta real del servidor y no
    /// una suposicion.
    fn delete_outcome(&mut self, id: &str) -> DeleteOutcome {
        let url = format!("{}/{}/_doc/{}", self.base_url, self.index_name(), id);
        delete_remote_then_cache(&mut self.cache, id, || {
            UreqClient.send(&HttpRequest::new(HttpMethod::Delete, &url))
        })
    }
}

impl VectorDb for ElasticsearchClient {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        let url = format!("{}/{}/_doc/{}", self.base_url, self.index_name(), id);
        let body = serde_json::json!({
            "vector": vector,
            "metadata": metadata
        });
        let what = "Elasticsearch insert failed";
        send_ok(
            &HttpRequest::new(HttpMethod::Put, &url)
                .json(&body)
                .map_err(|e| failed(what, e))?,
            what,
        )?;
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
        // `x-ndjson` y no JSON: el bulk de Elasticsearch es una linea por
        // operacion, asi que el cuerpo va crudo y la cabecera la pone el sitio.
        send_ok(
            &HttpRequest::new(HttpMethod::Post, &url)
                .header("Content-Type", "application/x-ndjson")
                .body(bulk_body.into_bytes()),
            "Elasticsearch bulk insert failed",
        )?;
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
        let resp = post_json_ok(&url, &body, &[], "Elasticsearch search failed")?;
        let json: serde_json::Value = resp
            .json()
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
        // `is_success()` y no solo `Ok(..)`: hoy un 404 llega como `Err` y cae
        // en `Ok(None)` --- «no esta», que es la respuesta correcta. Detras del
        // puerto el 404 es un `Ok`, y como el parseo de abajo esta hecho de
        // `unwrap_or_default()`, leeria el cuerpo del error y devolveria
        // `Some(StoredVector { vector: [], metadata: {} })`: un vector
        // INVENTADO, con embedding vacio, por el que una busqueda puntuaria.
        match UreqClient.send(&HttpRequest::get(&url)) {
            Ok(resp) if resp.is_success() => {
                let json: serde_json::Value = resp.json().map_err(|e| {
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
            // El no-2xx va por aqui, junto al fallo de transporte: las dos
            // cosas significan «no hay vector que devolver».
            Ok(_) | Err(_) => Ok(None),
        }
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        Ok(self.delete_outcome(id).removed_something())
    }

    fn delete_detailed(&mut self, id: &str) -> AiResult<DeleteOutcome> {
        Ok(self.delete_outcome(id))
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
        Ok(probe_is_healthy(&HttpRequest::get(&url)))
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

// Note: PineconeClient, ChromaClient, MilvusClient, WeaviateClient,
// RedisVectorClient and ElasticsearchClient are all auto-`Send + Sync` —
// every field is `String` / `Option<String>` / `VectorDbConfig` /
// `HashMap<String, StoredVector>`, which are themselves `Send + Sync`.
// The manual `unsafe impl`s that used to live here were unnecessary and
// have been removed (no `unsafe` needed to share these across threads).

#[cfg(test)]
mod tests {
    use super::*;

    /// Un no-2xx en `get` NO puede inventarse un vector.
    ///
    /// Es el sitio mas peligroso de migrar este modulo. Antes del puerto, un
    /// 404 llegaba como `Err` y caia en `Ok(None)` --- «no esta», que es la
    /// respuesta correcta. Detras del puerto el 404 es un `Ok`, y el parseo de
    /// `get` esta hecho de `unwrap_or_default()`: leeria el cuerpo del error,
    /// no encontraria ni `vector` ni `metadata`, y devolveria
    /// `Some(StoredVector { vector: [], metadata: {} })`.
    ///
    /// O sea un vector que **no existe**, con embedding vacio, por el que una
    /// busqueda semantica puntuaria --- y `get` diria `Some`, que es lo que el
    /// llamante usa para decidir si hace falta volver a indexar.
    ///
    /// **Hay tres** `get` con esta guarda escrita en linea, uno por backend, y
    /// por eso hay tres tests y no uno. La leccion es de V432, dos horas antes:
    /// un invariante repetido N veces necesita N tests, y un test escrito a
    /// mano para uno de los sitios se lee como cobertura del invariante
    /// cubriendo 1/N. (Los sondeos de salud no necesitan esto porque su guarda
    /// SI esta extraida, en `probe_is_healthy`: una ocurrencia, un test.)
    fn assert_a_404_is_absence_and_not_an_empty_vector(
        make: impl FnOnce(String) -> Box<dyn VectorDb>,
    ) {
        let server = crate::http_client::MockHttpServer::start();
        // El cuerpo parece una respuesta BUENA --- lleva `vector`, `metadata` y
        // el `_source` que usa Elasticsearch --- y el estado dice 404. Es el
        // caso separador, y hizo falta: con un cuerpo de error normal la
        // mutacion M12 SOBREVIVIA en Elasticsearch, porque su parseo mira si
        // existe `_source` y un cuerpo de error no lo tiene. El test pasaba por
        // el camino sano y no podia fallar.
        //
        // Lo que se afirma asi es el invariante entero: un no-2xx no devuelve
        // vector **ni aunque el cuerpo parezca uno**. Una pasarela o una cache
        // delante del backend pueden contestar exactamente esto.
        server.enqueue_json(
            404,
            serde_json::json!({
                "vector": [0.9, 0.9, 0.9],
                "metadata": {"se_cuela": true},
                "_source": {"vector": [0.9], "metadata": {"se_cuela": true}},
                "properties": {"se_cuela": true}
            }),
        );

        let store = make(server.url());
        let found = store.get("no-existe").expect("un 404 no es un fallo");

        assert!(
            found.is_none(),
            "un 404 significa que no esta, no un vector vacio: {:?}",
            found
        );
    }

    #[test]
    fn test_weaviate_get_does_not_invent_a_vector_out_of_an_error_body() {
        assert_a_404_is_absence_and_not_an_empty_vector(|url| {
            Box::new(WeaviateClient::new(url, VectorDbConfig::default()))
        });
    }

    #[test]
    fn test_redis_get_does_not_invent_a_vector_out_of_an_error_body() {
        assert_a_404_is_absence_and_not_an_empty_vector(|url| {
            Box::new(RedisVectorClient::new(url, VectorDbConfig::default()))
        });
    }

    #[test]
    fn test_elasticsearch_get_does_not_invent_a_vector_out_of_an_error_body() {
        assert_a_404_is_absence_and_not_an_empty_vector(|url| {
            Box::new(ElasticsearchClient::new(url, VectorDbConfig::default()))
        });
    }

    /// Y el otro lado de la misma moneda: un 500 no es un backend sano.
    ///
    /// Los seis sondeos de salud que hablan HTTP decian `Ok(_) => Ok(true)`, y
    /// eso valia porque `ureq` metia el no-2xx en el `Err`. Detras del puerto
    /// el `Ok(_)` casa tambien con un 500, asi que sin `is_success()` un
    /// backend que contesta «error interno» pasa por sano --- y el que elige
    /// backend se queda con el roto.
    #[test]
    fn test_health_check_does_not_call_a_500_healthy() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(500, "index corrupted");

        let store = WeaviateClient::new(server.url(), VectorDbConfig::default());
        assert!(
            !store.health_check().expect("el sondeo no propaga"),
            "un 500 no es sano"
        );
    }

    /// Lo que se GANA, y no solo lo que se conserva: el mensaje del rechazo.
    ///
    /// Antes, `send_json(..)?` daba `AiError::Other("Chroma add failed: http
    /// status: 400")` y el motivo se perdia, porque en `ureq` 3 un
    /// `Error::StatusCode` no lleva respuesta. Ahora el cuerpo viaja en el
    /// mensaje, que es donde el backend explica QUE rechazo.
    #[test]
    fn test_a_rejection_carries_the_body_that_explains_it() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(400, "dimension mismatch: expected 1536, got 3");

        let mut store = ChromaClient::new(server.url(), VectorDbConfig::default());
        let err = store
            .insert("v1", vec![0.1, 0.2, 0.3], serde_json::json!({}))
            .expect_err("un 400 sigue siendo un error");

        let text = err.to_string();
        assert!(
            text.contains("dimension mismatch"),
            "el cuerpo tiene que llegar al mensaje: {}",
            text
        );
        assert!(text.contains("400"), "y el estado tambien: {}", text);
    }

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
        let result = db.insert("v1", vec![0.5; 384], serde_json::json!({}));
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
    // ========================================================================
    // DeleteOutcome (N158)
    // ========================================================================

    /// El defecto del tipo: lo que `delete()` devuelve sale de aqui, y tiene que
    /// significar lo mismo que significaba antes.
    #[test]
    fn test_delete_outcome_maps_to_the_bool_delete_always_returned() {
        let confirmed = |server, removed_locally| DeleteOutcome {
            server,
            removed_locally,
        };
        assert!(confirmed(DeleteConfirmation::Deleted, false).removed_something());
        assert!(!confirmed(DeleteConfirmation::NotFound, false).removed_something());
        // Y el caso ORTOGONAL, que es el que tumbo la primera version: el
        // servidor dice que no esta y la copia local SI se quito.
        assert!(confirmed(DeleteConfirmation::NotFound, true).removed_something());
        let unconfirmed = DeleteConfirmation::Unconfirmed { reason: "x".into() };
        assert!(confirmed(unconfirmed.clone(), true).removed_something());
        assert!(!confirmed(unconfirmed, false).removed_something());
    }

    /// Y la pregunta que un booleano NO podia contestar.
    #[test]
    fn test_only_local_only_is_unconfirmed() {
        let of = |server| DeleteOutcome {
            server,
            removed_locally: true,
        };
        assert!(of(DeleteConfirmation::Deleted).confirmed_by_server());
        assert!(of(DeleteConfirmation::NotFound).confirmed_by_server());
        let unconfirmed = of(DeleteConfirmation::Unconfirmed {
            reason: "connection refused".into(),
        });
        assert!(!unconfirmed.confirmed_by_server());
        assert_eq!(unconfirmed.unconfirmed_reason(), Some("connection refused"));
        assert_eq!(of(DeleteConfirmation::Deleted).unconfirmed_reason(), None);
    }

    /// **El caso que motivo la tarea.** Con el servidor inalcanzable, el borrado
    /// NO esta confirmado --- y antes de N158 esto devolvia `Ok(true)`, o sea
    /// «borrado», con el vector todavia en el indice.
    #[test]
    fn test_a_dead_server_is_local_only_and_says_why() {
        // Puerto 1: reservado, nada escucha ahi.
        let mut db =
            ElasticsearchClient::new("http://127.0.0.1:1".to_string(), VectorDbConfig::default());
        // La cache se llena a mano, como hace `test_weaviate_cache_operations`:
        // un `insert` contra un puerto muerto falla y la dejaria vacia, y
        // entonces este test no probaria `delete` sino `insert`.
        db.cache.insert(
            "a".to_string(),
            StoredVector {
                id: "a".to_string(),
                vector: vec![0.5; 384],
                metadata: serde_json::json!({}),
                timestamp: 0,
            },
        );

        let outcome = db.delete_detailed("a").expect("el tipo informa, no falla");
        assert!(
            !outcome.confirmed_by_server(),
            "un servidor muerto no confirma nada: {:?}",
            outcome
        );
        let reason = outcome
            .unconfirmed_reason()
            .expect("un LocalOnly trae su motivo");
        assert!(!reason.trim().is_empty(), "el motivo no puede estar vacio");
        // Y la copia local SI se quito, que es lo que `delete()` reporta.
        assert!(outcome.removed_something());
    }

    /// El caso SEPARADOR del de arriba: un 404 de un servidor que SI contesta es
    /// `NotFound`, no `LocalOnly`. Sin este test, «siempre LocalOnly» pasaria.
    #[test]
    fn test_a_404_from_a_live_server_is_not_found_not_local_only() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(404, r#"{"result":"not_found"}"#);
        let mut db = ElasticsearchClient::new(server.url(), VectorDbConfig::default());

        let outcome = db.delete_detailed("missing").expect("contesta");
        assert_eq!(
            outcome.server,
            DeleteConfirmation::NotFound,
            "un 404 es el servidor diciendo que no esta"
        );
        assert!(outcome.confirmed_by_server());
        assert!(!outcome.removed_something());
    }

    /// Y el tercer caso: 2xx es confirmacion.
    #[test]
    fn test_a_2xx_is_a_confirmed_delete() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(200, r#"{"result":"deleted"}"#);
        let mut db = ElasticsearchClient::new(server.url(), VectorDbConfig::default());

        let outcome = db.delete_detailed("a").expect("contesta");
        assert_eq!(outcome.server, DeleteConfirmation::Deleted);
        assert!(outcome.confirmed_by_server());
    }

    /// Un no-2xx que NO es 404 --- un 401, lo que pasa con una clave caducada ---
    /// es `LocalOnly`, y el cuerpo viaja en el motivo porque es donde el backend
    /// explica el rechazo.
    #[test]
    fn test_a_401_carries_its_body_into_the_reason() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(500, "index is read-only");
        let mut db = ElasticsearchClient::new(server.url(), VectorDbConfig::default());

        let outcome = db.delete_detailed("a").expect("contesta");
        let reason = outcome
            .unconfirmed_reason()
            .unwrap_or_else(|| panic!("un 500 no es confirmacion: {:?}", outcome));
        assert!(reason.contains("500"), "reason = {}", reason);
        assert!(
            reason.contains("read-only"),
            "el cuerpo es donde el backend explica el rechazo: {}",
            reason
        );
    }

    /// `delete()` sigue devolviendo lo que devolvia. Es la mitad del contrato que
    /// no debe moverse: los llamantes hacen `db.delete(id).unwrap()` sobre un
    /// `bool` y ninguno se toca.
    #[test]
    fn test_delete_still_answers_the_same_bool() {
        let server = crate::http_client::MockHttpServer::start();
        server.enqueue_text(200, "{}");
        let mut db = ElasticsearchClient::new(server.url(), VectorDbConfig::default());
        db.insert("a", vec![0.5; 384], serde_json::json!({})).ok();
        assert!(db.delete("a").expect("borra"), "habia copia local");
    }

    /// Una implementacion SIN servidor remoto hereda el defecto del trait, y el
    /// defecto es correcto ahi: lo que dice `delete()` es toda la verdad.
    #[test]
    fn test_an_in_memory_db_inherits_the_default_and_never_says_local_only() {
        let mut db = InMemoryVectorDb::new(VectorDbConfig::default());
        db.insert("a", vec![0.5; 384], serde_json::json!({}))
            .expect("insert");

        let first = db.delete_detailed("a").expect("borra");
        assert_eq!(first.server, DeleteConfirmation::Deleted);
        assert!(first.removed_locally);
        let second = db.delete_detailed("a").expect("ya no esta");
        assert_eq!(second.server, DeleteConfirmation::NotFound);
        assert!(!second.removed_locally);
    }
}
