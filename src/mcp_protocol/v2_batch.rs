//! MCP JSON-RPC Batching.

use serde::{Deserialize, Serialize};

use super::types::McpError;

/// A standalone JSON-RPC 2.0 request used within batch operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: serde_json::Value, method: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// A standalone JSON-RPC 2.0 response used within batch operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, error: McpError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// Configuration for a batch of JSON-RPC requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub parallel_execution: bool,
    pub timeout_per_request_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            parallel_execution: true,
            timeout_per_request_ms: 30_000,
        }
    }
}

/// A batch of JSON-RPC requests with associated configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    pub requests: Vec<JsonRpcRequest>,
    pub config: BatchConfig,
}

/// The aggregated result of executing a batch of JSON-RPC requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub responses: Vec<JsonRpcResponse>,
    pub total_duration_ms: u64,
    pub errors: usize,
}

/// Executor that validates, creates, and correlates JSON-RPC batches.
pub struct BatchExecutor {
    config: BatchConfig,
}

impl BatchExecutor {
    /// Create a `BatchExecutor` with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Create a `BatchExecutor` using `BatchConfig::default()`.
    pub fn with_defaults() -> Self {
        Self {
            config: BatchConfig::default(),
        }
    }

    /// Validate that a batch meets the configured constraints.
    ///
    /// Returns `Err` if the batch is empty or exceeds `max_batch_size`.
    pub fn validate_batch(&self, batch: &BatchRequest) -> Result<(), String> {
        if batch.requests.is_empty() {
            return Err("Batch must contain at least one request".to_string());
        }
        if batch.requests.len() > self.config.max_batch_size {
            return Err(format!(
                "Batch size {} exceeds maximum of {}",
                batch.requests.len(),
                self.config.max_batch_size
            ));
        }
        Ok(())
    }

    /// Package a list of requests into a `BatchRequest` using the executor's config.
    pub fn create_batch(&self, requests: Vec<JsonRpcRequest>) -> BatchRequest {
        BatchRequest {
            requests,
            config: self.config.clone(),
        }
    }

    /// Correlate a list of responses to the original batch, computing summary stats.
    ///
    /// Responses are matched to requests by their JSON-RPC `id` field. Any response
    /// whose `error` field is `Some` is counted as an error.
    pub fn correlate_responses(
        &self,
        batch: &BatchRequest,
        responses: Vec<JsonRpcResponse>,
    ) -> BatchResponse {
        // Build a set of request IDs for correlation
        let request_ids: std::collections::HashSet<String> = batch
            .requests
            .iter()
            .map(|r| r.id.to_string())
            .collect();

        // Filter to only responses whose id matches a request in the batch
        let correlated: Vec<JsonRpcResponse> = responses
            .into_iter()
            .filter(|r| request_ids.contains(&r.id.to_string()))
            .collect();

        let errors = correlated.iter().filter(|r| r.error.is_some()).count();

        BatchResponse {
            responses: correlated,
            total_duration_ms: 0, // Caller should set real timing
            errors,
        }
    }

    /// Access the executor's configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}
