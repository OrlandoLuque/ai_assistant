//! HTTP client abstraction for provider communication.
//!
//! Defines a trait `HttpClient` that abstracts HTTP operations, allowing
//! the provider layer to be tested with mock clients. The default
//! implementation `UreqClient` wraps the `ureq` HTTP library.
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::http_client::{HttpClient, UreqClient};
//!
//! let client = UreqClient;
//! let result = client.get_json("http://localhost:11434/api/tags", 5);
//! ```

use anyhow::{Context, Result};

/// Abstraction over HTTP operations for provider communication.
///
/// Implementing this trait allows swapping the real HTTP client with a mock
/// for testing provider parsing logic without network access.
pub trait HttpClient: Send + Sync {
    /// Perform a GET request and return the parsed JSON response.
    fn get_json(&self, url: &str, timeout_secs: u64) -> Result<serde_json::Value>;

    /// Perform a POST request with a JSON body and return the parsed JSON response.
    fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<serde_json::Value>;

    /// Perform a POST request and return a streaming reader for the response body.
    ///
    /// Used for streaming generation responses where the server sends chunks
    /// incrementally (e.g., Ollama streaming, OpenAI SSE).
    fn post_streaming(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<Box<dyn std::io::Read + Send>>;
}

/// Default HTTP client using the `ureq` library.
///
/// This is the production client used by all provider functions.
/// Wraps `ureq::get()` and `ureq::post()` with configurable timeouts.
pub struct UreqClient;

impl HttpClient for UreqClient {
    fn get_json(&self, url: &str, timeout_secs: u64) -> Result<serde_json::Value> {
        let response = ureq::get(url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .call()
            .with_context(|| format!("GET request failed: {}", url))?;
        response.into_json()
            .context("Failed to parse JSON response")
    }

    fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        let response = ureq::post(url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .send_json(body)
            .with_context(|| format!("POST request failed: {}", url))?;
        response.into_json()
            .context("Failed to parse JSON response")
    }

    fn post_streaming(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<Box<dyn std::io::Read + Send>> {
        let response = ureq::post(url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .send_json(body)
            .with_context(|| format!("POST streaming request failed: {}", url))?;
        Ok(Box::new(response.into_reader()))
    }
}

// ============================================================================
// Parsing functions (shared between real and mock clients)
// ============================================================================

use crate::config::AiProvider;
use crate::models::{format_size, ModelInfo};

/// Parse an Ollama `/api/tags` response into a list of ModelInfo.
///
/// This is the core parsing logic extracted from `fetch_ollama_models` so it
/// can be tested with mock data without network access.
pub fn parse_ollama_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(model_list) = body.get("models").and_then(|m| m.as_array()) {
        for model in model_list {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                models.push(ModelInfo {
                    name: name.to_string(),
                    provider: AiProvider::Ollama,
                    size: model.get("size").and_then(|s| s.as_u64()).map(format_size),
                    modified_at: model.get("modified_at").and_then(|m| m.as_str()).map(|s| s.to_string()),
                });
            }
        }
    }
    models
}

/// Parse an OpenAI-compatible `/v1/models` response into a list of ModelInfo.
pub fn parse_openai_models(body: &serde_json::Value, provider: AiProvider) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        for model in data {
            if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                models.push(ModelInfo {
                    name: id.to_string(),
                    provider: provider.clone(),
                    size: None,
                    modified_at: None,
                });
            }
        }
    }
    models
}

/// Parse a Kobold.cpp `/api/v1/model` response into a list of ModelInfo.
pub fn parse_kobold_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(result) = body.get("result").and_then(|r| r.as_str()) {
        if !result.is_empty() && result != "Read Only" {
            models.push(ModelInfo {
                name: result.to_string(),
                provider: AiProvider::KoboldCpp,
                size: None,
                modified_at: None,
            });
        }
    }
    models
}

/// Fetch Ollama models using a provided HttpClient.
///
/// This is the testable version that accepts any HttpClient implementation.
/// The public `fetch_ollama_models` in providers.rs delegates to this.
pub fn fetch_ollama_models_with(client: &dyn HttpClient, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    let body = client.get_json(&url, 5)
        .context("Failed to connect to Ollama")?;
    Ok(parse_ollama_models(&body))
}

/// Fetch OpenAI-compatible models using a provided HttpClient.
pub fn fetch_openai_models_with(
    client: &dyn HttpClient,
    base_url: &str,
    provider: AiProvider,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url);
    let body = client.get_json(&url, 5)
        .context("Failed to connect to OpenAI-compatible API")?;
    Ok(parse_openai_models(&body, provider))
}

/// Fetch Kobold.cpp models using a provided HttpClient.
pub fn fetch_kobold_models_with(client: &dyn HttpClient, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);
    let body = client.get_json(&url, 5)
        .context("Failed to connect to Kobold.cpp")?;
    Ok(parse_kobold_models(&body))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock HTTP client for testing provider parsing without network access.
    struct MockHttpClient {
        responses: Mutex<Vec<Result<serde_json::Value>>>,
    }

    impl MockHttpClient {
        fn with_response(value: serde_json::Value) -> Self {
            Self {
                responses: Mutex::new(vec![Ok(value)]),
            }
        }

        fn with_error(msg: &str) -> Self {
            Self {
                responses: Mutex::new(vec![Err(anyhow::anyhow!("{}", msg))]),
            }
        }
    }

    impl HttpClient for MockHttpClient {
        fn get_json(&self, _url: &str, _timeout_secs: u64) -> Result<serde_json::Value> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No mock responses configured");
            }
            responses.remove(0)
        }

        fn post_json(
            &self,
            _url: &str,
            _body: &serde_json::Value,
            _timeout_secs: u64,
        ) -> Result<serde_json::Value> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No mock responses configured");
            }
            responses.remove(0)
        }

        fn post_streaming(
            &self,
            _url: &str,
            _body: &serde_json::Value,
            _timeout_secs: u64,
        ) -> Result<Box<dyn std::io::Read + Send>> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No mock responses configured");
            }
            let value = responses.remove(0)?;
            let bytes = serde_json::to_vec(&value)?;
            Ok(Box::new(std::io::Cursor::new(bytes)))
        }
    }

    #[test]
    fn test_parse_ollama_models_response() {
        let body = serde_json::json!({
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 4_000_000_000u64,
                    "modified_at": "2024-01-15T10:30:00Z"
                },
                {
                    "name": "qwen2.5:7b",
                    "size": 7_000_000_000u64,
                    "modified_at": "2024-02-01T08:00:00Z"
                }
            ]
        });

        let models = parse_ollama_models(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "llama3.2:latest");
        assert_eq!(models[0].provider, AiProvider::Ollama);
        assert!(models[0].size.is_some());
        assert_eq!(models[1].name, "qwen2.5:7b");
    }

    #[test]
    fn test_parse_ollama_empty_response() {
        let body = serde_json::json!({ "models": [] });
        let models = parse_ollama_models(&body);
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_openai_models_response() {
        let body = serde_json::json!({
            "data": [
                { "id": "gpt-4", "object": "model" },
                { "id": "gpt-3.5-turbo", "object": "model" }
            ]
        });

        let models = parse_openai_models(&body, AiProvider::LMStudio);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "gpt-4");
        assert_eq!(models[0].provider, AiProvider::LMStudio);
        assert_eq!(models[1].name, "gpt-3.5-turbo");
    }

    #[test]
    fn test_parse_kobold_models_response() {
        let body = serde_json::json!({ "result": "mistral-7b-instruct" });
        let models = parse_kobold_models(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "mistral-7b-instruct");
        assert_eq!(models[0].provider, AiProvider::KoboldCpp);
    }

    #[test]
    fn test_parse_kobold_read_only_ignored() {
        let body = serde_json::json!({ "result": "Read Only" });
        let models = parse_kobold_models(&body);
        assert!(models.is_empty());
    }

    #[test]
    fn test_fetch_ollama_with_mock() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "models": [
                { "name": "test-model:latest", "size": 1000000 }
            ]
        }));

        let models = fetch_ollama_models_with(&mock, "http://localhost:11434").unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test-model:latest");
    }

    #[test]
    fn test_fetch_openai_with_mock() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "data": [
                { "id": "local-model-1" },
                { "id": "local-model-2" }
            ]
        }));

        let models = fetch_openai_models_with(&mock, "http://localhost:1234", AiProvider::LMStudio).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_fetch_with_connection_error() {
        let mock = MockHttpClient::with_error("Connection refused");
        let result = fetch_ollama_models_with(&mock, "http://localhost:11434");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Ollama") || err.contains("Connection refused"));
    }

    #[test]
    fn test_mock_streaming_response() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "message": { "content": "Hello" },
            "done": false
        }));

        let reader = mock.post_streaming(
            "http://localhost:11434/api/chat",
            &serde_json::json!({}),
            300,
        ).unwrap();

        let mut content = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut content).unwrap();
        assert!(content.contains("Hello"));
    }

    #[test]
    fn test_parse_malformed_json_returns_empty() {
        // Missing "models" key
        let body = serde_json::json!({ "error": "bad request" });
        assert!(parse_ollama_models(&body).is_empty());

        // Missing "data" key
        assert!(parse_openai_models(&body, AiProvider::LMStudio).is_empty());

        // Missing "result" key
        assert!(parse_kobold_models(&body).is_empty());
    }
}
