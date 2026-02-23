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
        response
            .into_json()
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
        response
            .into_json()
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
                    modified_at: model
                        .get("modified_at")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string()),
                    capabilities: None,
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
                    capabilities: None,
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
                capabilities: None,
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
    let body = client
        .get_json(&url, 5)
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
    let body = client
        .get_json(&url, 5)
        .context("Failed to connect to OpenAI-compatible API")?;
    Ok(parse_openai_models(&body, provider))
}

/// Fetch Kobold.cpp models using a provided HttpClient.
pub fn fetch_kobold_models_with(client: &dyn HttpClient, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);
    let body = client
        .get_json(&url, 5)
        .context("Failed to connect to Kobold.cpp")?;
    Ok(parse_kobold_models(&body))
}

// ============================================================================
// Feature flag validation macro
// ============================================================================

/// Macro that validates feature flag combinations at compile time.
/// Produces a clear compilation error if invalid combinations are used.
///
/// Usage in lib.rs (added during wiring):
/// ```rust,ignore
/// validate_features!();
/// ```
#[macro_export]
macro_rules! validate_features {
    () => {
        // webrtc requires voice-agent
        #[cfg(all(feature = "webrtc", not(feature = "voice-agent")))]
        compile_error!(
            "Feature 'webrtc' requires 'voice-agent'. Add voice-agent to your Cargo.toml features."
        );

        // browser requires autonomous
        #[cfg(all(feature = "browser", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'browser' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // scheduler requires autonomous
        #[cfg(all(feature = "scheduler", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'scheduler' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // butler requires autonomous
        #[cfg(all(feature = "butler", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'butler' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // distributed-agents requires autonomous + distributed-network
        #[cfg(all(feature = "distributed-agents", not(feature = "autonomous")))]
        compile_error!("Feature 'distributed-agents' requires 'autonomous'.");

        #[cfg(all(feature = "distributed-agents", not(feature = "distributed-network")))]
        compile_error!("Feature 'distributed-agents' requires 'distributed-network'.");

        // whisper-local requires audio
        #[cfg(all(feature = "whisper-local", not(feature = "audio")))]
        compile_error!("Feature 'whisper-local' requires 'audio'.");
    };
}

// ============================================================================
// Mock HTTP Server (test-only)
// ============================================================================

/// A mock HTTP server that listens on localhost and serves queued responses.
/// Used for testing modules that make real HTTP calls (mcp_client, media_generation, etc.)
///
/// # Example
/// ```rust,no_run
/// let mut server = MockHttpServer::start();
/// server.enqueue_json(200, serde_json::json!({"result": "ok"}));
/// // Use server.url() as the base URL for the module under test
/// let url = server.url();
/// // ... make HTTP call to url ...
/// let received = server.last_request();
/// ```
#[cfg(test)]
pub struct MockHttpServer {
    addr: std::net::SocketAddr,
    /// Queued responses: (status_code, content_type, body)
    responses: std::sync::Arc<std::sync::Mutex<Vec<(u16, String, String)>>>,
    /// Received requests: (method, path, body)
    requests: std::sync::Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    /// Background thread handle
    _handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(test)]
impl MockHttpServer {
    /// Start a new mock server on a random port.
    pub fn start() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let responses = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        let resp_clone = responses.clone();
        let req_clone = requests.clone();

        let handle = std::thread::spawn(move || {
            // Set non-blocking with short timeout so the thread can exit
            listener.set_nonblocking(false).ok();

            // Accept connections in a loop
            for stream in listener.incoming() {
                match stream {
                    Ok(mut stream) => {
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                            .ok();

                        // Read HTTP request
                        let mut buf = vec![0u8; 8192];
                        let n = match std::io::Read::read(&mut stream, &mut buf) {
                            Ok(n) => n,
                            Err(_) => continue,
                        };
                        let request_str = String::from_utf8_lossy(&buf[..n]).to_string();

                        // Parse method, path, body
                        let lines: Vec<&str> = request_str.split("\r\n").collect();
                        let first_line = lines.first().unwrap_or(&"");
                        let parts: Vec<&str> = first_line.split(' ').collect();
                        let method = parts.first().unwrap_or(&"GET").to_string();
                        let path = parts.get(1).unwrap_or(&"/").to_string();

                        // Body is after the empty line
                        let body = if let Some(pos) = request_str.find("\r\n\r\n") {
                            request_str[pos + 4..].to_string()
                        } else {
                            String::new()
                        };

                        req_clone.lock().unwrap().push((method, path, body));

                        // Get next response from queue
                        let (status, content_type, resp_body) = {
                            let mut q = resp_clone.lock().unwrap();
                            if q.is_empty() {
                                (
                                    404,
                                    "text/plain".to_string(),
                                    "No responses queued".to_string(),
                                )
                            } else {
                                q.remove(0)
                            }
                        };

                        // Map status code to reason phrase
                        let reason = match status {
                            200 => "OK",
                            201 => "Created",
                            204 => "No Content",
                            400 => "Bad Request",
                            404 => "Not Found",
                            500 => "Internal Server Error",
                            _ => "Unknown",
                        };

                        // Write HTTP response
                        let response = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            status, reason, content_type, resp_body.len(), resp_body
                        );
                        std::io::Write::write_all(&mut stream, response.as_bytes()).ok();
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            addr,
            responses,
            requests,
            _handle: Some(handle),
        }
    }

    /// Get the server's base URL (e.g., "http://127.0.0.1:12345")
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Enqueue a JSON response
    pub fn enqueue_json(&self, status: u16, body: serde_json::Value) {
        self.responses.lock().unwrap().push((
            status,
            "application/json".to_string(),
            body.to_string(),
        ));
    }

    /// Enqueue a plain text response
    pub fn enqueue_text(&self, status: u16, body: &str) {
        self.responses.lock().unwrap().push((
            status,
            "text/plain".to_string(),
            body.to_string(),
        ));
    }

    /// Get the last received request (method, path, body)
    pub fn last_request(&self) -> Option<(String, String, String)> {
        self.requests.lock().unwrap().last().cloned()
    }

    /// Get all received requests
    pub fn all_requests(&self) -> Vec<(String, String, String)> {
        self.requests.lock().unwrap().clone()
    }

    /// Get the number of requests received
    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
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

        let models =
            fetch_openai_models_with(&mock, "http://localhost:1234", AiProvider::LMStudio).unwrap();
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

        let reader = mock
            .post_streaming(
                "http://localhost:11434/api/chat",
                &serde_json::json!({}),
                300,
            )
            .unwrap();

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

    // ========================================================================
    // MockHttpServer tests (Item 10.1 + 10.2)
    // ========================================================================

    #[test]
    fn test_mock_server_start() {
        let server = MockHttpServer::start();
        let url = server.url();
        assert!(url.starts_with("http://127.0.0.1:"));
        // Port should be > 0
        let port_str = url.rsplit(':').next().unwrap();
        let port: u16 = port_str.parse().unwrap();
        assert!(port > 0);
    }

    #[test]
    fn test_mock_server_basic_json_response() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"status": "ok"}));

        let client = UreqClient;
        let result = client.get_json(&format!("{}/health", server.url()), 5);
        assert!(result.is_ok());
        let body = result.unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[test]
    fn test_mock_server_post_json() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"id": 1}));

        let client = UreqClient;
        let result = client.post_json(
            &format!("{}/api/chat", server.url()),
            &serde_json::json!({"message": "hello"}),
            5,
        );
        assert!(result.is_ok());

        let (method, path, _body) = server.last_request().unwrap();
        assert_eq!(method, "POST");
        assert!(path.contains("/api/chat"));
    }

    #[test]
    fn test_mock_server_error_response() {
        let server = MockHttpServer::start();
        server.enqueue_json(500, serde_json::json!({"error": "internal"}));

        let client = UreqClient;
        let result = client.get_json(&format!("{}/fail", server.url()), 5);
        // ureq treats 500 as an error
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_server_multiple_requests() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"n": 1}));
        server.enqueue_json(200, serde_json::json!({"n": 2}));

        let client = UreqClient;
        let r1 = client
            .get_json(&format!("{}/a", server.url()), 5)
            .unwrap();
        let r2 = client
            .get_json(&format!("{}/b", server.url()), 5)
            .unwrap();

        assert_eq!(r1["n"], 1);
        assert_eq!(r2["n"], 2);
        assert_eq!(server.request_count(), 2);
    }

    #[test]
    fn test_mock_server_no_response_queued() {
        let server = MockHttpServer::start();
        // Don't enqueue anything — server returns 404
        let client = UreqClient;
        let result = client.get_json(&format!("{}/empty", server.url()), 5);
        // Should get 404 or error
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_server_text_response() {
        let server = MockHttpServer::start();
        server.enqueue_text(200, "Hello, world!");

        // Use a raw TCP connection to verify the text response
        let mut stream =
            std::net::TcpStream::connect(server.addr).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .ok();
        let request = "GET /text HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        std::io::Write::write_all(&mut stream, request.as_bytes()).unwrap();

        let mut response = String::new();
        std::io::Read::read_to_string(&mut stream, &mut response).unwrap();
        assert!(response.contains("Hello, world!"));
        assert!(response.contains("text/plain"));
    }

    #[test]
    fn test_mock_server_request_count() {
        let server = MockHttpServer::start();
        assert_eq!(server.request_count(), 0);

        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));

        let client = UreqClient;
        let _ = client.get_json(&format!("{}/1", server.url()), 5);
        assert_eq!(server.request_count(), 1);

        let _ = client.get_json(&format!("{}/2", server.url()), 5);
        assert_eq!(server.request_count(), 2);

        let _ = client.get_json(&format!("{}/3", server.url()), 5);
        assert_eq!(server.request_count(), 3);
    }

    #[test]
    fn test_mock_server_all_requests() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));

        let client = UreqClient;
        let _ = client.get_json(&format!("{}/first", server.url()), 5);
        let _ = client.post_json(
            &format!("{}/second", server.url()),
            &serde_json::json!({"key": "value"}),
            5,
        );

        let requests = server.all_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].0, "GET");
        assert!(requests[0].1.contains("/first"));
        assert_eq!(requests[1].0, "POST");
        assert!(requests[1].1.contains("/second"));
    }

    #[test]
    fn test_mock_server_post_streaming() {
        let server = MockHttpServer::start();
        server.enqueue_json(
            200,
            serde_json::json!({"message": {"content": "streamed"}, "done": false}),
        );

        let client = UreqClient;
        let reader = client
            .post_streaming(
                &format!("{}/api/chat", server.url()),
                &serde_json::json!({"model": "test", "stream": true}),
                5,
            )
            .unwrap();

        let mut content = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut content).unwrap();
        assert!(content.contains("streamed"));

        let (method, path, _) = server.last_request().unwrap();
        assert_eq!(method, "POST");
        assert!(path.contains("/api/chat"));
    }

    // ========================================================================
    // Feature flag validation macro test (Item 10.3)
    // ========================================================================

    #[test]
    fn test_validate_features_macro_exists() {
        // This test verifies the macro compiles.
        // Invalid feature combinations would cause compile errors.
        validate_features!();
    }
}
