// async_providers.rs — Async HTTP client and provider functions using reqwest + tokio.
//
// Gated behind `#[cfg(feature = "async-runtime")]`. Provides:
// - `AsyncHttpClient` trait mirroring `HttpClient` but async
// - `ReqwestClient` implementation using reqwest
// - Async model fetching for all providers
// - Async generation (non-streaming and streaming)
// - Blocking bridge for calling async code from sync contexts

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use std::future::Future;
use std::pin::Pin;

use crate::config::{AiConfig, AiProvider};
use crate::http_client::{parse_kobold_models, parse_ollama_models, parse_openai_models};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;

// ============================================================================
// Async HTTP Client Trait
// ============================================================================

/// Async version of [`HttpClient`](crate::http_client::HttpClient).
///
/// Provides non-blocking HTTP operations using `async`/`await`.
/// The default implementation [`ReqwestClient`] uses the `reqwest` library.
pub trait AsyncHttpClient: Send + Sync {
    /// Perform an async GET request and return parsed JSON.
    fn get_json(
        &self,
        url: &str,
        timeout_secs: u64,
    ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>>;

    /// Perform an async POST request with JSON body, return parsed JSON.
    fn post_json(
        &self,
        url: &str,
        body: &JsonValue,
        timeout_secs: u64,
    ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>>;
}

// ============================================================================
// Reqwest Client
// ============================================================================

/// Async HTTP client using `reqwest`.
///
/// Supports connection pooling, timeouts, and TLS.
/// All operations are truly async, backed by tokio.
#[derive(Clone)]
pub struct ReqwestClient {
    client: reqwest::Client,
    /// Optional API key to include as Bearer token.
    pub api_key: Option<String>,
}

impl ReqwestClient {
    /// Create a new client with default settings.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: None,
        }
    }

    /// Create a client with a custom reqwest::Client (e.g. for proxy settings).
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            api_key: None,
        }
    }

    /// Set an API key for Authorization header.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            builder.header("Authorization", format!("Bearer {}", key))
        } else {
            builder
        }
    }
}

impl Default for ReqwestClient {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncHttpClient for ReqwestClient {
    fn get_json(
        &self,
        url: &str,
        timeout_secs: u64,
    ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>> {
        let url = url.to_string();
        Box::pin(async move {
            let request = self
                .client
                .get(&url)
                .timeout(std::time::Duration::from_secs(timeout_secs));
            let request = self.apply_auth(request);
            let response = request
                .send()
                .await
                .with_context(|| format!("GET request failed: {}", url))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("HTTP {} from {}: {}", status, url, body);
            }
            response
                .json::<JsonValue>()
                .await
                .context("Failed to parse JSON response")
        })
    }

    fn post_json(
        &self,
        url: &str,
        body: &JsonValue,
        timeout_secs: u64,
    ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>> {
        let url = url.to_string();
        let body = body.clone();
        Box::pin(async move {
            let request = self
                .client
                .post(&url)
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .json(&body);
            let request = self.apply_auth(request);
            let response = request
                .send()
                .await
                .with_context(|| format!("POST request failed: {}", url))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("HTTP {} from {}: {}", status, url, body);
            }
            response
                .json::<JsonValue>()
                .await
                .context("Failed to parse JSON response")
        })
    }
}

// ============================================================================
// Async Provider Functions
// ============================================================================

/// Fetch models from any provider asynchronously.
///
/// Dispatches to the correct API endpoint based on the provider type.
pub async fn fetch_models_async(
    client: &dyn AsyncHttpClient,
    config: &AiConfig,
) -> Result<Vec<ModelInfo>> {
    let base_url = config.get_base_url();
    match config.provider {
        AiProvider::Ollama => fetch_ollama_models_async(client, &base_url).await,
        AiProvider::KoboldCpp => fetch_kobold_models_async(client, &base_url).await,
        _ => fetch_openai_models_async(client, &base_url, config.provider.clone()).await,
    }
}

/// Fetch Ollama models asynchronously.
pub async fn fetch_ollama_models_async(
    client: &dyn AsyncHttpClient,
    base_url: &str,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    let body = client
        .get_json(&url, 5)
        .await
        .context("Failed to connect to Ollama")?;
    Ok(parse_ollama_models(&body))
}

/// Fetch OpenAI-compatible models asynchronously.
pub async fn fetch_openai_models_async(
    client: &dyn AsyncHttpClient,
    base_url: &str,
    provider: AiProvider,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url);
    let body = client
        .get_json(&url, 5)
        .await
        .context("Failed to connect to OpenAI-compatible API")?;
    Ok(parse_openai_models(&body, provider))
}

/// Fetch Kobold.cpp models asynchronously.
pub async fn fetch_kobold_models_async(
    client: &dyn AsyncHttpClient,
    base_url: &str,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);
    let body = client
        .get_json(&url, 5)
        .await
        .context("Failed to connect to Kobold.cpp")?;
    Ok(parse_kobold_models(&body))
}

/// Generate a non-streaming response asynchronously.
///
/// Sends the message to the configured provider and waits for the full response.
/// Returns `AiResponse::Complete(text)` on success.
pub async fn generate_response_async(
    client: &dyn AsyncHttpClient,
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<AiResponse> {
    let base_url = config.get_base_url();
    let model = &config.selected_model;

    match config.provider {
        AiProvider::Ollama => {
            generate_ollama_async(client, &base_url, model, messages, system_prompt).await
        }
        _ => generate_openai_async(client, &base_url, model, messages, system_prompt).await,
    }
}

async fn generate_ollama_async(
    client: &dyn AsyncHttpClient,
    base_url: &str,
    model: &str,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<AiResponse> {
    let url = format!("{}/api/chat", base_url);

    let mut api_messages = Vec::new();
    if !system_prompt.is_empty() {
        api_messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt,
        }));
    }
    for msg in messages {
        api_messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content,
        }));
    }

    let body = serde_json::json!({
        "model": model,
        "messages": api_messages,
        "stream": false,
    });

    let response = client
        .post_json(&url, &body, 300)
        .await
        .context("Ollama generation failed")?;

    let content = response
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(AiResponse::Complete(content))
}

async fn generate_openai_async(
    client: &dyn AsyncHttpClient,
    base_url: &str,
    model: &str,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<AiResponse> {
    let url = format!("{}/v1/chat/completions", base_url);

    let mut api_messages = Vec::new();
    if !system_prompt.is_empty() {
        api_messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt,
        }));
    }
    for msg in messages {
        api_messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content,
        }));
    }

    let body = serde_json::json!({
        "model": model,
        "messages": api_messages,
        "stream": false,
    });

    let response = client
        .post_json(&url, &body, 300)
        .await
        .context("OpenAI-compatible generation failed")?;

    let content = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(AiResponse::Complete(content))
}

/// Generate a streaming response asynchronously, calling the callback for each token.
///
/// The callback receives each text chunk. Currently uses non-streaming internally
/// and delivers the full response as one chunk (full SSE parsing is provider-specific
/// and better handled at the integration layer).
pub async fn generate_response_streaming_async(
    client: &dyn AsyncHttpClient,
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
    on_token: impl Fn(&str) + Send + Sync,
) -> Result<AiResponse> {
    let response = generate_response_async(client, config, messages, system_prompt).await?;
    if let AiResponse::Complete(ref text) = response {
        on_token(text);
    }
    Ok(response)
}

// ============================================================================
// Blocking Bridge
// ============================================================================

/// Run an async operation on a new tokio runtime.
///
/// Creates a temporary multi-threaded runtime and blocks on the future.
/// Useful for calling async provider functions from synchronous code.
///
/// # Panics
///
/// Will panic if called from within an active tokio runtime.
/// Use `tokio::spawn` or direct `.await` in those cases.
pub fn block_on_async<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    // Check if we're already in a tokio context
    if tokio::runtime::Handle::try_current().is_ok() {
        // Inside a tokio runtime — spawn a thread to avoid nested block_on
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            let result = rt.block_on(future);
            let _ = tx.send(result);
        });
        rx.recv()
            .map_err(|_| anyhow::anyhow!("Async task thread panicked"))?
    } else {
        let rt =
            tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
        rt.block_on(future)
    }
}

/// Create a shared tokio runtime for reuse across the application.
///
/// Returns a multi-threaded runtime with default settings.
pub fn create_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock async HTTP client for testing.
    struct MockAsyncClient {
        responses: Mutex<Vec<Result<JsonValue>>>,
    }

    impl MockAsyncClient {
        fn with_response(value: JsonValue) -> Self {
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

    impl AsyncHttpClient for MockAsyncClient {
        fn get_json(
            &self,
            _url: &str,
            _timeout_secs: u64,
        ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>> {
            Box::pin(async move {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    anyhow::bail!("No mock responses");
                }
                responses.remove(0)
            })
        }

        fn post_json(
            &self,
            _url: &str,
            _body: &JsonValue,
            _timeout_secs: u64,
        ) -> Pin<Box<dyn Future<Output = Result<JsonValue>> + Send + '_>> {
            Box::pin(async move {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    anyhow::bail!("No mock responses");
                }
                responses.remove(0)
            })
        }
    }

    #[tokio::test]
    async fn test_fetch_ollama_models_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "models": [
                { "name": "llama3:latest", "size": 4000000000u64 },
                { "name": "phi3:mini", "size": 2000000000u64 },
            ]
        }));

        let models = fetch_ollama_models_async(&client, "http://localhost:11434")
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "llama3:latest");
        assert_eq!(models[0].provider, AiProvider::Ollama);
    }

    #[tokio::test]
    async fn test_fetch_openai_models_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "data": [
                { "id": "model-1" },
                { "id": "model-2" },
            ]
        }));

        let models =
            fetch_openai_models_async(&client, "http://localhost:1234", AiProvider::LMStudio)
                .await
                .unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "model-1");
    }

    #[tokio::test]
    async fn test_fetch_kobold_models_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "result": "mistral-7b"
        }));

        let models = fetch_kobold_models_async(&client, "http://localhost:5001")
            .await
            .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "mistral-7b");
    }

    #[tokio::test]
    async fn test_fetch_models_async_error() {
        let client = MockAsyncClient::with_error("Connection refused");
        let result = fetch_ollama_models_async(&client, "http://localhost:11434").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_ollama_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "eval_count": 12,
            "done": true
        }));

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            timestamp: chrono::Utc::now(),
        }];

        let config = AiConfig {
            provider: AiProvider::Ollama,
            selected_model: "llama3".to_string(),
            ..Default::default()
        };

        let response =
            generate_response_async(&client, &config, &messages, "You are helpful")
                .await
                .unwrap();
        match response {
            AiResponse::Complete(text) => assert_eq!(text, "Hello! How can I help you?"),
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_openai_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I am a helpful assistant."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": 25
            }
        }));

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Who are you?".to_string(),
            timestamp: chrono::Utc::now(),
        }];

        let config = AiConfig {
            provider: AiProvider::LMStudio,
            selected_model: "local-model".to_string(),
            ..Default::default()
        };

        let response = generate_response_async(&client, &config, &messages, "")
            .await
            .unwrap();
        match response {
            AiResponse::Complete(text) => assert_eq!(text, "I am a helpful assistant."),
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_streaming_async() {
        let client = MockAsyncClient::with_response(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "Streamed response"
            },
            "done": true
        }));

        let tokens = std::sync::Arc::new(Mutex::new(Vec::new()));
        let tokens_clone = tokens.clone();

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            timestamp: chrono::Utc::now(),
        }];

        let config = AiConfig {
            provider: AiProvider::Ollama,
            selected_model: "llama3".to_string(),
            ..Default::default()
        };

        let response = generate_response_streaming_async(
            &client,
            &config,
            &messages,
            "",
            move |token| {
                tokens_clone.lock().unwrap().push(token.to_string());
            },
        )
        .await
        .unwrap();

        match response {
            AiResponse::Complete(text) => assert_eq!(text, "Streamed response"),
            other => panic!("Expected Complete, got {:?}", other),
        }
        let received = tokens.lock().unwrap();
        assert!(!received.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_models_async_dispatch() {
        // Test Ollama dispatch
        let client = MockAsyncClient::with_response(serde_json::json!({
            "models": [{ "name": "test-model" }]
        }));
        let config = AiConfig {
            provider: AiProvider::Ollama,
            ..Default::default()
        };
        let models = fetch_models_async(&client, &config).await.unwrap();
        assert_eq!(models.len(), 1);

        // Test LMStudio dispatch
        let client = MockAsyncClient::with_response(serde_json::json!({
            "data": [{ "id": "local-model" }]
        }));
        let config = AiConfig {
            provider: AiProvider::LMStudio,
            ..Default::default()
        };
        let models = fetch_models_async(&client, &config).await.unwrap();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_block_on_async_simple() {
        let result = block_on_async(async { Ok(42) });
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_create_runtime() {
        let rt = create_runtime();
        assert!(rt.is_ok());
    }

    #[test]
    fn test_reqwest_client_creation() {
        let client = ReqwestClient::new();
        assert!(client.api_key.is_none());

        let client = ReqwestClient::new().with_api_key("sk-test123");
        assert_eq!(client.api_key.as_deref(), Some("sk-test123"));
    }
}
