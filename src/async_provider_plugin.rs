//! Async provider plugin trait and sync/async bridge adapters.
//!
//! Provides `AsyncProviderPlugin` -- the async counterpart to `ProviderPlugin`,
//! plus `SyncToAsyncAdapter` and `AsyncToSyncAdapter` for bridging between the two.
//!
//! Gated behind `#[cfg(feature = "async-runtime")]`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::AiConfig;
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;
use crate::tools::{ProviderCapabilities, ProviderPlugin, ToolCall, ToolDefinition};

// ============================================================================
// AsyncProviderPlugin Trait
// ============================================================================

/// Async version of [`ProviderPlugin`] for non-blocking provider operations.
///
/// Uses `Pin<Box<dyn Future>>` return types (no `async-trait` dependency).
/// All async methods take `&self` and return boxed futures. Sync accessors
/// (`name`, `capabilities`) are plain methods since they involve no I/O.
pub trait AsyncProviderPlugin: Send + Sync {
    /// Provider name (sync, no I/O).
    fn name(&self) -> &str;

    /// Provider capabilities (sync, no I/O).
    fn capabilities(&self) -> ProviderCapabilities;

    /// Check if the provider is available (async health check).
    fn is_available_async(&self) -> Pin<Box<dyn Future<Output = bool> + Send + '_>>;

    /// List available models from this provider.
    fn list_models_async(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>>> + Send + '_>>;

    /// Generate a complete (non-streaming) response.
    fn generate_async(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>>;

    /// Generate a streaming response, sending chunks over the provided channel.
    fn generate_streaming_async(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: tokio::sync::mpsc::Sender<AiResponse>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    /// Generate with tool-calling support.
    ///
    /// The default implementation returns an error indicating tools are not supported.
    fn generate_with_tools_async(
        &self,
        _config: &AiConfig,
        _messages: &[ChatMessage],
        _system_prompt: &str,
        _tools: &[ToolDefinition],
    ) -> Pin<Box<dyn Future<Output = Result<(String, Vec<ToolCall>)>> + Send + '_>> {
        Box::pin(async { anyhow::bail!("Tool calling not supported by this async provider") })
    }

    /// Generate embeddings for the given texts.
    ///
    /// The default implementation returns an error indicating embeddings are not supported.
    fn generate_embeddings_async(
        &self,
        _texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>> {
        Box::pin(async { anyhow::bail!("Embeddings not supported by this async provider") })
    }
}

// ============================================================================
// SyncToAsyncAdapter
// ============================================================================

/// Adapts a sync [`ProviderPlugin`] to work as an [`AsyncProviderPlugin`].
///
/// Blocking calls are dispatched to tokio's blocking thread pool via
/// [`tokio::task::spawn_blocking`], preventing them from stalling the async
/// runtime.
pub struct SyncToAsyncAdapter {
    inner: Arc<dyn ProviderPlugin>,
    name: String,
    caps: ProviderCapabilities,
}

impl SyncToAsyncAdapter {
    /// Wrap a shared provider plugin.
    pub fn new(provider: Arc<dyn ProviderPlugin>) -> Self {
        let name = provider.name().to_string();
        let caps = provider.capabilities();
        Self {
            inner: provider,
            name,
            caps,
        }
    }

    /// Wrap a boxed provider plugin.
    pub fn from_boxed(provider: Box<dyn ProviderPlugin>) -> Self {
        Self::new(Arc::from(provider))
    }
}

impl AsyncProviderPlugin for SyncToAsyncAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.caps.clone()
    }

    fn is_available_async(&self) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || inner.is_available())
                .await
                .unwrap_or(false)
        })
    }

    fn list_models_async(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || inner.list_models())
                .await
                .context("spawn_blocking join error")?
        })
    }

    fn generate_async(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let config = config.clone();
        let messages = messages.to_vec();
        let system_prompt = system_prompt.to_string();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || inner.generate(&config, &messages, &system_prompt))
                .await
                .context("spawn_blocking join error")?
        })
    }

    fn generate_streaming_async(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: tokio::sync::mpsc::Sender<AiResponse>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let config = config.clone();
        let messages = messages.to_vec();
        let system_prompt = system_prompt.to_string();
        Box::pin(async move {
            // Use a std channel inside spawn_blocking, then forward to the tokio channel.
            let (std_tx, std_rx) = std::sync::mpsc::channel::<AiResponse>();
            let blocking_result = tokio::task::spawn_blocking(move || {
                inner.generate_streaming(&config, &messages, &system_prompt, &std_tx)
            });

            // Forward received chunks while the blocking task runs.
            // We must poll both the blocking task and the std_rx receiver.
            let forward_handle = tokio::spawn(async move {
                while let Ok(response) = std_rx.recv() {
                    if tx.send(response).await.is_err() {
                        break; // receiver dropped
                    }
                }
            });

            let result = blocking_result
                .await
                .context("spawn_blocking join error")?;

            // Wait for forwarding to finish
            let _ = forward_handle.await;
            result
        })
    }

    fn generate_with_tools_async(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: &[ToolDefinition],
    ) -> Pin<Box<dyn Future<Output = Result<(String, Vec<ToolCall>)>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let config = config.clone();
        let messages = messages.to_vec();
        let system_prompt = system_prompt.to_string();
        let tools = tools.to_vec();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                inner.generate_with_tools(&config, &messages, &system_prompt, &tools)
            })
            .await
            .context("spawn_blocking join error")?
        })
    }

    fn generate_embeddings_async(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                inner.generate_embeddings(&refs)
            })
            .await
            .context("spawn_blocking join error")?
        })
    }
}

// ============================================================================
// AsyncToSyncAdapter
// ============================================================================

/// Adapts an [`AsyncProviderPlugin`] to work as a sync [`ProviderPlugin`].
///
/// Uses a dedicated tokio runtime to block on async operations. Useful when
/// you have an async provider but need to satisfy a sync `ProviderPlugin`
/// interface.
pub struct AsyncToSyncAdapter {
    inner: Arc<dyn AsyncProviderPlugin>,
    runtime: tokio::runtime::Runtime,
    name: String,
    caps: ProviderCapabilities,
}

impl AsyncToSyncAdapter {
    /// Create a new adapter wrapping the given async provider.
    ///
    /// Spawns a dedicated tokio runtime for blocking on async calls.
    pub fn new(provider: Arc<dyn AsyncProviderPlugin>) -> Result<Self> {
        let name = provider.name().to_string();
        let caps = provider.capabilities();
        let runtime = crate::async_providers::create_runtime()?;
        Ok(Self {
            inner: provider,
            runtime,
            name,
            caps,
        })
    }
}

impl ProviderPlugin for AsyncToSyncAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.caps.clone()
    }

    fn is_available(&self) -> bool {
        self.runtime.block_on(self.inner.is_available_async())
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.runtime.block_on(self.inner.list_models_async())
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        self.runtime
            .block_on(self.inner.generate_async(config, messages, system_prompt))
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &std::sync::mpsc::Sender<AiResponse>,
    ) -> Result<()> {
        let (async_tx, mut async_rx) = tokio::sync::mpsc::channel::<AiResponse>(64);
        let inner = Arc::clone(&self.inner);
        let config = config.clone();
        let messages = messages.to_vec();
        let system_prompt = system_prompt.to_string();
        let tx = tx.clone();

        self.runtime.block_on(async move {
            // Spawn the streaming task
            let stream_handle = tokio::spawn(async move {
                inner
                    .generate_streaming_async(&config, &messages, &system_prompt, async_tx)
                    .await
            });

            // Forward from tokio channel to std channel
            while let Some(response) = async_rx.recv().await {
                if tx.send(response).is_err() {
                    break; // receiver dropped
                }
            }

            stream_handle
                .await
                .context("streaming task join error")?
        })
    }

    fn generate_with_tools(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<(String, Vec<ToolCall>)> {
        self.runtime.block_on(
            self.inner
                .generate_with_tools_async(config, messages, system_prompt, tools),
        )
    }

    fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.runtime
            .block_on(self.inner.generate_embeddings_async(texts))
    }
}

// ============================================================================
// AsyncProviderRegistry
// ============================================================================

/// A registry for async provider plugins.
///
/// Allows registering, retrieving, and listing [`AsyncProviderPlugin`]
/// implementations. Supports setting a default provider.
pub struct AsyncProviderRegistry {
    providers: Vec<Box<dyn AsyncProviderPlugin>>,
    default_idx: Option<usize>,
}

impl AsyncProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            default_idx: None,
        }
    }

    /// Register a provider. If it is the first provider, it becomes the default.
    pub fn register(&mut self, provider: Box<dyn AsyncProviderPlugin>) {
        if self.providers.is_empty() {
            self.default_idx = Some(0);
        }
        self.providers.push(provider);
    }

    /// Look up a provider by name.
    pub fn get(&self, name: &str) -> Option<&dyn AsyncProviderPlugin> {
        self.providers
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// List the names of all registered providers.
    pub fn list(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }

    /// Set the default provider by name. Returns `true` if the provider was found.
    pub fn set_default(&mut self, name: &str) -> bool {
        if let Some(idx) = self.providers.iter().position(|p| p.name() == name) {
            self.default_idx = Some(idx);
            true
        } else {
            false
        }
    }

    /// Get the default provider, if one is set.
    pub fn default_provider(&self) -> Option<&dyn AsyncProviderPlugin> {
        self.default_idx
            .and_then(|idx| self.providers.get(idx))
            .map(|p| p.as_ref())
    }
}

impl Default for AsyncProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "async-runtime")]
mod tests {
    use super::*;
    use crate::config::AiProvider;

    // ---- Mock sync provider ------------------------------------------------

    struct MockSyncProvider;

    impl ProviderPlugin for MockSyncProvider {
        fn name(&self) -> &str {
            "mock-sync"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                streaming: true,
                tool_calling: false,
                vision: false,
                embeddings: false,
                json_mode: false,
                system_prompt: true,
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn list_models(&self) -> Result<Vec<ModelInfo>> {
            Ok(vec![ModelInfo {
                name: "mock-model".to_string(),
                provider: AiProvider::Ollama,
                size: None,
                modified_at: None,
                capabilities: None,
            }])
        }

        fn generate(
            &self,
            _config: &AiConfig,
            _messages: &[ChatMessage],
            _system_prompt: &str,
        ) -> Result<String> {
            Ok("mock response".to_string())
        }

        fn generate_streaming(
            &self,
            _config: &AiConfig,
            _messages: &[ChatMessage],
            _system_prompt: &str,
            tx: &std::sync::mpsc::Sender<AiResponse>,
        ) -> Result<()> {
            let _ = tx.send(AiResponse::Complete("mock stream".to_string()));
            Ok(())
        }

        fn generate_with_tools(
            &self,
            _config: &AiConfig,
            _messages: &[ChatMessage],
            _system_prompt: &str,
            _tools: &[ToolDefinition],
        ) -> Result<(String, Vec<ToolCall>)> {
            Ok(("mock tools".to_string(), vec![]))
        }

        fn generate_embeddings(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            anyhow::bail!("not supported")
        }
    }

    // ---- Mock async provider -----------------------------------------------

    struct MockAsyncProvider;

    impl AsyncProviderPlugin for MockAsyncProvider {
        fn name(&self) -> &str {
            "mock-async"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                streaming: false,
                tool_calling: true,
                vision: false,
                embeddings: false,
                json_mode: true,
                system_prompt: true,
            }
        }

        fn is_available_async(&self) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
            Box::pin(async { true })
        }

        fn list_models_async(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>>> + Send + '_>> {
            Box::pin(async {
                Ok(vec![ModelInfo {
                    name: "async-model".to_string(),
                    provider: AiProvider::Ollama,
                    size: None,
                    modified_at: None,
                    capabilities: None,
                }])
            })
        }

        fn generate_async(
            &self,
            _config: &AiConfig,
            _messages: &[ChatMessage],
            _system_prompt: &str,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
            Box::pin(async { Ok("async response".to_string()) })
        }

        fn generate_streaming_async(
            &self,
            _config: &AiConfig,
            _messages: &[ChatMessage],
            _system_prompt: &str,
            tx: tokio::sync::mpsc::Sender<AiResponse>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
            Box::pin(async move {
                let _ = tx.send(AiResponse::Complete("async stream".to_string())).await;
                Ok(())
            })
        }
    }

    // ---- Helper to create a default AiConfig --------------------------------

    fn test_config() -> AiConfig {
        AiConfig::default()
    }

    // ---- Tests --------------------------------------------------------------

    #[tokio::test]
    async fn test_sync_to_async_adapter_name() {
        let provider = Arc::new(MockSyncProvider) as Arc<dyn ProviderPlugin>;
        let adapter = SyncToAsyncAdapter::new(provider);
        assert_eq!(adapter.name(), "mock-sync");
    }

    #[tokio::test]
    async fn test_sync_to_async_adapter_capabilities() {
        let provider = Arc::new(MockSyncProvider) as Arc<dyn ProviderPlugin>;
        let adapter = SyncToAsyncAdapter::new(provider);
        let caps = adapter.capabilities();
        assert!(caps.streaming);
        assert!(!caps.tool_calling);
        assert!(!caps.vision);
        assert!(!caps.embeddings);
        assert!(!caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_async_to_sync_adapter_creation() {
        // Must be a sync test — AsyncToSyncAdapter creates its own tokio runtime,
        // and dropping a runtime inside an async context (e.g. #[tokio::test]) panics.
        let provider = Arc::new(MockAsyncProvider) as Arc<dyn AsyncProviderPlugin>;
        let adapter = AsyncToSyncAdapter::new(provider).expect("should create adapter");
        assert_eq!(adapter.name(), "mock-async");
    }

    #[tokio::test]
    async fn test_async_provider_registry_register() {
        let mut registry = AsyncProviderRegistry::new();
        registry.register(Box::new(MockAsyncProvider));
        assert!(registry.get("mock-async").is_some());
        assert_eq!(registry.list(), vec!["mock-async"]);
    }

    #[tokio::test]
    async fn test_async_provider_registry_default() {
        let mut registry = AsyncProviderRegistry::new();
        registry.register(Box::new(MockAsyncProvider));
        // First registered becomes default automatically.
        let default = registry.default_provider().expect("should have default");
        assert_eq!(default.name(), "mock-async");

        // set_default with unknown name returns false
        assert!(!registry.set_default("nonexistent"));

        // set_default with known name returns true
        assert!(registry.set_default("mock-async"));
    }

    #[tokio::test]
    async fn test_async_provider_registry_empty() {
        let registry = AsyncProviderRegistry::new();
        assert!(registry.get("anything").is_none());
        assert!(registry.default_provider().is_none());
        assert!(registry.list().is_empty());
    }

    #[tokio::test]
    async fn test_trait_object_safety() {
        // Verify that AsyncProviderPlugin is object-safe by constructing a Box<dyn>.
        let boxed: Box<dyn AsyncProviderPlugin> = Box::new(MockAsyncProvider);
        assert_eq!(boxed.name(), "mock-async");
        let available = boxed.is_available_async().await;
        assert!(available);
    }

    #[tokio::test]
    async fn test_sync_to_async_is_available() {
        let provider = Arc::new(MockSyncProvider) as Arc<dyn ProviderPlugin>;
        let adapter = SyncToAsyncAdapter::new(provider);
        let available = adapter.is_available_async().await;
        assert!(available);
    }

    #[tokio::test]
    async fn test_sync_to_async_list_models() {
        let provider = Arc::new(MockSyncProvider) as Arc<dyn ProviderPlugin>;
        let adapter = SyncToAsyncAdapter::new(provider);
        let models = adapter.list_models_async().await.expect("should list models");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "mock-model");
    }

    #[tokio::test]
    async fn test_sync_to_async_generate() {
        let provider = Arc::new(MockSyncProvider) as Arc<dyn ProviderPlugin>;
        let adapter = SyncToAsyncAdapter::new(provider);
        let config = test_config();
        let messages = vec![ChatMessage::user("hello")];
        let result = adapter
            .generate_async(&config, &messages, "system")
            .await
            .expect("should generate");
        assert_eq!(result, "mock response");
    }

    #[tokio::test]
    async fn test_default_embeddings_not_supported() {
        let provider = MockAsyncProvider;
        let result = provider.generate_embeddings_async(&["test"]).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not supported"),
            "expected 'not supported' in error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_default_tools_not_supported() {
        let provider = MockAsyncProvider;
        let config = test_config();
        let messages = vec![ChatMessage::user("hello")];
        let result = provider
            .generate_with_tools_async(&config, &messages, "sys", &[])
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not supported"),
            "expected 'not supported' in error, got: {}",
            err_msg
        );
    }
}
