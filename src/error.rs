//! Unified error handling for the AI assistant library
//!
//! This module provides a comprehensive error type hierarchy with:
//! - Specific error variants for each subsystem
//! - Helpful error messages with recovery suggestions
//! - Conversion from common error types
//! - Serializable error details for logging

use std::fmt;

/// Main error type for the AI assistant library
#[derive(Debug)]
pub enum AiError {
    /// Configuration errors
    Config(ConfigError),
    /// Provider/API errors
    Provider(ProviderError),
    /// RAG system errors
    Rag(RagError),
    /// Network/HTTP errors
    Network(NetworkError),
    /// Validation errors
    Validation(ValidationError),
    /// Resource limits exceeded
    ResourceLimit(ResourceLimitError),
    /// I/O errors
    Io(IoError),
    /// Serialization errors
    Serialization(SerializationError),
    /// Generic error with message
    Other(String),
}

impl std::error::Error for AiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AiError::Config(e) => Some(e),
            AiError::Provider(e) => Some(e),
            AiError::Rag(e) => Some(e),
            AiError::Network(e) => Some(e),
            AiError::Validation(e) => Some(e),
            AiError::ResourceLimit(e) => Some(e),
            AiError::Io(e) => Some(e),
            AiError::Serialization(e) => Some(e),
            AiError::Other(_) => None,
        }
    }
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::Config(e) => write!(f, "Configuration error: {}", e),
            AiError::Provider(e) => write!(f, "Provider error: {}", e),
            AiError::Rag(e) => write!(f, "RAG error: {}", e),
            AiError::Network(e) => write!(f, "Network error: {}", e),
            AiError::Validation(e) => write!(f, "Validation error: {}", e),
            AiError::ResourceLimit(e) => write!(f, "Resource limit error: {}", e),
            AiError::Io(e) => write!(f, "I/O error: {}", e),
            AiError::Serialization(e) => write!(f, "Serialization error: {}", e),
            AiError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl AiError {
    /// Create a new generic error
    pub fn other(msg: impl Into<String>) -> Self {
        AiError::Other(msg.into())
    }

    /// Get a recovery suggestion for this error
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            AiError::Config(e) => e.suggestion(),
            AiError::Provider(e) => e.suggestion(),
            AiError::Rag(e) => e.suggestion(),
            AiError::Network(e) => e.suggestion(),
            AiError::Validation(e) => e.suggestion(),
            AiError::ResourceLimit(e) => e.suggestion(),
            AiError::Io(e) => e.suggestion(),
            AiError::Serialization(_) => Some("Check that the data format is correct"),
            AiError::Other(_) => None,
        }
    }

    /// Check if this error is recoverable (can be retried)
    pub fn is_recoverable(&self) -> bool {
        match self {
            AiError::Network(e) => e.is_recoverable(),
            AiError::Provider(e) => e.is_recoverable(),
            AiError::ResourceLimit(e) => e.is_recoverable(),
            _ => false,
        }
    }

    /// Get the error code for logging/tracking
    pub fn code(&self) -> &'static str {
        match self {
            AiError::Config(_) => "CONFIG",
            AiError::Provider(_) => "PROVIDER",
            AiError::Rag(_) => "RAG",
            AiError::Network(_) => "NETWORK",
            AiError::Validation(_) => "VALIDATION",
            AiError::ResourceLimit(_) => "RESOURCE_LIMIT",
            AiError::Io(_) => "IO",
            AiError::Serialization(_) => "SERIALIZATION",
            AiError::Other(_) => "OTHER",
        }
    }
}

// === Configuration Errors ===

/// Errors related to configuration
#[derive(Debug)]
pub enum ConfigError {
    /// Missing required configuration value
    MissingValue { field: String, description: String },
    /// Invalid configuration value
    InvalidValue { field: String, value: String, expected: String },
    /// Failed to load configuration file
    LoadFailed { path: String, reason: String },
    /// Failed to save configuration file
    SaveFailed { path: String, reason: String },
    /// Unknown provider specified
    UnknownProvider(String),
}

impl std::error::Error for ConfigError {}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingValue { field, description } => {
                write!(f, "Missing required configuration '{}': {}", field, description)
            }
            ConfigError::InvalidValue { field, value, expected } => {
                write!(f, "Invalid value '{}' for '{}', expected: {}", value, field, expected)
            }
            ConfigError::LoadFailed { path, reason } => {
                write!(f, "Failed to load configuration from '{}': {}", path, reason)
            }
            ConfigError::SaveFailed { path, reason } => {
                write!(f, "Failed to save configuration to '{}': {}", path, reason)
            }
            ConfigError::UnknownProvider(name) => {
                write!(f, "Unknown AI provider: '{}'", name)
            }
        }
    }
}

impl ConfigError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ConfigError::MissingValue { .. } => {
                Some("Check your configuration file or provide the value programmatically")
            }
            ConfigError::InvalidValue { .. } => {
                Some("Verify the value format matches the expected type")
            }
            ConfigError::LoadFailed { .. } => {
                Some("Check that the file exists and has proper permissions")
            }
            ConfigError::SaveFailed { .. } => {
                Some("Check write permissions for the configuration directory")
            }
            ConfigError::UnknownProvider(_) => {
                Some("Available providers: ollama, lmstudio, textgenwebui, kobold, localai, openai")
            }
        }
    }
}

impl From<ConfigError> for AiError {
    fn from(e: ConfigError) -> Self {
        AiError::Config(e)
    }
}

// === Provider Errors ===

/// Errors related to AI providers (Ollama, LM Studio, etc.)
#[derive(Debug)]
pub enum ProviderError {
    /// Provider is not available/reachable
    Unavailable { provider: String, url: String },
    /// Model not found
    ModelNotFound { provider: String, model: String },
    /// Authentication failed
    AuthenticationFailed { provider: String, reason: String },
    /// Rate limited by provider
    RateLimited { provider: String, retry_after: Option<u64> },
    /// Context length exceeded
    ContextLengthExceeded { max_tokens: usize, used_tokens: usize },
    /// Invalid response from provider
    InvalidResponse { provider: String, reason: String },
    /// Provider returned an error
    ApiError { provider: String, status_code: u16, message: String },
    /// Generation was cancelled
    Cancelled,
    /// Streaming error
    StreamError { reason: String },
}

impl std::error::Error for ProviderError {}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Unavailable { provider, url } => {
                write!(f, "{} provider unavailable at '{}'", provider, url)
            }
            ProviderError::ModelNotFound { provider, model } => {
                write!(f, "Model '{}' not found on {} provider", model, provider)
            }
            ProviderError::AuthenticationFailed { provider, reason } => {
                write!(f, "Authentication failed for {}: {}", provider, reason)
            }
            ProviderError::RateLimited { provider, retry_after } => {
                if let Some(secs) = retry_after {
                    write!(f, "Rate limited by {}, retry after {} seconds", provider, secs)
                } else {
                    write!(f, "Rate limited by {}", provider)
                }
            }
            ProviderError::ContextLengthExceeded { max_tokens, used_tokens } => {
                write!(f, "Context length exceeded: {} tokens used, {} max", used_tokens, max_tokens)
            }
            ProviderError::InvalidResponse { provider, reason } => {
                write!(f, "Invalid response from {}: {}", provider, reason)
            }
            ProviderError::ApiError { provider, status_code, message } => {
                write!(f, "{} API error ({}): {}", provider, status_code, message)
            }
            ProviderError::Cancelled => {
                write!(f, "Generation was cancelled")
            }
            ProviderError::StreamError { reason } => {
                write!(f, "Streaming error: {}", reason)
            }
        }
    }
}

impl ProviderError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ProviderError::Unavailable { .. } => {
                Some("Check that the provider is running and the URL is correct")
            }
            ProviderError::ModelNotFound { .. } => {
                Some("Use fetch_models() to get available models, or pull the model first")
            }
            ProviderError::AuthenticationFailed { .. } => {
                Some("Check your API key or credentials")
            }
            ProviderError::RateLimited { .. } => {
                Some("Wait before retrying or reduce request frequency")
            }
            ProviderError::ContextLengthExceeded { .. } => {
                Some("Reduce conversation history, use summarization, or enable RAG")
            }
            ProviderError::InvalidResponse { .. } => {
                Some("This may be a provider bug - try a different model")
            }
            ProviderError::ApiError { status_code, .. } if *status_code >= 500 => {
                Some("Server error - retry after a moment")
            }
            ProviderError::ApiError { .. } => None,
            ProviderError::Cancelled => None,
            ProviderError::StreamError { .. } => {
                Some("Check network connection and try again")
            }
        }
    }

    pub fn is_recoverable(&self) -> bool {
        match self {
            ProviderError::Unavailable { .. } => true,
            ProviderError::RateLimited { .. } => true,
            ProviderError::ApiError { status_code, .. } => *status_code >= 500,
            _ => false,
        }
    }
}

impl From<ProviderError> for AiError {
    fn from(e: ProviderError) -> Self {
        AiError::Provider(e)
    }
}

// === RAG Errors ===

/// Errors related to the RAG system
#[derive(Debug)]
pub enum RagError {
    /// Database error
    Database { operation: String, reason: String },
    /// Document not found
    DocumentNotFound(String),
    /// Invalid document format
    InvalidDocument { source: String, reason: String },
    /// Indexing failed
    IndexingFailed { source: String, reason: String },
    /// Search failed
    SearchFailed { query: String, reason: String },
    /// Append-only mode violation
    AppendOnlyViolation { operation: String, source: String },
    /// Embedding error
    EmbeddingError(String),
}

impl std::error::Error for RagError {}

impl fmt::Display for RagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RagError::Database { operation, reason } => {
                write!(f, "Database error during '{}': {}", operation, reason)
            }
            RagError::DocumentNotFound(source) => {
                write!(f, "Document '{}' not found", source)
            }
            RagError::InvalidDocument { source, reason } => {
                write!(f, "Invalid document '{}': {}", source, reason)
            }
            RagError::IndexingFailed { source, reason } => {
                write!(f, "Failed to index '{}': {}", source, reason)
            }
            RagError::SearchFailed { query, reason } => {
                write!(f, "Search failed for '{}': {}", query, reason)
            }
            RagError::AppendOnlyViolation { operation, source } => {
                write!(f, "Cannot {} document '{}': append-only mode is enabled", operation, source)
            }
            RagError::EmbeddingError(reason) => {
                write!(f, "Embedding error: {}", reason)
            }
        }
    }
}

impl RagError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            RagError::Database { .. } => {
                Some("Check database file permissions and disk space")
            }
            RagError::DocumentNotFound(_) => {
                Some("Register the document first with register_knowledge_document()")
            }
            RagError::InvalidDocument { .. } => {
                Some("Ensure the document is valid UTF-8 text")
            }
            RagError::IndexingFailed { .. } => {
                Some("Check document content and try again")
            }
            RagError::SearchFailed { .. } => {
                Some("Try a simpler search query")
            }
            RagError::AppendOnlyViolation { .. } => {
                Some("Disable append-only mode with set_append_only_mode(false)")
            }
            RagError::EmbeddingError(_) => {
                Some("Check embedding configuration")
            }
        }
    }
}

impl From<RagError> for AiError {
    fn from(e: RagError) -> Self {
        AiError::Rag(e)
    }
}

// === Network Errors ===

/// Errors related to network operations
#[derive(Debug)]
pub enum NetworkError {
    /// Connection failed
    ConnectionFailed { url: String, reason: String },
    /// Request timeout
    Timeout { url: String, timeout_ms: u64 },
    /// DNS resolution failed
    DnsError { host: String },
    /// SSL/TLS error
    TlsError { url: String, reason: String },
    /// HTTP error
    HttpError { status: u16, message: String },
}

impl std::error::Error for NetworkError {}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailed { url, reason } => {
                write!(f, "Connection to '{}' failed: {}", url, reason)
            }
            NetworkError::Timeout { url, timeout_ms } => {
                write!(f, "Request to '{}' timed out after {}ms", url, timeout_ms)
            }
            NetworkError::DnsError { host } => {
                write!(f, "DNS resolution failed for '{}'", host)
            }
            NetworkError::TlsError { url, reason } => {
                write!(f, "TLS error for '{}': {}", url, reason)
            }
            NetworkError::HttpError { status, message } => {
                write!(f, "HTTP error {}: {}", status, message)
            }
        }
    }
}

impl NetworkError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            NetworkError::ConnectionFailed { .. } => {
                Some("Check network connectivity and server status")
            }
            NetworkError::Timeout { .. } => {
                Some("Increase timeout or check server responsiveness")
            }
            NetworkError::DnsError { .. } => {
                Some("Check hostname spelling and network configuration")
            }
            NetworkError::TlsError { .. } => {
                Some("Check SSL certificate or use HTTP for local servers")
            }
            NetworkError::HttpError { status, .. } if *status >= 500 => {
                Some("Server error - try again later")
            }
            NetworkError::HttpError { status, .. } if *status == 429 => {
                Some("Rate limited - reduce request frequency")
            }
            NetworkError::HttpError { .. } => None,
        }
    }

    pub fn is_recoverable(&self) -> bool {
        match self {
            NetworkError::ConnectionFailed { .. } => true,
            NetworkError::Timeout { .. } => true,
            NetworkError::HttpError { status, .. } => *status >= 500 || *status == 429,
            _ => false,
        }
    }
}

impl From<NetworkError> for AiError {
    fn from(e: NetworkError) -> Self {
        AiError::Network(e)
    }
}

// === Validation Errors ===

/// Errors related to input validation
#[derive(Debug)]
pub enum ValidationError {
    /// Empty input
    EmptyInput { field: String },
    /// Input too long
    TooLong { field: String, max_length: usize, actual_length: usize },
    /// Invalid format
    InvalidFormat { field: String, expected: String },
    /// Out of range
    OutOfRange { field: String, min: String, max: String, value: String },
    /// Custom validation error
    Custom { field: String, message: String },
}

impl std::error::Error for ValidationError {}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyInput { field } => {
                write!(f, "'{}' cannot be empty", field)
            }
            ValidationError::TooLong { field, max_length, actual_length } => {
                write!(f, "'{}' is too long ({} chars, max {})", field, actual_length, max_length)
            }
            ValidationError::InvalidFormat { field, expected } => {
                write!(f, "'{}' has invalid format, expected: {}", field, expected)
            }
            ValidationError::OutOfRange { field, min, max, value } => {
                write!(f, "'{}' value '{}' out of range [{}, {}]", field, value, min, max)
            }
            ValidationError::Custom { field, message } => {
                write!(f, "'{}': {}", field, message)
            }
        }
    }
}

impl ValidationError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ValidationError::EmptyInput { .. } => Some("Provide a non-empty value"),
            ValidationError::TooLong { .. } => Some("Shorten the input"),
            ValidationError::InvalidFormat { .. } => Some("Check the expected format"),
            ValidationError::OutOfRange { .. } => Some("Use a value within the valid range"),
            ValidationError::Custom { .. } => None,
        }
    }
}

impl From<ValidationError> for AiError {
    fn from(e: ValidationError) -> Self {
        AiError::Validation(e)
    }
}

// === Resource Limit Errors ===

/// Errors related to resource limits
#[derive(Debug)]
pub enum ResourceLimitError {
    /// Rate limit exceeded
    RateLimitExceeded { limit: u32, window_secs: u64, retry_after_secs: Option<u64> },
    /// Memory limit exceeded
    MemoryLimitExceeded { limit_mb: usize, used_mb: usize },
    /// Token limit exceeded
    TokenLimitExceeded { limit: usize, used: usize },
    /// Storage limit exceeded
    StorageLimitExceeded { limit_mb: usize, used_mb: usize },
    /// Concurrent request limit
    ConcurrentRequestLimit { limit: usize },
    /// Budget exceeded
    BudgetExceeded { budget: f64, used: f64, currency: String },
}

impl std::error::Error for ResourceLimitError {}

impl fmt::Display for ResourceLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceLimitError::RateLimitExceeded { limit, window_secs, retry_after_secs } => {
                if let Some(retry) = retry_after_secs {
                    write!(f, "Rate limit exceeded ({} requests per {}s), retry after {}s", limit, window_secs, retry)
                } else {
                    write!(f, "Rate limit exceeded ({} requests per {}s)", limit, window_secs)
                }
            }
            ResourceLimitError::MemoryLimitExceeded { limit_mb, used_mb } => {
                write!(f, "Memory limit exceeded: {}MB used of {}MB limit", used_mb, limit_mb)
            }
            ResourceLimitError::TokenLimitExceeded { limit, used } => {
                write!(f, "Token limit exceeded: {} used of {} limit", used, limit)
            }
            ResourceLimitError::StorageLimitExceeded { limit_mb, used_mb } => {
                write!(f, "Storage limit exceeded: {}MB used of {}MB limit", used_mb, limit_mb)
            }
            ResourceLimitError::ConcurrentRequestLimit { limit } => {
                write!(f, "Concurrent request limit exceeded: {} max", limit)
            }
            ResourceLimitError::BudgetExceeded { budget, used, currency } => {
                write!(f, "Budget exceeded: {:.2} {} used of {:.2} {} budget", used, currency, budget, currency)
            }
        }
    }
}

impl ResourceLimitError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ResourceLimitError::RateLimitExceeded { .. } => {
                Some("Wait before making more requests")
            }
            ResourceLimitError::MemoryLimitExceeded { .. } => {
                Some("Clear caches or reduce batch sizes")
            }
            ResourceLimitError::TokenLimitExceeded { .. } => {
                Some("Use summarization or RAG to reduce context size")
            }
            ResourceLimitError::StorageLimitExceeded { .. } => {
                Some("Delete old data or increase storage limit")
            }
            ResourceLimitError::ConcurrentRequestLimit { .. } => {
                Some("Wait for pending requests to complete")
            }
            ResourceLimitError::BudgetExceeded { .. } => {
                Some("Increase budget or use a cheaper model")
            }
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ResourceLimitError::RateLimitExceeded { .. }
                | ResourceLimitError::ConcurrentRequestLimit { .. }
        )
    }
}

impl From<ResourceLimitError> for AiError {
    fn from(e: ResourceLimitError) -> Self {
        AiError::ResourceLimit(e)
    }
}

// === I/O Errors ===

/// Errors related to I/O operations
#[derive(Debug)]
pub struct IoError {
    pub operation: String,
    pub path: Option<String>,
    pub reason: String,
}

impl std::error::Error for IoError {}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref path) = self.path {
            write!(f, "I/O error during '{}' on '{}': {}", self.operation, path, self.reason)
        } else {
            write!(f, "I/O error during '{}': {}", self.operation, self.reason)
        }
    }
}

impl IoError {
    pub fn new(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            path: None,
            reason: reason.into(),
        }
    }

    pub fn with_path(operation: impl Into<String>, path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            path: Some(path.into()),
            reason: reason.into(),
        }
    }

    pub fn suggestion(&self) -> Option<&'static str> {
        if self.reason.contains("permission") || self.reason.contains("Permission") {
            Some("Check file/directory permissions")
        } else if self.reason.contains("not found") || self.reason.contains("No such") {
            Some("Check that the file/directory exists")
        } else if self.reason.contains("space") || self.reason.contains("disk") {
            Some("Free up disk space")
        } else {
            None
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self {
            operation: "unknown".to_string(),
            path: None,
            reason: e.to_string(),
        }
    }
}

impl From<IoError> for AiError {
    fn from(e: IoError) -> Self {
        AiError::Io(e)
    }
}

impl From<std::io::Error> for AiError {
    fn from(e: std::io::Error) -> Self {
        AiError::Io(IoError::from(e))
    }
}

// === Serialization Errors ===

/// Errors related to serialization/deserialization
#[derive(Debug)]
pub struct SerializationError {
    pub format: String,
    pub operation: String,
    pub reason: String,
}

impl std::error::Error for SerializationError {}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} error: {}", self.format, self.operation, self.reason)
    }
}

impl SerializationError {
    pub fn json_serialize(reason: impl Into<String>) -> Self {
        Self {
            format: "JSON".to_string(),
            operation: "serialize".to_string(),
            reason: reason.into(),
        }
    }

    pub fn json_deserialize(reason: impl Into<String>) -> Self {
        Self {
            format: "JSON".to_string(),
            operation: "deserialize".to_string(),
            reason: reason.into(),
        }
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(e: serde_json::Error) -> Self {
        Self {
            format: "JSON".to_string(),
            operation: if e.is_syntax() || e.is_data() || e.is_eof() {
                "deserialize".to_string()
            } else {
                "serialize".to_string()
            },
            reason: e.to_string(),
        }
    }
}

impl From<SerializationError> for AiError {
    fn from(e: SerializationError) -> Self {
        AiError::Serialization(e)
    }
}

impl From<serde_json::Error> for AiError {
    fn from(e: serde_json::Error) -> Self {
        AiError::Serialization(SerializationError::from(e))
    }
}

// === Conversions from anyhow ===

impl From<anyhow::Error> for AiError {
    fn from(e: anyhow::Error) -> Self {
        AiError::Other(e.to_string())
    }
}

// === Result type alias ===

/// Convenient Result type for AI assistant operations
pub type AiResult<T> = std::result::Result<T, AiError>;

// === Error builder helpers ===

impl AiError {
    /// Create a provider unavailable error
    pub fn provider_unavailable(provider: impl Into<String>, url: impl Into<String>) -> Self {
        ProviderError::Unavailable {
            provider: provider.into(),
            url: url.into(),
        }.into()
    }

    /// Create a model not found error
    pub fn model_not_found(provider: impl Into<String>, model: impl Into<String>) -> Self {
        ProviderError::ModelNotFound {
            provider: provider.into(),
            model: model.into(),
        }.into()
    }

    /// Create a context length exceeded error
    pub fn context_exceeded(max_tokens: usize, used_tokens: usize) -> Self {
        ProviderError::ContextLengthExceeded { max_tokens, used_tokens }.into()
    }

    /// Create a rate limit error
    pub fn rate_limited(limit: u32, window_secs: u64) -> Self {
        ResourceLimitError::RateLimitExceeded {
            limit,
            window_secs,
            retry_after_secs: None,
        }.into()
    }

    /// Create an append-only violation error
    pub fn append_only_violation(operation: impl Into<String>, source: impl Into<String>) -> Self {
        RagError::AppendOnlyViolation {
            operation: operation.into(),
            source: source.into(),
        }.into()
    }

    /// Create a validation empty input error
    pub fn empty_input(field: impl Into<String>) -> Self {
        ValidationError::EmptyInput { field: field.into() }.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = AiError::provider_unavailable("Ollama", "http://localhost:11434");
        assert!(e.to_string().contains("Ollama"));
        assert!(e.suggestion().is_some());
    }

    #[test]
    fn test_error_code() {
        let e = AiError::context_exceeded(4096, 5000);
        assert_eq!(e.code(), "PROVIDER");
    }

    #[test]
    fn test_recoverable() {
        let e = AiError::rate_limited(100, 60);
        assert!(e.is_recoverable());

        let e = AiError::empty_input("message");
        assert!(!e.is_recoverable());
    }
}
