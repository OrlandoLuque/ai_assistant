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
#[non_exhaustive]
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
    /// Workflow engine errors
    Workflow(WorkflowError),
    /// Advanced memory system errors
    AdvancedMemory(AdvancedMemoryError),
    /// Agent-to-Agent protocol errors
    A2A(A2AError),
    /// Voice agent errors
    VoiceAgent(VoiceAgentError),
    /// Media generation errors
    MediaGeneration(MediaGenerationError),
    /// Distillation pipeline errors
    Distillation(DistillationError),
    /// Constrained decoding errors
    ConstrainedDecoding(ConstrainedDecodingError),
    /// Human-in-the-Loop errors
    Hitl(HitlError),
    /// Remote MCP client errors
    McpClient(McpClientError),
    /// Agent evaluation errors
    AgentEval(AgentEvalError),
    /// Red team / adversarial testing errors
    RedTeam(RedTeamError),
    /// MCTS planning errors
    Mcts(MctsError),
    /// Agent DevTools errors
    DevTools(DevToolsError),
    /// Evaluation benchmark suite errors
    EvalSuite(EvalSuiteError),
    /// Advanced routing errors
    AdvancedRouting(AdvancedRoutingError),
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
            AiError::Workflow(e) => Some(e),
            AiError::AdvancedMemory(e) => Some(e),
            AiError::A2A(e) => Some(e),
            AiError::VoiceAgent(e) => Some(e),
            AiError::MediaGeneration(e) => Some(e),
            AiError::Distillation(e) => Some(e),
            AiError::ConstrainedDecoding(e) => Some(e),
            AiError::Hitl(e) => Some(e),
            AiError::McpClient(e) => Some(e),
            AiError::AgentEval(e) => Some(e),
            AiError::RedTeam(e) => Some(e),
            AiError::Mcts(e) => Some(e),
            AiError::DevTools(e) => Some(e),
            AiError::EvalSuite(e) => Some(e),
            AiError::AdvancedRouting(e) => Some(e),
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
            AiError::Workflow(e) => write!(f, "Workflow error: {}", e),
            AiError::AdvancedMemory(e) => write!(f, "Memory error: {}", e),
            AiError::A2A(e) => write!(f, "A2A error: {}", e),
            AiError::VoiceAgent(e) => write!(f, "Voice agent error: {}", e),
            AiError::MediaGeneration(e) => write!(f, "Media generation error: {}", e),
            AiError::Distillation(e) => write!(f, "Distillation error: {}", e),
            AiError::ConstrainedDecoding(e) => write!(f, "Constrained decoding error: {}", e),
            AiError::Hitl(e) => write!(f, "HITL error: {}", e),
            AiError::McpClient(e) => write!(f, "MCP client error: {}", e),
            AiError::AgentEval(e) => write!(f, "Agent evaluation error: {}", e),
            AiError::RedTeam(e) => write!(f, "Red team error: {}", e),
            AiError::Mcts(e) => write!(f, "MCTS planning error: {}", e),
            AiError::DevTools(e) => write!(f, "DevTools error: {}", e),
            AiError::EvalSuite(e) => write!(f, "Eval suite error: {}", e),
            AiError::AdvancedRouting(e) => write!(f, "Advanced routing error: {}", e),
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
            AiError::Workflow(e) => e.suggestion(),
            AiError::AdvancedMemory(e) => e.suggestion(),
            AiError::A2A(e) => e.suggestion(),
            AiError::VoiceAgent(e) => e.suggestion(),
            AiError::MediaGeneration(e) => e.suggestion(),
            AiError::Distillation(e) => e.suggestion(),
            AiError::ConstrainedDecoding(e) => e.suggestion(),
            AiError::Hitl(e) => e.suggestion(),
            AiError::McpClient(e) => e.suggestion(),
            AiError::AgentEval(e) => e.suggestion(),
            AiError::RedTeam(e) => e.suggestion(),
            AiError::Mcts(e) => e.suggestion(),
            AiError::DevTools(e) => e.suggestion(),
            AiError::EvalSuite(e) => e.suggestion(),
            AiError::AdvancedRouting(e) => e.suggestion(),
            AiError::Other(_) => None,
        }
    }

    /// Check if this error is recoverable (can be retried)
    pub fn is_recoverable(&self) -> bool {
        match self {
            AiError::Network(e) => e.is_recoverable(),
            AiError::Provider(e) => e.is_recoverable(),
            AiError::ResourceLimit(e) => e.is_recoverable(),
            AiError::Workflow(e) => e.is_recoverable(),
            AiError::A2A(e) => e.is_recoverable(),
            AiError::VoiceAgent(e) => e.is_recoverable(),
            AiError::MediaGeneration(e) => e.is_recoverable(),
            AiError::Distillation(_) => false,
            AiError::ConstrainedDecoding(_) => false,
            AiError::Hitl(e) => e.is_recoverable(),
            AiError::McpClient(e) => e.is_recoverable(),
            AiError::AgentEval(_) => false,
            AiError::RedTeam(_) => false,
            AiError::Mcts(e) => e.is_recoverable(),
            AiError::DevTools(_) => false,
            AiError::EvalSuite(e) => e.is_recoverable(),
            AiError::AdvancedRouting(e) => e.is_recoverable(),
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
            AiError::Workflow(_) => "WORKFLOW",
            AiError::AdvancedMemory(_) => "MEMORY",
            AiError::A2A(_) => "A2A",
            AiError::VoiceAgent(_) => "VOICE_AGENT",
            AiError::MediaGeneration(_) => "MEDIA_GENERATION",
            AiError::Distillation(_) => "DISTILLATION",
            AiError::ConstrainedDecoding(_) => "CONSTRAINED_DECODING",
            AiError::Hitl(_) => "HITL",
            AiError::McpClient(_) => "MCP_CLIENT",
            AiError::AgentEval(_) => "AGENT_EVAL",
            AiError::RedTeam(_) => "RED_TEAM",
            AiError::Mcts(_) => "MCTS",
            AiError::DevTools(_) => "DEVTOOLS",
            AiError::EvalSuite(_) => "EVAL_SUITE",
            AiError::AdvancedRouting(_) => "ADVANCED_ROUTING",
            AiError::Other(_) => "OTHER",
        }
    }
}

// === Configuration Errors ===

/// Errors related to configuration
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// Missing required configuration value
    MissingValue { field: String, description: String },
    /// Invalid configuration value
    InvalidValue {
        field: String,
        value: String,
        expected: String,
    },
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
                write!(
                    f,
                    "Missing required configuration '{}': {}",
                    field, description
                )
            }
            ConfigError::InvalidValue {
                field,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid value '{}' for '{}', expected: {}",
                    value, field, expected
                )
            }
            ConfigError::LoadFailed { path, reason } => {
                write!(
                    f,
                    "Failed to load configuration from '{}': {}",
                    path, reason
                )
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
#[non_exhaustive]
pub enum ProviderError {
    /// Provider is not available/reachable
    Unavailable { provider: String, url: String },
    /// Model not found
    ModelNotFound { provider: String, model: String },
    /// Authentication failed
    AuthenticationFailed { provider: String, reason: String },
    /// Rate limited by provider
    RateLimited {
        provider: String,
        retry_after: Option<u64>,
    },
    /// Context length exceeded
    ContextLengthExceeded {
        max_tokens: usize,
        used_tokens: usize,
    },
    /// Invalid response from provider
    InvalidResponse { provider: String, reason: String },
    /// Provider returned an error
    ApiError {
        provider: String,
        status_code: u16,
        message: String,
    },
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
            ProviderError::RateLimited {
                provider,
                retry_after,
            } => {
                if let Some(secs) = retry_after {
                    write!(
                        f,
                        "Rate limited by {}, retry after {} seconds",
                        provider, secs
                    )
                } else {
                    write!(f, "Rate limited by {}", provider)
                }
            }
            ProviderError::ContextLengthExceeded {
                max_tokens,
                used_tokens,
            } => {
                write!(
                    f,
                    "Context length exceeded: {} tokens used, {} max",
                    used_tokens, max_tokens
                )
            }
            ProviderError::InvalidResponse { provider, reason } => {
                write!(f, "Invalid response from {}: {}", provider, reason)
            }
            ProviderError::ApiError {
                provider,
                status_code,
                message,
            } => {
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
            ProviderError::AuthenticationFailed { .. } => Some("Check your API key or credentials"),
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
            ProviderError::StreamError { .. } => Some("Check network connection and try again"),
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
#[non_exhaustive]
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
                write!(
                    f,
                    "Cannot {} document '{}': append-only mode is enabled",
                    operation, source
                )
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
            RagError::Database { .. } => Some("Check database file permissions and disk space"),
            RagError::DocumentNotFound(_) => {
                Some("Register the document first with register_knowledge_document()")
            }
            RagError::InvalidDocument { .. } => Some("Ensure the document is valid UTF-8 text"),
            RagError::IndexingFailed { .. } => Some("Check document content and try again"),
            RagError::SearchFailed { .. } => Some("Try a simpler search query"),
            RagError::AppendOnlyViolation { .. } => {
                Some("Disable append-only mode with set_append_only_mode(false)")
            }
            RagError::EmbeddingError(_) => Some("Check embedding configuration"),
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
#[non_exhaustive]
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
            NetworkError::Timeout { .. } => Some("Increase timeout or check server responsiveness"),
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
#[non_exhaustive]
pub enum ValidationError {
    /// Empty input
    EmptyInput { field: String },
    /// Input too long
    TooLong {
        field: String,
        max_length: usize,
        actual_length: usize,
    },
    /// Invalid format
    InvalidFormat { field: String, expected: String },
    /// Out of range
    OutOfRange {
        field: String,
        min: String,
        max: String,
        value: String,
    },
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
            ValidationError::TooLong {
                field,
                max_length,
                actual_length,
            } => {
                write!(
                    f,
                    "'{}' is too long ({} chars, max {})",
                    field, actual_length, max_length
                )
            }
            ValidationError::InvalidFormat { field, expected } => {
                write!(f, "'{}' has invalid format, expected: {}", field, expected)
            }
            ValidationError::OutOfRange {
                field,
                min,
                max,
                value,
            } => {
                write!(
                    f,
                    "'{}' value '{}' out of range [{}, {}]",
                    field, value, min, max
                )
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
#[non_exhaustive]
pub enum ResourceLimitError {
    /// Rate limit exceeded
    RateLimitExceeded {
        limit: u32,
        window_secs: u64,
        retry_after_secs: Option<u64>,
    },
    /// Memory limit exceeded
    MemoryLimitExceeded { limit_mb: usize, used_mb: usize },
    /// Token limit exceeded
    TokenLimitExceeded { limit: usize, used: usize },
    /// Storage limit exceeded
    StorageLimitExceeded { limit_mb: usize, used_mb: usize },
    /// Concurrent request limit
    ConcurrentRequestLimit { limit: usize },
    /// Budget exceeded
    BudgetExceeded {
        budget: f64,
        used: f64,
        currency: String,
    },
}

impl std::error::Error for ResourceLimitError {}

impl fmt::Display for ResourceLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceLimitError::RateLimitExceeded {
                limit,
                window_secs,
                retry_after_secs,
            } => {
                if let Some(retry) = retry_after_secs {
                    write!(
                        f,
                        "Rate limit exceeded ({} requests per {}s), retry after {}s",
                        limit, window_secs, retry
                    )
                } else {
                    write!(
                        f,
                        "Rate limit exceeded ({} requests per {}s)",
                        limit, window_secs
                    )
                }
            }
            ResourceLimitError::MemoryLimitExceeded { limit_mb, used_mb } => {
                write!(
                    f,
                    "Memory limit exceeded: {}MB used of {}MB limit",
                    used_mb, limit_mb
                )
            }
            ResourceLimitError::TokenLimitExceeded { limit, used } => {
                write!(f, "Token limit exceeded: {} used of {} limit", used, limit)
            }
            ResourceLimitError::StorageLimitExceeded { limit_mb, used_mb } => {
                write!(
                    f,
                    "Storage limit exceeded: {}MB used of {}MB limit",
                    used_mb, limit_mb
                )
            }
            ResourceLimitError::ConcurrentRequestLimit { limit } => {
                write!(f, "Concurrent request limit exceeded: {} max", limit)
            }
            ResourceLimitError::BudgetExceeded {
                budget,
                used,
                currency,
            } => {
                write!(
                    f,
                    "Budget exceeded: {:.2} {} used of {:.2} {} budget",
                    used, currency, budget, currency
                )
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
            write!(
                f,
                "I/O error during '{}' on '{}': {}",
                self.operation, path, self.reason
            )
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

    pub fn with_path(
        operation: impl Into<String>,
        path: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
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
        write!(
            f,
            "{} {} error: {}",
            self.format, self.operation, self.reason
        )
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

// === Workflow Errors ===

/// Errors related to the event-driven workflow engine
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkflowError {
    /// Node not found in workflow graph
    NodeNotFound { node_id: String },
    /// Cycle detected in workflow graph
    CycleDetected { path: Vec<String> },
    /// Event type mismatch during dispatch
    EventTypeMismatch { expected: String, got: String },
    /// Checkpoint save/load failure
    CheckpointFailed { workflow_id: String, reason: String },
    /// Node or workflow execution timed out
    TimeoutExceeded { node_id: String, timeout_ms: u64 },
    /// Breakpoint was hit during execution
    BreakpointHit { node_id: String },
    /// Workflow serialization/deserialization failed
    SerializationFailed { reason: String },
    /// Workflow is in an invalid state for the requested operation
    InvalidState { workflow_id: String, current_state: String, attempted_action: String },
}

impl std::error::Error for WorkflowError {}

impl fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkflowError::NodeNotFound { node_id } => {
                write!(f, "Workflow node '{}' not found", node_id)
            }
            WorkflowError::CycleDetected { path } => {
                write!(f, "Cycle detected in workflow: {}", path.join(" -> "))
            }
            WorkflowError::EventTypeMismatch { expected, got } => {
                write!(f, "Event type mismatch: expected '{}', got '{}'", expected, got)
            }
            WorkflowError::CheckpointFailed { workflow_id, reason } => {
                write!(f, "Checkpoint failed for workflow '{}': {}", workflow_id, reason)
            }
            WorkflowError::TimeoutExceeded { node_id, timeout_ms } => {
                write!(f, "Node '{}' timed out after {}ms", node_id, timeout_ms)
            }
            WorkflowError::BreakpointHit { node_id } => {
                write!(f, "Breakpoint hit at node '{}'", node_id)
            }
            WorkflowError::SerializationFailed { reason } => {
                write!(f, "Workflow serialization failed: {}", reason)
            }
            WorkflowError::InvalidState { workflow_id, current_state, attempted_action } => {
                write!(f, "Workflow '{}' in state '{}', cannot perform '{}'",
                    workflow_id, current_state, attempted_action)
            }
        }
    }
}

impl WorkflowError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            WorkflowError::NodeNotFound { .. } => {
                Some("Check that the node was added to the workflow graph")
            }
            WorkflowError::CycleDetected { .. } => {
                Some("Remove circular dependencies between workflow nodes")
            }
            WorkflowError::EventTypeMismatch { .. } => {
                Some("Check that connected nodes use compatible event types")
            }
            WorkflowError::CheckpointFailed { .. } => {
                Some("Check storage permissions and available disk space")
            }
            WorkflowError::TimeoutExceeded { .. } => {
                Some("Increase node timeout or optimize the node handler")
            }
            WorkflowError::BreakpointHit { .. } => {
                Some("Call resume() to continue execution past the breakpoint")
            }
            WorkflowError::SerializationFailed { .. } => {
                Some("Check that all workflow node handlers are serializable")
            }
            WorkflowError::InvalidState { .. } => {
                Some("Check the workflow lifecycle state before performing operations")
            }
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            WorkflowError::TimeoutExceeded { .. }
                | WorkflowError::CheckpointFailed { .. }
                | WorkflowError::BreakpointHit { .. }
        )
    }
}

impl From<WorkflowError> for AiError {
    fn from(e: WorkflowError) -> Self {
        AiError::Workflow(e)
    }
}

// === Advanced Memory Errors ===

/// Errors related to the advanced memory system (episodic, procedural, entity)
#[derive(Debug)]
#[non_exhaustive]
pub enum AdvancedMemoryError {
    /// Failed to store a memory entry
    StoreFailed { memory_type: String, reason: String },
    /// Failed to recall/query memories
    RecallFailed { query: String, reason: String },
    /// Memory consolidation process failed
    ConsolidationFailed { reason: String },
    /// Entity not found in entity memory
    EntityNotFound { name: String },
    /// Duplicate entity detected during insert
    DuplicateEntity { name: String, existing_id: String },
    /// Memory capacity limit reached
    CapacityExceeded { memory_type: String, limit: usize, current: usize },
}

impl std::error::Error for AdvancedMemoryError {}

impl fmt::Display for AdvancedMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvancedMemoryError::StoreFailed { memory_type, reason } => {
                write!(f, "Failed to store {} memory: {}", memory_type, reason)
            }
            AdvancedMemoryError::RecallFailed { query, reason } => {
                write!(f, "Failed to recall memories for '{}': {}", query, reason)
            }
            AdvancedMemoryError::ConsolidationFailed { reason } => {
                write!(f, "Memory consolidation failed: {}", reason)
            }
            AdvancedMemoryError::EntityNotFound { name } => {
                write!(f, "Entity '{}' not found in memory", name)
            }
            AdvancedMemoryError::DuplicateEntity { name, existing_id } => {
                write!(f, "Duplicate entity '{}' (existing id: {})", name, existing_id)
            }
            AdvancedMemoryError::CapacityExceeded { memory_type, limit, current } => {
                write!(f, "{} memory capacity exceeded: {} of {} limit", memory_type, current, limit)
            }
        }
    }
}

impl AdvancedMemoryError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            AdvancedMemoryError::StoreFailed { .. } => {
                Some("Check storage backend availability and permissions")
            }
            AdvancedMemoryError::RecallFailed { .. } => {
                Some("Simplify the query or check that memories have been stored")
            }
            AdvancedMemoryError::ConsolidationFailed { .. } => {
                Some("Check that there are enough episodes to consolidate")
            }
            AdvancedMemoryError::EntityNotFound { .. } => {
                Some("Store the entity first or check the entity name spelling")
            }
            AdvancedMemoryError::DuplicateEntity { .. } => {
                Some("Use merge_entity() to combine duplicates instead of insert")
            }
            AdvancedMemoryError::CapacityExceeded { .. } => {
                Some("Run consolidation or increase capacity limits")
            }
        }
    }
}

impl From<AdvancedMemoryError> for AiError {
    fn from(e: AdvancedMemoryError) -> Self {
        AiError::AdvancedMemory(e)
    }
}

// === A2A Protocol Errors ===

/// Errors related to the Agent-to-Agent protocol
#[derive(Debug)]
#[non_exhaustive]
pub enum A2AError {
    /// Task not found
    TaskNotFound { task_id: String },
    /// Invalid state transition for a task
    InvalidState { task_id: String, current: String, attempted: String },
    /// Agent not found in directory
    AgentNotFound { agent_id: String },
    /// Protocol-level error (malformed request, unsupported method, etc.)
    ProtocolError { method: String, reason: String },
    /// Agent discovery failed
    DiscoveryFailed { url: String, reason: String },
    /// Authentication failed for A2A communication
    AuthenticationFailed { agent_id: String, reason: String },
    /// Task was cancelled
    TaskCancelled { task_id: String },
}

impl std::error::Error for A2AError {}

impl fmt::Display for A2AError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            A2AError::TaskNotFound { task_id } => {
                write!(f, "A2A task '{}' not found", task_id)
            }
            A2AError::InvalidState { task_id, current, attempted } => {
                write!(f, "A2A task '{}' in state '{}', cannot transition to '{}'",
                    task_id, current, attempted)
            }
            A2AError::AgentNotFound { agent_id } => {
                write!(f, "A2A agent '{}' not found in directory", agent_id)
            }
            A2AError::ProtocolError { method, reason } => {
                write!(f, "A2A protocol error in '{}': {}", method, reason)
            }
            A2AError::DiscoveryFailed { url, reason } => {
                write!(f, "A2A agent discovery failed at '{}': {}", url, reason)
            }
            A2AError::AuthenticationFailed { agent_id, reason } => {
                write!(f, "A2A authentication failed for agent '{}': {}", agent_id, reason)
            }
            A2AError::TaskCancelled { task_id } => {
                write!(f, "A2A task '{}' was cancelled", task_id)
            }
        }
    }
}

impl A2AError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            A2AError::TaskNotFound { .. } => {
                Some("Check the task ID or create a new task with tasks/send")
            }
            A2AError::InvalidState { .. } => {
                Some("Check current task status before attempting state transitions")
            }
            A2AError::AgentNotFound { .. } => {
                Some("Discover agents first via .well-known/agent.json endpoint")
            }
            A2AError::ProtocolError { .. } => {
                Some("Check JSON-RPC request format and supported methods")
            }
            A2AError::DiscoveryFailed { .. } => {
                Some("Verify the agent URL is reachable and serves an agent card")
            }
            A2AError::AuthenticationFailed { .. } => {
                Some("Check authentication credentials or API key")
            }
            A2AError::TaskCancelled { .. } => None,
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            A2AError::DiscoveryFailed { .. }
                | A2AError::ProtocolError { .. }
        )
    }
}

impl From<A2AError> for AiError {
    fn from(e: A2AError) -> Self {
        AiError::A2A(e)
    }
}

// === Voice Agent Errors ===

/// Errors related to real-time voice agent operations
#[derive(Debug)]
#[non_exhaustive]
pub enum VoiceAgentError {
    /// Audio stream connection failed
    StreamFailed { reason: String },
    /// Voice activity detection error
    VadError { reason: String },
    /// STT transcription failed
    TranscriptionFailed { reason: String },
    /// TTS synthesis failed
    SynthesisFailed { reason: String },
    /// Session state is invalid for the requested operation
    InvalidSessionState { current: String, attempted: String },
    /// Audio format not supported
    UnsupportedFormat { format: String },
}

impl std::error::Error for VoiceAgentError {}

impl fmt::Display for VoiceAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceAgentError::StreamFailed { reason } => {
                write!(f, "Audio stream failed: {}", reason)
            }
            VoiceAgentError::VadError { reason } => {
                write!(f, "Voice activity detection error: {}", reason)
            }
            VoiceAgentError::TranscriptionFailed { reason } => {
                write!(f, "Transcription failed: {}", reason)
            }
            VoiceAgentError::SynthesisFailed { reason } => {
                write!(f, "Speech synthesis failed: {}", reason)
            }
            VoiceAgentError::InvalidSessionState { current, attempted } => {
                write!(f, "Voice session in state '{}', cannot perform '{}'", current, attempted)
            }
            VoiceAgentError::UnsupportedFormat { format } => {
                write!(f, "Unsupported audio format: {}", format)
            }
        }
    }
}

impl VoiceAgentError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            VoiceAgentError::StreamFailed { .. } => Some("Check audio device and network connection"),
            VoiceAgentError::VadError { .. } => Some("Adjust VAD sensitivity thresholds"),
            VoiceAgentError::TranscriptionFailed { .. } => Some("Check STT provider availability"),
            VoiceAgentError::SynthesisFailed { .. } => Some("Check TTS provider availability"),
            VoiceAgentError::InvalidSessionState { .. } => Some("Check voice session lifecycle state"),
            VoiceAgentError::UnsupportedFormat { .. } => Some("Use PCM 16-bit 16kHz or WAV format"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            VoiceAgentError::StreamFailed { .. }
                | VoiceAgentError::TranscriptionFailed { .. }
                | VoiceAgentError::SynthesisFailed { .. }
        )
    }
}

impl From<VoiceAgentError> for AiError {
    fn from(e: VoiceAgentError) -> Self {
        AiError::VoiceAgent(e)
    }
}

// === Media Generation Errors ===

/// Errors related to image and video generation
#[derive(Debug)]
#[non_exhaustive]
pub enum MediaGenerationError {
    /// Provider not available
    ProviderUnavailable { provider: String, reason: String },
    /// Generation job failed
    GenerationFailed { provider: String, reason: String },
    /// Job timed out
    JobTimeout { job_id: String, timeout_secs: u64 },
    /// Invalid generation parameters
    InvalidParams { param: String, reason: String },
    /// Unsupported output format
    UnsupportedFormat { format: String },
    /// Content policy violation (NSFW, etc.)
    ContentPolicyViolation { reason: String },
}

impl std::error::Error for MediaGenerationError {}

impl fmt::Display for MediaGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaGenerationError::ProviderUnavailable { provider, reason } => {
                write!(f, "Media provider '{}' unavailable: {}", provider, reason)
            }
            MediaGenerationError::GenerationFailed { provider, reason } => {
                write!(f, "Generation failed on '{}': {}", provider, reason)
            }
            MediaGenerationError::JobTimeout { job_id, timeout_secs } => {
                write!(f, "Generation job '{}' timed out after {}s", job_id, timeout_secs)
            }
            MediaGenerationError::InvalidParams { param, reason } => {
                write!(f, "Invalid generation parameter '{}': {}", param, reason)
            }
            MediaGenerationError::UnsupportedFormat { format } => {
                write!(f, "Unsupported media format: {}", format)
            }
            MediaGenerationError::ContentPolicyViolation { reason } => {
                write!(f, "Content policy violation: {}", reason)
            }
        }
    }
}

impl MediaGenerationError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            MediaGenerationError::ProviderUnavailable { .. } => Some("Check API key and provider status"),
            MediaGenerationError::GenerationFailed { .. } => Some("Try a different prompt or parameters"),
            MediaGenerationError::JobTimeout { .. } => Some("Increase timeout or try a simpler prompt"),
            MediaGenerationError::InvalidParams { .. } => Some("Check parameter ranges and valid values"),
            MediaGenerationError::UnsupportedFormat { .. } => Some("Use PNG, JPEG, WebP, MP4, or WebM"),
            MediaGenerationError::ContentPolicyViolation { .. } => Some("Modify the prompt to comply with content policies"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            MediaGenerationError::ProviderUnavailable { .. }
                | MediaGenerationError::JobTimeout { .. }
        )
    }
}

impl From<MediaGenerationError> for AiError {
    fn from(e: MediaGenerationError) -> Self {
        AiError::MediaGeneration(e)
    }
}

// === Distillation Errors ===

/// Errors related to the trace-to-distillation pipeline
#[derive(Debug)]
#[non_exhaustive]
pub enum DistillationError {
    /// Trajectory collection failed
    CollectionFailed { reason: String },
    /// Trajectory scoring failed
    ScoringFailed { reason: String },
    /// Dataset build failed
    DatasetBuildFailed { format: String, reason: String },
    /// No valid trajectories found after filtering
    NoValidTrajectories { min_score: f64, total_checked: usize },
    /// Flywheel cycle failed
    FlywheelFailed { cycle_id: String, reason: String },
    /// Storage backend error
    StorageError { operation: String, reason: String },
}

impl std::error::Error for DistillationError {}

impl fmt::Display for DistillationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistillationError::CollectionFailed { reason } => {
                write!(f, "Trajectory collection failed: {}", reason)
            }
            DistillationError::ScoringFailed { reason } => {
                write!(f, "Trajectory scoring failed: {}", reason)
            }
            DistillationError::DatasetBuildFailed { format, reason } => {
                write!(f, "Dataset build failed for format '{}': {}", format, reason)
            }
            DistillationError::NoValidTrajectories { min_score, total_checked } => {
                write!(f, "No trajectories met score threshold {:.2} (checked {})", min_score, total_checked)
            }
            DistillationError::FlywheelFailed { cycle_id, reason } => {
                write!(f, "Flywheel cycle '{}' failed: {}", cycle_id, reason)
            }
            DistillationError::StorageError { operation, reason } => {
                write!(f, "Distillation storage error during '{}': {}", operation, reason)
            }
        }
    }
}

impl DistillationError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            DistillationError::CollectionFailed { .. } => Some("Check that trajectory hooks are properly registered"),
            DistillationError::ScoringFailed { .. } => Some("Verify scorer configuration and trajectory format"),
            DistillationError::DatasetBuildFailed { .. } => Some("Check output path permissions and format config"),
            DistillationError::NoValidTrajectories { .. } => Some("Lower the score threshold or collect more trajectories"),
            DistillationError::FlywheelFailed { .. } => Some("Check flywheel trigger configuration"),
            DistillationError::StorageError { .. } => Some("Check storage backend availability and permissions"),
        }
    }
}

impl From<DistillationError> for AiError {
    fn from(e: DistillationError) -> Self {
        AiError::Distillation(e)
    }
}

// === Constrained Decoding Errors ===

/// Errors related to grammar-guided constrained decoding
#[derive(Debug)]
#[non_exhaustive]
pub enum ConstrainedDecodingError {
    /// Grammar compilation failed
    GrammarCompilationFailed { reason: String },
    /// JSON Schema conversion failed
    SchemaConversionFailed { path: String, reason: String },
    /// Streaming validation detected invalid output
    ValidationFailed { position: usize, expected: String, got: String },
    /// Provider does not support grammar-guided generation
    ProviderUnsupported { provider: String },
    /// Grammar syntax error
    GrammarSyntaxError { line: usize, message: String },
}

impl std::error::Error for ConstrainedDecodingError {}

impl fmt::Display for ConstrainedDecodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstrainedDecodingError::GrammarCompilationFailed { reason } => {
                write!(f, "Grammar compilation failed: {}", reason)
            }
            ConstrainedDecodingError::SchemaConversionFailed { path, reason } => {
                write!(f, "Schema conversion failed at '{}': {}", path, reason)
            }
            ConstrainedDecodingError::ValidationFailed { position, expected, got } => {
                write!(f, "Validation failed at position {}: expected {}, got '{}'", position, expected, got)
            }
            ConstrainedDecodingError::ProviderUnsupported { provider } => {
                write!(f, "Provider '{}' does not support constrained decoding", provider)
            }
            ConstrainedDecodingError::GrammarSyntaxError { line, message } => {
                write!(f, "Grammar syntax error at line {}: {}", line, message)
            }
        }
    }
}

impl ConstrainedDecodingError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ConstrainedDecodingError::GrammarCompilationFailed { .. } => Some("Check grammar syntax and rule definitions"),
            ConstrainedDecodingError::SchemaConversionFailed { .. } => Some("Verify JSON Schema is valid and supported"),
            ConstrainedDecodingError::ValidationFailed { .. } => Some("Check that the model output matches the expected schema"),
            ConstrainedDecodingError::ProviderUnsupported { .. } => Some("Use Ollama, LM Studio, or vLLM for grammar support"),
            ConstrainedDecodingError::GrammarSyntaxError { .. } => Some("Fix the grammar syntax at the indicated line"),
        }
    }
}

impl From<ConstrainedDecodingError> for AiError {
    fn from(e: ConstrainedDecodingError) -> Self {
        AiError::ConstrainedDecoding(e)
    }
}

// === HITL Errors ===

/// Errors related to Human-in-the-Loop operations
#[derive(Debug)]
#[non_exhaustive]
pub enum HitlError {
    /// Approval request timed out waiting for human response
    ApprovalTimeout { tool_name: String, timeout_secs: u64 },
    /// Action violated an approval policy
    PolicyViolation { policy_name: String, reason: String },
    /// No approval gate configured for the operation
    GateNotConfigured { operation: String },
    /// Human correction was rejected by the system
    CorrectionRejected { step_id: String, reason: String },
    /// Confidence estimation failed
    ConfidenceEstimationFailed { reason: String },
    /// Escalation target not available
    EscalationUnavailable { target: String, reason: String },
}

impl std::error::Error for HitlError {}

impl fmt::Display for HitlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HitlError::ApprovalTimeout { tool_name, timeout_secs } => {
                write!(f, "Approval timeout for tool '{}' after {}s", tool_name, timeout_secs)
            }
            HitlError::PolicyViolation { policy_name, reason } => {
                write!(f, "Policy '{}' violated: {}", policy_name, reason)
            }
            HitlError::GateNotConfigured { operation } => {
                write!(f, "No approval gate configured for '{}'", operation)
            }
            HitlError::CorrectionRejected { step_id, reason } => {
                write!(f, "Correction rejected for step '{}': {}", step_id, reason)
            }
            HitlError::ConfidenceEstimationFailed { reason } => {
                write!(f, "Confidence estimation failed: {}", reason)
            }
            HitlError::EscalationUnavailable { target, reason } => {
                write!(f, "Escalation target '{}' unavailable: {}", target, reason)
            }
        }
    }
}

impl HitlError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            HitlError::ApprovalTimeout { .. } => Some("Increase timeout or configure auto-approve for this tool"),
            HitlError::PolicyViolation { .. } => Some("Review approval policies or adjust action parameters"),
            HitlError::GateNotConfigured { .. } => Some("Register an approval gate before executing sensitive operations"),
            HitlError::CorrectionRejected { .. } => Some("Verify the correction format and target step"),
            HitlError::ConfidenceEstimationFailed { .. } => Some("Check confidence estimator configuration"),
            HitlError::EscalationUnavailable { .. } => Some("Ensure escalation handlers are registered and reachable"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            HitlError::ApprovalTimeout { .. }
                | HitlError::EscalationUnavailable { .. }
                | HitlError::ConfidenceEstimationFailed { .. }
        )
    }
}

impl From<HitlError> for AiError {
    fn from(e: HitlError) -> Self {
        AiError::Hitl(e)
    }
}

// === MCP Client Errors ===

/// Errors related to remote MCP client connections
#[derive(Debug)]
#[non_exhaustive]
pub enum McpClientError {
    /// Failed to connect to remote MCP server
    ConnectionFailed { url: String, reason: String },
    /// Authentication with MCP server failed
    AuthFailed { url: String, reason: String },
    /// MCP server returned an error
    ServerError { url: String, code: i64, message: String },
    /// Connection timed out
    Timeout { url: String, timeout_ms: u64 },
    /// Protocol version mismatch
    ProtocolMismatch { expected: String, got: String },
    /// Tool not found on remote server
    ToolNotFound { server: String, tool_name: String },
    /// Session expired or invalid
    SessionExpired { session_id: String },
}

impl std::error::Error for McpClientError {}

impl fmt::Display for McpClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpClientError::ConnectionFailed { url, reason } => {
                write!(f, "MCP connection to '{}' failed: {}", url, reason)
            }
            McpClientError::AuthFailed { url, reason } => {
                write!(f, "MCP authentication with '{}' failed: {}", url, reason)
            }
            McpClientError::ServerError { url, code, message } => {
                write!(f, "MCP server '{}' error {}: {}", url, code, message)
            }
            McpClientError::Timeout { url, timeout_ms } => {
                write!(f, "MCP connection to '{}' timed out after {}ms", url, timeout_ms)
            }
            McpClientError::ProtocolMismatch { expected, got } => {
                write!(f, "MCP protocol mismatch: expected '{}', got '{}'", expected, got)
            }
            McpClientError::ToolNotFound { server, tool_name } => {
                write!(f, "Tool '{}' not found on MCP server '{}'", tool_name, server)
            }
            McpClientError::SessionExpired { session_id } => {
                write!(f, "MCP session '{}' expired", session_id)
            }
        }
    }
}

impl McpClientError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            McpClientError::ConnectionFailed { .. } => Some("Check the MCP server URL and network connectivity"),
            McpClientError::AuthFailed { .. } => Some("Verify OAuth credentials or bearer token"),
            McpClientError::ServerError { .. } => Some("Check MCP server logs for details"),
            McpClientError::Timeout { .. } => Some("Increase timeout or check server availability"),
            McpClientError::ProtocolMismatch { .. } => Some("Update MCP client to match server protocol version"),
            McpClientError::ToolNotFound { .. } => Some("Verify tool name and refresh the tool registry"),
            McpClientError::SessionExpired { .. } => Some("Reconnect to the MCP server to create a new session"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            McpClientError::ConnectionFailed { .. }
                | McpClientError::Timeout { .. }
                | McpClientError::SessionExpired { .. }
        )
    }
}

impl From<McpClientError> for AiError {
    fn from(e: McpClientError) -> Self {
        AiError::McpClient(e)
    }
}

// === Agent Evaluation Errors ===

/// Errors related to agent trajectory evaluation
#[derive(Debug)]
#[non_exhaustive]
pub enum AgentEvalError {
    /// Trajectory is empty — nothing to evaluate
    TrajectoryEmpty { agent_id: String },
    /// A metric computation failed
    MetricFailed { metric_name: String, reason: String },
    /// Baseline trajectory not found for comparison
    BaselineNotFound { eval_id: String },
    /// Invalid evaluation configuration
    InvalidConfig { field: String, reason: String },
    /// Tool call matching failed
    ToolCallMatchFailed { expected: String, actual: String },
    /// Report generation failed
    ReportFailed { reason: String },
}

impl std::error::Error for AgentEvalError {}

impl fmt::Display for AgentEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentEvalError::TrajectoryEmpty { agent_id } => {
                write!(f, "Trajectory for agent '{}' is empty", agent_id)
            }
            AgentEvalError::MetricFailed { metric_name, reason } => {
                write!(f, "Metric '{}' failed: {}", metric_name, reason)
            }
            AgentEvalError::BaselineNotFound { eval_id } => {
                write!(f, "Baseline not found for evaluation '{}'", eval_id)
            }
            AgentEvalError::InvalidConfig { field, reason } => {
                write!(f, "Invalid eval config '{}': {}", field, reason)
            }
            AgentEvalError::ToolCallMatchFailed { expected, actual } => {
                write!(f, "Tool call mismatch: expected '{}', got '{}'", expected, actual)
            }
            AgentEvalError::ReportFailed { reason } => {
                write!(f, "Eval report generation failed: {}", reason)
            }
        }
    }
}

impl AgentEvalError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            AgentEvalError::TrajectoryEmpty { .. } => Some("Ensure the agent ran and TrajectoryRecorder was attached"),
            AgentEvalError::MetricFailed { .. } => Some("Check metric configuration and input data"),
            AgentEvalError::BaselineNotFound { .. } => Some("Create a baseline trajectory before running comparisons"),
            AgentEvalError::InvalidConfig { .. } => Some("Review evaluation configuration parameters"),
            AgentEvalError::ToolCallMatchFailed { .. } => Some("Check expected tool call definitions"),
            AgentEvalError::ReportFailed { .. } => Some("Ensure all metrics completed before generating the report"),
        }
    }
}

impl From<AgentEvalError> for AiError {
    fn from(e: AgentEvalError) -> Self {
        AiError::AgentEval(e)
    }
}

// === Red Team Errors ===

/// Errors related to automated red teaming
#[derive(Debug)]
#[non_exhaustive]
pub enum RedTeamError {
    /// Attack generation failed
    GenerationFailed { category: String, reason: String },
    /// Attack execution failed
    ExecutionFailed { attack_id: String, reason: String },
    /// Invalid attack category
    InvalidCategory { category: String },
    /// Defense evaluation failed
    DefenseEvalFailed { guard_name: String, reason: String },
    /// Report aggregation failed
    ReportFailed { reason: String },
}

impl std::error::Error for RedTeamError {}

impl fmt::Display for RedTeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RedTeamError::GenerationFailed { category, reason } => {
                write!(f, "Attack generation failed for '{}': {}", category, reason)
            }
            RedTeamError::ExecutionFailed { attack_id, reason } => {
                write!(f, "Attack '{}' execution failed: {}", attack_id, reason)
            }
            RedTeamError::InvalidCategory { category } => {
                write!(f, "Invalid attack category: '{}'", category)
            }
            RedTeamError::DefenseEvalFailed { guard_name, reason } => {
                write!(f, "Defense evaluation failed for '{}': {}", guard_name, reason)
            }
            RedTeamError::ReportFailed { reason } => {
                write!(f, "Red team report generation failed: {}", reason)
            }
        }
    }
}

impl RedTeamError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            RedTeamError::GenerationFailed { .. } => Some("Check attack templates and generator configuration"),
            RedTeamError::ExecutionFailed { .. } => Some("Verify the target agent is accessible"),
            RedTeamError::InvalidCategory { .. } => Some("Use a valid AttackCategory variant"),
            RedTeamError::DefenseEvalFailed { .. } => Some("Ensure guardrails are properly configured"),
            RedTeamError::ReportFailed { .. } => Some("Check that all attacks completed before reporting"),
        }
    }
}

impl From<RedTeamError> for AiError {
    fn from(e: RedTeamError) -> Self {
        AiError::RedTeam(e)
    }
}

// === MCTS Planning Errors ===

/// Errors related to Monte Carlo Tree Search planning
#[derive(Debug)]
#[non_exhaustive]
pub enum MctsError {
    /// Maximum iterations reached without finding a solution
    MaxIterations { iterations: usize, best_reward: f64 },
    /// No valid actions available from current state
    NoValidActions { state_description: String },
    /// Simulation failed
    SimulationFailed { depth: usize, reason: String },
    /// State transition error
    StateError { action: String, reason: String },
    /// Reward model error
    RewardModelError { step: usize, reason: String },
    /// Refinement loop exhausted
    RefinementExhausted { iterations: usize, last_improvement: f64 },
}

impl std::error::Error for MctsError {}

impl fmt::Display for MctsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MctsError::MaxIterations { iterations, best_reward } => {
                write!(f, "MCTS reached {} iterations, best reward: {:.4}", iterations, best_reward)
            }
            MctsError::NoValidActions { state_description } => {
                write!(f, "No valid actions from state: {}", state_description)
            }
            MctsError::SimulationFailed { depth, reason } => {
                write!(f, "MCTS simulation failed at depth {}: {}", depth, reason)
            }
            MctsError::StateError { action, reason } => {
                write!(f, "State error for action '{}': {}", action, reason)
            }
            MctsError::RewardModelError { step, reason } => {
                write!(f, "Reward model error at step {}: {}", step, reason)
            }
            MctsError::RefinementExhausted { iterations, last_improvement } => {
                write!(f, "Refinement exhausted after {} iterations (last improvement: {:.4})", iterations, last_improvement)
            }
        }
    }
}

impl MctsError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            MctsError::MaxIterations { .. } => Some("Increase max_iterations or adjust exploration constant"),
            MctsError::NoValidActions { .. } => Some("Check state implementation returns available actions"),
            MctsError::SimulationFailed { .. } => Some("Review simulation policy and state transitions"),
            MctsError::StateError { .. } => Some("Verify state transition logic for this action"),
            MctsError::RewardModelError { .. } => Some("Check reward model configuration and input format"),
            MctsError::RefinementExhausted { .. } => Some("Lower improvement threshold or increase iteration limit"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            MctsError::MaxIterations { .. }
                | MctsError::SimulationFailed { .. }
                | MctsError::RefinementExhausted { .. }
        )
    }
}

impl From<MctsError> for AiError {
    fn from(e: MctsError) -> Self {
        AiError::Mcts(e)
    }
}

// === DevTools Errors ===

/// Errors related to agent debugging and profiling tools
#[derive(Debug)]
#[non_exhaustive]
pub enum DevToolsError {
    /// Recording failed
    RecordingFailed { agent_id: String, reason: String },
    /// Replay failed
    ReplayFailed { recording_id: String, reason: String },
    /// Invalid breakpoint configuration
    BreakpointInvalid { description: String },
    /// State inspection failed
    InspectionFailed { agent_id: String, reason: String },
    /// Profiling data unavailable
    ProfilingUnavailable { reason: String },
}

impl std::error::Error for DevToolsError {}

impl fmt::Display for DevToolsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DevToolsError::RecordingFailed { agent_id, reason } => {
                write!(f, "Recording failed for agent '{}': {}", agent_id, reason)
            }
            DevToolsError::ReplayFailed { recording_id, reason } => {
                write!(f, "Replay failed for recording '{}': {}", recording_id, reason)
            }
            DevToolsError::BreakpointInvalid { description } => {
                write!(f, "Invalid breakpoint: {}", description)
            }
            DevToolsError::InspectionFailed { agent_id, reason } => {
                write!(f, "State inspection failed for agent '{}': {}", agent_id, reason)
            }
            DevToolsError::ProfilingUnavailable { reason } => {
                write!(f, "Profiling data unavailable: {}", reason)
            }
        }
    }
}

impl DevToolsError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            DevToolsError::RecordingFailed { .. } => Some("Ensure ExecutionRecorder is attached before agent runs"),
            DevToolsError::ReplayFailed { .. } => Some("Verify the recording file exists and is not corrupted"),
            DevToolsError::BreakpointInvalid { .. } => Some("Check breakpoint conditions and target identifiers"),
            DevToolsError::InspectionFailed { .. } => Some("Ensure the agent supports state inspection"),
            DevToolsError::ProfilingUnavailable { .. } => Some("Enable profiling in DevToolsConfig before running"),
        }
    }
}

impl From<DevToolsError> for AiError {
    fn from(e: DevToolsError) -> Self {
        AiError::DevTools(e)
    }
}

// === Eval Suite Errors ===

/// Errors related to evaluation benchmark suite execution
#[derive(Debug)]
#[non_exhaustive]
pub enum EvalSuiteError {
    /// Benchmark dataset file not found or unreadable
    DatasetLoadFailed { path: String, reason: String },
    /// Invalid problem format in dataset
    InvalidProblem { problem_id: String, reason: String },
    /// LLM generation failed during benchmark execution
    GenerationFailed { problem_id: String, reason: String },
    /// Response scoring failed
    ScoringFailed { problem_id: String, reason: String },
    /// No results available for analysis
    NoResults { reason: String },
    /// Insufficient data for statistical analysis
    InsufficientData { metric: String, samples: usize },
    /// Report generation failed
    ReportFailed { reason: String },
    /// Evaluation timed out
    Timeout { problem_id: String, timeout_secs: u64 },
    /// Configuration search failed
    SearchFailed { reason: String },
    /// Invalid agent configuration
    InvalidAgentConfig { field: String, reason: String },
}

impl std::error::Error for EvalSuiteError {}

impl fmt::Display for EvalSuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalSuiteError::DatasetLoadFailed { path, reason } => {
                write!(f, "Failed to load dataset '{}': {}", path, reason)
            }
            EvalSuiteError::InvalidProblem { problem_id, reason } => {
                write!(f, "Invalid problem '{}': {}", problem_id, reason)
            }
            EvalSuiteError::GenerationFailed { problem_id, reason } => {
                write!(f, "Generation failed for problem '{}': {}", problem_id, reason)
            }
            EvalSuiteError::ScoringFailed { problem_id, reason } => {
                write!(f, "Scoring failed for problem '{}': {}", problem_id, reason)
            }
            EvalSuiteError::NoResults { reason } => {
                write!(f, "No results available: {}", reason)
            }
            EvalSuiteError::InsufficientData { metric, samples } => {
                write!(f, "Insufficient data for '{}': only {} samples", metric, samples)
            }
            EvalSuiteError::ReportFailed { reason } => {
                write!(f, "Report generation failed: {}", reason)
            }
            EvalSuiteError::Timeout { problem_id, timeout_secs } => {
                write!(f, "Evaluation timed out for '{}' after {}s", problem_id, timeout_secs)
            }
            EvalSuiteError::SearchFailed { reason } => {
                write!(f, "Configuration search failed: {}", reason)
            }
            EvalSuiteError::InvalidAgentConfig { field, reason } => {
                write!(f, "Invalid agent config field '{}': {}", field, reason)
            }
        }
    }
}

impl EvalSuiteError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            EvalSuiteError::DatasetLoadFailed { .. } => Some("Check the dataset file path and format (JSONL or JSON)"),
            EvalSuiteError::InvalidProblem { .. } => Some("Verify the problem has all required fields (id, prompt, answer_format)"),
            EvalSuiteError::GenerationFailed { .. } => Some("Check that the LLM provider is accessible and the model is available"),
            EvalSuiteError::ScoringFailed { .. } => Some("Verify the scorer matches the problem's answer format"),
            EvalSuiteError::NoResults { .. } => Some("Run at least one benchmark before generating reports"),
            EvalSuiteError::InsufficientData { .. } => Some("Collect more samples or lower the significance threshold"),
            EvalSuiteError::ReportFailed { .. } => Some("Ensure all benchmark runs completed before generating the report"),
            EvalSuiteError::Timeout { .. } => Some("Increase the timeout or use a faster model"),
            EvalSuiteError::SearchFailed { .. } => Some("Check search dimensions and dataset, or increase max_evaluations budget"),
            EvalSuiteError::InvalidAgentConfig { .. } => Some("Verify the EvalAgentConfig fields (model identifiers, temperature range, etc.)"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            EvalSuiteError::GenerationFailed { .. }
                | EvalSuiteError::Timeout { .. }
                | EvalSuiteError::SearchFailed { .. }
        )
    }
}

impl From<EvalSuiteError> for AiError {
    fn from(e: EvalSuiteError) -> Self {
        AiError::EvalSuite(e)
    }
}

// === Advanced Routing Errors ===

/// Errors specific to the advanced routing system (bandits, NFA/DFA, DAGs, ensembles).
#[derive(Debug)]
#[non_exhaustive]
pub enum AdvancedRoutingError {
    /// Invalid routing configuration
    InvalidConfig { field: String, reason: String },
    /// Bandit arm not found
    ArmNotFound { arm_id: String },
    /// NFA/DFA compilation error
    CompilationError { reason: String },
    /// Cycle detected in routing DAG
    CycleDetected,
    /// Node not found in routing DAG
    NodeNotFound { node_id: String },
    /// Empty router ensemble (no sub-routers registered)
    EmptyEnsemble,
    /// Serialization or deserialization failed
    SerializationFailed { format: String, reason: String },
    /// Incompatible snapshot version
    IncompatibleVersion { expected: u32, found: u32 },
    /// No valid routing path found for the query
    NoRoutingPath { query: String, reason: String },
    /// Distributed merge conflict
    #[cfg(feature = "distributed")]
    MergeConflict { reason: String },
}

impl std::error::Error for AdvancedRoutingError {}

impl fmt::Display for AdvancedRoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvancedRoutingError::InvalidConfig { field, reason } => {
                write!(f, "Invalid routing config field '{}': {}", field, reason)
            }
            AdvancedRoutingError::ArmNotFound { arm_id } => {
                write!(f, "Bandit arm '{}' not found", arm_id)
            }
            AdvancedRoutingError::CompilationError { reason } => {
                write!(f, "NFA/DFA compilation failed: {}", reason)
            }
            AdvancedRoutingError::CycleDetected => {
                write!(f, "Cycle detected in routing DAG")
            }
            AdvancedRoutingError::NodeNotFound { node_id } => {
                write!(f, "Routing DAG node '{}' not found", node_id)
            }
            AdvancedRoutingError::EmptyEnsemble => {
                write!(f, "Ensemble router has no sub-routers")
            }
            AdvancedRoutingError::SerializationFailed { format, reason } => {
                write!(f, "Serialization failed ({}): {}", format, reason)
            }
            AdvancedRoutingError::IncompatibleVersion { expected, found } => {
                write!(f, "Incompatible snapshot version: expected {}, found {}", expected, found)
            }
            AdvancedRoutingError::NoRoutingPath { query, reason } => {
                write!(f, "No routing path for query '{}': {}", query, reason)
            }
            #[cfg(feature = "distributed")]
            AdvancedRoutingError::MergeConflict { reason } => {
                write!(f, "Distributed bandit merge conflict: {}", reason)
            }
        }
    }
}

impl AdvancedRoutingError {
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            AdvancedRoutingError::InvalidConfig { .. } => Some("Check the routing configuration fields and value ranges"),
            AdvancedRoutingError::ArmNotFound { .. } => Some("Register the arm with add_arm() before selecting"),
            AdvancedRoutingError::CompilationError { .. } => Some("Verify the NFA has valid states and transitions"),
            AdvancedRoutingError::CycleDetected => Some("Remove cycles from the routing DAG to make it acyclic"),
            AdvancedRoutingError::NodeNotFound { .. } => Some("Add the node with add_node() before referencing it"),
            AdvancedRoutingError::EmptyEnsemble => Some("Add at least one voter with add_voter() before routing"),
            AdvancedRoutingError::SerializationFailed { .. } => Some("Check that the data format matches the expected schema"),
            AdvancedRoutingError::IncompatibleVersion { .. } => Some("Export a new snapshot from the current version"),
            AdvancedRoutingError::NoRoutingPath { .. } => Some("Add transitions or accepting states that match the query features"),
            #[cfg(feature = "distributed")]
            AdvancedRoutingError::MergeConflict { .. } => Some("Ensure all nodes use compatible bandit configurations"),
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            AdvancedRoutingError::NoRoutingPath { .. }
                | AdvancedRoutingError::ArmNotFound { .. }
        )
    }
}

impl From<AdvancedRoutingError> for AiError {
    fn from(e: AdvancedRoutingError) -> Self {
        AiError::AdvancedRouting(e)
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

// === Contextual error wrapper ===

/// An error enriched with additional context describing what was being done
/// when the error occurred. Supports error chaining via `std::error::Error::source()`.
#[derive(Debug)]
pub struct ContextualError {
    /// The contextual message describing what operation was being performed
    pub context: String,
    /// The underlying error
    pub source: AiError,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl AiError {
    /// Wrap this error with additional context describing what was being done
    /// when the error occurred.
    ///
    /// # Example
    /// ```ignore
    /// let result = load_config(path)
    ///     .map_err(|e| e.with_context("loading main configuration"));
    /// ```
    pub fn with_context(self, context: impl Into<String>) -> ContextualError {
        ContextualError {
            context: context.into(),
            source: self,
        }
    }
}

/// Extension trait for `Result<T, AiError>` to add context conveniently.
pub trait ResultExt<T> {
    /// Add context to an error result, describing what operation was being performed.
    ///
    /// # Example
    /// ```ignore
    /// use ai_assistant::error::ResultExt;
    ///
    /// fn load_model(name: &str) -> AiResult<Model> {
    ///     fetch_from_registry(name)
    ///         .context(format!("loading model '{}'", name))
    /// }
    /// ```
    fn context(self, msg: impl Into<String>) -> Result<T, ContextualError>;

    /// Add lazy context to an error result (closure only called on error).
    fn with_context_fn<F>(self, f: F) -> Result<T, ContextualError>
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T, AiError> {
    fn context(self, msg: impl Into<String>) -> Result<T, ContextualError> {
        self.map_err(|e| e.with_context(msg))
    }

    fn with_context_fn<F>(self, f: F) -> Result<T, ContextualError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| e.with_context(f()))
    }
}

impl From<ContextualError> for AiError {
    fn from(ctx: ContextualError) -> Self {
        AiError::Other(format!("{}: {}", ctx.context, ctx.source))
    }
}

// === Error builder helpers ===

impl AiError {
    /// Create a provider unavailable error
    pub fn provider_unavailable(provider: impl Into<String>, url: impl Into<String>) -> Self {
        ProviderError::Unavailable {
            provider: provider.into(),
            url: url.into(),
        }
        .into()
    }

    /// Create a model not found error
    pub fn model_not_found(provider: impl Into<String>, model: impl Into<String>) -> Self {
        ProviderError::ModelNotFound {
            provider: provider.into(),
            model: model.into(),
        }
        .into()
    }

    /// Create a context length exceeded error
    pub fn context_exceeded(max_tokens: usize, used_tokens: usize) -> Self {
        ProviderError::ContextLengthExceeded {
            max_tokens,
            used_tokens,
        }
        .into()
    }

    /// Create a rate limit error
    pub fn rate_limited(limit: u32, window_secs: u64) -> Self {
        ResourceLimitError::RateLimitExceeded {
            limit,
            window_secs,
            retry_after_secs: None,
        }
        .into()
    }

    /// Create an append-only violation error
    pub fn append_only_violation(operation: impl Into<String>, source: impl Into<String>) -> Self {
        RagError::AppendOnlyViolation {
            operation: operation.into(),
            source: source.into(),
        }
        .into()
    }

    /// Create a validation empty input error
    pub fn empty_input(field: impl Into<String>) -> Self {
        ValidationError::EmptyInput {
            field: field.into(),
        }
        .into()
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

    #[test]
    fn test_all_error_display_variants() {
        let errors: Vec<AiError> = vec![
            AiError::Config(ConfigError::MissingValue {
                field: "key".into(),
                description: "required".into(),
            }),
            AiError::Provider(ProviderError::Cancelled),
            AiError::Rag(RagError::DocumentNotFound("doc.txt".into())),
            AiError::Network(NetworkError::DnsError {
                host: "example.com".into(),
            }),
            AiError::Validation(ValidationError::EmptyInput {
                field: "name".into(),
            }),
            AiError::ResourceLimit(ResourceLimitError::ConcurrentRequestLimit { limit: 5 }),
            AiError::Io(IoError::new("read", "permission denied")),
            AiError::Serialization(SerializationError::json_serialize("bad data")),
            AiError::Workflow(WorkflowError::BreakpointHit { node_id: "n".into() }),
            AiError::AdvancedMemory(AdvancedMemoryError::ConsolidationFailed { reason: "r".into() }),
            AiError::A2A(A2AError::TaskCancelled { task_id: "t".into() }),
            AiError::VoiceAgent(VoiceAgentError::StreamFailed { reason: "disconnected".into() }),
            AiError::MediaGeneration(MediaGenerationError::GenerationFailed { provider: "dalle".into(), reason: "timeout".into() }),
            AiError::Distillation(DistillationError::NoValidTrajectories { min_score: 0.8, total_checked: 100 }),
            AiError::ConstrainedDecoding(ConstrainedDecodingError::ProviderUnsupported { provider: "openai".into() }),
            AiError::Hitl(HitlError::ApprovalTimeout { tool_name: "delete".into(), timeout_secs: 30 }),
            AiError::McpClient(McpClientError::ConnectionFailed { url: "http://mcp.local".into(), reason: "refused".into() }),
            AiError::AgentEval(AgentEvalError::TrajectoryEmpty { agent_id: "agent-1".into() }),
            AiError::RedTeam(RedTeamError::GenerationFailed { category: "injection".into(), reason: "template".into() }),
            AiError::Mcts(MctsError::MaxIterations { iterations: 1000, best_reward: 0.75 }),
            AiError::DevTools(DevToolsError::RecordingFailed { agent_id: "a".into(), reason: "no recorder".into() }),
            AiError::EvalSuite(EvalSuiteError::DatasetLoadFailed { path: "bench.jsonl".into(), reason: "not found".into() }),
            AiError::Other("something went wrong".into()),
        ];

        for err in &errors {
            let display = err.to_string();
            assert!(
                !display.is_empty(),
                "Display for {:?} should be non-empty",
                err
            );
        }
    }

    #[test]
    fn test_config_error_suggestions() {
        let config_errors: Vec<ConfigError> = vec![
            ConfigError::MissingValue {
                field: "api_key".into(),
                description: "needed".into(),
            },
            ConfigError::InvalidValue {
                field: "port".into(),
                value: "abc".into(),
                expected: "integer".into(),
            },
            ConfigError::LoadFailed {
                path: "/tmp/config.toml".into(),
                reason: "not found".into(),
            },
            ConfigError::SaveFailed {
                path: "/tmp/config.toml".into(),
                reason: "read-only".into(),
            },
            ConfigError::UnknownProvider("foo_provider".into()),
        ];

        for err in &config_errors {
            let suggestion = err.suggestion();
            assert!(
                suggestion.is_some(),
                "ConfigError {:?} should have a suggestion",
                err
            );
            assert!(
                !suggestion.unwrap().is_empty(),
                "Suggestion for {:?} should be non-empty",
                err
            );
        }
    }

    #[test]
    fn test_network_error_recoverable() {
        // ConnectionFailed is recoverable
        let conn_err = NetworkError::ConnectionFailed {
            url: "http://localhost:8080".into(),
            reason: "refused".into(),
        };
        assert!(
            conn_err.is_recoverable(),
            "ConnectionFailed should be recoverable"
        );

        // Timeout is recoverable
        let timeout_err = NetworkError::Timeout {
            url: "http://localhost:8080".into(),
            timeout_ms: 5000,
        };
        assert!(
            timeout_err.is_recoverable(),
            "Timeout should be recoverable"
        );

        // DnsError is NOT recoverable
        let dns_err = NetworkError::DnsError {
            host: "bad.host".into(),
        };
        assert!(
            !dns_err.is_recoverable(),
            "DnsError should not be recoverable"
        );

        // TlsError is NOT recoverable
        let tls_err = NetworkError::TlsError {
            url: "https://example.com".into(),
            reason: "cert expired".into(),
        };
        assert!(
            !tls_err.is_recoverable(),
            "TlsError should not be recoverable"
        );

        // HttpError with 500 is recoverable, 404 is not
        let http_500 = NetworkError::HttpError {
            status: 500,
            message: "server error".into(),
        };
        assert!(http_500.is_recoverable(), "HTTP 500 should be recoverable");

        let http_404 = NetworkError::HttpError {
            status: 404,
            message: "not found".into(),
        };
        assert!(
            !http_404.is_recoverable(),
            "HTTP 404 should not be recoverable"
        );
    }

    #[test]
    fn test_from_conversions() {
        // ConfigError -> AiError::Config
        let config_err = ConfigError::UnknownProvider("test".into());
        let ai_err: AiError = config_err.into();
        assert_eq!(ai_err.code(), "CONFIG");

        // std::io::Error -> AiError::Io
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let ai_err: AiError = io_err.into();
        assert_eq!(ai_err.code(), "IO");
        assert!(ai_err.to_string().contains("file missing"));

        // ValidationError -> AiError::Validation
        let val_err = ValidationError::EmptyInput {
            field: "name".into(),
        };
        let ai_err: AiError = val_err.into();
        assert_eq!(ai_err.code(), "VALIDATION");

        // NetworkError -> AiError::Network
        let net_err = NetworkError::DnsError {
            host: "example.com".into(),
        };
        let ai_err: AiError = net_err.into();
        assert_eq!(ai_err.code(), "NETWORK");

        // SerializationError -> AiError::Serialization
        let ser_err = SerializationError::json_deserialize("unexpected token");
        let ai_err: AiError = ser_err.into();
        assert_eq!(ai_err.code(), "SERIALIZATION");
    }

    #[test]
    fn test_workflow_error_display_and_suggestion() {
        let errors: Vec<WorkflowError> = vec![
            WorkflowError::NodeNotFound { node_id: "step_1".into() },
            WorkflowError::CycleDetected { path: vec!["a".into(), "b".into(), "a".into()] },
            WorkflowError::EventTypeMismatch { expected: "QueryEvent".into(), got: "ResultEvent".into() },
            WorkflowError::CheckpointFailed { workflow_id: "wf-1".into(), reason: "disk full".into() },
            WorkflowError::TimeoutExceeded { node_id: "slow_node".into(), timeout_ms: 5000 },
            WorkflowError::BreakpointHit { node_id: "debug_node".into() },
            WorkflowError::SerializationFailed { reason: "invalid handler".into() },
            WorkflowError::InvalidState {
                workflow_id: "wf-1".into(),
                current_state: "completed".into(),
                attempted_action: "resume".into(),
            },
        ];

        for err in &errors {
            let display = err.to_string();
            assert!(!display.is_empty(), "Display for {:?} should be non-empty", err);
            assert!(err.suggestion().is_some(), "WorkflowError {:?} should have a suggestion", err);
        }
    }

    #[test]
    fn test_workflow_error_recoverable() {
        assert!(WorkflowError::TimeoutExceeded { node_id: "n".into(), timeout_ms: 100 }.is_recoverable());
        assert!(WorkflowError::CheckpointFailed { workflow_id: "w".into(), reason: "r".into() }.is_recoverable());
        assert!(WorkflowError::BreakpointHit { node_id: "n".into() }.is_recoverable());
        assert!(!WorkflowError::NodeNotFound { node_id: "n".into() }.is_recoverable());
        assert!(!WorkflowError::CycleDetected { path: vec![] }.is_recoverable());
    }

    #[test]
    fn test_advanced_memory_error_display_and_suggestion() {
        let errors: Vec<AdvancedMemoryError> = vec![
            AdvancedMemoryError::StoreFailed { memory_type: "episodic".into(), reason: "db locked".into() },
            AdvancedMemoryError::RecallFailed { query: "what happened".into(), reason: "no index".into() },
            AdvancedMemoryError::ConsolidationFailed { reason: "too few episodes".into() },
            AdvancedMemoryError::EntityNotFound { name: "Project X".into() },
            AdvancedMemoryError::DuplicateEntity { name: "Python".into(), existing_id: "e-42".into() },
            AdvancedMemoryError::CapacityExceeded { memory_type: "episodic".into(), limit: 1000, current: 1001 },
        ];

        for err in &errors {
            let display = err.to_string();
            assert!(!display.is_empty(), "Display for {:?} should be non-empty", err);
            assert!(err.suggestion().is_some(), "AdvancedMemoryError {:?} should have a suggestion", err);
        }
    }

    #[test]
    fn test_a2a_error_display_and_suggestion() {
        let errors: Vec<A2AError> = vec![
            A2AError::TaskNotFound { task_id: "task-123".into() },
            A2AError::InvalidState { task_id: "task-1".into(), current: "working".into(), attempted: "submitted".into() },
            A2AError::AgentNotFound { agent_id: "agent-42".into() },
            A2AError::ProtocolError { method: "tasks/send".into(), reason: "missing params".into() },
            A2AError::DiscoveryFailed { url: "https://example.com".into(), reason: "timeout".into() },
            A2AError::AuthenticationFailed { agent_id: "agent-1".into(), reason: "bad key".into() },
            A2AError::TaskCancelled { task_id: "task-99".into() },
        ];

        for err in &errors {
            let display = err.to_string();
            assert!(!display.is_empty(), "Display for {:?} should be non-empty", err);
            // TaskCancelled has no suggestion — that's ok
            if !matches!(err, A2AError::TaskCancelled { .. }) {
                assert!(err.suggestion().is_some(), "A2AError {:?} should have a suggestion", err);
            }
        }
    }

    #[test]
    fn test_a2a_error_recoverable() {
        assert!(A2AError::DiscoveryFailed { url: "u".into(), reason: "r".into() }.is_recoverable());
        assert!(A2AError::ProtocolError { method: "m".into(), reason: "r".into() }.is_recoverable());
        assert!(!A2AError::TaskNotFound { task_id: "t".into() }.is_recoverable());
        assert!(!A2AError::TaskCancelled { task_id: "t".into() }.is_recoverable());
    }

    #[test]
    fn test_new_error_from_conversions() {
        // WorkflowError -> AiError::Workflow
        let wf_err = WorkflowError::NodeNotFound { node_id: "test".into() };
        let ai_err: AiError = wf_err.into();
        assert_eq!(ai_err.code(), "WORKFLOW");
        assert!(ai_err.to_string().contains("test"));

        // AdvancedMemoryError -> AiError::AdvancedMemory
        let mem_err = AdvancedMemoryError::EntityNotFound { name: "entity".into() };
        let ai_err: AiError = mem_err.into();
        assert_eq!(ai_err.code(), "MEMORY");
        assert!(ai_err.to_string().contains("entity"));

        // A2AError -> AiError::A2A
        let a2a_err = A2AError::TaskNotFound { task_id: "task-1".into() };
        let ai_err: AiError = a2a_err.into();
        assert_eq!(ai_err.code(), "A2A");
        assert!(ai_err.to_string().contains("task-1"));
    }

    #[test]
    fn test_voice_agent_error_display_and_suggestion() {
        let errors: Vec<VoiceAgentError> = vec![
            VoiceAgentError::StreamFailed { reason: "disconnected".into() },
            VoiceAgentError::VadError { reason: "threshold".into() },
            VoiceAgentError::TranscriptionFailed { reason: "timeout".into() },
            VoiceAgentError::SynthesisFailed { reason: "no voice".into() },
            VoiceAgentError::InvalidSessionState { current: "idle".into(), attempted: "resume".into() },
            VoiceAgentError::UnsupportedFormat { format: "aac".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_voice_agent_error_recoverable() {
        assert!(VoiceAgentError::StreamFailed { reason: "r".into() }.is_recoverable());
        assert!(VoiceAgentError::TranscriptionFailed { reason: "r".into() }.is_recoverable());
        assert!(!VoiceAgentError::InvalidSessionState { current: "a".into(), attempted: "b".into() }.is_recoverable());
        assert!(!VoiceAgentError::UnsupportedFormat { format: "f".into() }.is_recoverable());
    }

    #[test]
    fn test_media_generation_error_display_and_suggestion() {
        let errors: Vec<MediaGenerationError> = vec![
            MediaGenerationError::ProviderUnavailable { provider: "dalle".into(), reason: "timeout".into() },
            MediaGenerationError::GenerationFailed { provider: "sd".into(), reason: "oom".into() },
            MediaGenerationError::JobTimeout { job_id: "j-1".into(), timeout_secs: 300 },
            MediaGenerationError::InvalidParams { param: "width".into(), reason: "too large".into() },
            MediaGenerationError::UnsupportedFormat { format: "bmp".into() },
            MediaGenerationError::ContentPolicyViolation { reason: "nsfw".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_media_generation_error_recoverable() {
        assert!(MediaGenerationError::ProviderUnavailable { provider: "p".into(), reason: "r".into() }.is_recoverable());
        assert!(MediaGenerationError::JobTimeout { job_id: "j".into(), timeout_secs: 60 }.is_recoverable());
        assert!(!MediaGenerationError::GenerationFailed { provider: "p".into(), reason: "r".into() }.is_recoverable());
        assert!(!MediaGenerationError::ContentPolicyViolation { reason: "r".into() }.is_recoverable());
    }

    #[test]
    fn test_distillation_error_display_and_suggestion() {
        let errors: Vec<DistillationError> = vec![
            DistillationError::CollectionFailed { reason: "no hooks".into() },
            DistillationError::ScoringFailed { reason: "nan".into() },
            DistillationError::DatasetBuildFailed { format: "openai".into(), reason: "io".into() },
            DistillationError::NoValidTrajectories { min_score: 0.9, total_checked: 50 },
            DistillationError::FlywheelFailed { cycle_id: "c-1".into(), reason: "trigger".into() },
            DistillationError::StorageError { operation: "write".into(), reason: "full".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_constrained_decoding_error_display_and_suggestion() {
        let errors: Vec<ConstrainedDecodingError> = vec![
            ConstrainedDecodingError::GrammarCompilationFailed { reason: "bad rule".into() },
            ConstrainedDecodingError::SchemaConversionFailed { path: "$.items".into(), reason: "unsupported".into() },
            ConstrainedDecodingError::ValidationFailed { position: 42, expected: "string".into(), got: "123".into() },
            ConstrainedDecodingError::ProviderUnsupported { provider: "openai".into() },
            ConstrainedDecodingError::GrammarSyntaxError { line: 5, message: "unexpected token".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_hitl_error_display_and_suggestion() {
        let errors: Vec<HitlError> = vec![
            HitlError::ApprovalTimeout { tool_name: "delete_file".into(), timeout_secs: 30 },
            HitlError::PolicyViolation { policy_name: "no-destructive".into(), reason: "tool is destructive".into() },
            HitlError::GateNotConfigured { operation: "deploy".into() },
            HitlError::CorrectionRejected { step_id: "s-1".into(), reason: "invalid format".into() },
            HitlError::ConfidenceEstimationFailed { reason: "no signals".into() },
            HitlError::EscalationUnavailable { target: "supervisor".into(), reason: "offline".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_hitl_error_recoverable() {
        assert!(HitlError::ApprovalTimeout { tool_name: "t".into(), timeout_secs: 10 }.is_recoverable());
        assert!(HitlError::EscalationUnavailable { target: "t".into(), reason: "r".into() }.is_recoverable());
        assert!(HitlError::ConfidenceEstimationFailed { reason: "r".into() }.is_recoverable());
        assert!(!HitlError::PolicyViolation { policy_name: "p".into(), reason: "r".into() }.is_recoverable());
        assert!(!HitlError::GateNotConfigured { operation: "o".into() }.is_recoverable());
        assert!(!HitlError::CorrectionRejected { step_id: "s".into(), reason: "r".into() }.is_recoverable());
    }

    #[test]
    fn test_mcp_client_error_display_and_suggestion() {
        let errors: Vec<McpClientError> = vec![
            McpClientError::ConnectionFailed { url: "http://mcp.local:3000".into(), reason: "refused".into() },
            McpClientError::AuthFailed { url: "http://mcp.local:3000".into(), reason: "invalid token".into() },
            McpClientError::ServerError { url: "http://mcp.local".into(), code: -32600, message: "invalid request".into() },
            McpClientError::Timeout { url: "http://mcp.local".into(), timeout_ms: 5000 },
            McpClientError::ProtocolMismatch { expected: "2025-11-05".into(), got: "2024-11-05".into() },
            McpClientError::ToolNotFound { server: "mcp.local".into(), tool_name: "search".into() },
            McpClientError::SessionExpired { session_id: "sess-123".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_mcp_client_error_recoverable() {
        assert!(McpClientError::ConnectionFailed { url: "u".into(), reason: "r".into() }.is_recoverable());
        assert!(McpClientError::Timeout { url: "u".into(), timeout_ms: 100 }.is_recoverable());
        assert!(McpClientError::SessionExpired { session_id: "s".into() }.is_recoverable());
        assert!(!McpClientError::AuthFailed { url: "u".into(), reason: "r".into() }.is_recoverable());
        assert!(!McpClientError::ProtocolMismatch { expected: "a".into(), got: "b".into() }.is_recoverable());
        assert!(!McpClientError::ToolNotFound { server: "s".into(), tool_name: "t".into() }.is_recoverable());
    }

    #[test]
    fn test_agent_eval_error_display_and_suggestion() {
        let errors: Vec<AgentEvalError> = vec![
            AgentEvalError::TrajectoryEmpty { agent_id: "agent-1".into() },
            AgentEvalError::MetricFailed { metric_name: "accuracy".into(), reason: "div by zero".into() },
            AgentEvalError::BaselineNotFound { eval_id: "eval-42".into() },
            AgentEvalError::InvalidConfig { field: "top_k".into(), reason: "must be > 0".into() },
            AgentEvalError::ToolCallMatchFailed { expected: "search".into(), actual: "browse".into() },
            AgentEvalError::ReportFailed { reason: "incomplete data".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_red_team_error_display_and_suggestion() {
        let errors: Vec<RedTeamError> = vec![
            RedTeamError::GenerationFailed { category: "jailbreak".into(), reason: "template parse".into() },
            RedTeamError::ExecutionFailed { attack_id: "atk-1".into(), reason: "target unreachable".into() },
            RedTeamError::InvalidCategory { category: "unknown".into() },
            RedTeamError::DefenseEvalFailed { guard_name: "pii".into(), reason: "timeout".into() },
            RedTeamError::ReportFailed { reason: "aggregation".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_mcts_error_display_and_suggestion() {
        let errors: Vec<MctsError> = vec![
            MctsError::MaxIterations { iterations: 1000, best_reward: 0.72 },
            MctsError::NoValidActions { state_description: "terminal state".into() },
            MctsError::SimulationFailed { depth: 5, reason: "invalid transition".into() },
            MctsError::StateError { action: "search".into(), reason: "state locked".into() },
            MctsError::RewardModelError { step: 3, reason: "nan score".into() },
            MctsError::RefinementExhausted { iterations: 10, last_improvement: 0.001 },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_mcts_error_recoverable() {
        assert!(MctsError::MaxIterations { iterations: 100, best_reward: 0.5 }.is_recoverable());
        assert!(MctsError::SimulationFailed { depth: 1, reason: "r".into() }.is_recoverable());
        assert!(MctsError::RefinementExhausted { iterations: 5, last_improvement: 0.0 }.is_recoverable());
        assert!(!MctsError::NoValidActions { state_description: "s".into() }.is_recoverable());
        assert!(!MctsError::StateError { action: "a".into(), reason: "r".into() }.is_recoverable());
    }

    #[test]
    fn test_devtools_error_display_and_suggestion() {
        let errors: Vec<DevToolsError> = vec![
            DevToolsError::RecordingFailed { agent_id: "agent-1".into(), reason: "no storage".into() },
            DevToolsError::ReplayFailed { recording_id: "rec-1".into(), reason: "corrupted".into() },
            DevToolsError::BreakpointInvalid { description: "unknown tool name".into() },
            DevToolsError::InspectionFailed { agent_id: "agent-2".into(), reason: "not running".into() },
            DevToolsError::ProfilingUnavailable { reason: "not enabled".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
    }

    #[test]
    fn test_eval_suite_error_display_and_suggestion() {
        let errors: Vec<EvalSuiteError> = vec![
            EvalSuiteError::DatasetLoadFailed { path: "bench.jsonl".into(), reason: "not found".into() },
            EvalSuiteError::InvalidProblem { problem_id: "humaneval/0".into(), reason: "missing prompt".into() },
            EvalSuiteError::GenerationFailed { problem_id: "mmlu/1".into(), reason: "provider down".into() },
            EvalSuiteError::ScoringFailed { problem_id: "gsm8k/5".into(), reason: "no reference".into() },
            EvalSuiteError::NoResults { reason: "no runs completed".into() },
            EvalSuiteError::InsufficientData { metric: "accuracy".into(), samples: 1 },
            EvalSuiteError::ReportFailed { reason: "incomplete data".into() },
            EvalSuiteError::Timeout { problem_id: "swe/10".into(), timeout_secs: 60 },
            EvalSuiteError::SearchFailed { reason: "budget exhausted".into() },
            EvalSuiteError::InvalidAgentConfig { field: "temperature".into(), reason: "must be >= 0".into() },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            assert!(err.suggestion().is_some());
        }
        // Recoverability
        assert!(EvalSuiteError::GenerationFailed { problem_id: "x".into(), reason: "r".into() }.is_recoverable());
        assert!(EvalSuiteError::Timeout { problem_id: "x".into(), timeout_secs: 30 }.is_recoverable());
        assert!(EvalSuiteError::SearchFailed { reason: "r".into() }.is_recoverable());
        assert!(!EvalSuiteError::DatasetLoadFailed { path: "x".into(), reason: "r".into() }.is_recoverable());
        assert!(!EvalSuiteError::NoResults { reason: "r".into() }.is_recoverable());
        assert!(!EvalSuiteError::InvalidAgentConfig { field: "f".into(), reason: "r".into() }.is_recoverable());
    }

    #[test]
    fn test_eval_suite_error_from_conversion() {
        let err = EvalSuiteError::NoResults { reason: "empty".into() };
        let ai_err: AiError = err.into();
        assert_eq!(ai_err.code(), "EVAL_SUITE");
    }

    #[test]
    fn test_v6_error_from_conversions() {
        let hitl_err = HitlError::ApprovalTimeout { tool_name: "t".into(), timeout_secs: 10 };
        let ai_err: AiError = hitl_err.into();
        assert_eq!(ai_err.code(), "HITL");

        let mcp_err = McpClientError::ConnectionFailed { url: "u".into(), reason: "r".into() };
        let ai_err: AiError = mcp_err.into();
        assert_eq!(ai_err.code(), "MCP_CLIENT");

        let eval_err = AgentEvalError::TrajectoryEmpty { agent_id: "a".into() };
        let ai_err: AiError = eval_err.into();
        assert_eq!(ai_err.code(), "AGENT_EVAL");

        let rt_err = RedTeamError::InvalidCategory { category: "c".into() };
        let ai_err: AiError = rt_err.into();
        assert_eq!(ai_err.code(), "RED_TEAM");

        let mcts_err = MctsError::NoValidActions { state_description: "s".into() };
        let ai_err: AiError = mcts_err.into();
        assert_eq!(ai_err.code(), "MCTS");

        let dt_err = DevToolsError::BreakpointInvalid { description: "d".into() };
        let ai_err: AiError = dt_err.into();
        assert_eq!(ai_err.code(), "DEVTOOLS");
    }

    #[test]
    fn test_new_v5_error_from_conversions() {
        let va_err = VoiceAgentError::StreamFailed { reason: "test".into() };
        let ai_err: AiError = va_err.into();
        assert_eq!(ai_err.code(), "VOICE_AGENT");

        let mg_err = MediaGenerationError::GenerationFailed { provider: "p".into(), reason: "r".into() };
        let ai_err: AiError = mg_err.into();
        assert_eq!(ai_err.code(), "MEDIA_GENERATION");

        let d_err = DistillationError::CollectionFailed { reason: "r".into() };
        let ai_err: AiError = d_err.into();
        assert_eq!(ai_err.code(), "DISTILLATION");

        let cd_err = ConstrainedDecodingError::ProviderUnsupported { provider: "p".into() };
        let ai_err: AiError = cd_err.into();
        assert_eq!(ai_err.code(), "CONSTRAINED_DECODING");
    }

    #[test]
    fn test_error_code_all_variants() {
        let cases: Vec<(AiError, &str)> = vec![
            (
                AiError::Config(ConfigError::UnknownProvider("x".into())),
                "CONFIG",
            ),
            (AiError::Provider(ProviderError::Cancelled), "PROVIDER"),
            (AiError::Rag(RagError::EmbeddingError("x".into())), "RAG"),
            (
                AiError::Network(NetworkError::DnsError { host: "x".into() }),
                "NETWORK",
            ),
            (
                AiError::Validation(ValidationError::EmptyInput { field: "x".into() }),
                "VALIDATION",
            ),
            (
                AiError::ResourceLimit(ResourceLimitError::ConcurrentRequestLimit { limit: 1 }),
                "RESOURCE_LIMIT",
            ),
            (AiError::Io(IoError::new("op", "reason")), "IO"),
            (
                AiError::Serialization(SerializationError::json_serialize("x")),
                "SERIALIZATION",
            ),
            (
                AiError::Workflow(WorkflowError::NodeNotFound { node_id: "x".into() }),
                "WORKFLOW",
            ),
            (
                AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound { name: "x".into() }),
                "MEMORY",
            ),
            (
                AiError::A2A(A2AError::TaskNotFound { task_id: "x".into() }),
                "A2A",
            ),
            (
                AiError::VoiceAgent(VoiceAgentError::StreamFailed { reason: "x".into() }),
                "VOICE_AGENT",
            ),
            (
                AiError::MediaGeneration(MediaGenerationError::GenerationFailed { provider: "x".into(), reason: "x".into() }),
                "MEDIA_GENERATION",
            ),
            (
                AiError::Distillation(DistillationError::CollectionFailed { reason: "x".into() }),
                "DISTILLATION",
            ),
            (
                AiError::ConstrainedDecoding(ConstrainedDecodingError::ProviderUnsupported { provider: "x".into() }),
                "CONSTRAINED_DECODING",
            ),
            (
                AiError::Hitl(HitlError::GateNotConfigured { operation: "x".into() }),
                "HITL",
            ),
            (
                AiError::McpClient(McpClientError::Timeout { url: "x".into(), timeout_ms: 5000 }),
                "MCP_CLIENT",
            ),
            (
                AiError::AgentEval(AgentEvalError::TrajectoryEmpty { agent_id: "x".into() }),
                "AGENT_EVAL",
            ),
            (
                AiError::RedTeam(RedTeamError::InvalidCategory { category: "x".into() }),
                "RED_TEAM",
            ),
            (
                AiError::Mcts(MctsError::NoValidActions { state_description: "x".into() }),
                "MCTS",
            ),
            (
                AiError::DevTools(DevToolsError::BreakpointInvalid { description: "x".into() }),
                "DEVTOOLS",
            ),
            (
                AiError::EvalSuite(EvalSuiteError::NoResults { reason: "x".into() }),
                "EVAL_SUITE",
            ),
            (AiError::Other("misc".into()), "OTHER"),
        ];

        for (err, expected_code) in &cases {
            assert_eq!(err.code(), *expected_code, "Wrong code for {:?}", err);
        }
    }

    // --- Error Context Tests (v8 item 10.1) ---

    #[test]
    fn test_with_context_wraps_error() {
        let err = AiError::other("something failed");
        let ctx = err.with_context("loading configuration");
        assert!(ctx.to_string().contains("loading configuration"));
        assert!(ctx.to_string().contains("something failed"));
    }

    #[test]
    fn test_contextual_error_source_chain() {
        let err = AiError::provider_unavailable("Ollama", "http://localhost:11434");
        let ctx = err.with_context("initializing assistant");
        // source() should return the original AiError
        let source = std::error::Error::source(&ctx);
        assert!(source.is_some());
        let source_display = source.unwrap().to_string();
        assert!(source_display.contains("Ollama"));
    }

    #[test]
    fn test_result_ext_context() {
        let result: AiResult<()> = Err(AiError::other("disk full"));
        let ctx_result = result.context("saving checkpoint");
        assert!(ctx_result.is_err());
        let err = ctx_result.unwrap_err();
        assert!(err.to_string().contains("saving checkpoint"));
        assert!(err.to_string().contains("disk full"));
    }

    #[test]
    fn test_result_ext_context_ok_passthrough() {
        let result: AiResult<i32> = Ok(42);
        let ctx_result = result.context("this context should not appear");
        assert_eq!(ctx_result.unwrap(), 42);
    }

    #[test]
    fn test_result_ext_with_context_fn_lazy() {
        let result: AiResult<()> = Err(AiError::other("timeout"));
        let ctx_result = result.with_context_fn(|| format!("connecting to {}", "remote-host"));
        assert!(ctx_result.is_err());
        let err = ctx_result.unwrap_err();
        assert!(err.to_string().contains("connecting to remote-host"));
    }

    #[test]
    fn test_contextual_error_into_ai_error() {
        let err = AiError::other("root cause");
        let ctx = err.with_context("during operation");
        let ai_err: AiError = ctx.into();
        assert!(ai_err.to_string().contains("during operation"));
        assert!(ai_err.to_string().contains("root cause"));
        assert_eq!(ai_err.code(), "OTHER");
    }

    #[test]
    fn test_with_context_preserves_error_info() {
        let err = AiError::rate_limited(100, 60);
        let original_display = err.to_string();
        let ctx = err.with_context("batch processing request 42");
        assert!(ctx.to_string().starts_with("batch processing request 42"));
        assert!(ctx.to_string().contains(&original_display));
    }

    #[test]
    fn test_contextual_error_debug() {
        let err = AiError::other("test");
        let ctx = err.with_context("debug check");
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("ContextualError"));
        assert!(debug_str.contains("debug check"));
    }
}
