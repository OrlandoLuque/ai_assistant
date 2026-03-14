//! MCP (Model Context Protocol) support
//!
//! Implementation of Anthropic's Model Context Protocol for standardized
//! tool use and context sharing between AI models and external systems.

// Submodules
pub mod types;
pub mod server;
pub mod client;
pub mod transport;
pub mod oauth;
pub mod session;
pub mod v2_transport;
pub mod v2_oauth;
pub mod v2_annotations;
pub mod v2_elicitation;
pub mod v2_batch;
pub mod v2_completion;
#[cfg(feature = "rag")]
pub mod knowledge_tools;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

// =============================================================================
// Re-exports: ALL public types accessible as mcp_protocol::TypeName
// =============================================================================

// --- types.rs ---
pub use types::{
    AudioContent,
    McpContent,
    McpError,
    McpMessageType,
    McpPagination,
    McpPaginationRequest,
    McpPrompt,
    McpPromptArgument,
    McpPromptMessage,
    McpRequest,
    McpResource,
    McpResourceContent,
    McpResponse,
    McpServerCapabilities,
    McpTool,
    McpToolAnnotation,
    McpToolsCapability,
    McpResourcesCapability,
    McpPromptsCapability,
    MCP_VERSION,
    MCP_VERSION_PREVIOUS,
};

// --- server.rs ---
pub use server::McpServer;

// --- client.rs ---
pub use client::McpClient;

// --- transport.rs ---
pub use transport::{McpStreamableSession, McpTransport};

// --- oauth.rs ---
pub use oauth::{
    McpAuthorizationRequest,
    McpOAuthConfig,
    McpOAuthGrantType,
    McpOAuthScope,
    McpOAuthTokenManager,
    McpTokenResponse,
};

// --- session.rs ---
pub use session::{
    SessionBelief,
    SessionBeliefs,
    SessionContext,
    SessionHighlights,
    SessionMcpManager,
    SessionRepairResult,
    SessionResourceInfo,
    SessionSummary,
};

// --- v2_transport.rs ---
pub use v2_transport::{
    InMemorySessionStore,
    McpSession,
    McpSessionStore,
    StreamableHttpTransport,
    TransportMode,
};

// --- v2_oauth.rs ---
pub use v2_oauth::{
    AuthorizationServerMetadata,
    DynamicClientRegistration,
    McpV2OAuthConfig,
    OAuthToken,
    OAuthTokenManager,
    PkceChallenge,
};

// --- v2_annotations.rs ---
pub use v2_annotations::{AnnotatedTool, ToolAnnotationRegistry, ToolAnnotations};

// --- v2_elicitation.rs ---
pub use v2_elicitation::{
    AutoAcceptHandler,
    ElicitAction,
    ElicitFieldSchema,
    ElicitFieldType,
    ElicitRequest,
    ElicitResponse,
    ElicitationHandler,
};

// --- v2_batch.rs ---
pub use v2_batch::{
    BatchConfig,
    BatchExecutor,
    BatchRequest,
    BatchResponse,
    JsonRpcRequest,
    JsonRpcResponse,
};

// --- v2_completion.rs ---
pub use v2_completion::{
    CompletionProvider,
    CompletionRefType,
    CompletionRegistry,
    CompletionRequest,
    CompletionResult,
    CompletionSuggestion,
    StaticCompletionProvider,
};
