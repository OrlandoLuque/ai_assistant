//! AI Assistant library for local LLM integration
//!
//! This crate provides a unified interface to connect to various local LLM providers
//! like Ollama, LM Studio, text-generation-webui, Kobold.cpp, LocalAI, and any
//! OpenAI-compatible API.
//!
//! # Cargo Features
//!
//! This crate uses Cargo features to allow selective compilation:
//!
//! - `core` (default): Basic assistant functionality
//! - `multi-agent`: Multi-agent orchestration and shared memory
//! - `security`: Guardrails, PII detection, content moderation
//! - `analytics`: Metrics, telemetry, engagement tracking
//! - `vision`: Multimodal/vision support
//! - `embeddings`: Local embeddings and vector database
//! - `advanced-streaming`: SSE, WebSocket streaming
//! - `adapters`: External API adapters (OpenAI, Anthropic, HuggingFace)
//! - `tools`: Tool calling, MCP protocol, agentic loops
//! - `documents`: Document parsing (EPUB, DOCX, HTML)
//! - `eval`: Evaluation, benchmarking, fine-tuning
//! - `rag`: SQLite FTS5-based knowledge base
//! - `egui-widgets`: Pre-built egui widgets for chat UI
//! - `full`: All features except egui-widgets
//!
//! # Example
//!
//! ```no_run
//! use ai_assistant::{AiAssistant, AiConfig};
//!
//! let mut assistant = AiAssistant::new();
//! assistant.fetch_models();
//! ```

// =============================================================================
// CORE MODULES (always available)
// =============================================================================

mod config;
mod messages;
mod models;
mod providers;
mod session;
mod context;
mod assistant;

// Core utilities
pub mod error;
pub mod progress;
pub mod config_file;
pub mod memory_management;
pub mod wasm;
pub mod async_support;
pub mod retry;
pub mod profiles;
pub mod templates;
pub mod formatting;
pub mod search;
pub mod caching;
pub mod structured;
pub mod diff;
pub mod streaming;
pub mod connection_pool;
pub mod cache_compression;
pub mod plugins;
pub mod batch;
pub mod fallback;
pub mod debug;
pub mod data_source_client;
pub mod crawl_policy;
pub mod persistence;
pub mod adaptive_thinking;
pub mod events;
pub mod internal_storage;
pub mod log_redaction;
pub mod http_client;
pub mod request_queue;
pub mod cloud_providers;
pub mod server;

// Core re-exports
pub use config::{AiConfig, AiProvider};
pub use messages::{ChatMessage, AiResponse};
pub use models::ModelInfo;
pub use session::{
    ChatSession, ChatSessionStore, UserPreferences, ResponseStyle,
    JournalSession, JournalEntry, JournalEntryType,
};
pub use context::{
    ContextUsage, estimate_tokens, get_model_context_size,
    get_model_context_size_cached, clear_context_size_cache, context_size_cache_len,
};
pub use assistant::{AiAssistant, SummaryResult};
pub use providers::{
    generate_response, generate_response_streaming,
    generate_response_streaming_cancellable,
    build_system_prompt, build_system_prompt_with_notes,
    fetch_model_context_size,
};

pub use error::{
    AiError, AiResult, ConfigError, ProviderError, RagError,
    NetworkError, ValidationError as AiValidationError, ResourceLimitError,
    IoError as AiIoError, SerializationError,
};

pub use progress::{
    Progress, ProgressCallback, ProgressReporter,
    MultiProgressTracker, OperationHandle, ProgressAggregator,
    ProgressCallbackBuilder, logging_callback, silent_callback,
};

pub use events::{
    AiEvent, EventHandler, EventBus, TimestampedEvent,
    FilteredHandler, LoggingHandler, LogLevel, CollectingHandler, EventTimer,
};

pub use config_file::{
    ConfigFile, ConfigFormat, ProviderConfig, UrlConfig,
    GenerationConfig, RagFileConfig, HybridConfig, SecurityConfig,
    CacheConfig as FileCacheConfig, LoggingConfig,
    load_config, save_config, default_config_path,
};

pub use memory_management::{
    BoundedCache, BoundedVec, EvictionPolicy, CacheStats as BoundedCacheStats,
    MemoryTracker, ComponentMemory, MemoryPressure, MemoryReport,
    format_bytes, MemoryEstimate,
};

pub use wasm::{is_wasm, PlatformCapabilities, Capability};

pub use async_support::{
    AsyncResult, BoxFuture, AsyncLoad, AsyncSave, AsyncIterator,
    BlockingHandle, spawn_blocking, block_on,
    YieldNow, yield_now, Sleep, sleep, Timeout, timeout,
    AsyncSender, AsyncReceiver, async_channel,
    AsyncStream, stream, AsyncMutex, AsyncMutexGuard,
    join, join3, select,
    AsyncRetryConfig, retry_async,
};

pub use retry::{
    RetryConfig, RetryExecutor, RetryResult, RetryAttempt, RetryableError,
    CircuitBreaker, CircuitState, ResilientExecutor,
    retry, retry_with_config,
};

pub use profiles::{ModelProfile, ProfileBuilder, ProfileManager, ProfileApplicator};

pub use adaptive_thinking::{
    AdaptiveThinkingConfig, ThinkingDepth, RagTierPriority,
    ClassificationSignals, ThinkingStrategy, QueryClassifier,
    ThinkingTagParser, ThinkingParseResult, parse_thinking_tags,
    is_trivial_query,
};

pub use templates::{
    PromptTemplate, TemplateVariable, VariableType, TemplateBuilder,
    TemplateManager, BuiltinTemplates,
};

pub use formatting::{
    ParsedResponse, CodeBlock, ParsedList, ListItem, ParsedTable, TableAlignment,
    ParsedLink, Heading, ResponseParser, ParserConfig,
    extract_first_code, extract_code_by_language, extract_first_json,
    parse_as_json, to_plain_text,
};

pub use search::{
    SearchResult as ConversationSearchResult, HighlightSpan, SearchContext, SearchQuery, SearchMode,
    ConversationSearcher, SearchStats, SearchQueryBuilder,
};

pub use caching::{
    CachedResponse, CacheConfig, CacheKey, ResponseCache,
    CacheStats as ResponseCacheStats, SemanticCache,
};

pub use structured::{
    SchemaType, SchemaProperty, JsonSchema, ValidationError, ValidationResult,
    SchemaValidator, StructuredOutputGenerator, StructuredParseResult,
    SchemaBuilder, StructuredRequest,
};

pub use diff::{
    ChangeType, DiffLine, DiffHunk, DiffResult, diff,
    ResponseComparison as DiffResponseComparison, WordStats, compare_responses as diff_compare_responses,
    InlineWordDiff, InlineDiffSegment, inline_word_diff,
};

pub use streaming::{
    StreamingConfig, StreamBuffer, StreamProducer, StreamConsumer,
    StreamError, StreamMetrics, BackpressureStream, Chunker, RateLimitedStream,
};

pub use connection_pool::{
    ConnectionPool, PoolConfig, PooledConnection, PooledConnectionGuard, PoolStats,
    global_pool, init_global_pool,
};

pub use cache_compression::{
    CompressionAlgorithm, CompressionLevel, CompressedData, CompressionError,
    compress, decompress, CompressedCache, CacheCompressionStats,
    compress_string, decompress_string, StreamingCompressor,
};

pub use internal_storage::{
    StorageFormat, InternalFileInfo,
    save_internal, load_internal, serialize_internal, deserialize_internal,
    detect_format, dump_as_json, convert_to_json, convert_json_to_binary, file_info,
};

pub use log_redaction::{
    redact, redact_with_config, contains_sensitive, RedactionConfig,
};

pub use http_client::{
    HttpClient, UreqClient,
    parse_ollama_models, parse_openai_models, parse_kobold_models,
    fetch_ollama_models_with, fetch_openai_models_with, fetch_kobold_models_with,
};

pub use server::{ServerConfig, AiServer, ServerHandle};

pub use cloud_providers::{
    resolve_api_key, generate_cloud_response, fetch_cloud_models,
    generate_openai_cloud, generate_anthropic_cloud,
    fetch_openai_cloud_models, fetch_anthropic_cloud_models,
};

pub use request_queue::{
    RequestPriority, QueuedRequest, RequestQueue, QueueStats as RequestQueueStats,
};

pub use plugins::{
    Plugin, PluginCapability, PluginContext, PluginInfo, PluginManager,
    MessageProcessorPlugin, LoggingPlugin,
};

pub use batch::{
    BatchProcessor, BatchConfig, BatchRequest, BatchResult,
    BatchBuilder, BatchStats, BatchResults,
};

pub use fallback::{
    FallbackChain, FallbackProvider, FallbackResult, FallbackError,
    ProviderState as FallbackProviderState, ProviderStatus,
};

pub use debug::{
    DebugLevel, DebugConfig, DebugLogger, DebugEntry, DebugReport,
    RequestInspector, CapturedRequest,
    global_debug, configure_global_debug, set_debug_level,
};

pub use persistence::{
    BackupConfig, BackupInfo, BackupManager,
    CompactionConfig, CompactionResult, DatabaseCompactor, DatabaseStats,
    MigrationConfig, MigrationResult, SessionMigrator, FullExport,
    PersistentCacheConfig, CacheStats, CacheEntryInfo,
};

pub use crawl_policy::{
    RobotsRule, RobotsDirectives, ParsedRobotsTxt,
    SitemapEntry, ChangeFrequency, ParsedSitemap,
    CrawlPolicyConfig, CrawlPolicy,
};

// =============================================================================
// MULTI-AGENT FEATURE
// =============================================================================

#[cfg(feature = "multi-agent")]
pub mod multi_agent;
#[cfg(feature = "multi-agent")]
pub mod task_decomposition;
#[cfg(feature = "multi-agent")]
pub mod agent_memory;
#[cfg(feature = "multi-agent")]
pub mod agent;

#[cfg(feature = "multi-agent")]
pub use multi_agent::{
    AgentOrchestrator, Agent, AgentStatus, AgentRole, AgentMessage, MessageType,
    AgentTask, TaskStatus, OrchestrationStrategy, OrchestrationStatus, OrchestrationError,
};

#[cfg(feature = "multi-agent")]
pub use task_decomposition::{
    TaskDecomposer, TaskNode, DecompositionStrategy, DecompositionStatus, FlatTask,
    DecompositionAnalysis,
};

#[cfg(feature = "multi-agent")]
pub use agent_memory::{
    SharedMemory, MemoryEntry as AgentMemoryEntry, MemoryType as AgentMemoryType,
    MemoryError, MemoryStats as AgentMemoryStats, ThreadSafeMemory,
};

#[cfg(feature = "multi-agent")]
pub use agent::{
    AgentState, AgentStep, AgentTool, AgentConfig, AgentContext, AgentResult,
    ReactAgent, PlanningAgent, PlanStep, PlanStepStatus,
    AgentExecutor, AgentCallback, LoggingCallback, create_builtin_agent_tools,
};

// =============================================================================
// ASYNC RUNTIME FEATURE
// =============================================================================

#[cfg(feature = "async-runtime")]
pub mod async_providers;

#[cfg(feature = "async-runtime")]
pub use async_providers::{
    AsyncHttpClient, ReqwestClient,
    fetch_models_async, fetch_ollama_models_async,
    fetch_openai_models_async, fetch_kobold_models_async,
    generate_response_async, generate_response_streaming_async,
    block_on_async, create_runtime,
};

// =============================================================================
// DISTRIBUTED FEATURE
// =============================================================================

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "distributed")]
pub use distributed::{
    // DHT
    NodeId, DhtNode, KBucket, RoutingTable, DhtValue, Dht,
    // CRDTs
    GCounter, PNCounter, LWWRegister, ORSet, LWWMap,
    // MapReduce
    DataChunk, MapOutput, ReduceOutput, MapReduceJob, MapReduceBuilder,
    // Coordinator
    DistributedCoordinator,
};

#[cfg(feature = "distributed-network")]
pub use distributed::NodeMessage;

// =============================================================================
// DISTRIBUTED NETWORKING (optional — QUIC transport, replication, failure detection)
// =============================================================================

#[cfg(feature = "distributed-network")]
pub mod consistent_hash;
#[cfg(feature = "distributed-network")]
pub mod failure_detector;
#[cfg(feature = "distributed-network")]
pub mod merkle_sync;
#[cfg(feature = "distributed-network")]
pub mod node_security;
#[cfg(feature = "distributed-network")]
pub mod distributed_network;

#[cfg(feature = "distributed-network")]
pub use consistent_hash::ConsistentHashRing;
#[cfg(feature = "distributed-network")]
pub use failure_detector::{PhiAccrualDetector, HeartbeatManager, HeartbeatConfig, NodeStatus};
#[cfg(feature = "distributed-network")]
pub use merkle_sync::{MerkleTree, MerkleProof, AntiEntropySync, SyncDelta as MerkleSyncDelta};
#[cfg(feature = "distributed-network")]
pub use node_security::{NodeIdentity, CertificateManager, JoinToken, ChallengeResponse};
#[cfg(feature = "distributed-network")]
pub use distributed_network::{
    NetworkNode, NetworkConfig, ReplicationConfig, WriteMode,
    DiscoveryConfig as NetworkDiscoveryConfig, NetworkEvent, NetworkStats,
    PeerState as NetworkPeerState, RingInfo,
};

// =============================================================================
// P2P NETWORKING (optional)
// =============================================================================

#[cfg(feature = "p2p")]
pub mod p2p;

#[cfg(feature = "p2p")]
pub use p2p::{
    P2PConfig, PeerDataTrust, TurnConfig,
    NatType, NatDiscoveryResult, NatTraversal,
    IceCandidateType, IceCandidate, IceState, IceAgent,
    PeerReputation, ReputationSystem,
    PeerMessage, PeerInfo, KnowledgeShare, ContradictionReport,
    PeerConnection, P2PManager, P2PStats,
};

// =============================================================================
// AUTONOMOUS AGENT FEATURE
// =============================================================================

#[cfg(feature = "autonomous")]
pub mod agent_policy;
#[cfg(feature = "autonomous")]
pub mod agent_sandbox;
#[cfg(feature = "autonomous")]
pub mod os_tools;
#[cfg(feature = "autonomous")]
pub mod user_interaction;
#[cfg(feature = "autonomous")]
pub mod task_board;
#[cfg(feature = "autonomous")]
pub mod interactive_commands;
#[cfg(feature = "autonomous")]
pub mod mode_manager;
#[cfg(feature = "autonomous")]
pub mod agent_profiles;
#[cfg(feature = "autonomous")]
pub mod autonomous_loop;
#[cfg(feature = "distributed-agents")]
pub mod distributed_agents;
#[cfg(feature = "butler")]
pub mod butler;
#[cfg(feature = "browser")]
pub mod browser_tools;
#[cfg(feature = "scheduler")]
pub mod scheduler;
#[cfg(feature = "scheduler")]
pub mod trigger_system;

#[cfg(feature = "autonomous")]
pub use agent_policy::{
    AutonomyLevel, InternetMode, RiskLevel as AgentRiskLevel, ActionType, ActionDescriptor,
    AgentPolicy, AgentPolicyBuilder, ApprovalHandler, AutoApproveAll, AutoDenyAll,
    ClosureApprovalHandler,
};

#[cfg(feature = "autonomous")]
pub use agent_sandbox::{
    AuditDecision, AuditEntry, SandboxError, SandboxValidator,
};

#[cfg(feature = "autonomous")]
pub use agent_profiles::{
    AgentProfile, ConversationProfile, WorkflowProfile, WorkflowPhase, ProfileRegistry,
};

#[cfg(feature = "autonomous")]
pub use user_interaction::{
    UserQuery, UserResponse, NotifyLevel, UserInteractionHandler,
    AutoApproveHandler as AutoApproveInteraction, CallbackInteractionHandler,
    BufferedInteractionHandler, PendingQuery, InteractionManager,
};

#[cfg(feature = "autonomous")]
pub use autonomous_loop::{
    AgentState as AutonomousAgentState, AutonomousAgent, AutonomousAgentBuilder,
    AgentResult as AutonomousAgentResult, IterationOutcome, AutonomousAgentConfig,
};

#[cfg(feature = "autonomous")]
pub use mode_manager::{OperationMode, ModeManager};

#[cfg(feature = "autonomous")]
pub use task_board::{TaskBoard, BoardCommand, TaskBoardListener, TaskExecutionState, TaskBoardSummary};

#[cfg(feature = "autonomous")]
pub use interactive_commands::{UserIntent, CommandResult, CommandProcessor};

#[cfg(feature = "autonomous")]
pub use multi_agent::{MultiAgentSession, SessionSummary as MultiAgentSessionSummary};

// =============================================================================
// MULTI-LAYER GRAPH (always available)
// =============================================================================

pub mod multi_layer_graph;
pub use multi_layer_graph::{
    GraphLayer, ConfidenceLevel, BeliefType, UserBelief,
    ContradictionSource, ContradictionResolution, Contradiction, ContradictionLog,
    LayeredEntity, BeliefExtractor, SessionGraph, UserGraph,
    InternetGraphEntry, InternetGraph, MultiLayerQueryResult,
    MultiLayerGraph, MultiLayerGraphStats,
};

// =============================================================================
// SECURITY FEATURE
// =============================================================================

#[cfg(feature = "security")]
pub mod security;
#[cfg(feature = "security")]
pub mod pii_detection;
#[cfg(feature = "security")]
pub mod content_moderation;
#[cfg(feature = "security")]
pub mod injection_detection;
#[cfg(feature = "security")]
pub mod access_control;
#[cfg(feature = "security")]
pub mod data_anonymization;
#[cfg(feature = "security")]
pub mod content_encryption;
#[cfg(feature = "security")]
pub mod advanced_guardrails;

#[cfg(feature = "security")]
pub use security::{
    RateLimitConfig, RateLimiter, RateLimitResult, RateLimitReason, RateLimitUsage, RateLimitStatus,
    SanitizationConfig, InputSanitizer, SanitizationResult, SanitizationWarning,
    AuditConfig, AuditLogger, AuditEvent, AuditEventType, AuditStats,
    HookManager, HookResult, HookChainResult,
};

#[cfg(feature = "security")]
pub use pii_detection::{
    PiiConfig, PiiDetector, PiiType, DetectedPii, RedactionStrategy,
    SensitivityLevel, PiiResult, CustomPiiPattern,
};

#[cfg(feature = "security")]
pub use content_moderation::{
    ModerationConfig, ContentModerator, ModerationCategory, ModerationResult,
    ModerationFlag, ModerationAction, ModerationStats,
};

#[cfg(feature = "security")]
pub use injection_detection::{
    InjectionConfig, InjectionDetector, InjectionType, InjectionDetection,
    RiskLevel, InjectionResult, Recommendation, DetectionSensitivity,
    CustomPattern as InjectionCustomPattern,
};

#[cfg(feature = "security")]
pub use access_control::{
    AccessControlManager, AccessControlEntry, Permission, ResourceType, Role,
    AccessCondition, AccessResult,
};

#[cfg(feature = "security")]
pub use data_anonymization::{
    DataAnonymizer, AnonymizationRule, AnonymizationStrategy, DataType as AnonymizationDataType,
    AnonymizationResult, Detection, BatchAnonymizer,
};

#[cfg(feature = "security")]
pub use content_encryption::{
    ContentEncryptor, EncryptionKey, EncryptedContent, EncryptionAlgorithm, EncryptionError,
    EncryptedMessageStore,
};

#[cfg(feature = "security")]
pub use advanced_guardrails::{
    ConstitutionalPrinciple, ConstitutionalConfig, ConstitutionalAI,
    ConstitutionalEvaluation, PrincipleViolation, default_principles,
    BiasDimension, BiasConfig, BiasDetector, BiasDetectionResult, BiasOccurrence,
    ToxicityCategory, ToxicityConfig, ToxicityDetector, ToxicityResult,
    AttackType, AttackDetector, AttackDetectionResult, DetectedAttack,
    GuardrailsManager, InputCheckResult, OutputCheckResult, FullSafetyCheck,
};

// =============================================================================
// ANALYTICS FEATURE
// =============================================================================

#[cfg(feature = "analytics")]
pub mod metrics;
#[cfg(feature = "analytics")]
pub mod analysis;
#[cfg(feature = "analytics")]
pub mod streaming_metrics;
#[cfg(feature = "analytics")]
pub mod latency_metrics;
#[cfg(feature = "analytics")]
pub mod telemetry;
#[cfg(feature = "analytics")]
pub mod conversation_analytics;
#[cfg(feature = "analytics")]
pub mod user_engagement;
#[cfg(feature = "analytics")]
pub mod response_effectiveness;
#[cfg(feature = "analytics")]
pub mod prometheus_metrics;
#[cfg(feature = "analytics")]
pub mod quality;
#[cfg(feature = "analytics")]
pub mod conversation_flow;

#[cfg(feature = "analytics")]
pub use metrics::{
    MetricsTracker, MessageMetrics, SessionMetrics, RagQualityMetrics,
    MessageMetricsBuilder, SearchCache, ConversationTestCase, TestCaseResult,
    TestSuite, TestSuiteResults, MetricsExport,
};

#[cfg(feature = "analytics")]
pub use analysis::{
    Sentiment, SentimentAnalysis, SentimentAnalyzer, SentimentTrend,
    ConversationSentimentAnalysis, Topic, TopicDetector,
    SummaryConfig, SessionSummary, SessionSummarizer,
};

#[cfg(feature = "analytics")]
pub use streaming_metrics::{
    StreamingMetrics, MetricsConfig, StreamingSnapshot, FinalMetrics,
    AggregatedMetrics, MetricsDisplay,
};

#[cfg(feature = "analytics")]
pub use latency_metrics::{
    LatencyTracker, LatencyRecord, LatencyStats, RequestTimer, LatencyTrend,
};

#[cfg(feature = "analytics")]
pub use telemetry::{
    TelemetryConfig, TelemetryEvent, TelemetryCollector, AggregatedMetrics as TelemetryMetrics,
    TimedOperation,
};

#[cfg(feature = "analytics")]
pub use conversation_analytics::{
    AnalyticsConfig, ConversationAnalytics, AnalyticsEvent, EventType, EventValue,
    AggregatedStats, PatternTracker, AnalyticsReport, ExportedEvent,
};

#[cfg(feature = "analytics")]
pub use user_engagement::{
    EngagementTracker, EngagementEvent, EngagementMetrics, EngagementManager, UserTrends,
};

#[cfg(feature = "analytics")]
pub use response_effectiveness::{
    EffectivenessScorer, EffectivenessScore, QAPair, UserFeedback as EffectivenessFeedback,
    ScoringWeights, BatchEvaluator, BatchResult as EffectivenessBatchResult,
};

#[cfg(feature = "analytics")]
pub use prometheus_metrics::{
    Counter, Gauge, Histogram, HistogramTimer, AiMetricsRegistry,
};

#[cfg(feature = "analytics")]
pub use quality::{
    QualityScore, QualityIssue, QualityIssueType, QualityConfig, QualityAnalyzer,
    ResponseComparison, compare_responses,
};

#[cfg(feature = "analytics")]
pub use conversation_flow::{
    FlowAnalyzer, ConversationTurn, FlowState, FlowAnalysis, TopicTransition,
};

// =============================================================================
// VISION FEATURE
// =============================================================================

#[cfg(feature = "vision")]
pub mod vision;

#[cfg(feature = "vision")]
pub use vision::{
    ImageInput, ImageData, ImageDetail, VisionMessage, VisionCapabilities,
    ImagePreprocessor, ImageBatch,
};

// =============================================================================
// EMBEDDINGS FEATURE
// =============================================================================

#[cfg(feature = "embeddings")]
pub mod embeddings;
#[cfg(feature = "embeddings")]
pub mod embedding_cache;
#[cfg(feature = "embeddings")]
pub mod vector_db;
#[cfg(feature = "embeddings")]
pub mod neural_embeddings;

#[cfg(feature = "embeddings")]
pub use embeddings::{
    EmbeddingConfig, LocalEmbedder, IndexedDocument, SemanticIndex, SearchResult,
    HybridSearchConfig, HybridSearcher, HybridSearchResult,
};

#[cfg(feature = "embeddings")]
pub use embedding_cache::{
    EmbeddingCacheConfig, EmbeddingCacheStats, EmbeddingCache,
    SharedEmbeddingCache, BatchEmbeddingCache, SimilarityEmbeddingCache,
    cosine_similarity, normalize_vector,
};

#[cfg(feature = "embeddings")]
pub use vector_db::{
    VectorDbConfig, DistanceMetric, StoredVector, VectorSearchResult,
    MetadataFilter, FilterOperation, VectorDb, InMemoryVectorDb,
    QdrantClient, VectorDbBuilder, VectorDbBackend, HybridVectorSearch,
    BackendInfo, VectorMigrationResult, migrate_vectors, string_id_to_u64,
};

// LanceDB vector database backend (Tier 2 — embedded, persistent)
#[cfg(feature = "vector-lancedb")]
pub mod vector_db_lance;
#[cfg(feature = "vector-lancedb")]
pub use vector_db_lance::LanceVectorDb;

#[cfg(feature = "distributed-network")]
pub use vector_db::DistributedVectorDb;

#[cfg(feature = "embeddings")]
pub use neural_embeddings::{
    EmbeddingVec, SparseEmbedding, PoolingStrategy, DenseEmbeddingConfig, DenseEmbedder,
    SparseEmbeddingConfig, SparseEmbedder, HybridRetriever, HybridEmbedding,
    QuantizationType, QuantizedEmbedding, DimensionalityReduction,
    CrossEncoder, RankedDocument, EmbeddingError,
    normalize_l2, cosine_similarity as neural_cosine_similarity,
    euclidean_distance, dot_product as neural_dot_product,
};

// =============================================================================
// ADVANCED STREAMING FEATURE
// =============================================================================

#[cfg(feature = "advanced-streaming")]
pub mod sse_streaming;
#[cfg(feature = "advanced-streaming")]
pub mod websocket_streaming;
#[cfg(feature = "advanced-streaming")]
pub mod streaming_compression;

#[cfg(feature = "advanced-streaming")]
pub use sse_streaming::{
    SseEvent, SseReader, SseWriter, SseClient, SseConnection, SseError,
    StreamChunk, StreamChoice, StreamDelta, StreamAggregator,
};

#[cfg(feature = "advanced-streaming")]
pub use websocket_streaming::{
    WsFrame, WsOpcode, WsCloseCode, WsAiMessage, WsUsage, WsState,
    WsStreamHandler, WsCallbacks, WsError, WsHandshake, BidirectionalStream,
};

#[cfg(feature = "advanced-streaming")]
pub use streaming_compression::{
    Algorithm as StreamCompressionAlgorithm, Level as StreamCompressionLevel,
    CompressionConfig as StreamCompressionConfig, CompressionResult as StreamCompressionResult,
    StreamCompressor, CompressionStats as StreamCompressionStats, StreamDecompressor,
    compress_string as stream_compress_string, decompress_string as stream_decompress_string,
};

// =============================================================================
// ADAPTERS FEATURE
// =============================================================================

#[cfg(feature = "adapters")]
pub mod openai_adapter;
#[cfg(feature = "adapters")]
pub mod anthropic_adapter;
#[cfg(feature = "adapters")]
pub mod huggingface_connector;
#[cfg(feature = "adapters")]
pub mod provider_plugins;

#[cfg(feature = "adapters")]
pub use openai_adapter::{
    OpenAIClient, OpenAIConfig, OpenAIRequest, OpenAIResponse, OpenAIMessage, OpenAIModel,
};

#[cfg(feature = "adapters")]
pub use anthropic_adapter::{
    AnthropicClient, AnthropicConfig, AnthropicRequest, AnthropicResponse,
    AnthropicMessage, AnthropicModel,
};

#[cfg(feature = "adapters")]
pub use huggingface_connector::{
    HfClient, HfConfig, HfRequest, HfResponse, HfTask,
};

#[cfg(feature = "adapters")]
pub use provider_plugins::{
    OllamaProvider, LmStudioProvider, TextGenWebUIProvider, KoboldCppProvider,
    OpenAICompatibleProvider, DiscoveryConfig, discover_providers,
    create_registry_with_discovery,
};

// =============================================================================
// TOOLS FEATURE
// =============================================================================

#[cfg(feature = "tools")]
pub mod unified_tools;
#[cfg(feature = "tools")]
pub mod tools;
#[cfg(feature = "tools")]
pub mod tool_use;
#[cfg(feature = "tools")]
pub mod tool_calling;
#[cfg(feature = "tools")]
pub mod function_calling;
#[cfg(feature = "tools")]
pub mod mcp_protocol;
#[cfg(feature = "tools")]
pub mod agentic_loop;
#[cfg(feature = "tools")]
pub mod model_integration;
#[cfg(feature = "tools")]
pub mod prompt_chaining;

#[cfg(feature = "tools")]
pub use unified_tools::{
    ParamType as UnifiedParamType, ParamSchema,
    ToolDef as UnifiedToolDef, ToolBuilder,
    ToolCall as UnifiedToolCall, ToolOutput as UnifiedToolOutput, ToolError as UnifiedToolError,
    ToolHandler as UnifiedToolHandler, ToolChoice as UnifiedToolChoice,
    ToolRegistry as UnifiedToolRegistry,
    ProviderCapabilities as UnifiedProviderCapabilities,
    ProviderPlugin as UnifiedProviderPlugin,
    ProviderRegistry as UnifiedProviderRegistry,
    parse_tool_calls as unified_parse_tool_calls,
    builtin_tools as unified_builtin_tools,
    evaluate_math,
};

#[cfg(feature = "tools")]
pub use tools::{
    ParameterType, ToolParameter, ToolDefinition, ToolCall, ToolResult,
    ToolRegistry, ProviderCapabilities, ProviderPlugin, ProviderRegistry,
    create_builtin_tools,
};

#[cfg(feature = "tools")]
pub use tool_use::{
    Tool as ToolUseTool, ToolParameter as ToolUseParameter, ToolRegistry as ToolUseRegistry,
    ToolCall as ToolUseCall, ToolResult as ToolUseResult, ToolError, ParameterType as ToolParameterType,
    parse_tool_calls,
};

#[cfg(feature = "tools")]
pub use tool_calling::{
    Tool as ToolDef, ToolParameter as ToolParam, ParameterType as ParamType,
    ToolCall as ToolInvocation, ToolResult as ToolOutput,
    ToolRegistry as ToolRepo, CommonTools,
};

#[cfg(feature = "tools")]
pub use function_calling::{
    FunctionDefinition, FunctionParameters, ParameterProperty, FunctionCall,
    FunctionResult, FunctionBuilder, FunctionRegistry, ToolChoice,
    parse_function_calls,
};

#[cfg(feature = "tools")]
pub use mcp_protocol::{
    McpServer, McpClient, McpRequest, McpResponse, McpError, McpTool, McpResource,
    McpResourceContent, McpPrompt, McpPromptMessage, McpContent, McpServerCapabilities,
    MCP_VERSION,
};

#[cfg(feature = "tools")]
pub use agentic_loop::{
    AgenticLoop, AgentConfig as LoopConfig, AgentState as LoopState,
    AgentStatus as LoopStatus, AgentMessage as LoopMessage, AgentRole as LoopRole,
    AgentLoopResult, IterationResult, AgentBuilder,
};

#[cfg(feature = "tools")]
pub use model_integration::{
    ChatMessage as IntegratedChatMessage, ChatRole, ChatResponse, FinishReason,
    TokenUsage, ModelError, ModelProvider, OllamaProvider as OllamaIntegrated,
    LMStudioProvider as LMStudioIntegrated, IntegratedModelClient,
    create_ollama_client, create_lm_studio_client,
};

#[cfg(feature = "tools")]
pub use prompt_chaining::{
    ChainConfig, PromptChain, ChainStep, StepCondition, ChainExecutor,
    ChainResult, StepResult, VariableExtraction, ExtractionMethod, ChainTemplates,
    ChainBuilder,
};

// =============================================================================
// DOCUMENTS FEATURE
// =============================================================================

#[cfg(feature = "documents")]
pub mod document_parsing;
#[cfg(feature = "documents")]
pub mod table_extraction;
#[cfg(feature = "documents")]
pub mod feed_monitor;
#[cfg(feature = "documents")]
pub mod html_extraction;

#[cfg(feature = "documents")]
pub use document_parsing::{
    DocumentFormat, DocumentSection, DocumentMetadata, ParsedDocument,
    DocumentParserConfig, DocumentParser, PdfTable, PageContent,
};

#[cfg(feature = "documents")]
pub use table_extraction::{
    TableCell, ExtractedTable, TableSourceFormat,
    TableExtractorConfig, TableExtractor,
};

#[cfg(feature = "documents")]
pub use feed_monitor::{
    FeedEntry, FeedMetadata, FeedFormat, ParsedFeed,
    FeedMonitorConfig, FeedMonitorState, FeedCheckResult,
    FeedParser, FeedMonitor,
};

#[cfg(feature = "documents")]
pub use html_extraction::{
    HtmlSelector, HtmlElement, HtmlMetadata,
    HtmlExtractionConfig, HtmlList, HtmlLink, HtmlExtractionResult,
    HtmlExtractor,
};

// =============================================================================
// EVAL FEATURE
// =============================================================================

#[cfg(feature = "eval")]
pub mod benchmark;
#[cfg(feature = "eval")]
pub mod evaluation;
#[cfg(feature = "eval")]
pub mod fine_tuning;
#[cfg(feature = "eval")]
pub mod hallucination_detection;
#[cfg(feature = "eval")]
pub mod confidence_scoring;
#[cfg(feature = "eval")]
pub mod model_ensemble;
#[cfg(feature = "eval")]
pub mod auto_model_selection;
#[cfg(feature = "eval")]
pub mod self_consistency;
#[cfg(feature = "eval")]
pub mod cot_parsing;
#[cfg(feature = "eval")]
pub mod output_validation;

#[cfg(feature = "eval")]
pub use benchmark::{
    BenchmarkConfig, BenchmarkStats, BenchmarkResult, BenchmarkSuite, BenchmarkRunner,
    run_all_benchmarks, compare_results, BenchmarkComparison,
};

#[cfg(feature = "eval")]
pub use evaluation::{
    MetricType, MetricResult, EvalSample, EvalResult, Evaluator,
    TextQualityEvaluator, RelevanceEvaluator, SafetyEvaluator,
    BenchmarkResult as EvalBenchmarkResult, Benchmarker, AbTestConfig, AbTestResult, AbTestManager,
    EvalSuite, EvalSummary,
};

#[cfg(feature = "eval")]
pub use fine_tuning::{
    TrainingFormat, TrainingDataset, ChatTrainingExample, ChatTrainingMessage,
    CompletionTrainingExample, AlpacaTrainingExample, ShareGPTExample, ShareGPTTurn,
    ValidationResult as DatasetValidationResult, ValidationError as DatasetValidationError,
    DatasetError, Hyperparameters, FineTuneStatus, FineTuneJob, FineTuneError, FineTuneEvent,
    OpenAIFineTuneClient, FileUploadResponse, CreateFineTuneRequest, FineTuneApiError,
    DatasetConverter, LoraConfig, LoraBias, LoraTaskType, LoraAdapter, LoraManager,
    TrainingMetrics, TrainingCallback, LoggingCallback as TrainingLoggingCallback,
    EarlyStoppingCallback,
};

#[cfg(feature = "eval")]
pub use hallucination_detection::{
    HallucinationConfig, HallucinationDetector, HallucinationType, HallucinationDetection,
    HallucinationResult, Claim, ClaimType,
};

#[cfg(feature = "eval")]
pub use confidence_scoring::{
    ConfidenceConfig, ConfidenceScorer, ConfidenceScore, Reliability,
    ConfidenceBreakdown, CertaintyIndicator, UncertaintyIndicator, UncertaintyType,
    CalibrationStats,
};

#[cfg(feature = "eval")]
pub use model_ensemble::{
    EnsembleConfig, Ensemble, EnsembleStrategy, EnsembleModel, ModelResponse,
    EnsembleResult, EnsembleBuilder,
};

#[cfg(feature = "eval")]
pub use auto_model_selection::{
    AutoSelectConfig, AutoModelSelector, ModelProfile as AutoModelProfile,
    ModelCapabilities as AutoModelCapabilities, TaskType as AutoTaskType,
    SelectionResult, Requirements as ModelRequirementsAuto, ModelStats,
};

#[cfg(feature = "eval")]
pub use self_consistency::{
    ConsistencyConfig, ConsistencyChecker, ConsistencyResult, AnswerGroup,
    VotingConsistency, VotingResult, Sample, ConsistencyAggregator,
};

#[cfg(feature = "eval")]
pub use cot_parsing::{
    CotConfig, CotParser, CotParseResult, ReasoningStep, StepType,
    CotValidator, ValidationResult as CotValidationResult,
};

#[cfg(feature = "eval")]
pub use output_validation::{
    ValidationConfig, OutputValidator, OutputFormat, ValidationResult as OutputValidationResult,
    SchemaValidator as OutputSchemaValidator, ValidationIssue, IssueSeverity, IssueType,
};

// =============================================================================
// RAG FEATURE
// =============================================================================

#[cfg(feature = "rag")]
pub mod rag;
#[cfg(feature = "rag")]
pub mod rag_advanced;
#[cfg(feature = "rag")]
pub mod rag_tiers;
#[cfg(feature = "rag")]
pub mod rag_debug;
#[cfg(feature = "rag")]
pub mod rag_pipeline;
#[cfg(feature = "rag")]
pub mod rag_methods;
#[cfg(feature = "rag")]
pub mod query_expansion;
#[cfg(feature = "rag")]
pub mod knowledge_graph;
#[cfg(feature = "rag")]
pub mod citations;
#[cfg(feature = "rag")]
pub mod auto_indexing;

#[cfg(feature = "rag")]
pub use rag::{
    RagDb, RagConfig, KnowledgeChunk, StoredMessage, User, DEFAULT_USER_ID,
    build_knowledge_context, build_conversation_context,
    KnowledgeExport, ExportedChunk, ExportedSource,
    HybridRagConfig, HybridKnowledgeResult,
    KnowledgeUsage, KnowledgeSourceUsage,
};

#[cfg(feature = "rag")]
pub use assistant::{IndexingResult, IndexingProgress, DocumentInfo, DocumentStats};

#[cfg(feature = "rag")]
pub use persistence::{PersistentCache, CacheEntry};

#[cfg(feature = "rag")]
pub use rag_advanced::{
    ChunkingStrategy, ChunkingConfig, SmartChunk, SmartChunker,
    DeduplicationResult, ChunkDeduplicator,
    RerankConfig, RankedChunk, ChunkReranker, ChunkMetadata,
};

#[cfg(feature = "rag")]
pub use rag_tiers::{
    RagTier, RagFeatures, RagConfig as RagTierConfig, RagRequirement,
    HybridWeights, RagStats, TierSelectionHints, UserPreference, QueryComplexity,
    auto_select_tier,
};

// Knowledge graph exports - use module path to avoid naming conflicts
// with similar types in other modules (e.g., multi_agent, entity_manager)
#[cfg(feature = "rag")]
pub use knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, KnowledgeGraphStore, KnowledgeGraphBuilder,
    Entity as KGEntity, EntityType as KGEntityType, Relation as KGRelation,
    EntityMention as KGEntityMention, GraphChunk, GraphStats as KGStats,
    EntityExtractor as KGEntityExtractor, LlmEntityExtractor, PatternEntityExtractor,
    ExtractionResult as KGExtractionResult, ExtractedEntity, ExtractedRelation,
    IndexingResult as KGIndexingResult, GraphQueryResult, KnowledgeGraphCallback,
};

#[cfg(feature = "rag")]
pub use rag_debug::{
    RagDebugLevel, RagDebugConfig, RagDebugLogger, RagDebugStep, RagDebugSession,
    RagSessionStats, RagQuerySession, AggregateRagStats, ScoreChange, AllSessionsExport,
    global_rag_debug, configure_global_rag_debug, enable_rag_debug, disable_rag_debug,
};

#[cfg(feature = "rag")]
pub use rag_pipeline::{
    RagPipeline, RagPipelineConfig, RagPipelineResult, RagPipelineStats, RagPipelineError,
    RetrievedChunk, ChunkPosition as PipelineChunkPosition,
    LlmCallback, EmbeddingCallback, RetrievalCallback, GraphCallback, GraphRelation,
};

#[cfg(feature = "rag")]
pub use rag_methods::{
    // Types
    ScoredItem, MethodResult, LlmGenerate, EmbeddingGenerate,
    // Query Enhancement
    QueryExpander as AdvancedQueryExpander, QueryExpanderConfig,
    MultiQueryDecomposer, MultiQueryConfig,
    HydeGenerator, HydeConfig,
    // Result Processing
    LlmReranker, LlmRerankerConfig,
    CrossEncoderReranker, CrossEncoderScore,
    RrfFusion, RrfConfig,
    ContextualCompressor, CompressionConfig,
    // Self-Improvement
    SelfRagEvaluator, SelfRagConfig, SelfReflectionResult, SelfReflectionAction,
    CragEvaluator, CragConfig, CragResult, CragAction,
    AdaptiveStrategySelector, AdaptiveStrategyConfig, RetrievalStrategy,
    // Advanced
    GraphRagRetriever, GraphRagConfig, Entity as GraphEntity, EntityMention, Relationship, GraphDatabase,
    RaptorRetriever, RaptorConfig, RaptorNode,
};

#[cfg(feature = "rag")]
pub use query_expansion::{
    ExpansionConfig, QueryExpander, ExpandedQuery, ExpansionSource,
    MultiQueryRetriever, ExpansionResult, ExpansionStats, ScoredResult,
};

#[cfg(feature = "rag")]
pub use citations::{
    CitationConfig, CitationGenerator, CitationStyle, Citation, Source, SourceType,
    CitedText, CitationVerifier, VerificationResult as CitationVerificationResult,
    UnverifiedCitation,
};

#[cfg(feature = "rag")]
pub use auto_indexing::{
    IndexChunkingStrategy, IndexedDocumentMeta, IndexableChunk,
    ChunkMetadata as IndexChunkMetadata, ChunkPosition, AutoIndexConfig,
    IndexState, AutoIndexer, IndexStats, IndexingResult as AutoIndexingResult,
};

#[cfg(feature = "rag")]
pub mod encrypted_knowledge;

#[cfg(feature = "rag")]
pub use encrypted_knowledge::{
    // Core types
    KpkgReader, KpkgBuilder, KpkgError, KpkgManifest,
    ExtractedDocument, KpkgIndexResult, KpkgIndexResultExt, RagDbKpkgExt,
    // Key providers
    AppKeyProvider, CustomKeyProvider, KeyProvider, KEY_SIZE, NONCE_SIZE,
    // Professional KPKG types
    ExamplePair, RagPackageConfig, KpkgMetadata,
};

// =============================================================================
// EGUI WIDGETS FEATURE
// =============================================================================

#[cfg(feature = "egui-widgets")]
pub mod widgets;

// =============================================================================
// ADDITIONAL MODULES (always available, lightweight)
// =============================================================================

pub mod i18n;
pub mod export;
pub mod conversation_control;
pub mod memory;
pub mod routing;
pub mod cost;
pub mod entities;
pub mod health_check;
pub mod distributed_rate_limit;
pub mod openapi_export;
pub mod quantization;
pub mod priority_queue;
pub mod keepalive;
pub mod few_shot;
pub mod prompt_optimizer;
pub mod token_budget;
pub mod regeneration;
pub mod summarization;
pub mod intent;
pub mod user_rate_limit;
pub mod api_key_rotation;
pub mod forecasting;
pub mod request_signing;
pub mod webhooks;
pub mod message_queue;
pub mod typing_indicator;
pub mod smart_suggestions;
pub mod conversation_templates;
pub mod context_window;
pub mod conversation_compaction;
pub mod memory_pinning;
pub mod response_ranking;
pub mod answer_extraction;
pub mod fact_verification;
pub mod conversation_snapshot;
pub mod incremental_sync;
pub mod conflict_resolution;
pub mod web_search;
pub mod content_versioning;
pub mod edit_operations;
pub mod patch_application;
pub mod text_transform;
pub mod code_editing;
pub mod translation_analysis;
pub mod entity_enrichment;
pub mod decision_tree;
pub mod task_planning;
pub mod request_coalescing;
pub mod prefetch;
pub mod model_warmup;

// Export all lightweight modules
pub use i18n::{
    DetectedLanguage, LanguageDetector, LanguagePreferences,
    LocalizedStrings, MultilingualPromptBuilder,
};

pub use export::{
    ExportFormat, ExportOptions, ExportedConversation, ExportedMessage,
    ConversationExporter, ImportOptions, ImportResult, ConversationImporter,
};

pub use conversation_control::{
    CancellationToken, MessageOperations, EditResult, RegenerateResult,
    BranchPoint, BranchManager, VariantManager, ResponseVariant,
};

pub use memory::{
    MemoryEntry, MemoryType, MemoryConfig, MemoryStore, WorkingMemory,
    MemoryManager, MemoryStats,
};

pub use routing::{
    TaskType, ModelRequirements, ModelCapabilityProfile,
    ModelRouter, RoutingDecision,
};

pub use cost::{
    ModelPricing, CostEstimate, CostTracker, CostEstimator,
    BudgetManager, BudgetStatus,
};

pub use entities::{
    EntityType, Entity, EntityExtractorConfig, CustomPattern, EntityExtractor,
    Fact, FactType, FactExtractorConfig, FactExtractor, FactStore,
};

pub use health_check::{
    HealthChecker, HealthCheckConfig, HealthCheckResult, HealthStatus,
    ProviderHealth, HealthSummary, HealthCheckType,
};

pub use distributed_rate_limit::{
    DistributedRateLimiter, RateLimitBackend, InMemoryBackend,
    RateLimitState, DistributedRateLimitResult,
};

pub use openapi_export::{
    OpenApiSpec, OpenApiInfo, OpenApiServer, OpenApiPathItem, OpenApiOperation,
    OpenApiParameter, OpenApiRequestBody, OpenApiResponse, OpenApiComponents,
    JsonSchema as OpenApiJsonSchema, OpenApiBuilder, OperationBuilder,
    generate_ai_assistant_spec, export_to_json, export_to_yaml,
};

pub use quantization::{
    QuantFormat, QuantizationDetector, MemoryRequirements, HardwareProfile,
    QuantRecommendation, ModelSize, GgufMetadata, FormatComparison,
};

pub use priority_queue::{
    Priority, PriorityRequest, PriorityQueue, QueueConfig, QueueStats,
    QueueError, SharedPriorityQueue, WorkerQueue,
};

pub use keepalive::{
    ConnectionState, KeepaliveConfig, ConnectionInfo, HeartbeatResult,
    KeepaliveEvent, KeepaliveManager, KeepaliveHandle, KeepaliveStats,
    ConnectionMonitor,
};

pub use few_shot::{
    ExampleCategory, Example, SelectionConfig, FewShotManager, FewShotStats,
    ExampleSets, ExampleBuilder,
};

pub use prompt_optimizer::{
    PromptVariant, OptimizerConfig, PromptOptimizer, Feedback as PromptFeedback,
    OptimizationStats, VariantReport, PromptShortener,
};

pub use token_budget::{
    BudgetPeriod, Budget, BudgetUsage, BudgetCheckResult, BudgetAlert,
    BudgetManager as TokenBudgetManager, BudgetStats as TokenBudgetStats,
    TokenEstimator, RequestPlanner, PlannedRequest,
};

pub use regeneration::{
    RegenerationFeedback, RegenerationIssue, ResponseStyle as RegenResponseStyle,
    LengthPreference, RegenerationRequest, RegenerationManager,
};

pub use summarization::{
    SummaryConfig as ConvSummaryConfig, SummaryStyle as ConvSummaryStyle,
    ConversationSummary, ConversationSummarizer,
};

pub use intent::{Intent, IntentResult, IntentClassifier};

pub use user_rate_limit::{
    UserRateLimitConfig, RateLimitCheckResult as UserRateLimitResult, UserRateLimiter,
};

pub use api_key_rotation::{
    KeyStatus, ApiKey, RotationConfig, ApiKeyManager, KeyStats,
};

pub use forecasting::{
    UsageDataPoint, UsageForecast, Trend, UsageForecaster, CapacityEstimate,
};

pub use request_signing::{
    SignatureAlgorithm, SignedRequest, RequestSigner, SignatureError,
};

pub use webhooks::{
    WebhookEvent, WebhookConfig, WebhookPayload, DeliveryResult,
    WebhookManager, WebhookStats,
};

pub use message_queue::{
    QueueMessage, MemoryQueue, QueueError as MessageQueueError,
    DeadLetterQueue,
};

pub use typing_indicator::{
    TypingState, TypingIndicator, AnimatedIndicator, ProgressIndicator,
};

pub use smart_suggestions::{
    SuggestionType, Suggestion, SuggestionGenerator, SuggestionConfig,
};

pub use conversation_templates::{
    TemplateCategory, ConversationTemplate, TemplateVariable as ConvTemplateVariable,
    TemplateLibrary,
};

pub use context_window::{
    ContextWindow, ContextWindowConfig, ContextMessage, EvictionStrategy as ContextEvictionStrategy,
};

pub use conversation_compaction::{
    ConversationCompactor, CompactableMessage,
    CompactionConfig as ConvCompactionConfig, CompactionResult as ConvCompactionResult,
};

pub use memory_pinning::{
    PinManager, PinnedItem, PinType, AutoPinner,
};

pub use response_ranking::{
    ResponseRanker, ResponseCandidate, RankingCriteria, ScoreBreakdown,
};

pub use answer_extraction::{
    AnswerExtractor, ExtractedAnswer, AnswerType, ExtractionConfig,
};

pub use fact_verification::{
    FactVerifier, VerifiedFact, VerificationStatus, FactSource, VerificationConfig,
    FactVerifierBuilder,
};

pub use conversation_snapshot::{
    ConversationSnapshot, SnapshotMetadata, SnapshotMessage, MemoryItem,
    SnapshotManager, SnapshotDiff, MemoryChange,
};

pub use incremental_sync::{
    IncrementalSyncManager, SyncEntry, SyncOperation, SyncState, SyncDelta, SyncLog,
    SyncError, TwoWaySyncCoordinator,
};

pub use conflict_resolution::{
    ConflictResolver, Conflict, ConflictType, ResolutionStrategy, Resolution,
    ConflictError, ThreeWayMerge, MergeResult, MergeConflictLine,
};

pub use web_search::{
    WebSearchManager, SearchProvider, SearchResult as WebSearchResult, SearchConfig,
    DuckDuckGoProvider, BraveSearchProvider, SearXNGProvider,
};

pub use content_versioning::{
    ContentSnapshot, ContentChange, ChangeType as VersionChangeType,
    VersionDiff, VersioningConfig, VersionHistory, ContentVersionStore,
};

pub use edit_operations::{
    Position, TextRange, EditKind, Edit, EditError,
    EditBuilder, TextEditor, LineEditor,
};

pub use patch_application::{
    PatchLine, PatchHunk, Patch, PatchConfig, PatchResult,
    PatchParseError, PatchApplyError, PatchApplicator,
};

pub use text_transform::{
    Transform, TransformResult, TextTransformer, TransformPipeline,
};

pub use code_editing::{
    LanguageConfig, CodeEditor, EditSuggestion, EditCategory,
    CodeSearch, SearchScope,
};

pub use translation_analysis::{
    GlossaryEntry, Glossary, TranslationIssueType, TranslationIssue,
    AlignedSegment, TranslationAnalysisResult, TranslationStats,
    TranslationAnalysisConfig, ComparisonPrompt, ComparisonResponse,
    TranslationAnalyzer,
};

pub use entity_enrichment::{
    EnrichableEntity, EnrichmentData, EnrichedEntity, MergeStrategy,
    DuplicateMatch, DuplicateReason, EnrichmentSource, EnrichmentConfig,
    EntityEnricher,
};

pub use decision_tree::{
    ConditionOperator, Condition, DecisionBranch, DecisionNodeType,
    DecisionNode, DecisionPath, DecisionTree, DecisionTreeBuilder,
};

pub use task_planning::{
    StepStatus, StepPriority, PlanStep as TaskPlanStep, StepNote,
    TaskPlan, PlanSummary, PlanBuilder,
};

pub use request_coalescing::{
    CoalescingConfig, RequestCoalescer, CoalescingKey, CoalescedResult,
    CoalescingHandle, SemanticCoalescer, CoalescableRequest, CoalescingStats,
};

pub use prefetch::{
    PrefetchConfig, Prefetcher, PrefetchCandidate, PrefetchedResponse, QueryPattern,
    ContextPredictor, PrefetchStats,
};

pub use model_warmup::{
    WarmupConfig, WarmupManager, WarmupStatus, ModelUsageStats, WarmupStats,
    ScheduledWarmup, WarmupTime,
};

// =============================================================================
// BINARY INTEGRITY FEATURE
// =============================================================================

pub mod binary_integrity;

pub use binary_integrity::{
    IntegrityResult, IntegrityConfig, IntegrityChecker,
    hash_file, hash_bytes, startup_integrity_check,
};
