// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! AI Assistant library for local and cloud LLM integration.
//!
//! This crate provides a unified interface to connect to various LLM providers:
//! Ollama, LM Studio, text-generation-webui, Kobold.cpp, LocalAI, OpenAI,
//! Anthropic, Google Gemini, AWS Bedrock, HuggingFace, and any OpenAI-compatible API.
//!
//! # Cargo Features
//!
//! This crate uses Cargo features for selective compilation. Features in `full`
//! are included when using `--features full`; others are opt-in.
//!
//! ## Included in `full`
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `core` | Basic assistant, providers, config, prompt management |
//! | `multi-agent` | Multi-agent orchestration, shared memory, 5 roles |
//! | `security` | Guardrails, PII detection, toxicity, attack detection |
//! | `analytics` | Metrics, telemetry, engagement tracking |
//! | `vision` | Multimodal/vision support |
//! | `embeddings` | Local embeddings, neural cross-encoder, HNSW |
//! | `advanced-streaming` | SSE, WebSocket RFC 6455, resumable streaming |
//! | `adapters` | External API adapters (OpenAI, Anthropic, HuggingFace) |
//! | `tools` | Tool calling, MCP protocol, agentic loops |
//! | `documents` | Document parsing (EPUB, DOCX, HTML, PDF, feeds) |
//! | `eval` | Evaluation, benchmarking, A/B testing |
//! | `rag` | SQLite FTS5 knowledge base, encrypted packages |
//! | `distributed` | DHT, CRDTs, MapReduce (parallel via rayon) |
//! | `binary-storage` | Bincode + gzip for internal data |
//! | `async-runtime` | Tokio + reqwest for async HTTP |
//! | `advanced-memory` | Episodic, procedural, entity memory |
//!
//! ## Opt-in features (not in `full`)
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `autonomous` | Autonomous agent: policy, sandbox, OS tools, profiles |
//! | `scheduler` | Cron-like task scheduling + event triggers |
//! | `butler` | Environment auto-detection and configuration |
//! | `browser` | Chrome DevTools Protocol automation |
//! | `distributed-agents` | Agents across distributed nodes |
//! | `distributed-network` | QUIC transport, consistent hashing, replication |
//! | `server-tls` | HTTPS for the embedded server (rustls) |
//! | `containers` | Docker-based sandboxed execution |
//! | `audio` | STT/TTS via cloud APIs |
//! | `whisper-local` | Offline STT via whisper.cpp |
//! | `workflows` | Event-driven workflow engine with checkpointing |
//! | `prompt-signatures` | DSPy-style prompt optimization |
//! | `a2a` | Google Agent-to-Agent protocol |
//! | `voice-agent` | Real-time bidirectional audio streaming |
//! | `media-generation` | Image/video generation (DALL-E, SD, Runway) |
//! | `distillation` | Trajectory collection, dataset building |
//! | `constrained-decoding` | Grammar-guided generation (GBNF, JSON Schema) |
//! | `hitl` | Human-in-the-Loop approval gates |
//! | `webrtc` | Real-time voice via SDP/ICE/RTP |
//! | `devtools` | Agent debugging, profiling, replay |
//! | `vector-lancedb` | LanceDB embedded vector database |
//! | `vector-pgvector` | PostgreSQL pgvector SQL generation |
//! | `cloud-connectors` | S3 and Google Drive integration |
//! | `aws-bedrock` | AWS Bedrock SigV4 authentication |
//! | `p2p` | Peer-to-peer networking with knowledge sharing |
//! | `code-sandbox` | Isolated code execution for agents |
//! | `integrity-check` | Binary integrity verification at startup |
//! | `wasm` | WebAssembly support (web-sys/js-sys) |
//! | `egui-widgets` | Pre-built egui widgets for chat UI |
//! | `server-axum` | Production axum server (HTTP/2, WS, SSE, tower) |
//! | `server-axum-tls` | HTTPS for the axum server (rustls) |
//! | `server-cluster` | Distributed cluster (CRDTs, QUIC mesh, health) |
//! | `server-openapi` | Swagger UI / OpenAPI at `/swagger-ui` |
//! | `redis-backend` | Redis for rate limiting, sessions, caching |
//!
//! # Example
//!
//! ```no_run
//! use ai_assistant::{AiAssistant, AiConfig};
//!
//! let mut assistant = AiAssistant::new();
//! assistant.fetch_models();
//! ```

// Clippy lint configuration — suppress warnings that are stylistic or would
// require large refactors without improving correctness.
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_without_default)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::inherent_to_string)]
#![allow(clippy::manual_strip)]
#![allow(clippy::single_match)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::manual_pattern_char_comparison)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::regex_creation_in_loops)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::io_other_error)]
#![allow(clippy::option_as_ref_deref)]
#![allow(clippy::let_and_return)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::map_clone)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::collapsible_str_replace)]
#![allow(clippy::double_ended_iterator_last)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::lines_filter_map_ok)]
#![allow(clippy::trim_split_whitespace)]
#![allow(clippy::while_let_on_iterator)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::useless_format)]
#![allow(clippy::unnecessary_get_then_check)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::unnecessary_filter_map)]
#![allow(clippy::manual_unwrap_or_default)]
#![allow(clippy::clone_on_ref_ptr)]
#![allow(clippy::iter_cloned_collect)]
#![allow(clippy::manual_hash_one)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::unused_unit)]
#![allow(clippy::needless_return)]
#![allow(clippy::approx_constant)]
#![allow(clippy::invalid_regex)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::explicit_into_iter_loop)]
#![allow(clippy::needless_doctest_main)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::iter_on_single_items)]
#![allow(clippy::map_entry)]

// =============================================================================
// CORE MODULES (always available)
// =============================================================================

mod assistant;
mod config;
mod context;
mod messages;
mod models;
mod providers;
mod session;

// Core utilities — always available regardless of feature flags
pub mod adaptive_thinking;
pub mod async_support;

/// Prelude module — import commonly-used types with `use ai_assistant::prelude::*;`
pub mod prelude;
pub mod batch;
pub mod cache_compression;
pub mod caching;
pub mod cloud_providers;
pub mod config_file;
pub mod connection_pool;
pub mod crawl_policy;
pub mod data_source_client;
pub mod debug;
pub mod diff;
pub mod error;
pub mod events;
pub mod fallback;
pub mod formatting;
pub mod http_client;
pub mod internal_storage;
pub mod secure_credentials;
pub mod llm_judge;
pub mod log_redaction;
pub mod memory_management;
pub mod persistence;
pub mod plugins;
pub mod profiles;
pub mod progress;
pub mod request_queue;
pub mod retry;
pub mod search;
pub mod server;
#[cfg(feature = "server-axum")]
pub mod server_axum;
#[cfg(feature = "server-axum")]
pub mod virtual_model;
#[cfg(feature = "server-cluster")]
pub mod cluster;
#[cfg(feature = "redis-backend")]
pub mod redis_backend;
pub mod streaming;
pub mod structured;
pub mod templates;
pub mod wasm;
pub mod wasm_hooks;

// Core re-exports
pub use assistant::{AiAssistant, SummaryResult};
pub use config::{AiConfig, AiProvider};
pub use context::{
    clear_context_size_cache, context_size_cache_len, estimate_tokens,
    estimate_tokens_for_model, get_model_context_size, get_model_context_size_cached,
    ContextUsage,
};
pub use messages::{AiResponse, ChatMessage};
pub use models::{ModelCapabilityInfo, ModelInfo, ModelRegistry};
pub use providers::{
    build_system_prompt, build_system_prompt_with_notes, fetch_model_context_size,
    generate_response, generate_response_streaming, generate_response_streaming_cancellable,
    ConnectionPoolHandle, PoolHandleConfig, ProviderConfig as LlmProviderConfig,
    ProviderHealthStatus, ProviderRateState, ProviderRegistry as LlmProviderRegistry,
    RateLimitHeaders, ResilientError, ResilientProviderRegistry,
    AuditEntry as ProviderAuditEntry, AuditSummary, AuditedProvider,
};
pub use session::{
    ChatSession, ChatSessionStore, JournalEntry, JournalEntryType, JournalSession, ResponseStyle,
    UserPreferences,
};

pub use error::{
    AiError, AiResult, ConfigError, ContextualError, IoError as AiIoError, NetworkError,
    ProviderError, RagError, ResourceLimitError, ResultExt, SerializationError,
    ValidationError as AiValidationError,
};

pub use progress::{
    logging_callback, silent_callback, MultiProgressTracker, OperationHandle, Progress,
    ProgressAggregator, ProgressCallback, ProgressCallbackBuilder, ProgressReporter,
};

pub use events::{
    AiEvent, CollectingHandler, EventBus, EventHandler, EventTimer, FilteredHandler, LogLevel,
    LoggingHandler, TimestampedEvent,
};

pub use config_file::{
    default_config_path, load_config, register_config_tools, save_config,
    CacheConfig as FileCacheConfig, ConfigFile, ConfigFormat, ConfigValidationError, ConfigWatcher,
    ContainersConfig, GenerationConfig, HybridConfig, LoggingConfig, ProviderConfig, RagFileConfig,
    ReloadResult, ReloadScope, SecurityConfig, UrlConfig,
};

pub use memory_management::{
    format_bytes, BoundedCache, BoundedVec, CacheStats as BoundedCacheStats, ComponentMemory,
    EvictionPolicy, MemoryEstimate, MemoryPressure, MemoryReport, MemoryTracker,
};

pub use wasm::{is_wasm, Capability, PlatformCapabilities};

pub use wasm_hooks::{
    AgentConfig as WasmAgentConfig, AgentState as WasmAgentState, ChatConfig as WasmChatConfig,
    ChatMessage as WasmChatMessage, ToolCallInfo as WasmToolCallInfo, UseAgentHook, UseChatHook,
    UseCompletionHook,
};

pub use async_support::{
    async_channel, block_on, join, join3, retry_async, select, sleep, spawn_blocking, stream,
    timeout, yield_now, AsyncIterator, AsyncLoad, AsyncMutex, AsyncMutexGuard, AsyncReceiver,
    AsyncResult, AsyncRetryConfig, AsyncSave, AsyncSender, AsyncStream, BlockingHandle, BoxFuture,
    Sleep, Timeout, YieldNow,
};

pub use retry::{
    retry, retry_with_config, CircuitBreaker, CircuitState, ResilientExecutor, RetryAttempt,
    RetryConfig, RetryExecutor, RetryResult, RetryableError,
};

pub use profiles::{ModelProfile, ProfileApplicator, ProfileBuilder, ProfileManager};

pub use adaptive_thinking::{
    is_trivial_query, parse_thinking_tags, AdaptiveThinkingConfig, ClassificationSignals,
    QueryClassifier, RagTierPriority, ThinkingDepth, ThinkingParseResult, ThinkingStrategy,
    ThinkingTagParser,
};

pub use templates::{
    BuiltinTemplates, PromptTemplate, TemplateBuilder, TemplateManager, TemplateVariable,
    VariableType,
};

pub use formatting::{
    extract_code_by_language, extract_first_code, extract_first_json, parse_as_json, to_plain_text,
    CodeBlock, Heading, ListItem, ParsedLink, ParsedList, ParsedResponse, ParsedTable,
    ParserConfig, ResponseParser, TableAlignment,
};

pub use search::{
    ConversationSearcher, HighlightSpan, SearchContext, SearchMode, SearchQuery,
    SearchQueryBuilder, SearchResult as ConversationSearchResult, SearchStats,
};

pub use caching::{
    BudgetCostAlert, CacheConfig, CacheKey, CacheMiddlewareStats, CacheStats as ResponseCacheStats,
    CachedLlmResponse, CachedResponse, CostTracker as CacheCostTracker, ResponseCache,
    ResponseCacheMiddleware, SemanticCache, UsageRecord,
};

pub use structured::{
    extract_json_from_response, EnforcementConfig, EnforcementResult, JsonSchema, SchemaBuilder,
    SchemaProperty, SchemaType, SchemaValidator, StructuredOutputEnforcer,
    StructuredOutputError, StructuredOutputGenerator, StructuredOutputRequest,
    StructuredOutputStrategy, StructuredParseResult, StructuredRequest, ValidationError,
    ValidationResult,
};

pub use diff::{
    compare_responses as diff_compare_responses, diff, inline_word_diff, ChangeType, DiffHunk,
    DiffLine, DiffResult, InlineDiffSegment, InlineWordDiff,
    ResponseComparison as DiffResponseComparison, WordStats,
};

pub use streaming::{
    BackpressureStream, Chunker, RateLimitedStream, StreamBuffer, StreamConsumer, StreamError,
    StreamMetrics, StreamProducer, StreamingConfig,
};

pub use connection_pool::{
    global_pool, init_global_pool, ConnectionPool, PoolConfig, PoolStats, PooledConnection,
    PooledConnectionGuard,
};

pub use cache_compression::{
    compress, compress_string, decompress, decompress_string, CacheCompressionStats,
    CompressedCache, CompressedData, CompressionAlgorithm, CompressionError, CompressionLevel,
    StreamingCompressor,
};

pub use internal_storage::{
    convert_json_to_binary, convert_to_json, deserialize_internal, detect_format, dump_as_json,
    file_info, load_internal, save_internal, serialize_internal, InternalFileInfo, StorageFormat,
};

pub use llm_judge::{
    BatchEvalResult, EvalCriterion, JudgeResult, LlmJudge, PairwiseResult,
    RagFaithfulnessResult,
};

pub use log_redaction::{contains_sensitive, redact, redact_with_config, RedactionConfig};

pub use http_client::{
    fetch_kobold_models_with, fetch_ollama_models_with, fetch_openai_models_with,
    parse_kobold_models, parse_ollama_models, parse_openai_models, HttpClient, UreqClient,
};

pub use server::{
    AiServer, AuditEntry as ServerAuditEntry, AuditEventType as ServerAuditEventType,
    AuditLog as ServerAuditLog, AuthConfig, AuthResult, CompactionEnrichmentConfig,
    ContextEnrichmentConfig, CorsConfig, CostEnrichmentConfig,
    EnrichmentConfig as ServerEnrichmentConfig, GuardrailEnrichmentConfig,
    ModelSelectionEnrichmentConfig, RagEnrichmentConfig, ServerConfig, ServerHandle,
    ServerRateLimiter, StructuredError, ThinkingEnrichmentConfig, TlsConfig,
};
#[cfg(feature = "server-tls")]
pub use server::load_tls_config;

pub use secure_credentials::{
    CallbackSource, CredentialError, CredentialResolver, CredentialSource, EnvVarSource, FileSource,
    SecureString, StaticSource,
};

pub use cloud_providers::{
    fetch_anthropic_cloud_models, fetch_cloud_models, fetch_openai_cloud_models,
    generate_anthropic_cloud, generate_cloud_response, generate_openai_cloud, resolve_api_key,
};

pub use request_queue::{
    QueueStats as RequestQueueStats, QueuedRequest, RequestPriority, RequestQueue,
};

pub use plugins::{
    IpAllowlistPlugin, LoggingPlugin, MessageProcessorPlugin, MetricsCollectorPlugin, Plugin,
    PluginCapability, PluginContext, PluginInfo, PluginManager, RequestLoggingPlugin,
};

pub use batch::{
    BatchBuilder, BatchConfig, BatchProcessor, BatchRequest, BatchResult, BatchResults, BatchStats,
};

pub use fallback::{
    FallbackChain, FallbackError, FallbackProvider, FallbackResult,
    ProviderState as FallbackProviderState, ProviderStatus,
};

pub use debug::{
    configure_global_debug, global_debug, set_debug_level, CapturedRequest, DebugConfig,
    DebugEntry, DebugLevel, DebugLogger, DebugReport, RequestInspector,
};

pub use persistence::{
    BackupConfig, BackupInfo, BackupManager, CacheEntryInfo, CacheStats, CompactionConfig,
    CompactionResult, DatabaseCompactor, DatabaseStats, FullExport, MigrationConfig,
    MigrationResult, PersistentCacheConfig, SessionMigrator,
};

pub use crawl_policy::{
    ChangeFrequency, CrawlPolicy, CrawlPolicyConfig, ParsedRobotsTxt, ParsedSitemap,
    RobotsDirectives, RobotsRule, SitemapEntry,
};

// =============================================================================
// MULTI-AGENT FEATURE
// =============================================================================

#[cfg(feature = "multi-agent")]
pub mod agent;
#[cfg(feature = "multi-agent")]
pub mod agent_memory;
#[cfg(feature = "multi-agent")]
pub mod multi_agent;
#[cfg(feature = "multi-agent")]
pub mod task_decomposition;

#[cfg(feature = "multi-agent")]
pub use multi_agent::{
    Agent, AgentMessage, AgentOrchestrator, AgentRole, AgentStatus, AgentTask, BusMessage,
    CollaborationSession, ContextEntry, ConversationPattern, ContextTransferPolicy,
    HandoffManager, HandoffRequest, HandoffResult, MessageBus, MessageType, OrchestrationError,
    OrchestrationStatus, OrchestrationStrategy, PatternAgent, PatternConfig, PatternMessage,
    PatternResult, PatternRunner, SharedContext, TaskDispatcher, TaskStatus,
    TerminationCondition,
};

#[cfg(feature = "multi-agent")]
pub use task_decomposition::{
    DecompositionAnalysis, DecompositionStatus, DecompositionStrategy, FlatTask, TaskDecomposer,
    TaskNode,
};

#[cfg(feature = "multi-agent")]
pub use agent_memory::{
    MemoryEntry as AgentMemoryEntry, MemoryError, MemoryStats as AgentMemoryStats,
    MemoryType as AgentMemoryType, SharedMemory, ThreadSafeMemory,
};

#[cfg(feature = "multi-agent")]
pub use agent::{
    create_builtin_agent_tools, AgentCallback, AgentConfig, AgentContext, AgentExecutor,
    AgentResult, AgentState, AgentStep, AgentTool, LoggingCallback, PlanStep, PlanStepStatus,
    PlanningAgent, ReactAgent,
};

// =============================================================================
// ASYNC RUNTIME FEATURE
// =============================================================================

#[cfg(feature = "async-runtime")]
pub mod async_provider_plugin;
#[cfg(feature = "async-runtime")]
pub mod async_providers;

#[cfg(feature = "async-runtime")]
pub use async_provider_plugin::{
    AsyncProviderPlugin, AsyncProviderRegistry, AsyncToSyncAdapter, SyncToAsyncAdapter,
};

#[cfg(feature = "async-runtime")]
pub use async_providers::{
    block_on_async, create_runtime, fetch_kobold_models_async, fetch_models_async,
    fetch_ollama_models_async, fetch_openai_models_async, generate_response_async,
    generate_response_streaming_async, AsyncHttpClient, ReqwestClient,
};

// =============================================================================
// DISTRIBUTED FEATURE
// =============================================================================

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "distributed")]
pub use distributed::{
    // MapReduce
    DataChunk,
    Dht,
    DhtNode,
    DhtValue,
    // Coordinator
    DistributedCoordinator,
    // CRDTs
    GCounter,
    KBucket,
    LWWMap,
    LWWRegister,
    MapOutput,
    MapReduceBuilder,
    MapReduceJob,
    // DHT
    NodeId,
    ORSet,
    PNCounter,
    ReduceOutput,
    RoutingTable,
};

#[cfg(feature = "distributed-network")]
pub use distributed::NodeMessage;

// =============================================================================
// DISTRIBUTED NETWORKING (optional — QUIC transport, replication, failure detection)
// =============================================================================

#[cfg(feature = "distributed-network")]
pub mod consistent_hash;
#[cfg(feature = "distributed-network")]
pub mod distributed_network;
#[cfg(feature = "distributed-network")]
pub mod failure_detector;
#[cfg(feature = "distributed-network")]
pub mod merkle_sync;
#[cfg(feature = "distributed-network")]
pub mod node_security;

#[cfg(feature = "distributed-network")]
pub use consistent_hash::ConsistentHashRing;
#[cfg(feature = "distributed-network")]
pub use distributed_network::{
    DiscoveryConfig as NetworkDiscoveryConfig, NetworkConfig, NetworkEvent, NetworkNode,
    NetworkStats, PeerState as NetworkPeerState, ReplicationConfig, RingInfo, WriteMode,
};
#[cfg(feature = "distributed-network")]
pub use failure_detector::{HeartbeatConfig, HeartbeatManager, NodeStatus, PhiAccrualDetector};
#[cfg(feature = "distributed-network")]
pub use merkle_sync::{AntiEntropySync, MerkleProof, MerkleTree, SyncDelta as MerkleSyncDelta};
#[cfg(feature = "distributed-network")]
pub use node_security::{CertificateManager, ChallengeResponse, JoinToken, NodeIdentity};

// =============================================================================
// P2P NETWORKING (optional)
// =============================================================================

#[cfg(feature = "p2p")]
pub mod p2p;

#[cfg(feature = "p2p")]
pub use p2p::{
    ContradictionReport, IceAgent, IceCandidate, IceCandidateType, IceState, KnowledgeShare,
    NatDiscoveryResult, NatTraversal, NatType, P2PConfig, P2PManager, P2PStats, PeerConnection,
    PeerDataTrust, PeerInfo, PeerMessage, PeerReputation, ReputationSystem, TurnConfig,
};

// =============================================================================
// AUTONOMOUS AGENT FEATURE
// =============================================================================

#[cfg(feature = "autonomous")]
pub mod agent_policy;
#[cfg(feature = "autonomous")]
pub mod agent_profiles;
#[cfg(feature = "autonomous")]
pub mod agent_sandbox;
#[cfg(feature = "autonomous")]
pub mod autonomous_loop;
#[cfg(feature = "browser")]
pub mod browser_tools;
#[cfg(feature = "butler")]
pub mod butler;
#[cfg(feature = "butler")]
pub use butler::{
    AdvisorConfig, AdvisorReport, AdvisorSummary, ButlerAdvisor, ButlerRecommendation,
    OptimizationCategory, RecommendationPriority,
};
#[cfg(feature = "distributed-agents")]
pub mod distributed_agents;
#[cfg(feature = "autonomous")]
pub mod interactive_commands;
#[cfg(feature = "autonomous")]
pub mod mode_manager;
#[cfg(feature = "autonomous")]
pub mod container_tools;
#[cfg(feature = "autonomous")]
pub mod mcts_planner;
#[cfg(feature = "autonomous")]
pub mod os_tools;
#[cfg(feature = "scheduler")]
pub mod scheduler;
#[cfg(feature = "autonomous")]
pub mod task_board;
#[cfg(feature = "scheduler")]
pub mod trigger_system;
#[cfg(feature = "autonomous")]
pub mod user_interaction;
#[cfg(feature = "scheduler")]
pub use trigger_system::{
    CronExpression, SchedulerConfig, SchedulerError, SchedulerRunner, SchedulerState, TsCronField,
};

#[cfg(feature = "browser")]
pub use browser_tools::{
    find_chrome_binary, register_browser_tools, BrowserError, BrowserSession,
    PageContent as BrowserPageContent,
};

#[cfg(feature = "distributed-agents")]
pub use distributed_agents::{
    AgentNodeInfo, DistributedAgentManager, DistributedTask, MapReduceAgentJob, MapReduceStatus,
    TaskDistributionStatus,
};

#[cfg(feature = "autonomous")]
pub use agent_policy::{
    ActionDescriptor, ActionType, AgentPolicy, AgentPolicyBuilder, ApprovalHandler, AutoApproveAll,
    AutoDenyAll, AutonomyLevel, ClosureApprovalHandler, InternetMode, RiskLevel as AgentRiskLevel,
};

#[cfg(feature = "autonomous")]
pub use agent_sandbox::{AuditDecision, AuditEntry, SandboxError, SandboxValidator};

#[cfg(feature = "autonomous")]
pub use agent_profiles::{
    AgentProfile, ConversationProfile, ProfileRegistry, WorkflowPhase, WorkflowProfile,
};

#[cfg(feature = "autonomous")]
pub use user_interaction::{
    AutoApproveHandler as AutoApproveInteraction, BufferedInteractionHandler,
    CallbackInteractionHandler, InteractionManager, NotifyLevel, PendingQuery,
    UserInteractionHandler, UserQuery, UserResponse,
};

#[cfg(feature = "autonomous")]
pub use autonomous_loop::{
    AgentResult as AutonomousAgentResult, AgentState as AutonomousAgentState, AutonomousAgent,
    AutonomousAgentBuilder, AutonomousAgentConfig, InterAgentMessage, IterationOutcome,
};

#[cfg(feature = "autonomous")]
pub use mode_manager::{ModeManager, OperationMode};

#[cfg(feature = "autonomous")]
pub use task_board::{
    BoardCommand, TaskBoard, TaskBoardListener, TaskBoardSummary, TaskExecutionState,
};

#[cfg(feature = "autonomous")]
pub use interactive_commands::{CommandProcessor, CommandResult, UserIntent};

#[cfg(feature = "autonomous")]
pub use multi_agent::{MultiAgentSession, SessionSummary as MultiAgentSessionSummary};

#[cfg(feature = "autonomous")]
pub use mcts_planner::{
    AgentMctsState, AggregationStrategy, ExecutionFeedback, LlmPRM, MctsConfig, MctsNode,
    MctsPlanner, MctsResult, MctsState, PrmAggregator, PrmRule, PrmRuleCheck,
    ProcessRewardModel, RefinementConfig, RefinementLoop, RefinementResult, RefinementStrategy,
    RuleBasedPRM, SimulationPolicy, StepScore,
};

// =============================================================================
// MULTI-LAYER GRAPH (always available)
// =============================================================================

pub mod multi_layer_graph;
pub use multi_layer_graph::{
    BeliefExtractor, BeliefType, ConfidenceLevel, ConflictPolicy, Contradiction, ContradictionLog,
    ContradictionResolution, ContradictionSource, GraphCluster, GraphDiff, GraphLayer,
    InferredRelation, InternetGraph, InternetGraphEntry, LayerConfig, LayeredEntity,
    MergeStrategy as GraphMergeStrategy, MultiLayerGraph, MultiLayerGraphStats,
    MultiLayerQueryResult, SessionGraph, SyncPolicy, UnifiedEntity, UnifiedRelation, UnifiedView,
    UserBelief, UserGraph,
};

// =============================================================================
// SECURITY FEATURE
// =============================================================================

#[cfg(feature = "security")]
pub mod access_control;
#[cfg(feature = "security")]
pub mod advanced_guardrails;
#[cfg(feature = "security")]
pub mod content_encryption;
#[cfg(feature = "security")]
pub mod content_moderation;
#[cfg(feature = "security")]
pub mod data_anonymization;
#[cfg(feature = "security")]
pub mod guardrail_pipeline;
#[cfg(feature = "security")]
pub mod injection_detection;
#[cfg(feature = "security")]
pub mod pii_detection;
#[cfg(feature = "security")]
pub mod security;

#[cfg(feature = "security")]
pub use security::{
    AuditConfig, AuditEvent, AuditEventType, AuditLogger, AuditStats, HookChainResult, HookManager,
    HookResult, InputSanitizer, RateLimitConfig, RateLimitReason, RateLimitResult, RateLimitStatus,
    RateLimitUsage, RateLimiter, SanitizationConfig, SanitizationResult, SanitizationWarning,
};

#[cfg(feature = "security")]
pub use pii_detection::{
    CustomPiiPattern, DetectedPii, PiiConfig, PiiDetector, PiiResult, PiiType, RedactionStrategy,
    SensitivityLevel,
};

#[cfg(feature = "security")]
pub use content_moderation::{
    ContentModerator, ModerationAction, ModerationCategory, ModerationConfig, ModerationFlag,
    ModerationResult, ModerationStats,
};

#[cfg(feature = "security")]
pub use injection_detection::{
    CustomPattern as InjectionCustomPattern, DetectionSensitivity, InjectionConfig,
    InjectionDetection, InjectionDetector, InjectionResult, InjectionType, Recommendation,
    RiskLevel,
};

#[cfg(feature = "security")]
pub use access_control::{
    AccessCondition, AccessControlEntry, AccessControlManager, AccessResult, Permission,
    ResourceType, Role,
};

#[cfg(feature = "security")]
pub use data_anonymization::{
    AnonymizationResult, AnonymizationRule, AnonymizationStrategy, BatchAnonymizer, DataAnonymizer,
    DataType as AnonymizationDataType, Detection,
};

#[cfg(feature = "security")]
pub use content_encryption::{
    ContentEncryptor, EncryptedContent, EncryptedMessageStore, EncryptionAlgorithm,
    EncryptionError, EncryptionKey,
};

#[cfg(feature = "security")]
pub use advanced_guardrails::{
    default_principles, AttackDetectionResult, AttackDetector, AttackType, BiasConfig,
    BiasDetectionResult, BiasDetector, BiasDimension, BiasOccurrence, ConstitutionalAI,
    ConstitutionalConfig, ConstitutionalEvaluation, ConstitutionalPrinciple, DetectedAttack,
    FullSafetyCheck, GuardrailsManager, InputCheckResult, OutputCheckResult, PrincipleViolation,
    ToxicityCategory, ToxicityConfig, ToxicityDetector, ToxicityResult,
};

#[cfg(feature = "security")]
pub use guardrail_pipeline::{
    AttackGuard, ContentLengthGuard, Guard, GuardAction, GuardCheckResult, GuardStage,
    GuardrailPipeline, NaturalLanguageGuard, OutputPiiConfig, OutputPiiGuard,
    OutputToxicityConfig, OutputToxicityGuard, PatternGuard, PiiGuard, PipelineResult,
    PolicyCompiler, PolicyPriority, PolicyScope, PolicyStatement, PolicyViolation,
    RateLimitGuard, SemanticChecker, StreamGuardAction, StreamingGuard,
    StreamingGuardrailConfig, StreamingGuardrailMetrics, StreamingGuardrailPipeline,
    StreamingPatternGuard, StreamingPiiGuard, StreamingToxicityGuard, ToxicityGuard,
};

// =============================================================================
// ANALYTICS FEATURE
// =============================================================================

#[cfg(feature = "analytics")]
pub mod analysis;
#[cfg(feature = "analytics")]
pub mod conversation_analytics;
#[cfg(feature = "analytics")]
pub mod conversation_flow;
#[cfg(feature = "analytics")]
pub mod latency_metrics;
#[cfg(feature = "analytics")]
pub mod metrics;
#[cfg(feature = "analytics")]
pub mod prometheus_metrics;
#[cfg(feature = "analytics")]
pub mod quality;
#[cfg(feature = "analytics")]
pub mod response_effectiveness;
#[cfg(feature = "analytics")]
pub mod streaming_metrics;
#[cfg(feature = "analytics")]
pub mod telemetry;
#[cfg(feature = "analytics")]
pub mod scalability_monitor;
#[cfg(feature = "analytics")]
pub mod user_engagement;

#[cfg(feature = "analytics")]
pub use metrics::{
    ConversationTestCase, MessageMetrics, MessageMetricsBuilder, MetricsExport, MetricsTracker,
    RagQualityMetrics, SearchCache, SessionMetrics, TestCaseResult, TestSuite, TestSuiteResults,
};

#[cfg(feature = "analytics")]
pub use analysis::{
    ConversationSentimentAnalysis, EmojiCategory, EmoticonAnalysis, EmoticonDetector,
    EmoticonMatch, Sentiment, SentimentAnalysis, SentimentAnalyzer, SentimentTrend,
    SessionSummarizer, SessionSummary, SummaryConfig, Topic, TopicDetector,
};

#[cfg(feature = "analytics")]
pub use streaming_metrics::{
    AggregatedMetrics, FinalMetrics, MetricsConfig, MetricsDisplay, StreamingMetrics,
    StreamingSnapshot,
};

#[cfg(feature = "analytics")]
pub use latency_metrics::{
    LatencyRecord, LatencyStats, LatencyTracker, LatencyTrend, RequestTimer,
};

#[cfg(feature = "analytics")]
pub use telemetry::{
    AggregatedMetrics as TelemetryMetrics, TelemetryCollector, TelemetryConfig, TelemetryEvent,
    TimedOperation,
};

#[cfg(feature = "analytics")]
pub use conversation_analytics::{
    AggregatedStats, AnalyticsConfig, AnalyticsEvent, AnalyticsReport, ConversationAnalytics,
    EventType, EventValue, ExportedEvent, PatternTracker,
};

#[cfg(feature = "analytics")]
pub use user_engagement::{
    EngagementEvent, EngagementManager, EngagementMetrics, EngagementTracker, UserTrends,
};

#[cfg(feature = "analytics")]
pub use response_effectiveness::{
    BatchEvaluator, BatchResult as EffectivenessBatchResult, EffectivenessScore,
    EffectivenessScorer, QAPair, ScoringWeights, UserFeedback as EffectivenessFeedback,
};

#[cfg(feature = "analytics")]
pub use prometheus_metrics::{AiMetricsRegistry, Counter, Gauge, Histogram, HistogramTimer};

#[cfg(feature = "analytics")]
pub use quality::{
    compare_responses, QualityAnalyzer, QualityConfig, QualityIssue, QualityIssueType,
    QualityScore, ResponseComparison,
};

#[cfg(feature = "analytics")]
pub use conversation_flow::{
    ConversationTurn, FlowAnalysis, FlowAnalyzer, FlowState, TopicTransition,
};

#[cfg(feature = "analytics")]
pub use scalability_monitor::{
    ScalabilityAction, ScalabilitySnapshot, ScalabilityWarning, Subsystem,
    WarningSeverity as ScalabilityWarningSeverity,
};

// =============================================================================
// VISION FEATURE
// =============================================================================

#[cfg(feature = "vision")]
pub mod vision;

#[cfg(feature = "vision")]
pub use vision::{
    ImageBatch, ImageData, ImageDetail, ImageInput, ImagePreprocessor, VisionCapabilities,
    VisionMessage,
};

// =============================================================================
// EMBEDDINGS FEATURE
// =============================================================================

#[cfg(feature = "embeddings")]
pub mod embedding_cache;
#[cfg(feature = "embeddings")]
pub mod embedding_providers;
#[cfg(feature = "embeddings")]
pub mod embeddings;
#[cfg(feature = "embeddings")]
pub mod hnsw;
#[cfg(feature = "embeddings")]
pub mod neural_embeddings;
#[cfg(feature = "embeddings")]
pub mod vector_db;

#[cfg(feature = "embeddings")]
pub use embeddings::{
    EmbeddingConfig, HybridSearchConfig, HybridSearchResult, HybridSearcher, IndexedDocument,
    LocalEmbedder, SearchResult, SemanticIndex,
};

#[cfg(feature = "embeddings")]
pub use embedding_cache::{
    cosine_similarity, normalize_vector, BatchEmbeddingCache, EmbeddingCache, EmbeddingCacheConfig,
    EmbeddingCacheStats, SharedEmbeddingCache, SimilarityEmbeddingCache,
};

#[cfg(feature = "embeddings")]
pub use vector_db::{
    migrate_vectors, string_id_to_u64, BackendInfo, DistanceMetric, FilterOperation,
    HybridVectorSearch, InMemoryVectorDb, MetadataFilter, QdrantClient, StoredVector, VectorDb,
    VectorDbBackend, VectorDbBuilder, VectorDbConfig, VectorMigrationResult, VectorSearchResult,
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
    cosine_similarity as neural_cosine_similarity, dot_product as neural_dot_product,
    euclidean_distance, normalize_l2, CrossEncoder, DenseEmbedder, DenseEmbeddingConfig,
    DimensionalityReduction, EmbeddingError, EmbeddingVec, HybridEmbedding, HybridRetriever,
    PoolingStrategy, QuantizationType, QuantizedEmbedding, RankedDocument, SparseEmbedder,
    SparseEmbedding, SparseEmbeddingConfig,
};

#[cfg(feature = "embeddings")]
pub use hnsw::{HnswConfig, HnswIndex, HnswVectorDb};

#[cfg(feature = "embeddings")]
pub use embedding_providers::{
    create_embedding_provider, EmbeddingProvider, HuggingFaceEmbeddings, LocalTfIdfEmbedding,
    OllamaEmbeddings, OpenAIEmbeddings,
};

// =============================================================================
// ADVANCED STREAMING FEATURE
// =============================================================================

#[cfg(feature = "advanced-streaming")]
pub mod sse_streaming;
#[cfg(feature = "advanced-streaming")]
pub mod streaming_compression;
#[cfg(feature = "advanced-streaming")]
pub mod websocket_streaming;

#[cfg(feature = "advanced-streaming")]
pub use sse_streaming::{
    SseClient, SseConnection, SseError, SseEvent, SseReader, SseWriter, StreamAggregator,
    StreamChoice, StreamChunk, StreamDelta,
};

#[cfg(feature = "advanced-streaming")]
pub use websocket_streaming::{
    BidirectionalStream, WsAiMessage, WsCallbacks, WsCloseCode, WsError, WsFrame, WsHandshake,
    WsOpcode, WsState, WsStreamHandler, WsUsage,
};

#[cfg(feature = "advanced-streaming")]
pub use streaming_compression::{
    compress_string as stream_compress_string, decompress_string as stream_decompress_string,
    Algorithm as StreamCompressionAlgorithm, CompressionConfig as StreamCompressionConfig,
    CompressionResult as StreamCompressionResult, CompressionStats as StreamCompressionStats,
    Level as StreamCompressionLevel, StreamCompressor, StreamDecompressor,
};

// Resumable streaming (SSE-based, checkpoint/resume for long-running streams)
#[cfg(feature = "advanced-streaming")]
pub mod resumable_streaming;

#[cfg(feature = "advanced-streaming")]
pub use resumable_streaming::{
    format_sse_event, ResumableStream, ResumableStreamConfig, StreamCheckpoint,
    StreamChunk as ResumableStreamChunk,
};

// =============================================================================
// ADAPTERS FEATURE
// =============================================================================

#[cfg(feature = "adapters")]
pub mod anthropic_adapter;
#[cfg(feature = "adapters")]
pub mod gemini_provider;
#[cfg(feature = "adapters")]
pub mod huggingface_connector;
#[cfg(feature = "adapters")]
pub mod openai_adapter;
#[cfg(feature = "adapters")]
pub mod provider_plugins;

#[cfg(feature = "adapters")]
pub use openai_adapter::{
    OpenAIClient, OpenAIConfig, OpenAIMessage, OpenAIModel, OpenAIRequest, OpenAIResponse,
};

#[cfg(feature = "adapters")]
pub use anthropic_adapter::{
    AnthropicClient, AnthropicConfig, AnthropicMessage, AnthropicModel, AnthropicRequest,
    AnthropicResponse,
};

#[cfg(feature = "adapters")]
pub use gemini_provider::GeminiProvider;

#[cfg(feature = "adapters")]
pub use huggingface_connector::{HfClient, HfConfig, HfRequest, HfResponse, HfTask};

#[cfg(feature = "adapters")]
pub use provider_plugins::{
    create_registry_with_discovery, discover_providers, DiscoveryConfig, KoboldCppProvider,
    LmStudioProvider, OllamaProvider, OpenAICompatibleProvider, TextGenWebUIProvider,
};

// =============================================================================
// TOOLS FEATURE
// =============================================================================

#[cfg(feature = "tools")]
pub mod agentic_loop;
#[cfg(feature = "tools")]
pub mod function_calling;
#[cfg(feature = "tools")]
pub mod mcp_client;
#[cfg(feature = "tools")]
pub mod mcp_protocol;
#[cfg(feature = "tools")]
pub mod model_integration;
#[cfg(feature = "tools")]
pub mod prompt_chaining;
#[cfg(feature = "tools")]
pub mod tool_calling;
#[cfg(feature = "tools")]
pub mod tool_use;
#[cfg(feature = "tools")]
pub mod tools;
#[cfg(feature = "tools")]
pub mod unified_tools;

#[cfg(feature = "tools")]
pub use unified_tools::{
    builtin_tools as unified_builtin_tools, evaluate_math,
    parse_tool_calls as unified_parse_tool_calls, ParamSchema, ParamType as UnifiedParamType,
    ProviderCapabilities as UnifiedProviderCapabilities, ProviderPlugin as UnifiedProviderPlugin,
    ProviderRegistry as UnifiedProviderRegistry, ToolBuilder, ToolCall as UnifiedToolCall,
    ToolChoice as UnifiedToolChoice, ToolDef as UnifiedToolDef, ToolError as UnifiedToolError,
    ToolHandler as UnifiedToolHandler, ToolOutput as UnifiedToolOutput,
    ToolRegistry as UnifiedToolRegistry,
};

#[cfg(feature = "tools")]
pub use tools::{
    create_builtin_tools, ApprovalGate, ApprovalStatus, ArgumentSource, OutputTransform,
    ParameterType, PendingApproval, ProviderCapabilities, ProviderPlugin, ProviderRegistry,
    RetryableToolCall, ToolCall, ToolChain, ToolChainResult, ToolChainStep, ToolDefinition,
    ToolParameter, ToolRegistry, ToolResult, ToolRetryConfig, ToolStepResult, ToolValidator,
    ValidationRule, ValidationType,
};

#[cfg(feature = "tools")]
pub use tool_use::{
    parse_tool_calls, ParameterType as ToolParameterType, Tool as ToolUseTool,
    ToolCall as ToolUseCall, ToolError, ToolParameter as ToolUseParameter,
    ToolRegistry as ToolUseRegistry, ToolResult as ToolUseResult,
};

#[cfg(feature = "tools")]
pub use tool_calling::{
    CommonTools, ParameterType as ParamType, Tool as ToolDef, ToolCall as ToolInvocation,
    ToolParameter as ToolParam, ToolRegistry as ToolRepo, ToolResult as ToolOutput,
};

#[cfg(feature = "tools")]
pub use function_calling::{
    parse_function_calls, FunctionBuilder, FunctionCall, FunctionDefinition, FunctionParameters,
    FunctionRegistry, FunctionResult, ParameterProperty, ToolChoice,
};

#[cfg(feature = "tools")]
pub use mcp_protocol::{
    AnnotatedTool, AuthorizationServerMetadata, DynamicClientRegistration,
    InMemorySessionStore, McpClient, McpContent, McpError, McpPrompt, McpPromptMessage,
    McpRequest, McpResource, McpResourceContent, McpResponse, McpServer, McpServerCapabilities,
    McpSession, McpSessionStore, McpTool, McpV2OAuthConfig, OAuthToken, OAuthTokenManager,
    PkceChallenge, StreamableHttpTransport, ToolAnnotationRegistry, ToolAnnotations,
    TransportMode, MCP_VERSION,
};

#[cfg(feature = "tools")]
pub use mcp_client::{
    ClientInfo as McpRemoteClientInfo, McpClientAuth, McpClientConfig, McpClientPool,
    RemoteMcpClient, RemoteResource, RemoteTool, RemoteToolAnnotations, RemoteToolRegistry,
    ResourceContent as McpResourceContentRemote,
    ServerCapabilities as McpRemoteServerCapabilities, ToolCallResult as McpToolCallResult,
    ToolResultContent as McpToolResultContent,
};

#[cfg(feature = "tools")]
pub use agentic_loop::{
    AgentBuilder, LoopConfig, AgentLoopResult, LoopMessage,
    LoopRole, LoopState, LoopStatus, AgenticLoop,
    IterationResult,
};

#[cfg(feature = "tools")]
pub use model_integration::{
    create_lm_studio_client, create_ollama_client, ChatMessage as IntegratedChatMessage,
    ChatResponse, ChatRole, FinishReason, IntegratedModelClient,
    LMStudioProvider as LMStudioIntegrated, ModelError, ModelProvider,
    OllamaProvider as OllamaIntegrated, TokenUsage,
};

#[cfg(feature = "tools")]
pub use prompt_chaining::{
    ChainBuilder, ChainConfig, ChainExecutor, ChainResult, ChainStep, ChainTemplates,
    ExtractionMethod, PromptChain, StepCondition, StepResult, VariableExtraction,
};

// =============================================================================
// DOCUMENTS FEATURE
// =============================================================================

#[cfg(feature = "documents")]
pub mod document_parsing;
#[cfg(feature = "documents")]
pub mod feed_monitor;
#[cfg(feature = "documents")]
pub mod html_extraction;
#[cfg(feature = "documents")]
pub mod table_extraction;

#[cfg(feature = "documents")]
pub use document_parsing::{
    DocumentFormat, DocumentImageAnalysis, DocumentMetadata, DocumentParser, DocumentParserConfig,
    DocumentSection, ExtractedImage, ImageExtractionConfig, ImageExtractor, ImageFormat,
    OcrPipeline, OcrPipelineConfig, PageContent, ParsedDocument, PdfTable, TemplateOcrBackend,
    TesseractConfig, TesseractOcrBackend,
};

#[cfg(feature = "documents")]
pub use table_extraction::{
    ExtractedTable, TableCell, TableExtractor, TableExtractorConfig, TableSourceFormat,
};

#[cfg(feature = "documents")]
pub use feed_monitor::{
    FeedCheckResult, FeedEntry, FeedFormat, FeedMetadata, FeedMonitor, FeedMonitorConfig,
    FeedMonitorState, FeedParser, ParsedFeed,
};

#[cfg(feature = "documents")]
pub use html_extraction::{
    HtmlElement, HtmlExtractionConfig, HtmlExtractionResult, HtmlExtractor, HtmlLink, HtmlList,
    HtmlMetadata, HtmlSelector,
};

// =============================================================================
// EVAL FEATURE
// =============================================================================

#[cfg(feature = "eval")]
pub mod ab_testing;
#[cfg(feature = "eval")]
pub mod agent_eval;
#[cfg(feature = "eval")]
pub mod auto_model_selection;
#[cfg(feature = "eval")]
pub mod benchmark;
#[cfg(feature = "eval")]
pub mod confidence_scoring;
#[cfg(feature = "eval")]
pub mod cot_parsing;
#[cfg(feature = "eval")]
pub mod evaluation;
#[cfg(feature = "eval")]
pub mod fine_tuning;
#[cfg(feature = "eval")]
pub mod hallucination_detection;
#[cfg(feature = "eval")]
pub mod model_ensemble;
#[cfg(feature = "eval")]
pub mod output_validation;
#[cfg(feature = "eval")]
pub mod red_team;
#[cfg(feature = "eval")]
pub mod self_consistency;

#[cfg(feature = "eval")]
pub use ab_testing::{
    AbTestError, Experiment as AbExperiment, ExperimentManager,
    ExperimentResult as AbExperimentResult, ExperimentStatus, ExperimentVariant, MetricRecord,
    SignificanceCalculator, VariantAssigner, VariantAssignment, VariantStats,
};

#[cfg(feature = "eval")]
pub use agent_eval::{
    AgentMetrics, AnalyzerConfig, EvalReport, EvalTrajectoryStep, ExpectedToolCall,
    MetricsComparison, ReportBuilder, StepActionType, StepBreakdown, ToolAccuracyMetrics,
    ToolCallEvaluator, ToolCallMatch, TrajectoryAnalyzer, TrajectoryRecorder,
};

#[cfg(feature = "eval")]
pub use benchmark::{
    compare_results, run_all_benchmarks, BenchmarkComparison, BenchmarkConfig, BenchmarkResult,
    BenchmarkRunner, BenchmarkStats, BenchmarkSuite,
};

#[cfg(feature = "eval")]
pub use evaluation::{
    AbTestConfig, AbTestManager, AbTestResult, BenchmarkResult as EvalBenchmarkResult, Benchmarker,
    EvalResult, EvalSample, EvalSuite, EvalSummary, Evaluator, MetricResult, MetricType,
    RelevanceEvaluator, SafetyEvaluator, TextQualityEvaluator,
};

#[cfg(feature = "eval")]
pub use fine_tuning::{
    AlpacaTrainingExample, ChatTrainingExample, ChatTrainingMessage, CompletionTrainingExample,
    CreateFineTuneRequest, DatasetConverter, DatasetError, EarlyStoppingCallback,
    FileUploadResponse, FineTuneApiError, FineTuneError, FineTuneEvent, FineTuneJob,
    FineTuneStatus, Hyperparameters, LoggingCallback as TrainingLoggingCallback, LoraAdapter,
    LoraBias, LoraConfig, LoraManager, LoraTaskType, OpenAIFineTuneClient, ShareGPTExample,
    ShareGPTTurn, TrainingCallback, TrainingDataset, TrainingFormat, TrainingMetrics,
    ValidationError as DatasetValidationError, ValidationResult as DatasetValidationResult,
};

#[cfg(feature = "eval")]
pub use hallucination_detection::{
    Claim, ClaimType, HallucinationConfig, HallucinationDetection, HallucinationDetector,
    HallucinationResult, HallucinationType,
};

#[cfg(feature = "eval")]
pub use confidence_scoring::{
    CalibrationStats, CertaintyIndicator, ConfidenceBreakdown, ConfidenceConfig, ConfidenceScore,
    ConfidenceScorer, Reliability, UncertaintyIndicator, UncertaintyType,
};

#[cfg(feature = "eval")]
pub use model_ensemble::{
    Ensemble, EnsembleBuilder, EnsembleConfig, EnsembleModel, EnsembleResult, EnsembleStrategy,
    ModelResponse,
};

#[cfg(feature = "eval")]
pub use auto_model_selection::{
    AutoModelSelector, AutoSelectConfig, CacheablePrompt, FallbackChain as SelectorFallbackChain,
    FallbackStrategy, ModelCapabilities as AutoModelCapabilities,
    ModelCostEntry as SelectorCostEntry, ModelCostRegistry as SelectorCostRegistry,
    ModelInvocation, ModelPerformanceStats, ModelProfile as AutoModelProfile, ModelStats,
    PerformanceTracker as SelectorPerformanceTracker, PipelineRouter,
    PipelineRoutingDecision, PipelineTaskType, PromptSegment,
    Requirements as ModelRequirementsAuto, RoutingRule as PipelineRoutingRule,
    SelectionResult, SmartSelector, TaskType as AutoTaskType,
};

#[cfg(feature = "eval")]
pub use self_consistency::{
    AnswerGroup, ConsistencyAggregator, ConsistencyChecker, ConsistencyConfig, ConsistencyResult,
    Sample, VotingConsistency, VotingResult,
};

#[cfg(feature = "eval")]
pub use cot_parsing::{
    CotConfig, CotParseResult, CotParser, CotValidator, ReasoningStep, StepType,
    ValidationResult as CotValidationResult,
};

#[cfg(feature = "eval")]
pub use output_validation::{
    IssueSeverity, IssueType, OutputFormat, OutputValidator,
    SchemaValidator as OutputSchemaValidator, ValidationConfig, ValidationIssue,
    ValidationResult as OutputValidationResult,
};

#[cfg(feature = "eval")]
pub use red_team::{
    AttackCategory, AttackGenerator, AttackInstance, AttackSeverity, AttackTemplate,
    CategoryReport, DefenseEvaluator, DetectionMethod, RedTeamConfig, RedTeamReport,
    RedTeamResult, RedTeamSuite,
};

// =============================================================================
// EVAL-SUITE FEATURE
// =============================================================================

#[cfg(feature = "eval-suite")]
pub mod eval_suite;

#[cfg(feature = "eval-suite")]
pub use eval_suite::{
    filter_by_contamination_cutoff, filter_by_language, make_code_edit_problem,
    make_competitive_problem, make_livecode_problem, make_terminal_problem, register_eval_tools,
    AblationEngine, AblationRecommendation, AblationResult, AblationStudy, AnswerFormat,
    BenchmarkDataset, BenchmarkProblem, BenchmarkRunResult, BenchmarkSuiteRunner,
    BenchmarkSuiteType, ComparisonConfig, ComparisonMatrix, ConfigMeasurement, ConfigSearchConfig,
    ConfigSearchEngine, ConfigSearchResult, CostBreakdown, DefaultScorer, EloCalculator,
    EvalAgentConfig, EvalSuiteReport, EvolutionSnapshot, ModelIdentifier, MultiModelGenerator,
    ProblemCategory, ProblemResult, ProblemScorer, ReportBuilder as EvalSuiteReportBuilder,
    ReportSummary, RunConfig, RunSummary, SearchCost, SearchDimension, SearchIteration,
    SearchObjective, Subtask, SubtaskAnalysis, SubtaskAnalyzer, SubtaskPerformance,
    TokenUsage as EvalTokenUsage,
};

// =============================================================================
// RAG FEATURE
// =============================================================================

#[cfg(feature = "rag")]
pub mod auto_indexing;
#[cfg(feature = "rag")]
pub mod citations;
#[cfg(feature = "rag")]
pub mod knowledge_graph;
#[cfg(feature = "rag")]
pub mod query_expansion;
#[cfg(feature = "rag")]
pub mod rag;
#[cfg(feature = "rag")]
pub mod rag_advanced;
#[cfg(feature = "rag")]
pub mod rag_debug;
#[cfg(feature = "rag")]
pub mod rag_methods;
#[cfg(feature = "rag")]
pub mod rag_pipeline;
#[cfg(feature = "rag")]
pub mod rag_tiers;

#[cfg(feature = "rag")]
pub use rag::{
    build_conversation_context, build_knowledge_context, ExportedChunk, ExportedSource,
    HybridKnowledgeResult, HybridRagConfig, KnowledgeChunk, KnowledgeExport, KnowledgeSourceUsage,
    KnowledgeUsage, RagConfig, RagDb, StoredMessage, User, DEFAULT_USER_ID,
};

#[cfg(feature = "rag")]
pub use assistant::{DocumentInfo, DocumentStats, IndexingProgress, IndexingResult};

#[cfg(feature = "rag")]
pub use persistence::{CacheEntry, PersistentCache};

#[cfg(feature = "rag")]
pub use rag_advanced::{
    ChunkDeduplicator, ChunkMetadata, ChunkReranker, ChunkingConfig, ChunkingStrategy,
    DeduplicationResult, RankedChunk, RerankConfig, SmartChunk, SmartChunker,
};

#[cfg(feature = "rag")]
pub use rag_tiers::{
    auto_select_tier, HybridWeights, QueryComplexity, RagTierConfig, RagFeatures,
    RagRequirement, RagStats, RagTier, TierSelectionHints, UserPreference,
};

// Knowledge graph exports - use module path to avoid naming conflicts
// with similar types in other modules (e.g., multi_agent, entity_manager)
#[cfg(feature = "rag")]
pub use knowledge_graph::{
    Entity as KGEntity, EntityExtractor as KGEntityExtractor, EntityMention as KGEntityMention,
    EntityType as KGEntityType, ExtractedEntity, ExtractedRelation,
    ExtractionResult as KGExtractionResult, GraphChunk, GraphQueryResult, GraphStats as KGStats,
    IndexingResult as KGIndexingResult, KnowledgeGraph, KnowledgeGraphBuilder,
    KnowledgeGraphCallback, KnowledgeGraphConfig, KnowledgeGraphStore, LlmEntityExtractor,
    PatternEntityExtractor, Relation as KGRelation,
};

#[cfg(feature = "rag")]
pub use rag_debug::{
    configure_global_rag_debug, disable_rag_debug, enable_rag_debug, global_rag_debug,
    AggregateRagStats, AllSessionsExport, RagDebugConfig, RagDebugLevel, RagDebugLogger,
    RagDebugSession, RagDebugStep, RagQuerySession, RagSessionStats, ScoreChange,
};

#[cfg(feature = "rag")]
pub use rag_pipeline::{
    PipelineChunkPosition, EmbeddingCallback, GraphCallback, GraphRelation,
    LlmCallback, RagPipeline, RagPipelineConfig, RagPipelineError, RagPipelineResult,
    RagPipelineStats, RetrievalCallback, RetrievedChunk,
};

#[cfg(feature = "rag")]
pub use rag_methods::{
    AdaptiveStrategyConfig,
    AdaptiveStrategySelector,
    CompressionConfig,
    ContextualCompressor,
    CragAction,
    CragConfig,
    CragEvaluator,
    CragResult,
    CrossEncoderReranker,
    CrossEncoderScore,
    EmbeddingGenerate,
    GraphEntity,
    EntityMention,
    GraphDatabase,
    GraphRagConfig,
    // Advanced
    GraphRagRetriever,
    HydeConfig,
    HydeGenerator,
    LlmGenerate,
    // Result Processing
    LlmReranker,
    LlmRerankerConfig,
    MethodResult,
    MultiQueryConfig,
    MultiQueryDecomposer,
    // Query Enhancement
    AdvancedQueryExpander,
    QueryExpanderConfig,
    RaptorConfig,
    RaptorNode,
    RaptorRetriever,
    Relationship,
    RetrievalStrategy,
    RrfConfig,
    RrfFusion,
    // Types
    ScoredItem,
    SelfRagConfig,
    // Self-Improvement
    SelfRagEvaluator,
    SelfReflectionAction,
    SelfReflectionResult,
};

#[cfg(feature = "rag")]
pub use query_expansion::{
    ExpandedQuery, ExpansionConfig, ExpansionResult, ExpansionSource, ExpansionStats,
    MultiQueryRetriever, QueryExpander, ScoredResult,
};

#[cfg(feature = "rag")]
pub use citations::{
    Citation, CitationConfig, CitationGenerator, CitationStyle, CitationVerifier, CitedText,
    Source, SourceType, UnverifiedCitation, CitationVerificationResult,
};

#[cfg(feature = "rag")]
pub use auto_indexing::{
    AutoIndexConfig, AutoIndexer, IndexChunkMetadata, ChunkPosition,
    IndexChunkingStrategy, IndexState, IndexStats, IndexableChunk, IndexedDocumentMeta,
    AutoIndexingResult,
};

#[cfg(feature = "rag")]
pub mod encrypted_knowledge;

#[cfg(feature = "rag")]
pub use encrypted_knowledge::{
    // Key providers
    AppKeyProvider,
    CustomKeyProvider,
    // Professional KPKG types
    ExamplePair,
    ExtractedDocument,
    KeyProvider,
    KpkgBuilder,
    KpkgError,
    KpkgIndexResult,
    KpkgIndexResultExt,
    KpkgManifest,
    KpkgMetadata,
    // Core types
    KpkgReader,
    RagDbKpkgExt,
    RagPackageConfig,
    KEY_SIZE,
    NONCE_SIZE,
};

// =============================================================================
// EGUI WIDGETS FEATURE
// =============================================================================

#[cfg(feature = "egui-widgets")]
pub mod widgets;

// =============================================================================
// ADDITIONAL MODULES (always available, lightweight)
// =============================================================================

pub mod agent_graph;
pub mod answer_extraction;
pub mod api_key_rotation;
pub mod code_editing;

// REPL/CLI engine
pub mod repl;
pub mod conflict_resolution;
pub mod content_versioning;
pub mod context_window;
pub mod conversation_compaction;
pub mod conversation_control;
pub mod conversation_snapshot;
pub mod conversation_templates;
pub mod cost;
pub mod cost_integration;
pub mod dag_executor;
pub mod decision_tree;
pub mod distributed_rate_limit;
pub mod edit_operations;
pub mod entities;
pub mod entity_enrichment;
pub mod export;
pub mod fact_verification;
pub mod few_shot;
pub mod forecasting;
pub mod health_check;
pub mod i18n;
pub mod incremental_sync;
pub mod intent;
pub mod keepalive;
pub mod memory;
pub mod memory_pinning;
pub mod message_queue;
pub mod model_warmup;
pub mod multimodal_rag;

pub use multimodal_rag::{
    ImageCaptionExtractor, ModalityType, MultiModalChunk, MultiModalConfig, MultiModalDocument,
    MultiModalPipeline, MultiModalResult, MultiModalRetriever,
};

pub mod openapi_export;
pub mod patch_application;
pub mod prefetch;
pub mod priority_queue;
pub mod prompt_optimizer;
pub mod quantization;
pub mod regeneration;
pub mod reranker;

pub use reranker::{
    CascadeReranker, CrossEncoderReranker as NeuralCrossEncoderReranker, DiversityReranker,
    Reranker, ReciprocalRankFusion, RerankerConfig, RerankerPipeline, ScoredDocument,
};

pub mod request_coalescing;
pub mod request_signing;
pub mod response_ranking;
pub mod routing;
pub mod advanced_routing;
pub mod smart_suggestions;
pub mod summarization;
pub mod task_planning;
pub mod text_transform;
pub mod token_budget;
pub mod token_counter;
pub mod translation_analysis;
pub mod typing_indicator;
pub mod user_rate_limit;
pub mod web_search;
pub mod ui_hooks;
pub mod webhooks;

pub use ui_hooks::{
    ChatHooks, ChatMessage as UiChatMessage, ChatSession as UiChatSession, ChatStatus,
    ChatStreamEvent, StreamAdapter, UsageInfo,
};

pub use dag_executor::{
    DagDefinition, DagEdge, DagError, DagExecutor, DagNode, DagNodeId, DagNodeStatus, EdgeCondition,
};

pub use agent_graph::{
    AgentEdge as GraphAgentEdge, AgentGraph, AgentNode as GraphAgentNode,
    EdgeType, ExecutionTrace, GraphAnalytics, GraphError, StepStatus as GraphStepStatus, TraceStep,
};

// Export all lightweight modules
pub use i18n::{
    DetectedLanguage, LanguageDetector, LanguagePreferences, LocalizedStrings,
    MultilingualPromptBuilder,
};

pub use export::{
    ConversationExporter, ConversationImporter, ExportFormat, ExportOptions, ExportedConversation,
    ExportedMessage, ImportOptions, ImportResult,
};

pub use conversation_control::{
    BranchManager, BranchPoint, CancellationToken, EditResult, MessageOperations, RegenerateResult,
    ResponseVariant, VariantManager,
};

pub use memory::{
    MemoryConfig, MemoryEntry, MemoryManager, MemoryStats, MemoryStore, MemoryType, WorkingMemory,
};

pub use routing::{
    ModelCapabilityProfile, ModelRequirements, ModelRouter, RoutingDecision, TaskType,
};

pub use advanced_routing::{
    AdaptivePerQueryRouter, AdvancedRoutingError, ArmFeedback, ArmVisibility, BanditArm,
    BanditConfig, BanditNfaSynthesizer, BanditRouter, BanditSnapshot, BanditStrategy, BetaParams,
    ContextSnapshot, ContextualDiscovery, ContextualObservation,
    DfaRouter, DfaSnapshot, DfaState, DiscoveredSplit,
    DiscoveryConfig as RoutingDiscoveryConfig, DomainSplit, EnsembleRouter,
    EnsembleStrategy as RoutingEnsembleStrategy, FeatureDimension, FeatureImportance,
    ModelTier, NfaDfaCompiler, NfaRouter, NfaRuleBuilder, NfaSnapshot, NfaState, NfaSymbol,
    PipelineConfig, PipelineSnapshot, QueryFeatureExtractor, QueryFeatures, RewardPolicy,
    RoutingContext, RoutingDag, RoutingDagNode, RoutingDagNodeType, RoutingOutcome,
    RoutingPipeline, RoutingPreferences, RoutingVoter, SnapshotFormat, SubRouterVote,
    merge_and_compile_nfas, register_routing_tools,
};

#[cfg(feature = "eval-suite")]
pub use advanced_routing::{BanditBootstrapper, EvalFeedbackMapper};

#[cfg(feature = "distributed")]
pub use advanced_routing::{
    BanditStateMerger, DistributedBanditState, DistributedNfaState, NfaStateMerger,
};

pub use cost::{
    BudgetManager, BudgetStatus, CostEstimate, CostEstimator, CostTracker, ModelPricing,
};

pub use cost_integration::{
    CostAwareConfig, CostDashboard, CostDecision, CostMiddleware, DefaultCostMiddleware,
    RequestCostEntry, RequestType,
};

pub use entities::{
    CustomPattern, Entity, EntityExtractor, EntityExtractorConfig, EntityType, Fact, FactExtractor,
    FactExtractorConfig, FactStore, FactType,
};

pub use health_check::{
    HealthCheckConfig, HealthCheckResult, HealthCheckType, HealthChecker, HealthStatus,
    HealthSummary, ProviderHealth,
};

pub use distributed_rate_limit::{
    DistributedRateLimitResult, DistributedRateLimiter, InMemoryBackend, RateLimitBackend,
    RateLimitState,
};

pub use openapi_export::{
    export_to_json, export_to_yaml, generate_ai_assistant_spec, JsonSchema as OpenApiJsonSchema,
    OpenApiBuilder, OpenApiComponents, OpenApiInfo, OpenApiOperation, OpenApiParameter,
    OpenApiPathItem, OpenApiRequestBody, OpenApiResponse, OpenApiServer, OpenApiSpec,
    OperationBuilder,
};

pub use quantization::{
    FormatComparison, GgufMetadata, HardwareProfile, MemoryRequirements, ModelSize, QuantFormat,
    QuantRecommendation, QuantizationDetector,
};

pub use priority_queue::{
    Priority, PriorityQueue, PriorityRequest, QueueConfig, QueueError, QueueStats,
    SharedPriorityQueue, WorkerQueue,
};

pub use keepalive::{
    ConnectionInfo, ConnectionMonitor, ConnectionState, HeartbeatResult, KeepaliveConfig,
    KeepaliveEvent, KeepaliveHandle, KeepaliveManager, KeepaliveStats,
};

pub use few_shot::{
    Example, ExampleBuilder, ExampleCategory, ExampleSets, FewShotManager, FewShotStats,
    SelectionConfig,
};

pub use prompt_optimizer::{
    Feedback as PromptFeedback, OptimizationStats, OptimizerConfig, PromptOptimizer,
    PromptShortener, PromptVariant, VariantReport,
};

pub use token_budget::{
    Budget, BudgetAlert, BudgetCheckResult, BudgetManager as TokenBudgetManager, BudgetPeriod,
    BudgetStats as TokenBudgetStats, BudgetUsage, PlannedRequest, RequestPlanner, TokenEstimator,
};

pub use token_counter::{
    ApproximateCounter, BpeTokenCounter, ProviderTokenCounter, TokenAllocation, TokenBudget,
    TokenCounter,
};

pub use regeneration::{
    LengthPreference, RegenerationFeedback, RegenerationIssue, RegenerationManager,
    RegenerationRequest, ResponseStyle as RegenResponseStyle,
};

pub use summarization::{
    ConversationSummarizer, ConversationSummary, SummaryConfig as ConvSummaryConfig,
    SummaryStyle as ConvSummaryStyle,
};

pub use intent::{Intent, IntentClassifier, IntentResult};

pub use user_rate_limit::{
    RateLimitCheckResult as UserRateLimitResult, UserRateLimitConfig, UserRateLimiter,
};

pub use api_key_rotation::{ApiKey, ApiKeyManager, KeyStats, KeyStatus, RotationConfig};

pub use forecasting::{CapacityEstimate, Trend, UsageDataPoint, UsageForecast, UsageForecaster};

pub use request_signing::{RequestSigner, SignatureAlgorithm, SignatureError, SignedRequest};

pub use webhooks::{
    verify_webhook_signature, DeliveryResult, WebhookConfig, WebhookEvent, WebhookManager,
    WebhookPayload, WebhookStats,
};

pub use message_queue::{
    DeadLetterQueue, MemoryQueue, ProcessingStats, QueueError as MessageQueueError, QueueMessage,
};

pub use typing_indicator::{AnimatedIndicator, ProgressIndicator, TypingIndicator, TypingState};

pub use smart_suggestions::{
    compute_relevance, Suggestion, SuggestionConfig, SuggestionGenerator, SuggestionType,
};

pub use conversation_templates::{
    ConversationTemplate, TemplateCategory, TemplateLibrary,
    TemplateVariable as ConvTemplateVariable,
};

pub use context_window::{
    AutoTokenConfig, ContextMessage, ContextOverflowMonitor, ContextWindow, ContextWindowConfig,
    EvictionStrategy as ContextEvictionStrategy,
    OverflowLevel as WindowOverflowLevel, OverflowThresholds as WindowOverflowThresholds,
};

pub use conversation_compaction::{
    CompactableMessage, CompactionConfig as ConvCompactionConfig,
    CompactionResult as ConvCompactionResult, ConversationCompactor,
};

pub use memory_pinning::{AutoPinner, PinManager, PinType, PinnedItem};

pub use response_ranking::{RankingCriteria, ResponseCandidate, ResponseRanker, ScoreBreakdown};

pub use answer_extraction::{AnswerExtractor, AnswerType, ExtractedAnswer, ExtractionConfig};

pub use fact_verification::{
    FactSource, FactVerifier, FactVerifierBuilder, VerificationConfig, VerificationStatus,
    VerifiedFact,
};

pub use conversation_snapshot::{
    ConversationSnapshot, MemoryChange, MemoryItem, SnapshotDiff, SnapshotManager, SnapshotMessage,
    SnapshotMetadata,
};

pub use incremental_sync::{
    IncrementalSyncManager, SyncDelta, SyncEntry, SyncError, SyncLog, SyncOperation, SyncState,
    TwoWaySyncCoordinator,
};

pub use conflict_resolution::{
    Conflict, ConflictError, ConflictResolver, ConflictType, MergeConflictLine, MergeResult,
    Resolution, ResolutionStrategy, ThreeWayMerge,
};

pub use web_search::{
    BraveSearchProvider, DuckDuckGoProvider, SearXNGProvider, SearchConfig, SearchProvider,
    SearchResult as WebSearchResult, WebSearchManager,
};

pub use content_versioning::{
    ChangeType as VersionChangeType, ContentChange, ContentSnapshot, ContentVersionStore,
    VersionDiff, VersionHistory, VersioningConfig,
};

pub use edit_operations::{
    Edit, EditBuilder, EditError, EditKind, LineEditor, Position, TextEditor, TextRange,
};

pub use patch_application::{
    Patch, PatchApplicator, PatchApplyError, PatchConfig, PatchHunk, PatchLine, PatchParseError,
    PatchResult,
};

pub use text_transform::{TextTransformer, Transform, TransformPipeline, TransformResult};

pub use code_editing::{
    CodeEditor, CodeSearch, EditCategory, EditSuggestion, LanguageConfig, SearchScope,
};

pub use translation_analysis::{
    AlignedSegment, ComparisonPrompt, ComparisonResponse, Glossary, GlossaryEntry,
    TranslationAnalysisConfig, TranslationAnalysisResult, TranslationAnalyzer, TranslationIssue,
    TranslationIssueType, TranslationStats,
};

pub use entity_enrichment::{
    DuplicateMatch, DuplicateReason, EnrichableEntity, EnrichedEntity, EnrichmentConfig,
    EnrichmentData, EnrichmentSource, EntityEnricher, MergeStrategy as EntityMergeStrategy,
};

pub use decision_tree::{
    Condition, ConditionOperator, DecisionBranch, DecisionNode, DecisionNodeType, DecisionPath,
    DecisionTree, DecisionTreeBuilder,
};

pub use task_planning::{
    PlanBuilder, PlanStep as TaskPlanStep, PlanSummary, StepNote, StepPriority, StepStatus,
    TaskPlan,
};

pub use request_coalescing::{
    CoalescableRequest, CoalescedResult, CoalescingConfig, CoalescingHandle, CoalescingKey,
    CoalescingStats, RequestCoalescer, SemanticCoalescer,
};

pub use prefetch::{
    ContextPredictor, PrefetchCandidate, PrefetchConfig, PrefetchStats, PrefetchedResponse,
    Prefetcher, QueryPattern,
};

pub use model_warmup::{
    ModelUsageStats, ScheduledWarmup, WarmupConfig, WarmupManager, WarmupStats, WarmupStatus,
    WarmupTime,
};

pub use repl::{
    format_message as format_repl_message, ReplAction, ReplCommand, ReplConfig, ReplEngine,
    ReplError, ReplSession,
};

// =============================================================================
// AWS BEDROCK PROVIDER (SigV4 auth)
// =============================================================================

pub mod aws_auth;

pub use aws_auth::{
    fetch_bedrock_models, AwsCredentials, BedrockMessage, BedrockRequest, SigV4Params,
    SignedRequest as AwsSignedRequest,
};

// =============================================================================
// PGVECTOR BACKEND
// =============================================================================

#[cfg(feature = "vector-pgvector")]
pub mod vector_db_pgvector;

#[cfg(feature = "vector-pgvector")]
pub use vector_db_pgvector::{PgVectorConfig, PgVectorDb};

// =============================================================================
// OPENTELEMETRY INTEGRATION
// =============================================================================

pub mod opentelemetry_integration;

pub use opentelemetry_integration::{
    create_llm_span, create_rag_span, create_tool_span, semantic_conventions, AgentSpan,
    AgentTracer, AiSpan, BudgetAlert as OtelBudgetAlert,
    BudgetCheckResult as OtelBudgetCheckResult, CostAttributor, CostBreakdownEntry, CostBudget,
    CostReport, ExportFormat as OtelExportFormat, ExportSpanStatus, ExportableSpan,
    GenAiAttributes, GenAiConventions, GenAiEvent, GenAiEventType, GenAiSpanBuilder, GenAiSystem,
    HistogramStats, MetricsCollector, ModelPricing as OtelModelPricing, OtelConfig, OtelTracer,
    OtlpHttpExporter, PricingTable, PrometheusMetrics, SpanAttribute, SpanAttributeValue,
    SpanExporter, SpanStatus, SpanTree, SpanTreeNode, TracingMiddleware,
};

// =============================================================================
// CODE SANDBOX
// =============================================================================

pub mod code_sandbox;

pub use code_sandbox::{
    detect_dangerous_commands, sanitize_env, CodeSandbox, ExecutionResult,
    Language as SandboxLanguage, SandboxConfig,
};

// =============================================================================
// CLOUD CONNECTORS (S3 + Google Drive)
// =============================================================================

#[cfg(feature = "cloud-connectors")]
pub mod cloud_connectors;

#[cfg(feature = "cloud-connectors")]
pub use cloud_connectors::{
    AzureBlobOperation, AzureBlobRequest, CloudObject, CloudStorage, GcsOperation, GcsRequest,
    GoogleDriveClient, GoogleDriveConfig, ListOptions, ListResult, S3Client, S3Config, S3Operation,
    S3Request, StorageConnector, StorageOperation, StorageRequest,
};

// =============================================================================
// BINARY INTEGRITY FEATURE
// =============================================================================

pub mod binary_integrity;

pub use binary_integrity::{
    hash_bytes, hash_file, startup_integrity_check, IntegrityChecker, IntegrityConfig,
    IntegrityResult,
};

// =============================================================================
// SHARED FOLDER (host/container file sharing)
// =============================================================================

#[cfg(feature = "containers")]
pub mod shared_folder;

#[cfg(feature = "containers")]
pub use shared_folder::SharedFolder;

// =============================================================================
// CONTAINER EXECUTOR (Docker-based sandboxed execution)
// =============================================================================

#[cfg(feature = "containers")]
pub mod container_executor;

#[cfg(feature = "containers")]
pub use container_executor::{
    ContainerCleanupPolicy, ContainerConfig, ContainerError, ContainerExecutor, ContainerRecord,
    ContainerStatus, CreateOptions, ExecResult, NetworkMode,
};

// MCP Docker tools (containers + tools)
#[cfg(all(feature = "containers", feature = "tools"))]
pub mod mcp_docker_tools;

#[cfg(all(feature = "containers", feature = "tools"))]
pub use mcp_docker_tools::register_mcp_docker_tools;

// =============================================================================
// CONTAINER SANDBOX (Docker-based isolated code execution)
// =============================================================================

#[cfg(feature = "containers")]
pub mod container_sandbox;

#[cfg(feature = "containers")]
pub use container_sandbox::{
    ContainerSandbox, ContainerSandboxConfig, ExecutionBackend,
};

// =============================================================================
// DOCUMENT PIPELINE (container-based document creation & conversion)
// =============================================================================

#[cfg(feature = "containers")]
pub mod document_pipeline;

#[cfg(feature = "containers")]
pub use document_pipeline::{
    DocumentError, DocumentPipeline, DocumentPipelineConfig, DocumentRequest, DocumentResult,
    OutputFormat as DocumentOutputFormat, SourceFormat,
};

// =============================================================================
// SPEECH (STT / TTS via cloud APIs)
// =============================================================================

#[cfg(feature = "audio")]
pub mod speech;

#[cfg(feature = "audio")]
pub use speech::{
    create_speech_provider, AudioFormat, CoquiTtsProvider, GoogleSpeechProvider,
    LocalSpeechProvider, OpenAISpeechProvider, PiperTtsProvider, SpeechConfig, SpeechProvider,
    SynthesisOptions, SynthesisResult, TranscriptionResult, TranscriptionSegment,
};

#[cfg(feature = "whisper-local")]
pub use speech::WhisperLocalProvider;

// =============================================================================
// A2A PROTOCOL (Google Agent-to-Agent)
// =============================================================================

#[cfg(feature = "a2a")]
pub mod a2a_protocol;

#[cfg(feature = "a2a")]
pub use a2a_protocol::{
    A2AArtifact, A2AClient, A2AMessage, A2APart, A2AServer, A2ATask, A2ATaskStatus,
    AgentCard, AgentDirectory, AgentSkill, DataPart, FilePart, JsonRpcError, JsonRpcRequest,
    JsonRpcResponse, MessageRole, PushNotification, PushNotificationConfig, TaskHandler,
    TaskStatusUpdate, TextPart,
};

// =============================================================================
// EVENT-DRIVEN WORKFLOW ENGINE
// =============================================================================

#[cfg(feature = "workflows")]
pub mod event_workflow;

#[cfg(feature = "workflows")]
pub use event_workflow::{
    Checkpointer, DurableBackend, DurableCheckpoint, DurableConfig, DurableExecutor, ErrorSnapshot,
    InMemoryCheckpointer, NodeHandler, RecoveryManager, RetentionPolicy, SimpleEvent,
    WorkflowBreakpoint, WorkflowCheckpoint, WorkflowDefinition, WorkflowEdgeDef, WorkflowEvent,
    WorkflowGraph, WorkflowNode, WorkflowNodeDef, WorkflowResult, WorkflowRunner, WorkflowState,
    WorkflowTool, WorkflowToolDefinition, WorkflowToolParam,
};

// =============================================================================
// DSPY-STYLE PROMPT SIGNATURES
// =============================================================================

#[cfg(feature = "prompt-signatures")]
pub mod prompt_signature;

#[cfg(feature = "prompt-signatures")]
pub use prompt_signature::{
    AdapterRouter, AssertedSignature, AssertionResult, BayesianOptimizer, BootstrapFewShot,
    ChatAdapter, CompletionAdapter, CompiledPrompt, ContainsAnswer, ContainsAssertion,
    CustomAssertion, DiscreteSearchStrategy, EvalMetric, EvaluationBudget, ExactMatch, F1Score,
    FieldType, FormatAssertion, FormattedMessage, FormattedPrompt, FunctionCallingAdapter,
    GEPAConfig, GEPAOptimizer, GridSearchOptimizer, ImprovementRule, InstructionProposer,
    JsonSchemaAssertion, LengthAssertion, LmAdapter, MIPROv2Config, MIPROv2Optimizer,
    OptimizationResult, ParetoFront, ParetoSolution, PromptAssertion, PromptExample,
    RandomSearchOptimizer, SelfReflector, Signature, SignatureField, TrainingExample,
};

// =============================================================================
// ADVANCED MEMORY (episodic, procedural, entity, consolidation)
// =============================================================================

#[cfg(feature = "advanced-memory")]
pub mod advanced_memory;

#[cfg(feature = "advanced-memory")]
pub use advanced_memory::{
    AdvancedMemoryManager, ConsolidationPipelineResult, ConsolidationResult,
    ConsolidationSchedule, EnhancedConsolidator, EntityRecord, EntityRelation, EntityStore,
    Episode, EpisodicStore, EvolutionConfig, EvolutionReport, EvolutionStatistics,
    FactExtractor as MemoryFactExtractor, FactStore as MemoryFactStore, FeedbackOutcome,
    LlmFactExtractor, MemoryConsolidator, PatternFactExtractor, Procedure, ProceduralStore,
    ProcedureEvolver, ProcedureFeedback, SemanticFact, TemporalEdge, TemporalEdgeType,
    TemporalGraph, TemporalQuery, TemporalQueryType,
    cosine_similarity as memory_cosine_similarity, new_episode,
    AutoPersistenceConfig,
};

#[cfg(feature = "advanced-memory")]
pub mod memory_service;

#[cfg(feature = "advanced-memory")]
pub use memory_service::{
    start_memory_service, EpisodicCmd, EntityCmd, MemoryCommand, MemoryHandle,
    MemoryServiceConfig, MemoryServiceHandle, PlanCmd, SystemCmd,
};

// =============================================================================
// ONLINE EVALUATION (feedback hooks, sampling, alerting)
// =============================================================================

#[cfg(feature = "eval")]
pub mod online_eval;

#[cfg(feature = "eval")]
pub use online_eval::{
    AlertConfig, AlertEvent, CostHook, EvalContext, EvalSamplingConfig, ExecutionFingerprint,
    FeedbackHook, FeedbackScore, LatencyHook, OnlineEvaluator, RelevanceHook, ToxicityHook,
};

// =============================================================================
// CONTEXT COMPOSER (token budgeting, overflow detection, compaction)
// =============================================================================

pub mod context_composer;

pub use context_composer::{
    BudgetAllocation, CompactableMessage as ComposerMessage, CompactedConversation,
    ComposedContext, ComposedSection, ContextCompiler, ContextComposer, ContextComposerConfig,
    ContextOverflowDetector, ContextSection,
    ConversationCompactor as ContextConversationCompactor, OverflowAction,
    OverflowLevel as ComposerOverflowLevel, OverflowThresholds as ComposerOverflowThresholds,
    SectionBudget, SectionPriority, TokenBudgetAllocator, ToolSearchIndex,
    estimate_tokens as composer_estimate_tokens, generate_mini_summary,
};

// =============================================================================
// VOICE AGENT (real-time bidirectional audio, VAD, turn management)
// =============================================================================

#[cfg(feature = "voice-agent")]
pub mod voice_agent;

#[cfg(feature = "voice-agent")]
pub use voice_agent::{
    AudioChunk, AudioFormat as VoiceAudioFormat, ConversationTurn as VoiceTurn, InMemoryTransport,
    InterruptionEvent, InterruptionPolicy, TurnManager, TurnPolicy, TurnSpeaker, VadConfig,
    VadDetector, VadEvent, VoiceAgent, VoiceAgentConfig, VoiceSession, VoiceSessionState,
    VoiceTransport,
};

// =============================================================================
// WEBRTC VOICE TRANSPORT (SDP, ICE, RTP)
// =============================================================================

#[cfg(all(feature = "webrtc", feature = "voice-agent"))]
pub use voice_agent::{
    IceCandidateType, RtpStreamConfig, SdpAnswer, SdpOffer, TurnServer, WebRtcAudioCodec,
    WebRtcConfig, WebRtcIceCandidate, WebRtcSessionStats, WebRtcTransport,
};

// =============================================================================
// MEDIA GENERATION (image & video generation providers)
// =============================================================================

#[cfg(feature = "media-generation")]
pub mod media_generation;

#[cfg(feature = "media-generation")]
pub use media_generation::{
    AspectRatio, DallEEditProvider, DallEProvider, FluxProvider, GeneratedImage, GeneratedVideo,
    ImageEditConfig, ImageEditOperation, ImageEditProvider, ImageFormat as MediaImageFormat,
    ImageGenConfig, ImageGenerationProvider, ImageProviderRouter, ImageQuality, ImageStyle,
    LocalDiffusionProvider, ReplicateVideoProvider, RunwayProvider, SoraProvider,
    StabilityEditProvider, StableDiffusionProvider, VideoFormat, VideoGenConfig,
    VideoGenerationProvider, VideoJob, VideoJobStatus, VideoProviderRouter, VideoResolution,
};

// =============================================================================
// DISTILLATION PIPELINE (trajectory collection, scoring, dataset building)
// =============================================================================

#[cfg(feature = "distillation")]
pub mod distillation;

#[cfg(feature = "distillation")]
pub use distillation::{
    CompositeScorer as DistillationCompositeScorer, CycleStatus, DataFlywheel, DatasetBuilder,
    DatasetConfig, DatasetEntry, DatasetFormat, DatasetMessage, DiversityScorer,
    EfficiencyScorer, FlatteningStrategy, FlywheelConfig, FlywheelCycle, FlywheelTrigger,
    InMemoryTrajectoryStore, JsonlTrajectoryStore, LogTrigger, OutcomeScorer, RequiredOutcome,
    StepType as DistillationStepType, Trajectory, TrajectoryCollector, TrajectoryDataset,
    TrajectoryFilter, TrajectoryId, TrajectoryOutcome, TrajectoryScorer, TrajectoryStep,
    TrajectoryStore, WebhookTrigger,
};

// =============================================================================
// AGENT DEFINITION (declarative agent configuration from JSON/TOML)
// =============================================================================

pub mod agent_definition;

pub use agent_definition::{
    AgentDefinition, AgentDefinitionLoader, AgentSpec, GuardrailSpec, MemorySpec, ToolRef,
    ValidationWarning, WarningSeverity,
};

#[cfg(feature = "autonomous")]
pub mod agent_wiring;

#[cfg(feature = "autonomous")]
pub use agent_wiring::{
    agent_from_definition, chat_to_loop_message, create_agent_from_definition,
    create_agent_from_definition_with_options, filter_tool_registry, loop_message_to_pair,
    make_response_generator, make_response_generator_factory, parse_agent_role,
    role_system_prompt, score_agent_for_task, AgentCreateOptions, AgentCreationError,
    AgentPool, IterationHook, PoolAgentStatus, PoolTask, PoolTaskResult, ResponseGenerator,
    ResponseGeneratorFactory, SupervisorConfig, TriggerReason,
};

#[cfg(all(feature = "autonomous", feature = "devtools"))]
pub use agent_wiring::{plan_next_actions, AgentPlanningState};

// =============================================================================
// CONSTRAINED DECODING (grammar-guided generation, GBNF, JSON Schema)
// =============================================================================

#[cfg(feature = "constrained-decoding")]
pub mod constrained_decoding;

#[cfg(feature = "constrained-decoding")]
pub use constrained_decoding::{
    Grammar, GrammarAlternative, GrammarBuilder, GrammarConstraint, GrammarElement, GrammarRule,
    ProviderGrammarFormat, RepeatKind, SchemaToGrammar, StreamingValidationConfig,
    StreamingValidator, ValidationState as GrammarValidationState,
};

// =============================================================================
// HUMAN-IN-THE-LOOP (tool approval gates, confidence escalation, corrections)
// =============================================================================

#[cfg(feature = "hitl")]
pub mod hitl;

#[cfg(feature = "hitl")]
pub use hitl::{
    ApprovalDecision, ApprovalLog, ApprovalLogEntry, ApprovalPolicy, ApprovalRequest,
    AutoApproveGate, AutoDenyGate, CallbackApprovalGate, ConfidenceEstimator, ConfidenceSignal,
    Correction, CorrectionHistory, CorrectionType, EscalationAction, EscalationEvaluator,
    EscalationPolicy, EscalationThreshold, EscalationTrigger, HitlApprovalGate, ImpactLevel,
    MinimumEstimator, PolicyAction, PolicyCondition, PolicyEngine, PolicyLoader, PolicyRule,
    WeightedAverageEstimator,
};

// =============================================================================
// AGENT DEVTOOLS (debugging, profiling, execution replay)
// =============================================================================

#[cfg(feature = "devtools")]
pub mod agent_devtools;

#[cfg(feature = "devtools")]
pub use agent_devtools::{
    AgentDebugger, Breakpoint, DebugEvent, DebugEventType, DevToolsConfig, ExecutionRecorder,
    ExecutionReplay, PerformanceProfiler, ProfileSummary, StateDiff, StateInspector, StateSnapshot,
    StepProfile,
};

// =============================================================================
// COMPILE-TIME FEATURE FLAG VALIDATION
// =============================================================================

validate_features!();
