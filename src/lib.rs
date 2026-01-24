//! AI Assistant library for local LLM integration
//!
//! This crate provides a unified interface to connect to various local LLM providers
//! like Ollama, LM Studio, text-generation-webui, Kobold.cpp, LocalAI, and any
//! OpenAI-compatible API.
//!
//! # Core Features
//!
//! - **Multi-provider support**: Automatically discovers and connects to available providers
//! - **Streaming responses**: Real-time streaming with backpressure via [`StreamBuffer`]
//! - **Session management**: Save and load conversation sessions
//! - **Context management**: Automatic context usage tracking and summarization
//! - **Preference extraction**: Learn user preferences from conversations
//!
//! # Analysis & Quality
//!
//! - **Sentiment analysis**: Detect emotional tone with [`SentimentAnalyzer`]
//! - **Topic detection**: Identify conversation topics with [`TopicDetector`]
//! - **Quality analysis**: Evaluate response quality with [`QualityAnalyzer`]
//! - **Entity extraction**: Extract emails, URLs, phones with [`EntityExtractor`]
//! - **Fact tracking**: Extract and reinforce facts with [`FactExtractor`] and [`FactStore`]
//!
//! # Security & Safety
//!
//! - **Rate limiting**: Local and distributed rate limiting with [`RateLimiter`] and [`DistributedRateLimiter`]
//! - **Input sanitization**: Remove prompt injection with [`InputSanitizer`]
//! - **Audit logging**: Complete audit trail with [`AuditLogger`]
//! - **Hook system**: Pre/post processing with [`HookManager`]
//!
//! # Performance & Reliability
//!
//! - **Connection pooling**: Reuse HTTP connections with [`ConnectionPool`]
//! - **Cache compression**: Compress cached data with gzip/deflate via [`CompressedCache`]
//! - **Latency metrics**: Track per-provider latency with percentiles via [`LatencyTracker`]
//! - **Health checks**: Monitor provider availability with [`HealthChecker`]
//! - **Automatic fallback**: Failover between providers with [`FallbackChain`]
//! - **Batch processing**: Process multiple requests in parallel with [`BatchProcessor`]
//!
//! # Extensibility
//!
//! - **Plugin system**: Extend functionality with [`Plugin`] trait and [`PluginManager`]
//! - **Tool/Function calling**: Define and execute tools with [`ToolUseRegistry`]
//! - **OpenAPI export**: Generate API specs with [`generate_ai_assistant_spec`]
//!
//! # Persistence & Export
//!
//! - **Backup manager**: Automatic backups with [`BackupManager`]
//! - **Database compaction**: Optimize storage with [`DatabaseCompactor`]
//! - **Multi-format export**: JSON, Markdown, CSV, HTML with [`ConversationExporter`]
//!
//! # Internationalization
//!
//! - **Language detection**: 10+ languages with [`LanguageDetector`]
//! - **Localized strings**: Built-in translations with [`LocalizedStrings`]
//! - **Multilingual prompts**: Language-specific prompts with [`MultilingualPromptBuilder`]
//!
//! # Provider Plugins
//!
//! - **Ollama**: [`OllamaProvider`]
//! - **LM Studio**: [`LmStudioProvider`]
//! - **text-generation-webui**: [`TextGenWebUIProvider`]
//! - **Kobold.cpp**: [`KoboldCppProvider`]
//! - **OpenAI-compatible**: [`OpenAICompatibleProvider`]
//!
//! # Development & Debugging
//!
//! - **Tool calling**: Define custom tools with [`ToolRegistry`]
//! - **Benchmarking**: Performance testing with [`run_all_benchmarks`]
//! - **Metrics tracking**: Quality metrics with [`MetricsTracker`]
//! - **Debug mode**: Verbose logging with [`DebugLogger`] and [`set_debug_level`]
//! - **Telemetry**: Optional metrics collection with [`TelemetryCollector`]
//!
//! # Vision & Multimodal
//!
//! - **Image input**: Send images for analysis with [`ImageInput`] and [`VisionMessage`]
//! - **Image preprocessing**: Resize and optimize images with [`ImagePreprocessor`]
//! - **Capability detection**: Check model vision support with [`VisionCapabilities`]
//!
//! # Advanced LLM Features
//!
//! - **Request coalescing**: Combine duplicate requests with [`RequestCoalescer`]
//! - **Prefetching**: Predictive response caching with [`Prefetcher`]
//! - **Model warmup**: Reduce cold start latency with [`WarmupManager`]
//! - **Prompt chaining**: Multi-step workflows with [`ChainExecutor`]
//! - **Self-consistency**: Multi-sampling consensus with [`ConsistencyChecker`]
//! - **CoT parsing**: Extract reasoning steps with [`CotParser`]
//! - **Output validation**: Validate format and schema with [`OutputValidator`]
//!
//! # RAG Enhancements
//!
//! - **Query expansion**: Expand queries for better retrieval with [`QueryExpander`]
//! - **Citation generation**: Add references to responses with [`CitationGenerator`]
//!
//! # Security & Privacy
//!
//! - **PII detection**: Detect and redact personal info with [`PiiDetector`]
//! - **Content moderation**: Filter harmful content with [`ContentModerator`]
//! - **Injection detection**: Detect prompt injection with [`InjectionDetector`]
//!
//! # Response Quality
//!
//! - **Hallucination detection**: Detect factual errors with [`HallucinationDetector`]
//! - **Confidence scoring**: Score response reliability with [`ConfidenceScorer`]
//!
//! # Multi-Model
//!
//! - **Model ensemble**: Combine multiple models with [`Ensemble`]
//! - **Auto model selection**: Route tasks to best model with [`AutoModelSelector`]
//!
//! # Monitoring & Analytics
//!
//! - **Prometheus metrics**: Export metrics with [`AiMetricsRegistry`]
//! - **Conversation analytics**: Track usage patterns with [`ConversationAnalytics`]
//!
//! # Optional Cargo Features
//!
//! - `egui-widgets`: Pre-built egui widgets for chat UI integration
//! - `rag`: SQLite FTS5-based knowledge base with hybrid search
//!
//! # Example
//!
//! ```no_run
//! use ai_assistant::{AiAssistant, AiConfig};
//!
//! let mut assistant = AiAssistant::new();
//!
//! // Fetch available models from all providers
//! assistant.fetch_models();
//!
//! // Wait for models to load (in a real app, poll in update loop)
//! while assistant.is_fetching_models {
//!     std::thread::sleep(std::time::Duration::from_millis(100));
//!     assistant.poll_models();
//! }
//!
//! // Send a message
//! assistant.send_message("Hello!".to_string(), "");
//!
//! // Poll for response (in a real app, do this in update loop)
//! loop {
//!     if let Some(response) = assistant.poll_response() {
//!         match response {
//!             ai_assistant::AiResponse::Chunk(text) => print!("{}", text),
//!             ai_assistant::AiResponse::Complete(text) => {
//!                 println!("\nDone: {}", text);
//!                 break;
//!             }
//!             ai_assistant::AiResponse::Error(e) => {
//!                 eprintln!("Error: {}", e);
//!                 break;
//!             }
//!             _ => {}
//!         }
//!     }
//!     std::thread::sleep(std::time::Duration::from_millis(10));
//! }
//! ```

mod config;
mod messages;
mod models;
mod providers;
mod session;
mod context;
mod assistant;

// New modules for improved functionality
pub mod data_source_client;
pub mod error;
pub mod progress;
pub mod config_file;
pub mod memory_management;
pub mod wasm;
pub mod async_support;

// Metrics module
pub mod metrics;

// Security module (rate limiting, sanitization, audit logging, hooks)
pub mod security;

// Analysis module (sentiment, topics, auto-summarization)
pub mod analysis;

// Conversation control (cancellation, regeneration, branching)
pub mod conversation_control;

// Advanced RAG features (intelligent chunking, deduplication, re-ranking)
pub mod rag_advanced;

// Persistence utilities (backup, compaction, migration)
pub mod persistence;

// Tool calling and provider plugins
pub mod tools;

// Local embeddings for semantic search
pub mod embeddings;

// Entity extraction and fact tracking
pub mod entities;

// Real provider plugin implementations
pub mod provider_plugins;

// Response quality analysis
pub mod quality;

// Internationalization and multi-language support
pub mod i18n;

// Advanced export/import functionality
pub mod export;

// Benchmarking framework
pub mod benchmark;

// Retry with exponential backoff
pub mod retry;

// Model profiles (creative, precise, balanced)
pub mod profiles;

// Prompt templates with variables
pub mod templates;

// Response formatting and parsing
pub mod formatting;

// Conversation search
pub mod search;

// Streaming metrics (tokens/second)
pub mod streaming_metrics;

// Function calling for OpenAI-compatible APIs
pub mod function_calling;

// Vision support for multimodal models
pub mod vision;

// Conversation memory (long-term storage)
pub mod memory;

// Model routing (intelligent model selection)
pub mod routing;

// Cost estimation for API usage
pub mod cost;

// Response caching with TTL
pub mod caching;

// Structured output with JSON schema
pub mod structured;

// Agent framework for multi-step tasks
pub mod agent;

// Embedding cache
pub mod embedding_cache;

// Diff viewer for comparing responses
pub mod diff;

// Vector database integration
pub mod vector_db;

// New performance and reliability modules
pub mod streaming;
pub mod connection_pool;
pub mod cache_compression;
pub mod latency_metrics;
pub mod plugins;
pub mod batch;
pub mod fallback;
pub mod health_check;
pub mod tool_use;
pub mod distributed_rate_limit;
pub mod telemetry;
pub mod debug;
pub mod openapi_export;

// Crawl policy (robots.txt, rate limiting, sitemap discovery)
pub mod crawl_policy;

// RAG module (optional feature)
#[cfg(feature = "rag")]
pub mod rag;

// Re-export main types
pub use config::{AiConfig, AiProvider};
pub use messages::{ChatMessage, AiResponse};
pub use models::ModelInfo;
pub use session::{ChatSession, ChatSessionStore, UserPreferences, ResponseStyle};
pub use context::{ContextUsage, estimate_tokens, get_model_context_size};
pub use assistant::{AiAssistant, SummaryResult};
pub use providers::{
    generate_response, generate_response_streaming,
    generate_response_streaming_cancellable,
};

// RAG exports (optional feature)
#[cfg(feature = "rag")]
pub use rag::{
    RagDb, RagConfig, KnowledgeChunk, StoredMessage, User, DEFAULT_USER_ID,
    build_knowledge_context, build_conversation_context,
    KnowledgeExport, ExportedChunk, ExportedSource,
    HybridRagConfig, HybridKnowledgeResult,
    KnowledgeUsage, KnowledgeSourceUsage,
};

// RAG assistant extensions (optional feature)
#[cfg(feature = "rag")]
pub use assistant::{IndexingResult, IndexingProgress, DocumentInfo, DocumentStats};

// Metrics exports
pub use metrics::{
    MetricsTracker, MessageMetrics, SessionMetrics, RagQualityMetrics,
    MessageMetricsBuilder, SearchCache, ConversationTestCase, TestCaseResult,
    TestSuite, TestSuiteResults, MetricsExport,
};

// Security exports
pub use security::{
    RateLimitConfig, RateLimiter, RateLimitResult, RateLimitReason, RateLimitUsage, RateLimitStatus,
    SanitizationConfig, InputSanitizer, SanitizationResult, SanitizationWarning,
    AuditConfig, AuditLogger, AuditEvent, AuditEventType, AuditStats,
    HookManager, HookResult, HookChainResult,
};

// Analysis exports
pub use analysis::{
    Sentiment, SentimentAnalysis, SentimentAnalyzer, SentimentTrend,
    ConversationSentimentAnalysis, Topic, TopicDetector,
    SummaryConfig, SessionSummary, SessionSummarizer,
};

// Conversation control exports
pub use conversation_control::{
    CancellationToken, MessageOperations, EditResult, RegenerateResult,
    BranchPoint, BranchManager, VariantManager, ResponseVariant,
};

// RAG advanced exports
pub use rag_advanced::{
    ChunkingStrategy, ChunkingConfig, SmartChunk, SmartChunker,
    DeduplicationResult, ChunkDeduplicator,
    RerankConfig, RankedChunk, ChunkReranker, ChunkMetadata,
};

// Persistence exports
pub use persistence::{
    BackupConfig, BackupInfo, BackupManager,
    CompactionConfig, CompactionResult, DatabaseCompactor, DatabaseStats,
    MigrationConfig, MigrationResult, SessionMigrator, FullExport,
    PersistentCacheConfig, CacheStats, CacheEntryInfo,
};

// Persistent cache (requires rag feature for SQLite)
#[cfg(feature = "rag")]
pub use persistence::{PersistentCache, CacheEntry};

// Tool calling exports
pub use tools::{
    ParameterType, ToolParameter, ToolDefinition, ToolCall, ToolResult,
    ToolRegistry, ProviderCapabilities, ProviderPlugin, ProviderRegistry,
    create_builtin_tools,
};

// Embeddings exports
pub use embeddings::{
    EmbeddingConfig, LocalEmbedder, IndexedDocument, SemanticIndex, SearchResult,
    HybridSearchConfig, HybridSearcher, HybridSearchResult,
};

// Entity extraction exports
pub use entities::{
    EntityType, Entity, EntityExtractorConfig, CustomPattern, EntityExtractor,
    Fact, FactType, FactExtractorConfig, FactExtractor, FactStore,
};

// Provider plugin exports
pub use provider_plugins::{
    OllamaProvider, LmStudioProvider, TextGenWebUIProvider, KoboldCppProvider,
    OpenAICompatibleProvider, DiscoveryConfig, discover_providers,
    create_registry_with_discovery,
};

// Quality analysis exports
pub use quality::{
    QualityScore, QualityIssue, QualityIssueType, QualityConfig, QualityAnalyzer,
    ResponseComparison, compare_responses,
};

// i18n exports
pub use i18n::{
    DetectedLanguage, LanguageDetector, LanguagePreferences,
    LocalizedStrings, MultilingualPromptBuilder,
};

// Export/import exports
pub use export::{
    ExportFormat, ExportOptions, ExportedConversation, ExportedMessage,
    ConversationExporter, ImportOptions, ImportResult, ConversationImporter,
};

// Benchmark exports
pub use benchmark::{
    BenchmarkConfig, BenchmarkStats, BenchmarkResult, BenchmarkSuite, BenchmarkRunner,
    run_all_benchmarks, compare_results, BenchmarkComparison,
};

// Retry exports
pub use retry::{
    RetryConfig, RetryExecutor, RetryResult, RetryAttempt, RetryableError,
    CircuitBreaker, CircuitState, ResilientExecutor,
    retry, retry_with_config,
};

// Profile exports
pub use profiles::{
    ModelProfile, ProfileBuilder, ProfileManager, ProfileApplicator,
};

// Template exports
pub use templates::{
    PromptTemplate, TemplateVariable, VariableType, TemplateBuilder,
    TemplateManager, BuiltinTemplates,
};

// Formatting exports
pub use formatting::{
    ParsedResponse, CodeBlock, ParsedList, ListItem, ParsedTable, TableAlignment,
    ParsedLink, Heading, ResponseParser, ParserConfig,
    extract_first_code, extract_code_by_language, extract_first_json,
    parse_as_json, to_plain_text,
};

// Search exports
pub use search::{
    SearchResult as ConversationSearchResult, HighlightSpan, SearchContext, SearchQuery, SearchMode,
    ConversationSearcher, SearchStats, SearchQueryBuilder,
};

// Streaming metrics exports
pub use streaming_metrics::{
    StreamingMetrics, MetricsConfig, StreamingSnapshot, FinalMetrics,
    AggregatedMetrics, MetricsDisplay,
};

// Function calling exports
pub use function_calling::{
    FunctionDefinition, FunctionParameters, ParameterProperty, FunctionCall,
    FunctionResult, FunctionBuilder, FunctionRegistry, ToolChoice,
    parse_function_calls,
};

// Vision exports
pub use vision::{
    ImageInput, ImageData, ImageDetail, VisionMessage, VisionCapabilities,
    ImagePreprocessor, ImageBatch,
};

// Memory exports
pub use memory::{
    MemoryEntry, MemoryType, MemoryConfig, MemoryStore, WorkingMemory,
    MemoryManager, MemoryStats,
};

// Routing exports
pub use routing::{
    TaskType, ModelRequirements, ModelCapabilityProfile,
    ModelRouter, RoutingDecision,
};

// Cost exports
pub use cost::{
    ModelPricing, CostEstimate, CostTracker, CostEstimator,
    BudgetManager, BudgetStatus,
};

// Caching exports
pub use caching::{
    CachedResponse, CacheConfig, CacheKey, ResponseCache,
    CacheStats as ResponseCacheStats, SemanticCache,
};

// Structured output exports
pub use structured::{
    SchemaType, SchemaProperty, JsonSchema, ValidationError, ValidationResult,
    SchemaValidator, StructuredOutputGenerator, StructuredParseResult,
    SchemaBuilder, StructuredRequest,
};

// Agent exports
pub use agent::{
    AgentState, AgentStep, AgentTool, AgentConfig, AgentContext, AgentResult,
    ReactAgent, PlanningAgent, PlanStep, PlanStepStatus,
    AgentExecutor, AgentCallback, LoggingCallback, create_builtin_agent_tools,
};

// Embedding cache exports
pub use embedding_cache::{
    EmbeddingCacheConfig, EmbeddingCacheStats, EmbeddingCache,
    SharedEmbeddingCache, BatchEmbeddingCache, SimilarityEmbeddingCache,
    cosine_similarity, normalize_vector,
};

// Diff exports
pub use diff::{
    ChangeType, DiffLine, DiffHunk, DiffResult, diff,
    ResponseComparison as DiffResponseComparison, WordStats, compare_responses as diff_compare_responses,
    InlineWordDiff, InlineDiffSegment, inline_word_diff,
};

// Error handling exports
pub use error::{
    AiError, AiResult, ConfigError, ProviderError, RagError,
    NetworkError, ValidationError as AiValidationError, ResourceLimitError,
    IoError as AiIoError, SerializationError,
};

// Progress reporting exports
pub use progress::{
    Progress, ProgressCallback, ProgressReporter,
    MultiProgressTracker, OperationHandle, ProgressAggregator,
    ProgressCallbackBuilder, logging_callback, silent_callback,
};

// Config file exports
pub use config_file::{
    ConfigFile, ConfigFormat, ProviderConfig, UrlConfig,
    GenerationConfig, RagFileConfig, HybridConfig, SecurityConfig,
    CacheConfig as FileCacheConfig, LoggingConfig,
    load_config, save_config, default_config_path,
};

// Memory management exports
pub use memory_management::{
    BoundedCache, BoundedVec, EvictionPolicy, CacheStats as BoundedCacheStats,
    MemoryTracker, ComponentMemory, MemoryPressure, MemoryReport,
    format_bytes, MemoryEstimate,
};

// WASM compatibility exports
pub use wasm::{
    is_wasm, PlatformCapabilities, Capability,
};

// Vector database exports
pub use vector_db::{
    VectorDbConfig, DistanceMetric, StoredVector, VectorSearchResult,
    MetadataFilter, FilterOperation, VectorDb, InMemoryVectorDb,
    QdrantClient, VectorDbBuilder, VectorDbBackend, HybridVectorSearch,
};

// Async support exports
pub use async_support::{
    AsyncResult, BoxFuture, AsyncLoad, AsyncSave, AsyncIterator,
    BlockingHandle, spawn_blocking, block_on,
    YieldNow, yield_now, Sleep, sleep, Timeout, timeout,
    AsyncSender, AsyncReceiver, async_channel,
    AsyncStream, stream, AsyncMutex, AsyncMutexGuard,
    join, join3, select,
    AsyncRetryConfig, retry_async,
};

// Streaming exports
pub use streaming::{
    StreamingConfig, StreamBuffer, StreamProducer, StreamConsumer,
    StreamError, StreamMetrics, BackpressureStream, Chunker, RateLimitedStream,
};

// Connection pool exports
pub use connection_pool::{
    ConnectionPool, PoolConfig, PooledConnection, PooledConnectionGuard, PoolStats,
    global_pool, init_global_pool,
};

// Cache compression exports
pub use cache_compression::{
    CompressionAlgorithm, CompressionLevel, CompressedData, CompressionError,
    compress, decompress, CompressedCache, CacheCompressionStats,
    compress_string, decompress_string, StreamingCompressor,
};

// Latency metrics exports
pub use latency_metrics::{
    LatencyTracker, LatencyRecord, LatencyStats, RequestTimer, LatencyTrend,
};

// Plugin system exports
pub use plugins::{
    Plugin, PluginCapability, PluginContext, PluginInfo, PluginManager,
    MessageProcessorPlugin, LoggingPlugin,
};

// Batch processing exports
pub use batch::{
    BatchProcessor, BatchConfig, BatchRequest, BatchResult,
    BatchBuilder, BatchStats, BatchResults,
};

// Fallback exports
pub use fallback::{
    FallbackChain, FallbackProvider, FallbackResult, FallbackError,
    ProviderState as FallbackProviderState, ProviderStatus,
};

// Health check exports
pub use health_check::{
    HealthChecker, HealthCheckConfig, HealthCheckResult, HealthStatus,
    ProviderHealth, HealthSummary, HealthCheckType,
};

// Tool use exports
pub use tool_use::{
    Tool as ToolUseTool, ToolParameter as ToolUseParameter, ToolRegistry as ToolUseRegistry,
    ToolCall as ToolUseCall, ToolResult as ToolUseResult, ToolError, ParameterType as ToolParameterType,
    parse_tool_calls,
};

// Distributed rate limiting exports
pub use distributed_rate_limit::{
    DistributedRateLimiter, RateLimitBackend, InMemoryBackend,
    RateLimitState, DistributedRateLimitResult,
};

// Telemetry exports
pub use telemetry::{
    TelemetryConfig, TelemetryEvent, TelemetryCollector, AggregatedMetrics as TelemetryMetrics,
    TimedOperation,
};

// Debug exports
pub use debug::{
    DebugLevel, DebugConfig, DebugLogger, DebugEntry, DebugReport,
    RequestInspector, CapturedRequest,
    global_debug, configure_global_debug, set_debug_level,
};

// OpenAPI exports
pub use openapi_export::{
    OpenApiSpec, OpenApiInfo, OpenApiServer, OpenApiPathItem, OpenApiOperation,
    OpenApiParameter, OpenApiRequestBody, OpenApiResponse, OpenApiComponents,
    JsonSchema as OpenApiJsonSchema, OpenApiBuilder, OperationBuilder,
    generate_ai_assistant_spec, export_to_json, export_to_yaml,
};

// New advanced modules (2024)
pub mod request_coalescing;
pub mod prefetch;
pub mod model_warmup;
pub mod prompt_chaining;
pub mod self_consistency;
pub mod cot_parsing;
pub mod output_validation;
pub mod query_expansion;
pub mod citations;
pub mod pii_detection;
pub mod content_moderation;
pub mod injection_detection;
pub mod hallucination_detection;
pub mod confidence_scoring;
pub mod model_ensemble;
pub mod auto_model_selection;
pub mod prometheus_metrics;
pub mod conversation_analytics;

// Request coalescing exports
pub use request_coalescing::{
    CoalescingConfig, RequestCoalescer, CoalescingKey, CoalescedResult,
    CoalescingHandle, SemanticCoalescer, CoalescableRequest, CoalescingStats,
};

// Prefetch exports
pub use prefetch::{
    PrefetchConfig, Prefetcher, PrefetchCandidate, PrefetchedResponse, QueryPattern,
    ContextPredictor, PrefetchStats,
};

// Model warmup exports
pub use model_warmup::{
    WarmupConfig, WarmupManager, WarmupStatus, ModelUsageStats, WarmupStats,
    ScheduledWarmup, WarmupTime,
};

// Prompt chaining exports
pub use prompt_chaining::{
    ChainConfig, PromptChain, ChainStep, StepCondition, ChainExecutor,
    ChainResult, StepResult, VariableExtraction, ExtractionMethod, ChainTemplates,
    ChainBuilder,
};

// Self-consistency exports
pub use self_consistency::{
    ConsistencyConfig, ConsistencyChecker, ConsistencyResult, AnswerGroup,
    VotingConsistency, VotingResult, Sample, ConsistencyAggregator,
};

// Chain-of-thought parsing exports
pub use cot_parsing::{
    CotConfig, CotParser, CotParseResult, ReasoningStep, StepType,
    CotValidator, ValidationResult as CotValidationResult,
};

// Output validation exports
pub use output_validation::{
    ValidationConfig, OutputValidator, OutputFormat, ValidationResult as OutputValidationResult,
    SchemaValidator as OutputSchemaValidator, ValidationIssue, IssueSeverity, IssueType,
};

// Query expansion exports
pub use query_expansion::{
    ExpansionConfig, QueryExpander, ExpandedQuery, ExpansionSource,
    MultiQueryRetriever, ExpansionResult, ExpansionStats, ScoredResult,
};

// Citation exports
pub use citations::{
    CitationConfig, CitationGenerator, CitationStyle, Citation, Source, SourceType,
    CitedText, CitationVerifier, VerificationResult as CitationVerificationResult,
    UnverifiedCitation,
};

// PII detection exports
pub use pii_detection::{
    PiiConfig, PiiDetector, PiiType, DetectedPii, RedactionStrategy,
    SensitivityLevel, PiiResult, CustomPiiPattern,
};

// Content moderation exports
pub use content_moderation::{
    ModerationConfig, ContentModerator, ModerationCategory, ModerationResult,
    ModerationFlag, ModerationAction, ModerationStats,
};

// Injection detection exports
pub use injection_detection::{
    InjectionConfig, InjectionDetector, InjectionType, InjectionDetection,
    RiskLevel, InjectionResult, Recommendation, DetectionSensitivity,
    CustomPattern as InjectionCustomPattern,
};

// Hallucination detection exports
pub use hallucination_detection::{
    HallucinationConfig, HallucinationDetector, HallucinationType, HallucinationDetection,
    HallucinationResult, Claim, ClaimType,
};

// Confidence scoring exports
pub use confidence_scoring::{
    ConfidenceConfig, ConfidenceScorer, ConfidenceScore, Reliability,
    ConfidenceBreakdown, CertaintyIndicator, UncertaintyIndicator, UncertaintyType,
    CalibrationStats,
};

// Model ensemble exports
pub use model_ensemble::{
    EnsembleConfig, Ensemble, EnsembleStrategy, EnsembleModel, ModelResponse,
    EnsembleResult, EnsembleBuilder,
};

// Automatic model selection exports
pub use auto_model_selection::{
    AutoSelectConfig, AutoModelSelector, ModelProfile as AutoModelProfile,
    ModelCapabilities as AutoModelCapabilities, TaskType as AutoTaskType,
    SelectionResult, Requirements as ModelRequirementsAuto, ModelStats,
};

// Prometheus metrics exports
pub use prometheus_metrics::{
    Counter, Gauge, Histogram, HistogramTimer, AiMetricsRegistry,
};

// Conversation analytics exports
pub use conversation_analytics::{
    AnalyticsConfig, ConversationAnalytics, AnalyticsEvent, EventType, EventValue,
    AggregatedStats, PatternTracker, AnalyticsReport, ExportedEvent,
};

// Additional advanced modules (2025)
pub mod quantization;
pub mod priority_queue;
pub mod streaming_compression;
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

// Quantization exports
pub use quantization::{
    QuantFormat, QuantizationDetector, MemoryRequirements, HardwareProfile,
    QuantRecommendation, ModelSize, GgufMetadata, FormatComparison,
};

// Priority queue exports
pub use priority_queue::{
    Priority, PriorityRequest, PriorityQueue, QueueConfig, QueueStats,
    QueueError, SharedPriorityQueue, WorkerQueue,
};

// Streaming compression exports
pub use streaming_compression::{
    Algorithm as StreamCompressionAlgorithm, Level as StreamCompressionLevel,
    CompressionConfig as StreamCompressionConfig, CompressionResult as StreamCompressionResult,
    StreamCompressor, CompressionStats as StreamCompressionStats, StreamDecompressor,
    compress_string as stream_compress_string, decompress_string as stream_decompress_string,
};

// Keepalive exports
pub use keepalive::{
    ConnectionState, KeepaliveConfig, ConnectionInfo, HeartbeatResult,
    KeepaliveEvent, KeepaliveManager, KeepaliveHandle, KeepaliveStats,
    ConnectionMonitor,
};

// Few-shot exports
pub use few_shot::{
    ExampleCategory, Example, SelectionConfig, FewShotManager, FewShotStats,
    ExampleSets, ExampleBuilder,
};

// Prompt optimizer exports
pub use prompt_optimizer::{
    PromptVariant, OptimizerConfig, PromptOptimizer, Feedback as PromptFeedback,
    OptimizationStats, VariantReport, PromptShortener,
};

// Token budget exports
pub use token_budget::{
    BudgetPeriod, Budget, BudgetUsage, BudgetCheckResult, BudgetAlert,
    BudgetManager as TokenBudgetManager, BudgetStats as TokenBudgetStats,
    TokenEstimator, RequestPlanner, PlannedRequest,
};

// Regeneration exports
pub use regeneration::{
    RegenerationFeedback, RegenerationIssue, ResponseStyle as RegenResponseStyle,
    LengthPreference, RegenerationRequest, RegenerationManager,
};

// Summarization exports
pub use summarization::{
    SummaryConfig as ConvSummaryConfig, SummaryStyle as ConvSummaryStyle,
    ConversationSummary, ConversationSummarizer,
};

// Intent exports
pub use intent::{Intent, IntentResult, IntentClassifier};

// User rate limit exports
pub use user_rate_limit::{
    UserRateLimitConfig, RateLimitCheckResult as UserRateLimitResult, UserRateLimiter,
};

// API key rotation exports
pub use api_key_rotation::{
    KeyStatus, ApiKey, RotationConfig, ApiKeyManager, KeyStats,
};

// Forecasting exports
pub use forecasting::{
    UsageDataPoint, UsageForecast, Trend, UsageForecaster, CapacityEstimate,
};

// Request signing exports
pub use request_signing::{
    SignatureAlgorithm, SignedRequest, RequestSigner, SignatureError,
};

// Webhook exports
pub use webhooks::{
    WebhookEvent, WebhookConfig, WebhookPayload, DeliveryResult,
    WebhookManager, WebhookStats,
};

// Message queue exports
pub use message_queue::{
    QueueMessage, MemoryQueue, QueueError as MessageQueueError,
    DeadLetterQueue,
};

// Typing indicator exports
pub use typing_indicator::{
    TypingState, TypingIndicator, AnimatedIndicator, ProgressIndicator,
};

// Smart suggestions exports
pub use smart_suggestions::{
    SuggestionType, Suggestion, SuggestionGenerator, SuggestionConfig,
};

// Conversation templates exports
pub use conversation_templates::{
    TemplateCategory, ConversationTemplate, TemplateVariable as ConvTemplateVariable,
    TemplateLibrary,
};

// Context and memory management (2025 additions)
pub mod context_window;
pub mod conversation_compaction;
pub mod memory_pinning;

// Response optimization
pub mod response_ranking;
pub mod answer_extraction;
pub mod fact_verification;

// External API adapters
pub mod openai_adapter;
pub mod anthropic_adapter;
pub mod huggingface_connector;

// Analytics and flow
pub mod conversation_flow;
pub mod user_engagement;
pub mod response_effectiveness;

// Security and privacy
pub mod content_encryption;
pub mod access_control;
pub mod data_anonymization;

// Multi-agent system
pub mod multi_agent;
pub mod task_decomposition;
pub mod agent_memory;

// Persistence and sync
pub mod conversation_snapshot;
pub mod incremental_sync;
pub mod conflict_resolution;

// Web search
pub mod web_search;

// Tool calling and agentic systems
pub mod tool_calling;
pub mod agentic_loop;
pub mod model_integration;

// MCP Protocol
pub mod mcp_protocol;

// Advanced streaming
pub mod sse_streaming;
pub mod websocket_streaming;

// Evaluation and benchmarking
pub mod evaluation;

// Fine-tuning and LoRA
pub mod fine_tuning;

// Neural embeddings (advanced)
pub mod neural_embeddings;

// Advanced guardrails (Constitutional AI, bias detection)
pub mod advanced_guardrails;

// Context window exports
pub use context_window::{
    ContextWindow, ContextWindowConfig, ContextMessage, EvictionStrategy as ContextEvictionStrategy,
};

// Conversation compaction exports
pub use conversation_compaction::{
    ConversationCompactor, CompactableMessage,
    CompactionConfig as ConvCompactionConfig, CompactionResult as ConvCompactionResult,
};

// Memory pinning exports
pub use memory_pinning::{
    PinManager, PinnedItem, PinType, AutoPinner,
};

// Response ranking exports
pub use response_ranking::{
    ResponseRanker, ResponseCandidate, RankingCriteria, ScoreBreakdown,
};

// Answer extraction exports
pub use answer_extraction::{
    AnswerExtractor, ExtractedAnswer, AnswerType, ExtractionConfig,
};

// Fact verification exports
pub use fact_verification::{
    FactVerifier, VerifiedFact, VerificationStatus, FactSource, VerificationConfig,
    FactVerifierBuilder,
};

// OpenAI adapter exports
pub use openai_adapter::{
    OpenAIClient, OpenAIConfig, OpenAIRequest, OpenAIResponse, OpenAIMessage, OpenAIModel,
};

// Anthropic adapter exports
pub use anthropic_adapter::{
    AnthropicClient, AnthropicConfig, AnthropicRequest, AnthropicResponse,
    AnthropicMessage, AnthropicModel,
};

// HuggingFace connector exports
pub use huggingface_connector::{
    HfClient, HfConfig, HfRequest, HfResponse, HfTask,
};

// Conversation flow exports
pub use conversation_flow::{
    FlowAnalyzer, ConversationTurn, FlowState, FlowAnalysis, TopicTransition,
};

// User engagement exports
pub use user_engagement::{
    EngagementTracker, EngagementEvent, EngagementMetrics, EngagementManager, UserTrends,
};

// Response effectiveness exports
pub use response_effectiveness::{
    EffectivenessScorer, EffectivenessScore, QAPair, UserFeedback as EffectivenessFeedback,
    ScoringWeights, BatchEvaluator, BatchResult as EffectivenessBatchResult,
};

// Content encryption exports
pub use content_encryption::{
    ContentEncryptor, EncryptionKey, EncryptedContent, EncryptionAlgorithm, EncryptionError,
    EncryptedMessageStore,
};

// Access control exports
pub use access_control::{
    AccessControlManager, AccessControlEntry, Permission, ResourceType, Role,
    AccessCondition, AccessResult,
};

// Data anonymization exports
pub use data_anonymization::{
    DataAnonymizer, AnonymizationRule, AnonymizationStrategy, DataType as AnonymizationDataType,
    AnonymizationResult, Detection, BatchAnonymizer,
};

// Multi-agent exports
pub use multi_agent::{
    AgentOrchestrator, Agent, AgentStatus, AgentRole, AgentMessage, MessageType,
    AgentTask, TaskStatus, OrchestrationStrategy, OrchestrationStatus, OrchestrationError,
};

// Task decomposition exports
pub use task_decomposition::{
    TaskDecomposer, TaskNode, DecompositionStrategy, DecompositionStatus, FlatTask,
    DecompositionAnalysis,
};

// Agent memory exports
pub use agent_memory::{
    SharedMemory, MemoryEntry as AgentMemoryEntry, MemoryType as AgentMemoryType,
    MemoryError, MemoryStats as AgentMemoryStats, ThreadSafeMemory,
};

// Conversation snapshot exports
pub use conversation_snapshot::{
    ConversationSnapshot, SnapshotMetadata, SnapshotMessage, MemoryItem,
    SnapshotManager, SnapshotDiff, MemoryChange,
};

// Incremental sync exports
pub use incremental_sync::{
    IncrementalSyncManager, SyncEntry, SyncOperation, SyncState, SyncDelta, SyncLog,
    SyncError, TwoWaySyncCoordinator,
};

// Conflict resolution exports
pub use conflict_resolution::{
    ConflictResolver, Conflict, ConflictType, ResolutionStrategy, Resolution,
    ConflictError, ThreeWayMerge, MergeResult, MergeConflictLine,
};

// Web search exports
pub use web_search::{
    WebSearchManager, SearchProvider, SearchResult as WebSearchResult, SearchConfig,
    DuckDuckGoProvider, BraveSearchProvider, SearXNGProvider,
};

// Tool calling exports (new system)
pub use tool_calling::{
    Tool as ToolDef, ToolParameter as ToolParam, ParameterType as ParamType,
    ToolCall as ToolInvocation, ToolResult as ToolOutput,
    ToolRegistry as ToolRepo, CommonTools,
};

// Agentic loop exports
pub use agentic_loop::{
    AgenticLoop, AgentConfig as LoopConfig, AgentState as LoopState,
    AgentStatus as LoopStatus, AgentMessage as LoopMessage, AgentRole as LoopRole,
    AgentLoopResult, IterationResult, AgentBuilder,
};

// Model integration exports
pub use model_integration::{
    ChatMessage as IntegratedChatMessage, ChatRole, ChatResponse, FinishReason,
    TokenUsage, ModelError, ModelProvider, OllamaProvider as OllamaIntegrated,
    LMStudioProvider as LMStudioIntegrated, IntegratedModelClient,
    create_ollama_client, create_lm_studio_client,
};

// MCP Protocol exports
pub use mcp_protocol::{
    McpServer, McpClient, McpRequest, McpResponse, McpError, McpTool, McpResource,
    McpResourceContent, McpPrompt, McpPromptMessage, McpContent, McpServerCapabilities,
    MCP_VERSION,
};

// SSE Streaming exports
pub use sse_streaming::{
    SseEvent, SseReader, SseWriter, SseClient, SseConnection, SseError,
    StreamChunk, StreamChoice, StreamDelta, StreamAggregator,
};

// WebSocket streaming exports
pub use websocket_streaming::{
    WsFrame, WsOpcode, WsCloseCode, WsAiMessage, WsUsage, WsState,
    WsStreamHandler, WsCallbacks, WsError, WsHandshake, BidirectionalStream,
};

// Evaluation exports
pub use evaluation::{
    MetricType, MetricResult, EvalSample, EvalResult, Evaluator,
    TextQualityEvaluator, RelevanceEvaluator, SafetyEvaluator,
    BenchmarkResult as EvalBenchmarkResult, Benchmarker, AbTestConfig, AbTestResult, AbTestManager,
    EvalSuite, EvalSummary,
};

// Fine-tuning exports
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

// Neural embeddings exports
pub use neural_embeddings::{
    EmbeddingVec, SparseEmbedding, PoolingStrategy, DenseEmbeddingConfig, DenseEmbedder,
    SparseEmbeddingConfig, SparseEmbedder, HybridRetriever, HybridEmbedding,
    QuantizationType, QuantizedEmbedding, DimensionalityReduction,
    CrossEncoder, RankedDocument, EmbeddingError,
    normalize_l2, cosine_similarity as neural_cosine_similarity,
    euclidean_distance, dot_product as neural_dot_product,
};

// Advanced guardrails exports
pub use advanced_guardrails::{
    ConstitutionalPrinciple, ConstitutionalConfig, ConstitutionalAI,
    ConstitutionalEvaluation, PrincipleViolation, default_principles,
    BiasDimension, BiasConfig, BiasDetector, BiasDetectionResult, BiasOccurrence,
    ToxicityCategory, ToxicityConfig, ToxicityDetector, ToxicityResult,
    AttackType, AttackDetector, AttackDetectionResult, DetectedAttack,
    GuardrailsManager, InputCheckResult, OutputCheckResult, FullSafetyCheck,
};

// Document parsing (EPUB, DOCX, ODT, HTML)
pub mod document_parsing;

// Table extraction (Markdown, ASCII, HTML, TSV)
pub mod table_extraction;

// RSS/Atom feed monitoring
pub mod feed_monitor;

// HTML structured data extraction
pub mod html_extraction;

// Content versioning and change tracking
pub mod content_versioning;

// Translation quality analysis
pub mod translation_analysis;

// Entity enrichment (dedup, auto-tagging, cross-reference)
pub mod entity_enrichment;

// Decision trees for conditional flows
pub mod decision_tree;

// Task planning with steps and sub-steps
pub mod task_planning;

// Automatic RAG indexing (requires rag feature)
#[cfg(feature = "rag")]
pub mod auto_indexing;

// Document parsing exports
pub use document_parsing::{
    DocumentFormat, DocumentSection, DocumentMetadata, ParsedDocument,
    DocumentParserConfig, DocumentParser,
};

// Table extraction exports
pub use table_extraction::{
    TableCell, ExtractedTable, TableSourceFormat,
    TableExtractorConfig, TableExtractor,
};

// Crawl policy exports
pub use crawl_policy::{
    RobotsRule, RobotsDirectives, ParsedRobotsTxt,
    SitemapEntry, ChangeFrequency, ParsedSitemap,
    CrawlPolicyConfig, CrawlPolicy,
};

// Feed monitor exports
pub use feed_monitor::{
    FeedEntry, FeedMetadata, FeedFormat, ParsedFeed,
    FeedMonitorConfig, FeedMonitorState, FeedCheckResult,
    FeedParser, FeedMonitor,
};

// HTML extraction exports
pub use html_extraction::{
    HtmlSelector, HtmlElement, HtmlMetadata,
    HtmlExtractionConfig, HtmlList, HtmlLink, HtmlExtractionResult,
    HtmlExtractor,
};

// Content versioning exports
pub use content_versioning::{
    ContentSnapshot, ContentChange, ChangeType as VersionChangeType,
    VersionDiff, VersioningConfig, VersionHistory, ContentVersionStore,
};

// Translation analysis exports
pub use translation_analysis::{
    GlossaryEntry, Glossary, TranslationIssueType, TranslationIssue,
    AlignedSegment, TranslationAnalysisResult, TranslationStats,
    TranslationAnalysisConfig, ComparisonPrompt, ComparisonResponse,
    TranslationAnalyzer,
};

// Entity enrichment exports
pub use entity_enrichment::{
    EnrichableEntity, EnrichmentData, EnrichedEntity, MergeStrategy,
    DuplicateMatch, DuplicateReason, EnrichmentSource, EnrichmentConfig,
    EntityEnricher,
};

// Decision tree exports
pub use decision_tree::{
    ConditionOperator, Condition, DecisionBranch, DecisionNodeType,
    DecisionNode, DecisionPath, DecisionTree, DecisionTreeBuilder,
};

// Task planning exports
pub use task_planning::{
    StepStatus, StepPriority, PlanStep as TaskPlanStep, StepNote,
    TaskPlan, PlanSummary, PlanBuilder,
};

// Auto-indexing exports (rag feature)
#[cfg(feature = "rag")]
pub use auto_indexing::{
    IndexChunkingStrategy, IndexedDocumentMeta, IndexableChunk,
    ChunkMetadata as IndexChunkMetadata, ChunkPosition, AutoIndexConfig,
    IndexState, AutoIndexer, IndexStats, IndexingResult as AutoIndexingResult,
};

// Optional egui widgets
#[cfg(feature = "egui-widgets")]
pub mod widgets;
