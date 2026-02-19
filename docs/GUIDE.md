# AI Assistant Crate - Complete Feature Guide

This guide covers every feature in the `ai_assistant` crate. Each section explains **what** the concept is, **why** it matters, and **how** it works here with code examples.

---

## Table of Contents

1. [Providers](#1-providers)
2. [Streaming](#2-streaming)
3. [Synchronous Generation](#3-synchronous-generation)
4. [Sessions](#4-sessions)
5. [Context Window](#5-context-window)
6. [Preference Extraction](#6-preference-extraction)
7. [RAG (Retrieval-Augmented Generation)](#7-rag-retrieval-augmented-generation)
8. [Multi-User Support](#8-multi-user-support)
9. [Notes System](#9-notes-system)
10. [Analysis Tools](#10-analysis-tools)
11. [Security Features](#11-security-features)
12. [Performance Features](#12-performance-features)
13. [Decision Trees](#13-decision-trees)
14. [Tool Use / Tool Registry](#14-tool-use--tool-registry)
15. [Function Calling (OpenAI-compatible)](#15-function-calling-openai-compatible)
16. [Agent Framework](#16-agent-framework)
17. [Plugins](#17-plugins)
18. [Persistence & Export](#18-persistence--export)
19. [Vision & Multimodal](#19-vision--multimodal)
20. [Structured Output](#20-structured-output)
21. [Model Profiles](#21-model-profiles)
22. [Prompt Templates](#22-prompt-templates)
23. [Model Routing](#23-model-routing)
24. [Cost Estimation](#24-cost-estimation)
25. [Response Caching](#25-response-caching)
26. [Conversation Memory](#26-conversation-memory)
27. [Embedding Cache](#27-embedding-cache)
28. [Streaming Metrics](#28-streaming-metrics)
29. [Retry & Circuit Breaker](#29-retry--circuit-breaker)
30. [Diff Viewer](#30-diff-viewer)
31. [Advanced LLM Techniques](#31-advanced-llm-techniques)
32. [Security & Privacy (Advanced)](#32-security--privacy-advanced)
33. [Quality & Reliability](#33-quality--reliability)
34. [Internationalization](#34-internationalization)
35. [Benchmarking](#35-benchmarking)
36. [Provider Discovery](#36-provider-discovery)
37. [egui Widgets](#37-egui-widgets)
38. [Monitoring](#38-monitoring)
39. [Adaptive Thinking](#39-adaptive-thinking)
40. [Provider Failover](#40-provider-failover)
41. [Retry with Backoff](#41-retry-with-backoff)
42. [Conversation Compaction](#42-conversation-compaction)
43. [API Key Rotation](#43-api-key-rotation)
44. [Context Size Cache](#44-context-size-cache)
45. [Journal Sessions (JSONL)](#45-journal-sessions-jsonl)
46. [Encrypted Sessions](#46-encrypted-sessions)
47. [Binary Storage](#47-binary-storage)
48. [Log Redaction](#48-log-redaction)
49. [HTTP Client Abstraction](#49-http-client-abstraction)
50. [Vector Database Backends](#50-vector-database-backends)
51. [Distributed Computing](#51-distributed-computing)
52. [Distributed Networking](#52-distributed-networking)
53. [Autonomous Agents](#53-autonomous-agents)
54. [Knowledge Graphs](#54-knowledge-graphs)
55. [Multi-Layer Knowledge Graphs](#55-multi-layer-knowledge-graphs)
56. [Document Parsing](#56-document-parsing)
57. [Feed Monitor](#57-feed-monitor)
58. [Content Versioning](#58-content-versioning)
59. [P2P Networking](#59-p2p-networking)
60. [WebSocket Streaming](#60-websocket-streaming)
61. [MCP Protocol](#61-mcp-protocol)
62. [HTTP API Server](#62-http-api-server)
63. [Event System](#63-event-system)
64. [Content Encryption](#64-content-encryption)
65. [Access Control](#65-access-control)
66. [Request Signing](#66-request-signing)
67. [Request Queue](#67-request-queue)
68. [Webhooks](#68-webhooks)
69. [Model Ensemble](#69-model-ensemble)
70. [Prompt Optimizer](#70-prompt-optimizer)
71. [Quantization](#71-quantization)
72. [WASM Support](#72-wasm-support)
73. [OpenAPI Export](#73-openapi-export)

---

## 1. Providers

**What**: A "provider" is a local LLM server that runs on your machine. It receives text prompts and returns AI-generated responses.

**Common providers**:
- **Ollama** (port 11434) - Popular, downloads models for you
- **LM Studio** (port 1234) - GUI app with built-in server
- **text-generation-webui** (port 5000) - Flexible, many model formats
- **Kobold.cpp** (port 5001) - Optimized for CPU inference
- **LocalAI** (port 8080) - Drop-in OpenAI replacement
- **Custom** - Any OpenAI-compatible endpoint

**How it works here**: The crate auto-discovers which providers are running by checking their default ports.

```rust
use ai_assistant::{AiAssistant, AiConfig, AiProvider};

let mut assistant = AiAssistant::new();
assistant.fetch_models(); // Scans all known ports for active providers

// Or configure a specific provider
let mut config = AiConfig::default();
config.provider = AiProvider::OpenAICompatible {
    base_url: "http://my-server:8000".to_string()
};
assistant.load_config(config);
```

| Provider | API Type | Default URL | Streaming |
|----------|----------|-------------|-----------|
| Ollama | Native | `localhost:11434` | Yes |
| LM Studio | OpenAI-compatible | `localhost:1234` | Yes |
| text-generation-webui | OpenAI-compatible | `localhost:5000` | Yes |
| Kobold.cpp | Native | `localhost:5001` | No* |
| LocalAI | OpenAI-compatible | `localhost:8080` | Yes |

**Key types**: `AiConfig`, `AiProvider`, `ModelInfo`

---

## 2. Streaming

**What**: Instead of waiting for the entire response, streaming delivers it word-by-word as it's generated.

**Why**: The user sees text appearing in real-time instead of a blank screen for seconds.

**How it works here**: Responses arrive as `AiResponse::Chunk(text)` events through a channel. You poll for them in your main loop.

```rust
assistant.send_message("Explain gravity".to_string(), "");

// In your update loop:
if let Some(response) = assistant.poll_response() {
    match response {
        AiResponse::Chunk(text) => print!("{}", text),     // Partial text
        AiResponse::Complete(full) => println!("{}", full), // All done
        AiResponse::Error(e) => eprintln!("{}", e),
        _ => {}
    }
}
```

**Backpressure**: If your UI can't process chunks fast enough, `StreamBuffer` accumulates them and delivers at a controlled rate.

**Cancellation**: Use `CancellationToken` to stop generation mid-response:

```rust
assistant.cancel_generation(); // Stops the current generation
```

---

## 3. Synchronous Generation

**What**: A blocking alternative to streaming when you just need the full response.

**Why**: Simpler for scripts, tests, or background processing where streaming UI isn't needed.

```rust
let response = assistant.generate_sync(
    "Explain ownership in Rust".to_string(),
    ""  // optional knowledge context
)?;
println!("Response: {}", response);
```

---

## 4. Sessions

**What**: A "session" is a saved conversation - messages, metadata, model info, timestamps.

**Why**: Users can close the app and resume conversations later.

```rust
// Save current session
assistant.save_current_session();

// Create new session (saves current first)
assistant.new_session();

// Save/load to file
assistant.save_sessions_to_file(Path::new("ai_sessions.json"))?;
assistant.load_sessions_from_file(Path::new("ai_sessions.json"))?;

// List sessions
for session in assistant.get_sessions() {
    println!("{}: {} ({} messages)", session.id, session.name, session.messages.len());
}

// Load/delete specific sessions
assistant.load_session("session_1234567890");
assistant.delete_session("session_1234567890");
```

**Key types**: `ChatSession`, `ChatMessage`

---

## 5. Context Window

**What**: Every LLM has a maximum number of "tokens" it can process at once. This is the "context window" - typically 2K to 128K tokens.

**Why**: If your conversation exceeds the window, older messages are lost. You need strategies to handle this.

**How it works here**:
- **Token estimation**: Approximates count without needing a tokenizer
- **Context tracking**: Monitors usage percentage
- **Automatic summarization**: When context gets full (~70%), older messages are summarized

```rust
let usage = assistant.calculate_context_usage(&knowledge_context);
println!("Usage: {:.1}% ({}/{} tokens)", usage.usage_percent, usage.total_tokens, usage.max_tokens);
println!("  System: {}", usage.system_tokens);
println!("  Knowledge: {}", usage.knowledge_tokens);
println!("  Conversation: {}", usage.conversation_tokens);

if usage.is_critical {
    println!("Context almost full!");
}

// Trigger summarization when needed
assistant.summarize_old_messages(&knowledge);
assistant.start_background_summarization();
assistant.poll_summarization(); // In update loop
```

**Known model context sizes**:
| Model | Context |
|-------|---------|
| Llama 3.2, 3.1 | 128K |
| Qwen 2.5 (32B+) | 128K |
| Phi-3 | 128K |
| Mistral, Mixtral | 32K |
| Qwen 2.5 (smaller) | 32K |
| DeepSeek | 32K |
| CodeLlama | 16K |
| Gemma 2 | 8K |
| Others | 8K (default) |

**Key types**: `ContextTracker`, `ContextUsage`

---

## 6. Preference Extraction

**What**: Learning what the user likes from their messages, without explicit configuration.

**Example**: "I prefer short answers" -> system remembers and adjusts responses.

```rust
// Built-in preferences:
pub struct UserPreferences {
    pub response_style: ResponseStyle, // Concise, Normal, Detailed, Technical
    pub ships_owned: Vec<String>,
    pub target_ship: Option<String>,
    pub interests: Vec<String>,
}

// Custom extraction logic
assistant.extract_preferences_with(|messages, prefs| {
    for msg in messages {
        if msg.content.contains("be brief") {
            prefs.response_style = ResponseStyle::Concise;
        }
    }
});
```

---

## 7. RAG (Retrieval-Augmented Generation)

**What**: RAG stores your documents in a searchable database and retrieves relevant chunks when the user asks a question. These chunks are injected into the LLM prompt as context.

**Why**: LLMs have fixed training data. RAG lets you give them access to YOUR documents without fine-tuning.

**Feature flag**: Requires `features = ["rag"]`

### Document Registration (Recommended)

```rust
let mut assistant = AiAssistant::new();

// Set database path (lazy initialization)
assistant.set_rag_path(Path::new("./ai_data.db"));

// Register documents - they're indexed automatically on first query
let guide = std::fs::read_to_string("docs/guide.md")?;
assistant.register_knowledge_document("User Guide", &guide);

let faq = std::fs::read_to_string("docs/faq.md")?;
assistant.register_knowledge_document("FAQ", &faq);

// Build context (auto-indexes pending documents, retrieves relevant chunks)
let (knowledge_context, conversation_context) = assistant.build_rag_context("How do I configure X?");

// Use in message
assistant.send_message("How do I configure X?".to_string(), &knowledge_context);
```

### Direct Indexing (Alternative)

```rust
assistant.init_rag(Path::new("./ai_data.db"))?;

let content = std::fs::read_to_string("docs/guide.md")?;
let chunks = assistant.index_knowledge_document("User Guide", &content)?;
println!("Indexed {} chunks", chunks);

let (chunk_count, total_tokens) = assistant.get_knowledge_stats()?;
```

### Document Management

```rust
// List, check pending, unregister, delete
let sources = assistant.get_registered_sources();
if assistant.has_pending_documents() {
    let results = assistant.process_pending_documents();
}
assistant.unregister_knowledge_document("Old Guide");
assistant.delete_knowledge_document("Outdated FAQ")?;
```

### Conversation RAG

Store and retrieve conversation history for "infinite" context:

```rust
assistant.set_conversation_rag_enabled(true);

// Store messages
assistant.store_message_in_rag(&msg, true)?; // true = in current context

// Archive old messages (keeps them searchable)
assistant.archive_messages_to_rag(4)?;

let (total, archived, archived_tokens) = assistant.get_conversation_rag_stats()?;
```

### Database Schema

| Table | Purpose |
|-------|---------|
| `knowledge_chunks` | Indexed document chunks |
| `knowledge_fts` | FTS5 full-text search index |
| `knowledge_sources` | Document metadata + content hash |
| `conversation_messages` | Per-user message history |
| `conversation_fts` | FTS5 for conversation search |
| `users` | User info + global notes |
| `knowledge_notes` | Per-user notes per document |
| `session_notes` | Per-user notes per session |

### RAG Tiers (Advanced)

The RAG system supports multiple tiers with increasing sophistication. Higher tiers use more LLM calls but improve retrieval quality.

```rust
use ai_assistant::{
    RagTier, RagTierConfig, RagFeatures, RagPipeline, RagPipelineConfig,
    RagDebugConfig, RagDebugLevel, enable_rag_debug,
};

// Option 1: Use a predefined tier
let config = RagTierConfig::with_tier(RagTier::Enhanced);
println!("{}", config.summary()); // "RAG Config: Enhanced tier, 8 features enabled, 1-2 LLM calls"

// Option 2: Custom feature selection
let mut features = RagFeatures::none();
features.fts_search = true;
features.semantic_search = true;
features.hybrid_search = true;
features.reranking = true;  // Add reranking without full Enhanced tier

let config = RagTierConfig::with_features(features);

// Check requirements
let requirements = config.check_requirements();
for req in requirements {
    println!("Need: {} - {}", req.display_name(), req.description());
}

// Estimate LLM call cost
let (min_calls, max_calls) = config.estimate_extra_calls();
println!("This config will use {}-{} extra LLM calls per query",
    min_calls, max_calls.unwrap_or(usize::MAX));
```

**Available Tiers**:

| Tier | Features | LLM Calls | Best For |
|------|----------|-----------|----------|
| `Disabled` | None | 0 | Testing, no retrieval |
| `Fast` | FTS5 only | 0 | Low latency, simple queries |
| `Semantic` | FTS5 + embeddings + hybrid | 0 | Better recall |
| `Enhanced` | + query expansion + reranking | 1-2 | Balanced quality/cost |
| `Thorough` | + multi-query + compression | 3-5 | High accuracy |
| `Agentic` | + iterative agent | Unbounded | Complex research |
| `Graph` | + knowledge graph | N+ | Relationship queries |
| `Full` | All features | N+ | Maximum capability |
| `Custom` | User-defined | Varies | Fine-tuned control |

### RAG Debug Logging

Enable detailed logging to understand what the RAG system is doing:

```rust
use ai_assistant::{RagDebugConfig, RagDebugLevel, enable_rag_debug, global_rag_debug};

// Quick enable
enable_rag_debug(RagDebugLevel::Detailed);

// Or configure in detail
let debug_config = RagDebugConfig {
    enabled: true,
    level: RagDebugLevel::Detailed,
    log_to_file: true,
    log_path: Some("./rag_debug".into()),
    log_to_stderr: true,
    log_chunks: true,
    log_llm_details: true,
    log_scores: true,
    ..Default::default()
};

// Apply config
configure_global_rag_debug(debug_config);

// After queries, export all sessions
let stats = global_rag_debug().aggregate_stats();
println!("Total sessions: {}, Avg LLM calls: {:.1}",
    stats.total_sessions, stats.avg_llm_calls_per_session);

global_rag_debug().export_all("./all_rag_sessions.json")?;
```

### RAG Pipeline (Low-Level)

For full control, use the pipeline directly:

```rust
use ai_assistant::{RagPipeline, RagPipelineConfig, RagTier};

// Create pipeline
let config = RagPipelineConfig::for_tier(RagTier::Enhanced);
let mut pipeline = RagPipeline::with_config(config);

// Process query (requires implementing callback traits)
let result = pipeline.process(
    "What is the Aurora's cargo capacity?",
    &my_llm,          // impl LlmCallback
    Some(&my_embedder), // impl EmbeddingCallback
    &my_retriever,    // impl RetrievalCallback
    None,             // impl GraphCallback (optional)
)?;

println!("Retrieved {} chunks, {} tokens",
    result.chunks.len(), result.token_count);
println!("Sources: {:?}", result.sources);
println!("Stats: {} LLM calls in {}ms",
    result.stats.llm_calls, result.stats.total_duration_ms);
```

### Individual RAG Methods

Use advanced methods standalone:

```rust
use ai_assistant::{
    AdvancedQueryExpander, MultiQueryDecomposer, HydeGenerator,
    LlmReranker, RrfFusion, ContextualCompressor,
    SelfRagEvaluator, CragEvaluator, AdaptiveStrategySelector,
};

// Query expansion
let expander = AdvancedQueryExpander::new();
let variants = expander.expand("Aurora specifications", &llm)?;

// Multi-query decomposition
let decomposer = MultiQueryDecomposer::new();
let sub_queries = decomposer.decompose("Compare Aurora and Mustang for cargo and combat", &llm)?;

// HyDE
let hyde = HydeGenerator::new();
let (hypothetical_doc, embedding) = hyde.generate_with_embedding("best starter ship", &llm, &embedder)?;

// RRF fusion of multiple result sets
let fusion = RrfFusion::new();
let fused = fusion.fuse_strings(vec![keyword_results, semantic_results]);

// Self-reflection
let evaluator = SelfRagEvaluator::new();
let result = evaluator.evaluate("What's the Aurora's price?", &context, &llm)?;
if !result.is_sufficient {
    // Trigger re-retrieval...
}

// Adaptive strategy selection
let selector = AdaptiveStrategySelector::new();
let strategy = selector.select_with_llm("Compare ship specifications", &llm)?;
```

### Encrypted Knowledge Packages (KPKG)

**What**: Distribute knowledge bases securely as encrypted packages. Each package contains documents, metadata, AI configuration, and RAG settings in a single encrypted file.

**Package structure**:
```
.kpkg file (AES-256-GCM encrypted):
├── manifest.json      # Metadata, AI config, RAG settings
├── doc1.md           # Knowledge document
├── doc2.txt          # Another document
└── subfolder/
    └── doc3.md       # Nested documents supported
```

**Manifest fields**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Package name |
| `description` | String | Package description |
| `version` | String | Version string |
| `default_priority` | i32 | Default document priority |
| `priorities` | Map | Per-document priorities |
| `system_prompt` | String? | System prompt for AI |
| `persona` | String? | AI persona description |
| `examples` | Array | Few-shot learning examples |
| `rag_config` | Object? | RAG configuration |
| `metadata` | Object? | Author, license, tags, etc. |

**Creating packages with KpkgBuilder**:

```rust
use ai_assistant::{KpkgBuilder, AppKeyProvider, ExamplePair};

let package = KpkgBuilder::<AppKeyProvider>::with_app_key()
    // Basic info
    .name("My Knowledge Base")
    .description("A comprehensive guide")
    .version("1.0.0")

    // AI Configuration
    .system_prompt("You are a knowledgeable assistant. Be accurate and helpful.")
    .persona("Expert with deep domain knowledge")

    // Few-shot examples (helps the AI understand expected format)
    .add_example(
        "What is X?",
        "X is [definition]. Here's how it works: [explanation]."
    )
    .add_example_with_category(
        "How do I do Y?",
        "To do Y, follow these steps: 1. First... 2. Then...",
        "how-to"  // Category helps organize examples
    )

    // Document priorities (higher = more relevant in RAG)
    .default_priority(5)
    .add_document("intro.md", content, Some(10))  // High priority
    .add_document("details.md", content, None)     // Uses default

    // RAG Configuration
    .chunk_size(512)           // Tokens per chunk
    .chunk_overlap(50)         // Overlap between chunks
    .top_k(5)                  // Results to retrieve
    .min_relevance(0.3)        // Minimum similarity score
    .priority_boost(10)        // Boost all docs in this package

    // Metadata
    .author("Author Name")
    .language("en")
    .license("MIT")
    .url("https://example.com")
    .add_tag("documentation")
    .add_tag("guide")
    .with_current_timestamps()

    .build()?;

std::fs::write("knowledge.kpkg", &package)?;
```

**Reading and indexing packages**:

```rust
use ai_assistant::{KpkgReader, AppKeyProvider, RagDbKpkgExt};

let data = std::fs::read("knowledge.kpkg")?;
let reader = KpkgReader::<AppKeyProvider>::with_app_key();

// Read just manifest (lightweight inspection)
let manifest = reader.read_manifest_only(&data)?;
println!("Package: {} v{}", manifest.name, manifest.version);
println!("System prompt: {:?}", manifest.system_prompt);
println!("Examples: {}", manifest.examples.len());

// Read documents with manifest
let (docs, manifest) = reader.read_with_manifest(&data)?;
for doc in &docs {
    println!("{}: {} bytes (priority {})", doc.path, doc.content.len(), doc.priority);
}

// Index into RAG with extended result
let rag_db = assistant.rag_db().unwrap();
let result = rag_db.index_kpkg_ext(&data)?;

// Access result helpers
println!("Indexed {} docs", result.documents_indexed());
println!("Created {} chunks", result.chunks_created());

// Build effective system prompt from manifest
if let Some(prompt) = result.build_effective_system_prompt() {
    // Combines system_prompt + persona
    assistant.config.system_prompt = prompt;
}

// Get formatted examples for prompt injection
let examples = result.format_examples_for_prompt();
// Returns "User: Q1\nAssistant: A1\n\nUser: Q2\nAssistant: A2"
```

**Best practices for system_prompt and persona**:

1. **System prompt** should define the assistant's role and constraints:
   - "You are a Star Citizen expert. Answer based only on the knowledge provided."
   - "Be concise but thorough. If unsure, say so."

2. **Persona** adds personality and expertise:
   - "Veteran pilot with 10 years of experience"
   - "Friendly teacher who explains concepts clearly"

3. **Examples** teach the AI your expected format:
   - Include 2-5 diverse examples
   - Cover different question types
   - Use categories to organize similar examples

**Custom encryption keys**:

```rust
use ai_assistant::{KpkgBuilder, KpkgReader, CustomKeyProvider};

// Create with custom passphrase
let package = KpkgBuilder::with_key_provider(CustomKeyProvider::new("secret123"))
    .name("Private Knowledge")
    .add_document("doc.md", "content", None)
    .build()?;

// Read with same passphrase
let reader = KpkgReader::with_key_provider(CustomKeyProvider::new("secret123"));
let docs = reader.read(&package)?;
```

---

## 8. Multi-User Support

**What**: The RAG system isolates data per user - each user has their own notes, preferences, and conversation history.

```rust
// Set user (default is "default" for single-user apps)
assistant.set_user_id("user_123");
let global_notes = assistant.ensure_user()?; // Creates user if needed

// All RAG operations now use this user_id
```

---

## 9. Notes System

**What**: Users can annotate their sessions, store global preferences, and add notes to specific knowledge documents.

**Three note types**:
- **Global notes**: Persistent across all sessions (e.g., "I prefer concise answers")
- **Session notes**: Specific to one chat (e.g., "Discussing ship upgrades")
- **Knowledge notes**: Annotations on documents (e.g., "Focus on Warbond CCUs")

```rust
// Global notes
assistant.set_rag_global_notes("I prefer concise answers")?;
let notes = assistant.get_rag_global_notes();

// Session notes
assistant.set_rag_session_notes("Discussing upgrades")?;

// Knowledge notes
assistant.set_knowledge_notes("CCU Guide", "Focus on Warbond")?;
let guide_notes = assistant.get_knowledge_notes("CCU Guide");

// Include notes in messages
let knowledge_notes_ctx = assistant.build_knowledge_notes_context();
assistant.send_message_with_notes(
    "What ship?".to_string(),
    &knowledge_context,
    &session_notes,
    &knowledge_notes_ctx,
);
```

---

## 10. Analysis Tools

### Sentiment Analysis

Detect emotional tone of text:

```rust
use ai_assistant::analysis::SentimentAnalyzer;

let analyzer = SentimentAnalyzer::new();
let sentiment = analyzer.analyze("I love this ship!");
// sentiment.score > 0 = positive, < 0 = negative
```

### Topic Detection

Identify what a conversation is about:

```rust
let topics = analyzer.detect_topics(&messages);
// ["combat", "ships", "upgrades"]
```

### Entity Extraction

Pull structured data from text:

```rust
use ai_assistant::{EntityExtractor, EntityExtractorConfig, EntityType};

let extractor = EntityExtractor::new(EntityExtractorConfig::default());
let entities = extractor.extract(
    "Contact user@example.com or visit https://example.com for v2.0"
);

for entity in entities {
    println!("{:?}: {} (confidence: {:.2})", entity.entity_type, entity.text, entity.confidence);
}
```

Supported types: `Email`, `Url`, `Phone`, `Version`, `ProgrammingLanguage`, `Money`, `Percentage`, `Date`, `Time`, `Organization`, `Person`, `Location`, `Custom`.

### Fact Tracking

Extract and reinforce facts from conversations:

```rust
use ai_assistant::{FactExtractor, FactExtractorConfig, FactStore};

let extractor = FactExtractor::new(FactExtractorConfig::default());
let facts = extractor.extract("I prefer Rust over Python. My goal is to learn systems programming.");

let mut store = FactStore::new();
for fact in facts {
    store.add(fact);
}

// Facts mentioned multiple times get higher confidence
store.reinforce("preference:rust");
```

### Quality Analysis

Evaluate response quality:

```rust
use ai_assistant::{QualityAnalyzer, QualityConfig};

let analyzer = QualityAnalyzer::new(QualityConfig::default());
let score = analyzer.analyze("What is Rust?", "Rust is a systems programming language...", None);

println!("Overall: {:.0}%", score.overall * 100.0);
println!("Relevance: {:.0}%", score.relevance * 100.0);
println!("Coherence: {:.0}%", score.coherence * 100.0);
println!("Issues: {:?}", score.issues);
```

---

## 11. Security Features

### Rate Limiting
Prevents too many requests in a short time. Uses token bucket algorithm - N tokens per window, each request costs one.

### Input Sanitization
Removes potential prompt injection attacks where malicious text tries to override the system prompt.

### Audit Logging
Records every request/response for accountability and debugging.

### Hook System
Pre/post processing hooks that run before/after LLM calls:

```rust
hooks.add_pre_hook("logger", |msg| {
    println!("Sending: {}", msg.content);
    Ok(msg)
});
```

---

## 12. Performance Features

### Connection Pooling
Reuses HTTP connections instead of creating new ones per request.

### Cache Compression
Compresses cached responses with gzip to save disk space.

### Latency Tracking
Measures response times with percentiles (p50, p95, p99).

### Health Checks
Periodically pings providers to check availability.

### Fallback Chain
If primary provider fails, automatically tries the next one.

### Batch Processing
Sends multiple requests in parallel for bulk operations.

---

## 13. Decision Trees

**What**: A flowchart-like structure where each node makes a decision, performs an action, or produces a result.

**Why**: Useful for structured workflows where the next step depends on previous results.

### Node Types

| Type | Purpose | Example |
|------|---------|---------|
| **Condition** | Branch on variable values | "Is age >= 18?" |
| **Action** | Record an action | "Log check-in" |
| **Terminal** | End with a result | "Access granted" |
| **Question** | Ask user for input | "What's your name?" |
| **Prompt** | Send to LLM, store response | "Summarize this" |
| **Function** | Call registered Rust function | "calculate_price()" |
| **Sequence** | Run children in order, all must succeed | A then B then C |
| **Selector** | Try children, first success wins | Try A, else B, else C |
| **Parallel** | Run all children, collect results | A and B simultaneously |
| **SubTree** | Call another tree | "Run validation tree" |
| **LlmCondition** | LLM decides branching | "Is this spam?" |

### Building a Tree

```rust
use ai_assistant::decision_tree::*;
use serde_json::json;

let tree = DecisionTreeBuilder::new("my_tree", "User Check")
    .root("check_age")
    .condition_node("check_age", vec![
        DecisionBranch {
            condition: Condition::new("age", ConditionOperator::GreaterOrEqual, json!(18)),
            target_node_id: "adult".to_string(),
            label: Some("Adult".to_string()),
        }
    ], Some("minor".to_string()))
    .terminal_node("adult", json!("allowed"), Some("Adult".to_string()))
    .terminal_node("minor", json!("denied"), Some("Minor".to_string()))
    .build();
```

### Sync Execution

```rust
let mut context = HashMap::new();
context.insert("age".to_string(), json!(25));

let path = tree.evaluate(&context);
// path.result == Some(json!("allowed"))
```

### Async Execution (TreeExecutor)

For trees with Prompt/LlmCondition nodes:

```rust
let (tx, rx) = mpsc::channel();
let mut executor = TreeExecutor::new(tree, tx);

executor.register_function("get_price", |args| {
    Ok(json!({"price": 99.99}))
});

executor.run();

match executor.state() {
    ExecutorState::WaitingForLlm { node_id } => {
        executor.resume_with_response("LLM answer");
    }
    ExecutorState::Completed => println!("Done!"),
    _ => {}
}
```

### Template Substitution

`{{variable}}` placeholders replaced with context values:
```
// Context: {"user": "Alice", "topic": "ships"}
// Template: "Tell {{user}} about {{topic}}"
// Result: "Tell Alice about ships"
```

### Serialization & Visualization

```rust
let json = tree.to_json();                    // Serialize to JSON
let tree = DecisionTree::from_json(&json)?;   // Deserialize
let mermaid = tree.to_mermaid();              // Export to Mermaid flowchart
```

---

## 14. Tool Use / Tool Registry

**What**: Define tools (functions with typed parameters) that the LLM can call during conversation.

```rust
use ai_assistant::tool_use::{Tool, ToolRegistry, ToolParameter};

let mut registry = ToolRegistry::new();

registry.register(
    Tool::new("get_weather", "Get weather for a city")
        .with_param(ToolParameter::string("city", "City name"))
        .with_handler(|args| {
            let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
            Ok(serde_json::json!({"temp": 22, "city": city}))
        })
);

let result = registry.execute("get_weather", serde_json::json!({"city": "Madrid"}));
```

**Key types**: `Tool`, `ToolRegistry`, `ToolParameter`, `ToolCall`, `ToolResult`

---

## 15. Function Calling (OpenAI-compatible)

**What**: The OpenAI function calling format - defines functions as JSON schemas that the model can choose to invoke.

**Difference from Tool Use**: This module generates the OpenAI-compatible JSON format for the API request, while Tool Use is the internal execution system.

```rust
use ai_assistant::{FunctionBuilder, FunctionRegistry, ToolChoice};

let mut registry = FunctionRegistry::new();

registry.register(
    FunctionBuilder::new("get_weather")
        .description("Get current weather")
        .add_string_param("location", "City name", true)
        .add_enum_param("unit", &["celsius", "fahrenheit"], false)
        .build()
);

let request = registry.build_request(ToolChoice::Auto);
// Produces OpenAI-compatible JSON with function definitions
```

---

## 16. Agent Framework

**What**: A multi-step execution framework where the LLM can reason, act, and observe in a loop (ReAct pattern: Reason-Act-Observe).

**Why**: Complex tasks often require multiple steps - search, calculate, lookup - before producing a final answer.

```rust
use ai_assistant::{ReactAgent, AgentConfig, create_builtin_agent_tools};

let mut agent = ReactAgent::new(AgentConfig {
    max_steps: 10,
    verbose: true,
    ..Default::default()
});

// Add built-in tools (calculator, search, etc.)
for tool in create_builtin_agent_tools() {
    agent.add_tool(tool);
}

// Execute a step
let observation = agent.execute_step(
    "I need to calculate".to_string(),
    "calculator".to_string(),
    "2 + 2 * 3".to_string(),
)?;

if agent.should_stop() {
    println!("Agent finished");
}
```

---

## 17. Plugins

**What**: Extension system for adding functionality without modifying core code.

**Lifecycle hooks**:
- `on_message_received` - Before processing user input
- `on_response_generated` - After getting LLM response
- `on_session_start` / `on_session_end`

---

## 18. Persistence & Export

### Backup Manager
Automatic backups with configurable retention (last N, or older than N days).

### Database Compaction
Removes old data, merges fragments, reduces file size.

### Multi-Format Export

```rust
use ai_assistant::{ConversationExporter, ExportFormat, ExportOptions};

let exporter = ConversationExporter::new(ExportOptions {
    format: ExportFormat::Markdown,
    include_metadata: true,
    redact_pii: true,
    ..Default::default()
});

let output = exporter.export(&messages, Some(&metadata))?;
// Also supports: JSON, CSV, HTML
```

---

## 19. Vision & Multimodal

**What**: Some LLMs process images alongside text (e.g., LLaVA). This is "multimodal" input.

```rust
use ai_assistant::{VisionMessage, ImageInput, ImageDetail};

let message = VisionMessage::new("What's in this image?")
    .add_image(ImageInput::from_file("photo.jpg")?)
    .with_detail(ImageDetail::High);

let openai_format = message.to_openai_format();
```

- `ImageInput` encodes images as base64
- `ImagePreprocessor` resizes large images to fit model limits
- `VisionCapabilities` checks if current model supports images

---

## 20. Structured Output

**What**: Force LLM responses into a specific JSON schema that can be validated.

**Why**: When you need machine-readable output (not free text), schemas ensure the response has the right fields and types.

```rust
use ai_assistant::{JsonSchema, SchemaProperty, SchemaValidator, SchemaBuilder};

// Define schema
let schema = JsonSchema::new("sentiment")
    .with_property("sentiment", SchemaProperty::string()
        .with_enum(vec!["positive", "negative", "neutral"]))
    .with_property("confidence", SchemaProperty::number()
        .with_minimum(0.0).with_maximum(1.0))
    .with_required("sentiment")
    .with_required("confidence");

// Generate instruction prompt
let prompt = format!("{}\n{}", user_question, schema.to_prompt());

// Validate response
let validation = SchemaValidator::validate(&json_response, &schema);
if !validation.valid {
    println!("Errors: {:?}", validation.errors);
}

// Pre-built schemas
let sentiment_schema = SchemaBuilder::sentiment_analysis();
let entity_schema = SchemaBuilder::entity_extraction();
```

---

## 21. Model Profiles

**What**: Pre-defined generation settings (temperature, top_p, etc.) for different use cases.

**Why**: A "coding" task needs low temperature (precise), while "creative writing" needs high temperature (varied).

```rust
use ai_assistant::{ProfileManager, ProfileApplicator};

let manager = ProfileManager::with_defaults();

// Available: balanced, creative, precise, coding, conversational, concise, detailed
let profile = manager.get("coding").unwrap();
println!("Temperature: {}", profile.temperature);

let applicator = ProfileApplicator::new(&config);
let config = applicator.apply(profile);
```

---

## 22. Prompt Templates

**What**: Reusable prompt patterns with `{{variable}}` substitution.

**Why**: Common tasks (code review, translation, summarization) always need similar prompts. Templates avoid repetition.

```rust
use ai_assistant::{TemplateManager, BuiltinTemplates};

let manager = TemplateManager::with_builtins();

// Available: code_review, translation, explain, bug_fix, summarize, api_docs, refactor
let prompt = manager.render("code_review", &[
    ("code", "fn main() { ... }"),
    ("language", "Rust"),
])?;
```

---

## 23. Model Routing

**What**: Automatically selects the best available model based on the task type.

**Why**: A coding task benefits from a code-specialized model, while a creative task benefits from a general model. Don't make the user choose.

```rust
use ai_assistant::{ModelRouter, ModelRequirements, TaskType};

let router = ModelRouter::new();

// Auto-detect task type from the query
let task = ModelRouter::detect_task_type("Write code to sort an array");
assert_eq!(task, TaskType::Coding);

// Select best model for the task
let requirements = ModelRequirements::for_task(TaskType::Coding)
    .with_min_context(8000);

let best = router.select_best(&available_models, &requirements);
```

---

## 24. Cost Estimation

**What**: Track and budget API usage costs per model and provider.

**Why**: When using paid APIs, you need visibility into spending and the ability to set limits.

```rust
use ai_assistant::{CostEstimator, CostTracker, BudgetManager};

let estimator = CostEstimator::new();
let mut tracker = CostTracker::new();

// Estimate cost for a request
let estimate = estimator.estimate("gpt-4", "openai", 1000, 500);
println!("Cost: {}", estimate.format()); // $0.0450 USD

tracker.add(estimate);
println!("{}", tracker.summary());

// Budget limits
let budget = BudgetManager::new()
    .with_daily_limit(5.0)
    .with_request_limit(0.50);

if budget.check(0.10).is_exceeded() {
    println!("Budget exceeded!");
}
```

---

## 25. Response Caching

**What**: Cache LLM responses so identical (or similar) questions don't need regeneration.

**Why**: LLM calls are slow and expensive. Caching identical or near-identical queries saves both.

```rust
use ai_assistant::{ResponseCache, CacheConfig};

let mut cache = ResponseCache::new(CacheConfig {
    max_entries: 1000,
    default_ttl: Duration::from_secs(3600),
    fuzzy_matching: true,         // Similar queries match cached responses
    similarity_threshold: 0.85,   // How similar is "similar enough"
    ..Default::default()
});

// Cache a response
cache.put("What is Rust?", "llama-3", "Rust is...", 50, Some("factual"));

// Get from cache (fuzzy matching)
if let Some(response) = cache.get("What's Rust?", "llama-3") {
    println!("Cached: {}", response.content);
}

println!("Hit rate: {:.1}%", cache.stats().hit_rate() * 100.0);
```

---

## 26. Conversation Memory

**What**: Long-term memory with decay - remembers facts, preferences, and goals across sessions.

**Why**: Conversations are ephemeral. Memory lets the assistant remember important things even after sessions end.

```rust
use ai_assistant::{MemoryManager, MemoryConfig};

let mut memory = MemoryManager::new(MemoryConfig::default());

// Remember with importance weight (decays over time)
memory.remember_fact("User prefers Rust", 0.8);
memory.remember_preference("Concise explanations");
memory.remember_goal("Learn systems programming");

// Recall relevant memories for a query
let memories = memory.recall("What programming language?");

// Build context for prompts (within token budget)
let context = memory.build_context("Tell me about Rust", 500);
```

---

## 27. Embedding Cache

**What**: Cache computed embedding vectors so the same text doesn't need re-embedding.

**Why**: Embedding computation (converting text to vectors) is expensive. Caching avoids redundant work.

```rust
use ai_assistant::{SharedEmbeddingCache, EmbeddingCacheConfig, cosine_similarity};

let cache = SharedEmbeddingCache::new(EmbeddingCacheConfig {
    max_entries: 10000,
    ttl: Duration::from_secs(86400),
    ..Default::default()
});

cache.set("hello world", "text-embedding", vec![0.1, 0.2, 0.3]);

if let Some(embedding) = cache.get("hello world", "text-embedding") {
    let similarity = cosine_similarity(&embedding, &other_embedding);
}

println!("Hit rate: {:.1}%", cache.stats().unwrap().hit_rate() * 100.0);
```

---

## 28. Streaming Metrics

**What**: Real-time metrics about generation performance.

**Why**: Know how fast your LLM is generating, detect slowdowns, and track time-to-first-token.

```rust
use ai_assistant::{StreamingMetrics, MetricsConfig};

let mut metrics = StreamingMetrics::new(MetricsConfig::default());
metrics.start();

// Record tokens as they arrive
metrics.record_token();

let snapshot = metrics.snapshot();
println!("Tokens/second: {:.1}", snapshot.tokens_per_second);
println!("Time to first token: {}ms", snapshot.time_to_first_token_ms);
```

---

## 29. Retry & Circuit Breaker

**What**: Automatic retry with exponential backoff, plus circuit breaker to stop hammering a failing service.

**Why**: Network requests fail. Retries handle transient failures. Circuit breakers prevent cascading failures when a service is down.

```rust
use ai_assistant::{RetryConfig, retry, CircuitBreaker, ResilientExecutor};

// Simple retry with backoff
let result = retry(|| some_fallible_operation(), RetryConfig::default())?;

// Circuit breaker: opens after 5 failures, waits 30s before retrying
let breaker = CircuitBreaker::new(5, Duration::from_secs(30));
let executor = ResilientExecutor::new(breaker);

let result = executor.execute(|| api_call())?;
```

---

## 30. Diff Viewer

**What**: Compare text and responses, showing additions, deletions, and similarity.

**Why**: Useful for comparing different model responses or tracking how text changes.

```rust
use ai_assistant::{diff, diff_compare_responses};

let result = diff("old text\nline 2", "new text\nline 2\nline 3");
println!("Additions: {}, Deletions: {}", result.additions, result.deletions);
println!("{}", result.to_unified("old.txt", "new.txt"));

let comparison = diff_compare_responses("The quick brown fox", "The fast brown fox jumps");
println!("Similarity: {:.1}%", comparison.similarity * 100.0);
println!("Common phrases: {:?}", comparison.common_phrases);
```

---

## 31. Advanced LLM Techniques

### Prompt Chaining
Breaking a complex task into sequential LLM calls where each output feeds the next.

### Chain of Thought (CoT) Parsing
Extracting reasoning steps from a model's "thinking out loud" response.

### Self-Consistency
Asking the same question multiple times and taking the most common answer.

### Request Coalescing
If multiple users ask the same question simultaneously, share one LLM call.

### Model Warmup
Pre-load a model before it's needed to avoid cold-start latency.

---

## 32. Security & Privacy (Advanced)

### PII Detection
Finds Personally Identifiable Information (names, emails, phones) so they can be redacted before sending to the LLM.

### Content Moderation
Filters harmful or inappropriate content in both input and output.

### Injection Detection
Catches "prompt injection" attacks where input tries to override the system prompt.

---

## 33. Quality & Reliability

### Hallucination Detection
Compares responses against known facts to catch fabricated information.

### Confidence Scoring
Assigns 0-1 confidence to responses. Low-confidence responses get flagged for review.

### Model Ensemble
Asks multiple models the same question and combines answers for reliability.

---

## 34. Internationalization

### Language Detection

```rust
use ai_assistant::LanguageDetector;

let detector = LanguageDetector::new();
let result = detector.detect("Bonjour, comment allez-vous?");
println!("{} ({:.0}%)", result.name, result.confidence * 100.0);
// French (85%)
```

Supports: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean.

### Localized Strings
Built-in translations for UI text (errors, status indicators, etc.).

### Multilingual Prompts
Adapts system prompts to the user's detected language.

---

## 35. Benchmarking

**What**: Built-in performance benchmarks for all analysis components.

```rust
use ai_assistant::{run_all_benchmarks, BenchmarkConfig, compare_results};

let config = BenchmarkConfig {
    warmup_iterations: 5,
    iterations: 100,
    ..Default::default()
};

let suite = run_all_benchmarks(config);
println!("{}", suite.summary());

for result in &suite.results {
    println!("{}: mean={:.2}ms, p95={:.2}ms",
        result.name, result.stats.mean_ms, result.stats.p95_ms);
}

// Compare runs for regression detection
let comparison = compare_results(&baseline, &current);
println!("Regression: {}", comparison.has_regression);
```

Built-in benchmarks: token estimation, entity extraction, quality analysis, language detection, sentiment, topic detection.

---

## 36. Provider Discovery

**What**: Auto-discover running LLM providers on the network.

```rust
use ai_assistant::{discover_providers, DiscoveryConfig, create_registry_with_discovery};

let config = DiscoveryConfig::default();
let providers = discover_providers(&config);

for provider in &providers {
    println!("Found: {} at {}", provider.name(), provider.base_url());
    if provider.is_available() {
        let models = provider.list_models()?;
        println!("  Models: {:?}", models.iter().map(|m| &m.name).collect::<Vec<_>>());
    }
}
```

---

## 37. egui Widgets

Enable with `features = ["egui-widgets"]`. Pre-built UI components for chat interfaces:

| Widget | Purpose |
|--------|---------|
| `chat_message()` | Render a message bubble |
| `streaming_response()` | Show current response with spinner |
| `model_selector()` | Model dropdown |
| `context_usage_bar()` | Context window usage bar |
| `session_list()` | Sidebar session list |
| `chat_input()` | Single-line input with send |
| `chat_input_multiline()` | Multi-line with Ctrl+Enter |
| `suggestions()` | Suggestion button row |
| `welcome_screen()` | Centered welcome with suggestions |
| `connection_status()` | Provider connection indicator |
| `rag_controls()` | RAG enable/disable checkboxes |
| `rag_status_compact()` | Compact RAG stats |
| `rag_status_panel()` | Detailed RAG grid |
| `NotesManager` | Complete notes management |
| `context_full_hint()` | Context full warning |

```rust
use ai_assistant::widgets::*;

fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
    let colors = ChatColors::default();

    connection_status(ui, self.assistant.is_fetching_models, self.assistant.available_models.len());

    if let Some(model) = model_selector(ui, &mut self.selected, &self.assistant.available_models) {
        self.assistant.config.selected_model = model.name;
    }

    context_usage_bar(ui, &usage, 120.0);

    egui::ScrollArea::vertical().show(ui, |ui| {
        for msg in self.assistant.get_display_messages() {
            chat_message(ui, msg, &colors, 400.0);
        }
    });

    if let Some(text) = chat_input(ui, &mut self.input, self.assistant.is_generating, "Ask...") {
        self.assistant.send_message(text, &self.knowledge);
    }
}
```

---

## 38. Monitoring

### Latency Metrics
Response times with percentiles (p50, p95, p99) per provider.

### Conversation Analytics
Messages per session, response length, topics, time distribution.

### Prometheus Metrics
Export in Prometheus format for Grafana dashboards.

---

## 39. Adaptive Thinking

**What**: Automatically adjusts the reasoning depth, temperature, max tokens, RAG tier, and chain-of-thought prompting based on heuristic analysis of the user's query — without any LLM call.

**Why**: A greeting like "hi" doesn't need low-temperature chain-of-thought reasoning, and a deep comparison query shouldn't use high temperature with minimal tokens. Adaptive thinking makes every query use the right parameters automatically, improving both quality and efficiency.

**How it works here**: The `QueryClassifier` analyzes structural signals (word count, question marks, comparison/analysis keywords, concept count, multi-part detection) combined with intent detection from `IntentClassifier` to produce a `ThinkingStrategy`. This strategy is applied before the LLM call — modifying the system prompt, temperature, and max tokens.

### Enable Adaptive Thinking

```rust
use ai_assistant::AiAssistant;

let mut assistant = AiAssistant::new();

// Enable with defaults (disabled by default for backward compatibility)
assistant.enable_adaptive_thinking();

// Now every send_message call automatically classifies the query
// and adjusts parameters before the LLM call
assistant.send_message("hello".to_string(), "");
// → Trivial: temp=0.8, max_tokens=256, no CoT

assistant.send_message(
    "Compare the advantages of CRDTs vs OT for collaborative editing".to_string(),
    ""
);
// → Complex: temp=0.4, max_tokens=4096, step-by-step CoT injected
```

### Manual Classification

```rust
use ai_assistant::adaptive_thinking::*;

let config = AdaptiveThinkingConfig { enabled: true, ..Default::default() };
let classifier = QueryClassifier::new(config);

let strategy = classifier.classify("What is Rust?");
println!("Depth: {:?}", strategy.depth);           // Simple
println!("Temperature: {}", strategy.temperature);  // 0.7
println!("CoT: {}", strategy.system_prompt_addition); // (empty)

let strategy = classifier.classify("Analyze the trade-offs between microservices and monolith");
println!("Depth: {:?}", strategy.depth);           // Expert
println!("Temperature: {}", strategy.temperature);  // 0.2
println!("RAG hint: {}", strategy.rag_complexity_hint); // reasoning
```

### Inspect Classification Signals

```rust
let strategy = classifier.classify("Compare HashMap vs BTreeMap, and when should I use each?");

let signals = &strategy.signals;
println!("Word count: {}", signals.word_count);
println!("Has comparison: {}", signals.has_comparison);
println!("Is multi-part: {}", signals.is_multi_part);
println!("Concept count: {}", signals.concept_count);
println!("Detected intent: {}", signals.detected_intent);
```

### Custom Configuration

```rust
use std::collections::HashMap;
use ai_assistant::adaptive_thinking::*;

let mut config = AdaptiveThinkingConfig {
    enabled: true,
    // Clamp depth to Simple..Complex (never Trivial, never Expert)
    min_depth: ThinkingDepth::Simple,
    max_depth: ThinkingDepth::Complex,
    // Custom temperature overrides
    temperature_map: Some(HashMap::from([
        (ThinkingDepth::Simple, 0.9),
        (ThinkingDepth::Moderate, 0.7),
        (ThinkingDepth::Complex, 0.3),
    ])),
    // CoT instructions in Spanish
    cot_instructions_override: Some(HashMap::from([
        (ThinkingDepth::Moderate, "Razona brevemente antes de responder.".to_string()),
        (ThinkingDepth::Complex, "Piensa paso a paso. Verifica cada paso.".to_string()),
    ])),
    ..Default::default()
};

assistant.set_adaptive_thinking(config);
```

### RAG Tier Priority

When adaptive thinking suggests a RAG tier but the user has also set one explicitly, the `rag_tier_priority` field controls which wins:

| Priority | Behavior |
|----------|----------|
| `Adaptive` (default) | Adaptive suggestion wins; warning logged if conflict |
| `Explicit` | User's explicit tier wins; warning logged if conflict |
| `Highest` | Whichever tier is more thorough wins |

```rust
let mut config = AdaptiveThinkingConfig {
    enabled: true,
    rag_tier_priority: RagTierPriority::Highest,
    ..Default::default()
};
```

### Thinking Tag Parsing

Models like DeepSeek-R1 and QwQ emit `<think>...</think>` blocks with their internal reasoning. The `ThinkingTagParser` strips these from the visible response:

```rust
use ai_assistant::adaptive_thinking::{ThinkingTagParser, parse_thinking_tags};

// Non-streaming: convenience function
let result = parse_thinking_tags(
    "<think>Let me reason about this...</think>The answer is 42."
);
assert_eq!(result.visible_response, "The answer is 42.");
assert_eq!(result.thinking.as_deref(), Some("Let me reason about this..."));

// Streaming: process chunk by chunk
let mut parser = ThinkingTagParser::new(true); // true = strip thinking tags
let visible1 = parser.process_chunk("<thi");
let visible2 = parser.process_chunk("nk>internal reasoning</think>Visible text");
let visible3 = parser.finalize();

let result = parser.result();
println!("Visible: {}", result.visible_response);
println!("Thinking: {:?}", result.thinking);
```

When `transparent_thinking_parse` is enabled in config (default: true), `poll_response` automatically routes chunks through the parser — the caller sees only the visible response.

### Accessing Last Strategy

After a message is sent with adaptive thinking enabled, inspect what was applied:

```rust
if let Some(strategy) = &assistant.last_thinking_strategy {
    println!("Used depth: {:?}", strategy.depth);
    println!("Temperature: {}", strategy.temperature);
}

if let Some(thinking) = &assistant.last_thinking_result {
    if let Some(ref thought) = thinking.thinking {
        println!("Model's internal reasoning: {}", thought);
    }
}
```

### Depth Levels Reference

| Depth    | Temp | max_tokens | RAG tier   | CoT prompt     | Example query |
|----------|------|------------|------------|----------------|---------------|
| Trivial  | 0.8  | 256        | simple     | (none)         | "hello", "thanks" |
| Simple   | 0.7  | 1024       | simple     | (none)         | "What is Rust?" |
| Moderate | 0.6  | 2048       | standard   | Brief reasoning | "Explain async/await" |
| Complex  | 0.4  | 4096       | complex    | Step-by-step   | "Compare X vs Y" |
| Expert   | 0.2  | (none)     | reasoning  | Rigorous verification | Deep multi-concept analysis |

---

## 40. Provider Failover

**What**: Automatic fallback to alternative providers when the primary one fails.

```rust
use ai_assistant::{AiAssistant, AiProvider};

let mut assistant = AiAssistant::new();

// Configure fallback chain: try LM Studio, then LocalAI
assistant.configure_fallback(vec![
    (AiProvider::LMStudio, "qwen2.5:7b".to_string()),
    (AiProvider::LocalAI, "llama3:8b".to_string()),
]);
assistant.enable_fallback();

// If primary fails, fallbacks are tried automatically
assistant.send_message("Hello".to_string(), "");

// Check which provider actually responded
if let Some(provider) = assistant.last_provider_used() {
    println!("Response from: {}", provider);
}
```

---

## 41. Retry with Backoff

**What**: Automatic retry with exponential backoff for transient network errors.

**How**: All provider HTTP calls are wrapped with `RetryConfig`. Fetch operations use `RetryConfig::fast()` (2 retries), generation uses `RetryConfig::default()` (3 retries). Only retryable errors (connection refused, timeout, 5xx) are retried.

```rust
use ai_assistant::{RetryConfig, retry_with_config};

// Custom retry for your own operations
let result = retry_with_config(RetryConfig::default(), || {
    some_network_operation()
})?;

// AiConfig carries a retry_config field for provider calls
let mut config = AiConfig::default();
config.retry_config = RetryConfig::fast(); // 2 retries with short delays
```

---

## 42. Conversation Compaction

**What**: Lightweight heuristic compaction of conversation history (no LLM call needed).

```rust
let mut assistant = AiAssistant::new();

// Enable auto-compaction (runs before each send_message)
assistant.enable_auto_compaction();

// Or configure thresholds
use ai_assistant::CompactionConfig;
let config = CompactionConfig {
    max_messages: 50,     // Trigger at 50 messages
    target_messages: 20,  // Compact down to 20
    preserve_recent: 10,  // Always keep last 10
    preserve_first: 2,    // Always keep first 2
    ..Default::default()
};
assistant.set_compaction_config(config);

// Manual compaction
let result = assistant.compact_conversation();
println!("Removed {} messages", result.messages_removed);
```

---

## 43. API Key Rotation

**What**: Round-robin API key management with automatic rotation on rate limits.

```rust
use ai_assistant::{ApiKeyManager, ApiKey, RotationConfig};

let mut assistant = AiAssistant::new();

// Add keys for a provider
assistant.add_api_key("openai", ApiKey::new("sk-key1"));
assistant.add_api_key("openai", ApiKey::new("sk-key2"));

// Get current key (round-robin)
let key = assistant.get_current_api_key("openai");

// Mark as rate-limited (auto-rotates to next key)
assistant.mark_key_rate_limited("openai");
```

---

## 44. Context Size Cache

**What**: Global cache for model context window sizes, avoiding repeated API calls.

```rust
use ai_assistant::{get_model_context_size_cached, clear_context_size_cache};

// Lookup with provider API as fetcher (cached after first call)
let size = get_model_context_size_cached("llama3.2:7b", |name| {
    // fetch_model_context_size(&config, name)  // Provider API call
    None  // Falls back to static table: 128_000
});

// Clear cache (e.g., when switching providers)
clear_context_size_cache();
```

---

## 45. Journal Sessions (JSONL)

**What**: Append-only JSONL session format — each message is one line, no full rewrite.

```rust
use ai_assistant::{JournalSession, JournalEntry, ChatMessage};

let journal = JournalSession::new("session.jsonl");

// Append messages (O(1) per message)
journal.append_message(&ChatMessage::user("Hello"))?;
journal.append_message(&ChatMessage::assistant("Hi!"))?;

// Count without loading all data
let count = journal.message_count()?;

// Load all messages
let messages = journal.load_messages()?;

// Compact: summary + keep last 10 messages
journal.compact("Summary of old conversation", 10)?;

// Migrate from ChatSession to journal
let session = ChatSession::new("My Session");
let journal = session.to_journal("session.jsonl")?;
```

---

## 46. Encrypted Sessions

**What**: AES-256-GCM encrypted session storage (requires `rag` feature).

```rust
use ai_assistant::ChatSessionStore;
use std::path::Path;

let key: [u8; 32] = derive_key_from_password("my_password");

// Save encrypted
store.save_encrypted(Path::new("sessions.enc"), &key)?;

// Load encrypted (returns empty store if file doesn't exist)
let store = ChatSessionStore::load_encrypted(Path::new("sessions.enc"), &key)?;
```

---

## 47. Binary Storage

**What**: Internal storage abstraction using bincode + gzip compression.

```rust
use ai_assistant::internal_storage::{save_internal, load_internal, dump_as_json, file_info};
use std::path::Path;

// Save (auto-selects binary or JSON based on features)
save_internal(&my_data, Path::new("data.bin"))?;

// Load (auto-detects format — reads both legacy JSON and binary)
let data: MyData = load_internal(Path::new("data.bin"))?;

// Debug: dump binary file as JSON
let json = dump_as_json::<MyData>(Path::new("data.bin"))?;

// Inspect file metadata
let info = file_info(Path::new("data.bin"))?;
println!("Format: {}, Size: {} bytes", info.format, info.size_bytes);
```

---

## 48. Log Redaction

**What**: Strip sensitive data from log output.

```rust
use ai_assistant::log_redaction::{redact, safe_log};

let safe = redact("Authorization: Bearer eyJhbG...");
// "Authorization: Bearer ***REDACTED***"

let safe = redact("postgres://user:s3cret@db:5432/mydb");
// "postgres://user:***@db:5432/mydb"

// Macro: redacts before printing
safe_log!("API response: {}", sensitive_data);
```

---

## 49. HTTP Client Abstraction

**What**: Testable HTTP client trait for provider communication.

```rust
use ai_assistant::http_client::{HttpClient, UreqClient};

// Production: use the real client
let client = UreqClient;
let models = fetch_ollama_models_with(&client, "http://localhost:11434")?;

// Tests: use a mock
#[cfg(test)]
let mock = MockHttpClient::with_response(json!({"models": []}));
let models = fetch_ollama_models_with(&mock, "http://localhost:11434")?;
```

---

## 50. Vector Database Backends

**What**: Pluggable vector storage backends for semantic search at different scales.

**Feature flags**:
- `embeddings` — enables the base VectorDb trait and InMemoryVectorDb
- `vector-lancedb` — adds LanceDB embedded backend

### InMemoryVectorDb (Tier 0)

```rust
use ai_assistant::vector_db::{VectorDbBuilder, VectorDbBackend, VectorDbConfig};

let config = VectorDbConfig { dimension: 384, ..Default::default() };
let mut db = VectorDbBuilder::new()
    .backend(VectorDbBackend::InMemory)
    .config(config)
    .build()?;

// Insert vectors
let metadata = std::collections::HashMap::from([
    ("source".to_string(), "doc1.md".to_string()),
]);
db.insert("vec_1", vec![0.1; 384], metadata)?;

// Search
let results = db.search(&vec![0.1; 384], 5, None)?;
for result in &results {
    println!("{}: score={:.4}", result.id, result.score);
}

// Health check
let health = db.health_check()?;
println!("Backend: {}, healthy: {}", health.backend, health.is_healthy);
```

### LanceDB Backend (Tier 2)

**Feature flag**: `vector-lancedb`

```rust
use ai_assistant::vector_db::{VectorDbBuilder, VectorDbConfig};

let config = VectorDbConfig { dimension: 384, ..Default::default() };
let mut db = VectorDbBuilder::new()
    .lance("./my_vectors")  // Path to LanceDB directory
    .config(config)
    .build()?;

// Same API as InMemory — all VectorDb trait methods work
db.insert("vec_1", vec![0.1; 384], metadata)?;
let results = db.search(&vec![0.1; 384], 5, None)?;

// Backend info
let info = db.backend_info();
println!("{}: tier {}, persistence={}", info.name, info.tier, info.supports_persistence);
// "LanceDB: tier 2, persistence=true"
```

### Qdrant Backend (Tier 3)

```rust
use ai_assistant::vector_db::{VectorDbBuilder, VectorDbConfig};

let config = VectorDbConfig { dimension: 384, ..Default::default() };
let mut db = VectorDbBuilder::new()
    .qdrant("http://localhost:6333", "my_collection")
    .config(config)
    .build()?;

// Same VectorDb trait API
db.insert("vec_1", vec![0.1; 384], metadata)?;
let results = db.search(&vec![0.1; 384], 5, None)?;
```

### Migrating Between Backends

```rust
use ai_assistant::vector_db::migrate_vectors;

// Export from InMemory, import to LanceDB
let result = migrate_vectors(&*source_db, &mut *target_db)?;
println!("Migrated: {} exported, {} imported", result.exported, result.imported);
```

### Hybrid Search (Vector + Keyword)

```rust
use ai_assistant::vector_db::{HybridVectorSearch, InMemoryVectorDb};

let vector_db = InMemoryVectorDb::new(VectorDbConfig { dimension: 384, ..Default::default() });
let mut hybrid = HybridVectorSearch::new(vector_db, 0.7); // 0.7 = vector weight

hybrid.add_document("doc1", vec![0.1; 384], "The quick brown fox", metadata)?;
let results = hybrid.hybrid_search(&vec![0.1; 384], "brown fox", 5)?;
```

### Backend Comparison

| Backend | Feature Flag | Persistence | Server | Scale | Use Case |
|---------|-------------|-------------|--------|-------|----------|
| InMemory | `embeddings` | No | No | <50K vectors | Prototypes, tests |
| LanceDB | `vector-lancedb` | Yes | No | 50K-10M | Production, single-user |
| Qdrant | `embeddings` | Yes | Yes | 10M+ | Multi-user, clusters |

---

## 51. Distributed Computing

**What**: Parallel data processing, conflict-free replicated data types (CRDTs), and distributed hash table (DHT) for multi-node coordination.

**Feature flag**: `distributed`

### Parallel MapReduce

```rust
use ai_assistant::distributed::{MapReduceJob, MapReduceConfig};

let config = MapReduceConfig::default();
let mut job = MapReduceJob::new(config);

// Word count example
job.set_input(vec![
    "hello world hello".to_string(),
    "world rust hello".to_string(),
]);

job.set_map_fn(|chunk| {
    chunk.split_whitespace()
        .map(|word| (word.to_string(), 1u64))
        .collect()
});

job.set_reduce_fn(|_key, values| {
    values.iter().sum::<u64>()
});

let results = job.execute()?;
// Results computed in parallel using rayon's work-stealing thread pool
for (word, count) in &results {
    println!("{}: {}", word, count);
}
```

### CRDTs (Conflict-Free Replicated Data Types)

```rust
use ai_assistant::distributed::{GCounter, PNCounter, LWWRegister, ORSet};

// GCounter: distributed increment-only counter
let mut counter_a = GCounter::new("node_a");
let mut counter_b = GCounter::new("node_b");
counter_a.increment(5);
counter_b.increment(3);
counter_a.merge(&counter_b);
assert_eq!(counter_a.value(), 8);

// PNCounter: increment and decrement
let mut pn = PNCounter::new("node_a");
pn.increment(10);
pn.decrement(3);
assert_eq!(pn.value(), 7);

// LWWRegister: last-writer-wins register
let mut reg = LWWRegister::new("initial_value".to_string());
reg.set("updated_value".to_string());
assert_eq!(reg.get(), "updated_value");

// ORSet: observed-remove set
let mut set = ORSet::new();
set.add("item_a", "node_1");
set.add("item_b", "node_1");
set.remove("item_a", "node_1");
assert!(set.contains("item_b"));
assert!(!set.contains("item_a"));
```

### DHT (Distributed Hash Table)

```rust
use ai_assistant::distributed::{DhtNode, NodeId};

let node_id = NodeId::random();
let mut dht = DhtNode::new(node_id);

// Store and retrieve values
dht.store("key_1", b"value_1".to_vec());
let value = dht.get("key_1");

// Find closest nodes to a key (Kademlia XOR distance)
let closest = dht.find_closest(&target_id, 3);
```

### Distributed Coordinator

```rust
use ai_assistant::distributed::DistributedCoordinator;

let mut coordinator = DistributedCoordinator::new("node_1");

// Combines DHT + MapReduce + CRDTs
coordinator.store("shared_counter", counter_data);
let result = coordinator.execute_map_reduce(job)?;
```

---

## 52. Distributed Networking

**What**: Real QUIC-based networking for multi-node clusters with mutual TLS, consistent hashing, failure detection, Merkle tree sync, replication, and LAN discovery.

**Feature flag**: `distributed-network` (separate from `distributed` — adds quinn, rustls, rcgen, sha2)

### Starting a Network Node

```rust
use ai_assistant::distributed_network::{NetworkNode, NetworkConfig, ReplicationConfig};
use std::path::PathBuf;

let config = NetworkConfig {
    listen_addr: "0.0.0.0:0".parse().unwrap(), // Auto-assigned port
    identity_dir: PathBuf::from("./node_identity"),
    ..NetworkConfig::default()
};

let node = NetworkNode::new(config).expect("Failed to create node");
println!("Node {} listening on {}", node.node_id(), node.local_addr());
```

### Connecting Nodes

```rust
// Node 2 connects to Node 1
let peer_id = node2.connect(node1.local_addr()).unwrap();
println!("Connected to peer: {}", peer_id);

// Store data with automatic replication
node1.store("key", b"value".to_vec()).unwrap();

// Retrieve from any node
let value = node2.get("key").unwrap();
```

### Consistent Hashing

```rust
use ai_assistant::ConsistentHashRing;
use ai_assistant::NodeId;

let mut ring = ConsistentHashRing::new(64, 3); // 64 vnodes, replication=3
ring.add_node(node_a);
ring.add_node(node_b);
ring.add_node(node_c);

// Get 3 distinct nodes responsible for a key
let nodes = ring.get_nodes("my_key", 3);
```

### Failure Detection

```rust
use ai_assistant::failure_detector::{PhiAccrualDetector, HeartbeatManager, HeartbeatConfig};

let config = HeartbeatConfig::default(); // phi_threshold: 8.0
let mut mgr = HeartbeatManager::new(config);

mgr.record_heartbeat(&peer_id);
match mgr.check_node(&peer_id) {
    NodeStatus::Alive => println!("Peer is healthy"),
    NodeStatus::Suspicious(phi) => println!("Peer may be failing (phi={})", phi),
    NodeStatus::Dead(phi) => println!("Peer is dead (phi={})", phi),
    NodeStatus::Unknown => println!("No heartbeats recorded"),
}
```

### Merkle Tree Sync

```rust
use ai_assistant::merkle_sync::{MerkleTree, AntiEntropySync};
use std::collections::BTreeMap;

let mut data = BTreeMap::new();
data.insert("key1".into(), b"value1".to_vec());
data.insert("key2".into(), b"value2".to_vec());

let tree = MerkleTree::from_data(&data);
let proof = tree.proof("key1").unwrap();
assert!(MerkleTree::verify_proof(tree.root_hash().as_ref().unwrap(), &proof));
```

### Node Security

```rust
use ai_assistant::node_security::{CertificateManager, JoinToken};

// Generate CA and node certificate
let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
let identity = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();

// Create a join token (valid 24 hours, max 5 uses)
let mut token = JoinToken::generate(24, Some(5));
assert!(token.consume()); // Use one admission
```

### LAN Discovery

```rust
use ai_assistant::distributed_network::{NetworkConfig, DiscoveryConfig};

let config = NetworkConfig {
    discovery: DiscoveryConfig {
        enable_broadcast: true,       // Enable UDP broadcast
        broadcast_port: 9876,         // LAN discovery port
        broadcast_interval: Duration::from_secs(10),
        enable_peer_exchange: true,   // Ask peers for their peers
    },
    ..NetworkConfig::default()
};

// Nodes on the same LAN will auto-discover each other
let node = NetworkNode::new(config).unwrap();
// Poll for discovery events
for event in node.poll_events() {
    match event {
        NetworkEvent::PeerConnected(id, addr) => {
            println!("Auto-discovered peer {} at {}", id, addr);
        }
        _ => {}
    }
}
```

### Join Tokens (Cluster Security)

```rust
// Admin node generates a join token (valid 24h, max 10 uses)
let token = node.generate_join_token(24, Some(10));
let token_string = token.encode(); // Share this with new nodes

// New node joins with the token
let config = NetworkConfig {
    join_token: Some(token_string),
    bootstrap_peers: vec![admin_addr],
    ..NetworkConfig::default()
};
let new_node = NetworkNode::new(config).unwrap();
new_node.connect(admin_addr).unwrap();
```

### Reputation & Probation

```rust
// Check peer reputation (0.0 = untrusted, 1.0 = fully trusted)
if let Some(rep) = node.peer_reputation(&peer_id) {
    println!("Peer reputation: {:.3}", rep);
}

// Check if peer is still in probation (new nodes start in probation)
if let Some(true) = node.peer_in_probation(&peer_id) {
    println!("Peer is new, still in probation period");
}

// Check how many peers need anti-entropy sync
let pending = node.sync_status();
println!("{} peers pending sync", pending);
```

### Network Events

```rust
// Poll for cluster events
for event in node.poll_events() {
    match event {
        NetworkEvent::PeerConnected(id, addr) => println!("Connected: {} at {}", id, addr),
        NetworkEvent::PeerDisconnected(id) => println!("Disconnected: {}", id),
        NetworkEvent::PeerFailed(id, phi) => println!("Failed: {} (phi={})", id, phi),
        NetworkEvent::ReplicationComplete(key, copies) => {
            println!("Key '{}' replicated to {} nodes", key, copies);
        }
        NetworkEvent::JoinRequestReceived(id, addr) => {
            println!("Join request from {} at {}", id, addr);
        }
        NetworkEvent::MessageReceived(id, msg) => println!("Message from {}", id),
        NetworkEvent::Error(e) => eprintln!("Error: {}", e),
    }
}
```

### Distributed Vector Search

```rust
use ai_assistant::vector_db::{InMemoryVectorDb, VectorDbConfig, DistributedVectorDb};

let local_db = InMemoryVectorDb::new(VectorDbConfig::default());
let distributed_db = DistributedVectorDb::new(local_db, network_node.clone());

// Search across ALL cluster nodes
let results = distributed_db.distributed_search(&query_vector, 10).unwrap();
```

---

## Architecture Summary

```
User Code
    |
    v
AiAssistant (main entry point)
    |
    +-- Config (provider URLs, model preferences)
    +-- Session (conversation history, persistence)
    +-- Context (token tracking, summarization)
    +-- Preferences (learned user preferences)
    |
    +-- Providers (Ollama, LM Studio, etc.)
    |       |
    |       v
    |   HTTP requests to local LLM servers
    |
    +-- RAG (knowledge base, conversation storage)
    +-- Memory (long-term facts, decay)
    +-- Security (rate limit, sanitize, audit)
    +-- Analysis (sentiment, topics, quality, entities, facts)
    +-- Tools (function calling, registry)
    +-- Decision Trees (conditional workflows, async execution)
    +-- Agent (ReAct multi-step execution)
    +-- Plugins (extensibility)
    +-- Persistence (backup, export, compaction)
    +-- Cache (response cache, embedding cache)
    +-- Metrics (streaming, latency, cost)
    +-- Error Handling (zero .unwrap() — lock poison recovery, NaN-safe sorting)
    +-- Vision (multimodal, image preprocessing)
    +-- Structured Output (JSON schema, validation)
    +-- Adaptive Thinking (depth classification, CoT injection, thinking tags)
    +-- Fallback (automatic provider failover)
    +-- Retry (exponential backoff, circuit breaker)
    +-- Compaction (lightweight conversation compaction)
    +-- API Key Rotation (round-robin, rate-limit aware)
    +-- Internal Storage (bincode+gzip, auto-detect legacy JSON)
    +-- Log Redaction (safe_log!, pattern-based redaction)
    +-- HTTP Client (testable trait, mock support)
    +-- Vector DB (InMemory, LanceDB, Qdrant — pluggable backends)
    +-- Distributed (MapReduce, CRDTs, DHT, coordinator)
    +-- Distributed Network (QUIC transport, consistent hashing, failure detection,
    |       Merkle sync, replication, mutual TLS, LAN discovery)
    +-- Distributed VectorDb (fan-out search, vector replication across nodes)
    +-- Autonomous Agents (self-directed execution, 5 autonomy levels, task boards,
    |       scheduling, browser automation, environment detection, distributed tasks)
```

---

## Quick Start

```rust
use ai_assistant::{AiAssistant, AiResponse};

fn main() {
    let mut assistant = AiAssistant::new();

    // Discover providers
    assistant.fetch_models();
    while assistant.is_fetching_models {
        std::thread::sleep(std::time::Duration::from_millis(100));
        assistant.poll_models();
    }

    // Select first model
    if let Some(model) = assistant.available_models.first() {
        assistant.config.selected_model = model.name.clone();
        assistant.config.provider = model.provider.clone();
    }

    // Send message
    assistant.send_message("Hello!".to_string(), "");

    // Poll streaming response
    loop {
        if let Some(response) = assistant.poll_response() {
            match response {
                AiResponse::Chunk(text) => print!("{}", text),
                AiResponse::Complete(_) => break,
                AiResponse::Error(e) => { eprintln!("{}", e); break; }
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
```

---

## 53. Autonomous Agents

**What**: A complete autonomous agent system that allows AI agents to operate independently, make decisions, execute tools, manage tasks, and collaborate — with configurable safety policies and human oversight.

**Feature flags**: `autonomous`, `scheduler`, `butler`, `browser`, `distributed-agents`. None are included in `full` — opt-in only.

### Architecture Overview

The autonomous system consists of 14 modules organized in layers:

| Layer | Modules | Purpose |
|-------|---------|---------|
| Core autonomy | `autonomous_loop`, `mode_manager`, `agent_sandbox` | Agent execution loop, autonomy levels, tool validation |
| User interaction | `user_interaction`, `interactive_commands` | Human-in-the-loop, natural language command parsing |
| Task management | `task_board` | Kanban-style task tracking with priorities |
| Profiles & policies | `agent_profiles`, `agent_policy` | Pre-configured agent behaviors and safety constraints |
| Scheduling | `scheduler`, `trigger_system` | Cron-based scheduling, event-driven triggers |
| Environment | `butler`, `os_tools`, `browser_tools` | Auto-detection, OS operations, browser automation |
| Distribution | `distributed_agents` | Multi-node task distribution |

### Autonomy Levels

Five levels of agent autonomy, from most restricted to most free:

```rust
use ai_assistant::mode_manager::OperationMode;

// Chat: Respond only, no tools
// Assistant: Use pre-approved tools
// Programming: Read/write files, run code
// AssemblyLine: Execute multi-step plans
// Autonomous: Full self-direction
```

### Agent Policies

Safety constraints that limit what an agent can do:

```rust
use ai_assistant::agent_policy::{AgentPolicyBuilder, InternetMode, AutonomyLevel, RiskLevel};

let policy = AgentPolicyBuilder::new()
    .autonomy(AutonomyLevel::Standard)
    .internet(InternetMode::AllowList(vec!["api.github.com".into()]))
    .require_approval_above(RiskLevel::Medium)
    .allowed_commands(vec!["cargo".into(), "git".into()])
    .max_cost_usd(5.0)
    .max_iterations(100)
    .max_runtime_secs(3600)
    .build();
```

### Agent Profiles

Pre-configured agent personalities with default policies:

```rust
use ai_assistant::agent_profiles::ProfileRegistry;

let registry = ProfileRegistry::with_defaults();

// Built-in profiles: coding-assistant, research-agent, devops-agent, paranoid
let profile = registry.get_agent_profile("coding-assistant").unwrap();
println!("Mode: {:?}", profile.mode);       // Programming
println!("Internet: {:?}", profile.policy.internet); // AllowList
```

### Autonomous Loop

The core execution loop that drives agent behavior:

```rust
use ai_assistant::autonomous_loop::{AutonomousAgent, AutonomousAgentConfig, CostConfig};
use std::collections::HashMap;

let config = AutonomousAgentConfig::builder()
    .task("Analyze the project structure and suggest improvements")
    .max_iterations(50)
    .with_cost_config(CostConfig {
        default_cost_per_call: 0.001,
        tool_costs: HashMap::from([("browser_navigate".into(), 0.01)]),
        cost_callback: None,
    })
    .build();
```

The loop supports multi-format tool call parsing:
- **JSON array**: `[{"name":"tool","arguments":{"key":"val"}}]`
- **OpenAI-style**: `{"tool_calls":[{"function":{"name":"x","arguments":"{...}"}}]}`
- **XML tool_use**: `<tool_use><name>x</name><arguments>{"key":"val"}</arguments></tool_use>`

### Task Board

Kanban-style task management for agents:

```rust
use ai_assistant::task_board::{TaskBoard, BoardCommand, StepPriority};

let mut board = TaskBoard::new("Sprint 1");
board.execute(BoardCommand::AddTask {
    title: "Implement authentication".into(),
    description: "Add JWT-based auth to the API".into(),
    priority: StepPriority::High,
});

// Undo the last command
match board.undo_last() {
    Ok(msg) => println!("Undone: {}", msg),
    Err(e) => println!("Cannot undo: {}", e),
}

// Undo via interactive commands (English or Spanish)
// > undo
// > deshacer
// Supports reversing: AddTask, StartTask, PauseTask, ResumeTask, CancelTask, CompleteTask
```

### Scheduling & Triggers

Run agents on schedules or in response to events:

```rust
use ai_assistant::scheduler::{Scheduler, ScheduledJob, CronSchedule, ScheduledAction};

let mut scheduler = Scheduler::new();
let job = ScheduledJob::new(
    "Daily backup",
    CronSchedule::parse("0 2 * * *").unwrap(), // 2 AM daily
    ScheduledAction::RunShell { command: "backup.sh".into() },
);
scheduler.add_job(job);
```

### Butler (Environment Detection)

Auto-detects local resources with real connectivity checks:

```rust
use ai_assistant::butler::Butler;

let butler = Butler::new();
let env = butler.scan();
// Checks: Ollama (HTTP), LM Studio (HTTP), GPU (nvidia-smi),
// Chrome (path check), Docker (docker info), Network (HTTP HEAD)
for (name, result) in &env.detections {
    println!("{}: detected={}", name, result.detected);
}
```

### Browser Automation (Real CDP)

Real Chrome DevTools Protocol communication via WebSocket:

```rust
use ai_assistant::browser_tools::BrowserSession;

let mut session = BrowserSession::new();
session.launch(9222)?;  // Launches Chrome headless
let page = session.navigate("https://example.com")?;
println!("Title: {}", page.title);
let text = session.get_text("h1")?;
session.click("#submit")?;
session.type_text("#search", "query")?;
let screenshot_b64 = session.screenshot()?;
session.close()?;
```

### Multi-Agent Sessions

Multiple agents collaborating on a shared task:

```rust
use ai_assistant::AiAssistant;

let mut assistant = AiAssistant::default();
let session = assistant.create_multi_agent_session("Build a web app");
session.add_agent("architect", "design-agent");
session.add_agent("developer", "coding-assistant");
session.add_agent("reviewer", "paranoid");
```

### Distributed Agents

Task distribution across multiple nodes (requires `distributed-agents` feature):

```rust
use ai_assistant::distributed_agents::{DistributedAgentManager, NodeId};

let node_id = NodeId::from_string("worker-1");
let mut manager = DistributedAgentManager::new(node_id);
let task_id = manager.submit_task("Process dataset", "data-worker", 5);
```

---

## 54. Knowledge Graphs

**What**: Extracts entities and relations from documents using LLM-based analysis and stores them in a SQLite-backed graph, enabling graph-traversal-enhanced retrieval (Graph RAG). Entities span organizations, products, persons, locations, concepts, and events, connected by typed relations such as `manufactures` or `located_in`.

### Building and Indexing

Open or create a database, then index documents with an `LlmEntityExtractor`. The extractor wraps any function that accepts `(system_prompt, user_prompt)` and returns a string response from your LLM.

```rust
use ai_assistant::knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, LlmEntityExtractor, EntityType,
};

let config = KnowledgeGraphConfig {
    max_traversal_depth: 2,
    min_relation_confidence: 0.6,
    ..KnowledgeGraphConfig::default()
};

let mut graph = KnowledgeGraph::open("knowledge.db", config)?;

let extractor = LlmEntityExtractor::new(|system, user| {
    my_llm_call(system, user) // returns Result<String>
})
.with_entity_types(vec![EntityType::Organization, EntityType::Product]);

graph.index_document("doc_ships_01", "Aegis Dynamics manufactures the Sabre fighter.", &extractor)?;

let stats = graph.stats()?;
println!("Entities: {}, Relations: {}", stats.total_entities, stats.total_relations);
```

### Querying and Graph RAG Integration

```rust
let result = graph.query("What ships does Aegis make?", &extractor)?;
for chunk in &result.chunks {
    println!("[{}] {}", chunk.source_doc, chunk.content);
}

// Integrate with RagPipeline
let callback = graph.as_graph_callback(&extractor);
// pass callback to RagPipeline::with_graph_callback(...)
```

**Feature flag**: `rag`

---

## 55. Multi-Layer Knowledge Graphs

**What**: Organises graph data into four priority-ranked layers -- `Knowledge` (verified packs), `User` (stated beliefs), `Internet` (web-sourced, with TTL), and `Session` (per-conversation context). The coordinator automatically extracts user beliefs from messages and detects contradictions between layers.

### Session and User Graphs

```rust
use ai_assistant::multi_layer_graph::{MultiLayerGraph, BeliefExtractor};

let mut mlg = MultiLayerGraph::new();

let entities = vec!["Sabre".to_string(), "Aegis".to_string()];
mlg.process_user_message("session-42", "I own a Sabre and I think it's the best fighter.", &entities);

for belief in &mlg.user_graph.beliefs {
    println!("[{:?}] {}", belief.belief_type, belief.statement);
}

let session = mlg.get_or_create_session("session-42");
println!("Session entities: {:?}", session.entity_names());
```

### Contradiction Detection and Persistence

```rust
use ai_assistant::multi_layer_graph::MultiLayerGraph;
use std::path::PathBuf;

let mut mlg = MultiLayerGraph::with_persistence(
    PathBuf::from("user_graph.json"),
    PathBuf::from("internet_graph.json"),
    PathBuf::from("contradictions.json"),
);

if let Some(conflict) = mlg.add_internet_data(
    "Sabre", "manufacturer", "Aegis",
    "https://example.com/sabre",
    Some("Aegis Dynamics"),
) {
    println!("Conflict on '{}': {} vs {}", conflict.attribute,
             conflict.primary_value, conflict.conflicting_value);
}
```

---

## 56. Document Parsing

**What**: Extracts plain text, metadata, and structured sections from EPUB, DOCX, ODT, PDF, HTML, and plain-text files. HTML and plain text are always available; the richer binary formats require the `documents` feature flag.

### Parsing Files

```rust
use ai_assistant::document_parsing::{DocumentParser, DocumentParserConfig};
use std::path::Path;

let config = DocumentParserConfig {
    extract_sections: true,
    extract_metadata: true,
    extract_tables: false,
    ..DocumentParserConfig::default()
};

let parser = DocumentParser::new(config);
let doc = parser.parse_file(Path::new("report.pdf"))?;

println!("Title: {:?}", doc.metadata.title);
println!("Words: {}", doc.word_count);
for title in doc.section_titles() {
    println!("  Section: {}", title);
}
```

### Parsing Raw Strings and Bytes

```rust
use ai_assistant::document_parsing::{DocumentParser, DocumentParserConfig, DocumentFormat};

let parser = DocumentParser::new(DocumentParserConfig::default());

let doc = parser.parse_string(
    "<h1>Introduction</h1><p>Hello world.</p>",
    DocumentFormat::Html,
)?;
println!("{}", doc.text);

let epub_bytes = std::fs::read("book.epub")?;
let doc = parser.parse_bytes(&epub_bytes, DocumentFormat::Epub)?;
println!("Sections: {}", doc.sections.len());
```

**Feature flag**: `documents` (required for EPUB, DOCX, ODT, PDF)

---

## 57. Feed Monitor

**What**: Parses RSS 2.0 and Atom feeds using regex-based XML extraction and tracks new entries across repeated checks. `FeedMonitor` maintains stateful tracking so only genuinely new entries are reported on subsequent polls.

### Parsing a Feed Directly

```rust
use ai_assistant::feed_monitor::{FeedParser, FeedFormat};

let xml = reqwest::blocking::get("https://example.com/rss.xml")?.text()?;
let feed = FeedParser::parse(&xml)?;

println!("Feed: {} ({} entries)", feed.metadata.title, feed.entry_count);
for entry in &feed.entries {
    println!("  [{}] {}", entry.published.map(|d| d.to_string()).unwrap_or_default(), entry.title);
}
```

### Monitoring Multiple Feeds

```rust
use ai_assistant::feed_monitor::{FeedMonitor, FeedMonitorConfig};
use std::collections::HashMap;

let mut feeds = HashMap::new();
feeds.insert("rust-blog".to_string(), "https://blog.rust-lang.org/feed.xml".to_string());

let config = FeedMonitorConfig {
    feeds,
    check_interval_secs: 1800,
    max_entries_per_feed: 50,
    ..FeedMonitorConfig::default()
};

let mut monitor = FeedMonitor::new(config);

for result in monitor.check_all() {
    if result.success {
        println!("{}: {} new entries", result.feed_name, result.new_entries.len());
    }
}
```

---

## 58. Content Versioning

**What**: Tracks arbitrary string content over time by storing `ContentSnapshot` instances and computing line-based diffs. An optional similarity threshold prevents storing versions that differ too little.

### Storing Versions

```rust
use ai_assistant::content_versioning::{ContentVersionStore, VersioningConfig};

let config = VersioningConfig {
    max_versions: 20,
    auto_diff: true,
    change_threshold: 0.05,
    store_content: true,
};

let mut store = ContentVersionStore::new(config);

let v1 = store.add_version("page/home", "Hello world.");
let v2 = store.add_version("page/home", "Hello world. New paragraph added.");
println!("v2: {:?}", v2); // Some(...) because content changed
```

### Diffing Versions

```rust
let diff = store.diff_latest("page/home")?;
println!("{}", diff.summary());
print!("{}", diff.to_unified_diff());

if let Some(history) = store.history("page/home") {
    println!("Versions stored: {}", history.version_count());
}
```

**Feature flag**: `rag` (for the SQLite-backed store variant)

---

## 59. P2P Networking

**What**: Peer-to-peer networking layer for distributed AI assistant instances. Provides NAT traversal via STUN/UPnP/NAT-PMP, peer discovery via bootstrap nodes, a trust-based reputation system, and knowledge sharing between peers.

**Feature flag**: `p2p` (requires `distributed`)

### Configuration and Startup

```rust
use ai_assistant::p2p::{P2PConfig, P2PManager, PeerDataTrust};

let config = P2PConfig {
    enabled: true,
    peer_data_trust: PeerDataTrust::ConsensusRequired(3),
    bootstrap_nodes: vec!["203.0.113.1:12345".to_string()],
    enable_mdns: true,
    max_peers: 50,
    min_reputation: 0.4,
    ..P2PConfig::default()
};

let mut manager = P2PManager::new(config);
manager.start()?;

println!("Local peer ID: {}", manager.local_peer_id());
println!("Connected peers: {}", manager.peer_count());

manager.stop();
```

### NAT Traversal

```rust
use ai_assistant::p2p::{NatTraversal, P2PConfig};

let mut nat = NatTraversal::new(P2PConfig::default());

let result = nat.discover_nat()?;
println!("NAT type: {:?}", result.nat_type);

if let Ok(port) = nat.try_upnp_mapping(12345, 12345) {
    println!("UPnP mapped external port {}", port);
}

if let Some(addr) = nat.get_connectable_address() {
    println!("Share this address with peers: {}", addr);
}
```

### ICE Connectivity and Reputation

```rust
use ai_assistant::p2p::{IceAgent, IceCandidate, ReputationSystem};
use std::net::SocketAddr;

let mut ice = IceAgent::new();
let local_addr: SocketAddr = "0.0.0.0:12345".parse().unwrap();
ice.add_local_candidate(IceCandidate::host(local_addr));
ice.start_checks();

let mut rep = ReputationSystem::new(0.3);
let peer = rep.get_or_create("peer_abc");
peer.record_success();
peer.record_correct_contribution();
println!("Score: {:.2}, trusted: {}", peer.score, peer.is_trusted(0.3));

for top_peer in rep.get_top_peers(5) {
    println!("{}: {:.2}", top_peer.peer_id, top_peer.score);
}
```

---

## 60. WebSocket Streaming

**What**: RFC 6455 WebSocket implementation for real-time bidirectional AI communication. Provides low-level frame encoding/decoding, a handshake builder, and a high-level stream handler that reassembles streaming chunks.

### Frame Construction and Handshake

```rust
use ai_assistant::websocket_streaming::{WsFrame, WsHandshake, WsOpcode};

let handshake = WsHandshake::new("localhost:8090", "/ws")
    .with_protocol("ai-assistant")
    .with_header("Authorization", "Bearer mytoken");

let http_request = handshake.build_request();
let expected_accept = handshake.expected_accept();

let text_frame = WsFrame::text("hello");
let ping_frame = WsFrame::ping(b"keepalive");
let close_frame = WsFrame::close(1000, "normal closure");

let encoded = text_frame.encode(); // ready to write to TcpStream
```

### Stream Handler with Callbacks

```rust
use ai_assistant::websocket_streaming::WsStreamHandler;

let mut handler = WsStreamHandler::new();
handler.set_open();

handler.set_on_chunk(|id, content| {
    print!("{}", content);
});
handler.set_on_complete(|id, full_text| {
    println!("\n[{}] complete ({} chars)", id, full_text.len());
});

handler.handle_message(r#"{"type":"stream_start","id":"msg_1","model":"llama3"}"#)?;
handler.handle_message(r#"{"type":"stream_chunk","id":"msg_1","content":"Hello"}"#)?;
handler.handle_message(r#"{"type":"stream_end","id":"msg_1"}"#)?;
```

---

## 61. MCP Protocol

**What**: Implementation of Anthropic's Model Context Protocol (MCP, version 2024-11-05) over JSON-RPC 2.0. Provides both a server side (register tools, resources, and prompts) and a client side (call tools, read resources, retrieve prompts from remote MCP servers).

### MCP Server

```rust
use ai_assistant::mcp_protocol::{McpServer, McpTool, McpResource, McpResourceContent};

let mut server = McpServer::new("my-assistant", "1.0.0");

let tool = McpTool::new("search", "Search the knowledge base")
    .with_property("query", "string", "The search query", true)
    .with_property("limit", "integer", "Max results", false);

server.register_tool(tool, |args| {
    let query = args["query"].as_str().unwrap_or("");
    Ok(serde_json::json!({ "results": [format!("Result for: {}", query)] }))
});

let request_json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
let response_json = server.handle_message(request_json);
```

### MCP Client

```rust
use ai_assistant::mcp_protocol::McpClient;

let mut client = McpClient::new("http://localhost:9000/mcp");
let capabilities = client.initialize()?;

let tools = client.list_tools()?;
for tool in &tools {
    println!("Tool: {} - {}", tool.name, tool.description);
}

let result = client.call_tool("search", serde_json::json!({
    "query": "Rust ownership"
}))?;
```

---

## 62. HTTP API Server

**What**: Lightweight embedded HTTP server that exposes an `AiAssistant` instance as a REST API using only `std::net::TcpListener`. Endpoints: `GET /health`, `GET /models`, `POST /chat`, `GET /config`, `POST /config`.

### Blocking Server

```rust
use ai_assistant::server::{AiServer, ServerConfig};

let config = ServerConfig {
    host: "127.0.0.1".to_string(),
    port: 8090,
    max_body_size: 1_048_576,
    read_timeout_secs: 30,
};

let server = AiServer::new(config);
server.run_blocking()?;
```

### Background Server

```rust
use ai_assistant::server::{AiServer, ServerConfig};
use ai_assistant::{AiAssistant, AiConfig};

let assistant = AiAssistant::with_config(AiConfig::default());
let server = AiServer::with_assistant(ServerConfig::default(), assistant);
let handle = server.start_background()?;

println!("Server running at {}", handle.url());
// POST http://<addr>/chat  {"message": "Hello"}
```

---

## 63. Event System

**What**: A synchronous, multi-handler event bus covering the full `AiAssistant` lifecycle. Events span seven categories: `response`, `provider`, `session`, `context`, `model`, `rag`, and `tool`.

### Registering Handlers

```rust
use ai_assistant::events::{EventBus, AiEvent, EventHandler};
use std::sync::Arc;

let mut bus = EventBus::with_history(200);

bus.on(|event| {
    println!("[{}] {}", event.category(), event.name());
});

bus.emit(AiEvent::ProviderAttempt {
    provider: "ollama".to_string(),
    model: "llama3".to_string(),
});
bus.emit(AiEvent::ResponseComplete { response_length: 512 });

println!("Total provider events: {}", bus.count_events("provider"));
```

### Built-in Handlers

```rust
use ai_assistant::events::{EventBus, LoggingHandler, LogLevel, CollectingHandler};
use std::sync::Arc;

let mut bus = EventBus::new();
bus.add_handler(Arc::new(LoggingHandler::new(LogLevel::Info)));

let collector = CollectingHandler::new();
bus.add_handler(Arc::new(collector));
```

### History and Replay

```rust
use ai_assistant::events::{EventBus, AiEvent};

let bus = EventBus::with_history(500);
// ... emit events ...

let history = bus.history();
for entry in &history {
    println!("[{}ms] {}", entry.timestamp_ms, entry.event.name());
}
bus.clear_history();
```

---

## 64. Content Encryption

**What**: Encrypt and decrypt conversation content using symmetric key encryption. Supports `Aes256Gcm`, `ChaCha20Poly1305`, and a simple `Xor` fallback. With the `rag` feature, AES-256-GCM uses real authenticated encryption.

### Creating Keys and Encrypting Content

```rust
use ai_assistant::content_encryption::{
    ContentEncryptor, EncryptionKey, EncryptionAlgorithm,
};

let mut encryptor = ContentEncryptor::new();

let key = EncryptionKey::new("key-1", vec![0u8; 32], EncryptionAlgorithm::Aes256Gcm);
encryptor.add_key(key);

let encrypted = encryptor.encrypt_string("Hello, secret world!").unwrap();
let plaintext = encryptor.decrypt_string(&encrypted).unwrap();
assert_eq!(plaintext, "Hello, secret world!");
```

### Key Rotation

```rust
use ai_assistant::content_encryption::{ContentEncryptor, EncryptionKey, EncryptionAlgorithm};

let mut encryptor = ContentEncryptor::new();

let key_a = EncryptionKey::new("key-a", vec![1u8; 32], EncryptionAlgorithm::Aes256Gcm);
let key_b = EncryptionKey::new("key-b", vec![2u8; 32], EncryptionAlgorithm::Aes256Gcm);

encryptor.add_key(key_a);
encryptor.add_key(key_b);
encryptor.set_active_key("key-b").unwrap();
// Old ciphertexts encrypted with key-a still decrypt correctly
```

**Feature flag**: `rag` (enables real AES-256-GCM authenticated encryption)

---

## 65. Access Control

**What**: Role-based access control (RBAC) for resources. Supports role inheritance, explicit deny rules, conditional access (MFA, IP ranges, usage caps), and fine-grained per-resource-instance entries.

### Defining Roles and Checking Permissions

```rust
use ai_assistant::access_control::{AccessControlManager, Permission, ResourceType};

let mut acl = AccessControlManager::new();

acl.assign_role("alice", "editor");
acl.assign_role("bob", "viewer");

let result = acl.check_permission(
    "alice",
    ResourceType::Conversation,
    Permission::Write,
    None,
);
```

### Custom Roles with Inheritance

```rust
use ai_assistant::access_control::{AccessControlManager, Role, Permission, ResourceType};

let mut acl = AccessControlManager::new();

let analyst = Role::new("analyst")
    .inherits_from("viewer")
    .with_permission(ResourceType::Memory, Permission::Read)
    .with_permission(ResourceType::Model, Permission::Execute);

acl.add_role(analyst);
acl.assign_role("carol", "analyst");
```

### Per-Resource Entries and Deny Rules

```rust
use ai_assistant::access_control::{
    AccessControlManager, AccessControlEntry, AccessCondition,
    Permission, ResourceType,
};

let mut acl = AccessControlManager::new();

let entry = AccessControlEntry::new("dave", ResourceType::Conversation)
    .with_permissions(&[Permission::Read, Permission::Write])
    .for_resource("conv-42")
    .with_condition(AccessCondition::RequiresMfa);

acl.add_entry(entry);
acl.deny("dave", ResourceType::Conversation, Permission::Delete);
```

---

## 66. Request Signing

**What**: Sign outgoing requests with a shared secret and verify incoming requests to confirm authenticity. Uses a timestamp-and-nonce scheme with constant-time comparison.

### Signing and Verifying

```rust
use ai_assistant::request_signing::{RequestSigner, SignatureAlgorithm, SignatureError};

let signer = RequestSigner::new(b"my-shared-secret", SignatureAlgorithm::HmacSha256);

let signed = signer.sign(r#"{"model":"llama3","prompt":"Hello"}"#);
println!("Signature: {}", signed.signature);

match signer.verify(&signed, 60) {
    Ok(()) => println!("Request is authentic"),
    Err(SignatureError::Expired) => eprintln!("Replay attempt"),
    Err(SignatureError::Invalid) => eprintln!("Signature mismatch"),
    Err(SignatureError::MissingFields) => eprintln!("Incomplete request"),
}
```

---

## 67. Request Queue

**What**: A thread-safe priority queue for managing concurrent AI generation requests. Requests are dequeued in `High > Normal > Low` order with FIFO within the same level.

### Basic Enqueue and Dequeue

```rust
use ai_assistant::request_queue::{RequestQueue, QueuedRequest, RequestPriority};

let queue = RequestQueue::new(100);

queue.enqueue(QueuedRequest::new("Summarize this").with_priority(RequestPriority::Low));
queue.enqueue(QueuedRequest::new("Cancel operation").with_priority(RequestPriority::High));

let first = queue.try_dequeue().unwrap();
assert_eq!(first.priority, RequestPriority::High);
```

### Monitoring Queue Health

```rust
let stats = queue.stats();
println!("Pending: {}, Processed: {}, Dropped: {}",
    stats.pending, stats.total_processed, stats.total_dropped);
```

---

## 68. Webhooks

**What**: Send HTTP event notifications to external endpoints. Supports per-webhook event filtering, shared-secret signing, custom headers, configurable timeouts, and automatic retry.

### Registering and Sending

```rust
use ai_assistant::webhooks::{WebhookManager, WebhookConfig, WebhookEvent, WebhookPayload};

let mut manager = WebhookManager::new();

let config = WebhookConfig::new("https://example.com/hooks/ai")
    .with_events(vec![WebhookEvent::MessageReceived, WebhookEvent::ErrorOccurred])
    .with_secret("my-webhook-secret")
    .with_header("Authorization", "Bearer token123");

manager.register(config);

let payload = WebhookPayload::new(
    WebhookEvent::MessageReceived,
    serde_json::json!({"session_id": "sess-1", "message": "Hello"}),
);

let results = manager.send(payload);
for result in &results {
    println!("Success: {}, attempts: {}", result.success, result.attempts);
}

let stats = manager.stats();
println!("Success rate: {:.1}%", stats.success_rate * 100.0);
```

---

## 69. Model Ensemble

**What**: Combines outputs from multiple AI models using configurable strategies (voting, weighted averaging, best-of-N, cascade) to improve accuracy and reliability.

### Building an Ensemble

```rust
use ai_assistant::model_ensemble::{Ensemble, EnsembleConfig, EnsembleModel, EnsembleStrategy};

let config = EnsembleConfig {
    strategy: EnsembleStrategy::BestOfN,
    model_timeout: std::time::Duration::from_secs(20),
    min_models: 2,
    parallel: true,
    models: vec![
        EnsembleModel::new("llama3:8b", "ollama").with_weight(1.0),
        EnsembleModel::new("mistral:7b", "ollama").with_weight(0.9),
    ],
};

let ensemble = Ensemble::new(config);

let result = ensemble.execute("Explain backpropagation", |prompt, model_id, provider| {
    Ok(format!("[{provider}/{model_id}] response to: {prompt}"))
});

println!("Winner: {:?}", result.winning_model);
println!("Agreement: {:.0}%", result.agreement * 100.0);
```

### Available Strategies

| Strategy | Description |
|---|---|
| `Voting` | Majority vote across model responses |
| `WeightedVoting` | Vote weighted by `EnsembleModel::weight` |
| `BestOfN` | Pick highest-scoring response |
| `Cascade` | Try models in order, stop on first success |
| `Routing` | Route query to the most specialised model |

---

## 70. Prompt Optimizer

**What**: A/B tests prompt templates and learns from feedback to automatically select the most effective variant, with configurable exploration vs. exploitation balance.

### Registering Variants and Selecting

```rust
use ai_assistant::prompt_optimizer::{PromptOptimizer, OptimizerConfig, Feedback};

let mut optimizer = PromptOptimizer::new(OptimizerConfig::default());

optimizer.add_variant("concise",  "Answer briefly: {query}");
optimizer.add_variant("detailed", "Provide a thorough answer to: {query}. Include examples.");
optimizer.add_variant("cot",      "Think step by step, then answer: {query}");

if let Some(variant) = optimizer.select_best("What is attention in transformers?") {
    let prompt = variant.apply_query("What is attention in transformers?");
    println!("Using '{}': {}", variant.name, prompt);
}

// Record feedback to close the loop
let id = "concise".to_string();
optimizer.record_feedback(&id, Feedback {
    success: true,
    quality_score: Some(0.85),
    response_time_ms: Some(1200),
    token_count: Some(148),
});
```

---

## 71. Quantization

**What**: Detects quantization formats (GGUF, GPTQ, AWQ, EXL2), calculates VRAM/RAM requirements, and recommends the optimal format for a given hardware profile.

### Detecting a Format

```rust
use ai_assistant::quantization::{QuantizationDetector, QuantFormat};

let detector = QuantizationDetector::new();

let format = detector.detect_format("llama-3-13b-q4_k_m.gguf");
if let Some(fmt) = format {
    println!("Format: {}, Bits: {}", fmt.name(), fmt.bits_per_weight());
    println!("Quality: {:.1}%", fmt.quality_retention() * 100.0);

    let mem = detector.estimate_memory("13B", &fmt);
    println!("Total: {:.1} GB, CPU ok: {}", mem.total_gb, mem.cpu_compatible);
}
```

### Hardware-Aware Recommendations

```rust
use ai_assistant::quantization::{QuantizationDetector, HardwareProfile};

let detector = QuantizationDetector::new();

let hw = HardwareProfile::nvidia(8.0, 32.0); // 8 GB VRAM, 32 GB RAM
let rec = detector.recommend_quantization("13B", &hw);
println!("Recommended: {} (confidence: {:.0}%)", rec.format.name(), rec.confidence * 100.0);
println!("Reason: {}", rec.reason);
```

---

## 72. WASM Support

**What**: Platform detection, capability queries, and cross-platform utilities (time, HTTP) that compile for both native and `wasm32` targets.

**Feature flag**: `wasm` (enables `js_sys` integration on `wasm32`)

### Platform Detection

```rust
use ai_assistant::wasm::{is_wasm, PlatformCapabilities, Capability};

if is_wasm() {
    println!("Running in WebAssembly");
}

let caps = PlatformCapabilities::current();
if !caps.has(Capability::Filesystem) {
    println!("No filesystem - use in-memory storage");
}
if caps.has_all(&[Capability::Network, Capability::SystemTime]) {
    println!("Can make timed network requests");
}

println!("filesystem: {}, threads: {}, sqlite: {}", caps.filesystem, caps.threads, caps.sqlite);
```

---

## 73. OpenAPI Export

**What**: Builds OpenAPI 3.0 specifications programmatically using a fluent builder API and exports them as JSON or YAML.

### Building a Spec

```rust
use ai_assistant::openapi_export::{
    OpenApiBuilder, OperationBuilder, OpenApiPathItem, JsonSchema,
};

let spec = OpenApiBuilder::new()
    .title("My AI API")
    .description("Local LLM gateway")
    .version("1.0.0")
    .server("http://localhost:8080", Some("Local development"))
    .tag("chat", Some("Chat completion endpoints"))
    .schema("ChatRequest",
        JsonSchema::object()
            .with_property("model",  JsonSchema::string())
            .with_property("prompt", JsonSchema::string())
            .with_required(vec!["model", "prompt"]),
    )
    .path("/v1/chat", OpenApiPathItem {
        post: Some(
            OperationBuilder::new()
                .summary("Chat completion")
                .tag("chat")
                .operation_id("chatComplete")
                .build()
        ),
        ..Default::default()
    })
    .build();
```

### Exporting

```rust
use ai_assistant::openapi_export::{export_to_json, export_to_yaml, generate_ai_assistant_spec};

let spec = generate_ai_assistant_spec();
let json = export_to_json(&spec).expect("serialization failed");
let yaml = export_to_yaml(&spec).expect("serialization failed");
println!("Exported {} paths", spec.paths.len());
```
