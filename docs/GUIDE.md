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
    +-- Vision (multimodal, image preprocessing)
    +-- Structured Output (JSON schema, validation)
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
