<!-- Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander) -->

> **This software is licensed under [PolyForm Noncommercial 1.0.0](LICENSE). For commercial use, please contact [orlando.luque@gmail.com](mailto:orlando.luque@gmail.com).**

# ai_assistant

A comprehensive Rust library for LLM integration — local and cloud. 13+ providers, 5-tier RAG, multi-agent orchestration, autonomous agents, distributed computing, and 40+ MCP tools. Single crate, zero-runtime, ~353K LOC.

![Rust](https://img.shields.io/badge/Rust-Edition%202021-orange)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)
![Tests](https://img.shields.io/badge/tests-6696-brightgreen)
![LOC](https://img.shields.io/badge/LOC-353K-blue)

> **Status: Experimental**
>
> This project compiles and passes 6,696 tests, but it has not been validated in production
> or real-world usage. Developed iteratively with AI assistance, some modules may be
> incomplete, loosely wired, or tested only at a surface level. A comprehensive review is
> pending. **Use at your own risk.**

## Quick Start

### Option A: Download Pre-Built Binaries (No Rust Needed)

Download the latest release from [**GitHub Releases**](https://github.com/OrlandoLuque/ai_assistant/releases), extract, and run:

```bash
ai_assistant_cli.exe
```

### Option B: Install from crates.io

```bash
cargo install ai_assistant --bin ai_assistant_cli --features "full,butler"
```

### Option C: Clone & Build

```bash
git clone https://github.com/OrlandoLuque/ai_assistant.git
cd ai_assistant
cargo build --release --bin ai_assistant_cli --features "full,butler"
```

> **Full guide**: See [**docs/GETTING_STARTED.md**](docs/GETTING_STARTED.md) for detailed instructions, server setup, code snippets in multiple languages, and more.

## Included Binaries

| Binary | Description |
|--------|-------------|
| `ai_assistant_cli` | Interactive REPL with auto-detection of local LLMs |
| `ai_gui` | Desktop GUI with chat and .kpkg support (WIP) |
| `ai_assistant_server` | HTTP API server — OpenAI-compatible drop-in replacement |
| `kpkg_tool` | Create, inspect, and extract encrypted knowledge packages |
| `ai_cluster_node` | Distributed cluster node with QUIC mesh and CRDTs |

## Features

### Core Features
- **Multi-provider support**: 13+ LLM providers — Ollama, LM Studio, OpenAI, Anthropic, Google Gemini, Mistral, HuggingFace, AWS Bedrock, DeepSeek, Groq, Together AI, and any OpenAI-compatible API
- **Provider failover**: Automatic fallback with configurable retry and API key rotation
- **Streaming responses**: Real-time SSE and WebSocket (RFC 6455) streaming with cancellation support
- **Resumable streaming**: Checkpoint/replay for long-running generations with `Last-Event-ID`
- **Session management**: Save/load sessions (JSON, binary, JSONL journal, AES-256-GCM encrypted)
- **Context management**: Auto-truncation, compaction, summarization, context size cache
- **Adaptive thinking**: Automatic reasoning depth adjustment (5 levels: Trivial to Expert)

### RAG & Knowledge (`rag` feature)
- **5-tier RAG**: Self-RAG, CRAG, Graph RAG, RAPTOR, auto-selection (28 configurable features)
- **7 vector DB backends**: InMemory, Qdrant, LanceDB, Pinecone, Chroma, Milvus, pgvector
- **Document parsing**: PDF, EPUB, DOCX, ODT, HTML, TXT, CSV, EML, PPTX, XLSX
- **Knowledge graphs**: Entity/relation extraction, multi-layer graphs, graph traversal
- **Encrypted packages**: AES-256-GCM knowledge packages (.kpkg)
- **Hybrid search**: BM25 + semantic similarity, query expansion, reranking (MMR, RRF, cascade)

### OpenAI-Compatible API Server
- **Drop-in replacement**: Serve `/v1/chat/completions` and `/v1/models` — works with Open WebUI, LangChain, LiteLLM, Cursor, etc.
- **Enrichment pipeline**: 52 configurable fields across 7 sub-configs (guardrails, RAG, context, compaction, model selection, cost, thinking)
- **Selective guards**: Toggle individual input/output guardrails (attack, PII, toxicity, rate limit, patterns)
- **Budget manager**: Daily/monthly/per-request cost limits with automatic HTTP 429

### Multi-Agent & Autonomous
- **5-role orchestration**: Coordinator, Researcher, Analyst, Writer, Reviewer
- **Autonomous agent**: 5 autonomy levels, policy-based sandbox, cron scheduler
- **Browser automation**: Chrome DevTools Protocol (CDP) via WebSocket
- **Butler Advisor**: 30 optimization recommendations across 6 categories
- **Distributed agents**: Task distribution across nodes with heartbeats and MapReduce

### Advanced Model Routing (`full` feature)
- **Bandit algorithms**: Thompson Sampling, UCB1, epsilon-greedy
- **NFA/DFA routing**: Automaton-based rule matching with Hopcroft minimization
- **Composite rewards**: Configurable quality + latency + cost scoring
- **Per-query preferences**: Arm exclusion/boosting, weight overrides
- **10 MCP tools**: Runtime routing management

### Security & Safety
- **RBAC**: MFA, CIDR ranges, time windows, usage limits
- **Guardrail pipeline**: Constitutional AI, bias detection, toxicity (9 categories), injection detection (6 types)
- **PII detection**: 4 redaction strategies with configurable per-type toggles
- **Content encryption**: AES-256-GCM for sessions and knowledge
- **Output guardrails**: PII redaction and toxicity filtering on responses

### Streaming & Protocols
- **SSE streaming**: With gzip compression and aggregation
- **WebSocket**: RFC 6455 with handshake from scratch
- **MCP protocol**: 2025-03-26 spec with 40+ tools, annotations, pagination
- **A2A protocol**: Agent-to-agent communication

### Distributed Computing (`distributed`, `distributed-network` features)
- **CRDTs**: GCounter, PNCounter, LWWRegister, ORSet, LWWMap
- **DHT**: Kademlia with consistent hashing
- **QUIC/TLS 1.3**: Mutual TLS transport with node security
- **P2P**: STUN/UPnP/NAT-PMP, ICE, knowledge broadcast, consensus

### Analytics & Observability
- **OpenTelemetry**: GenAI semantic conventions for traces, spans, metrics
- **Prometheus metrics**: Request counts, latency, status codes
- **Eval suite**: Benchmark runner with 10+ suites (LiveCodeBench, AiderPolyglot, APPS, etc.)
- **LLM-as-judge**: Multi-criteria evaluation
- **A/B testing**: Experiment framework with Welch's t-test significance

### More Features
- **Voice agent**: STT/TTS with cloud and local providers
- **Media generation**: Image/audio/video generation pipelines
- **Event workflows**: DAG-based event-driven workflows with conditions
- **Prompt signatures**: DSPy-style optimizable prompt templates
- **Constrained decoding**: Grammar-guided generation (JSON, regex, CFG)
- **HITL**: Human-in-the-loop approval gates
- **Distillation**: Teacher-student model training data collection
- **WASM support**: web-sys, js-sys, wasm-bindgen
- **egui widgets**: Pre-built chat UI components

### Code Quality
- **Zero `.unwrap()` in production**: Proper error handling across all files
- **6,696 tests**: Comprehensive unit tests with 0 clippy warnings
- **313 source files**: Fully implemented — zero stubs or TODOs
- **20+ feature flags**: Granular dependency control
- **~353,000 lines of Rust**: Single crate, modular architecture
- **Security audited**: 32 vulnerabilities identified and fixed (2-pass audit)

## Building from Source

```bash
# Clone the repository
git clone https://github.com/OrlandoLuque/ai_assistant.git
cd ai_assistant

# Build with default features
cargo build

# Build with all lightweight features
cargo build --features full

# Build with autonomous agents
cargo build --features "full,autonomous,scheduler,butler,browser"

# Build with everything (including heavy deps)
cargo build --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio"

# Run tests
cargo test --features "full,autonomous,scheduler,butler,browser,containers" --lib
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# From crates.io (stable release):
ai_assistant = { version = "0.1", features = ["full"] }

# From git (latest development version):
ai_assistant = { git = "https://github.com/OrlandoLuque/ai_assistant", features = ["full"] }

# With autonomous agents:
ai_assistant = { version = "0.1", features = ["full", "autonomous", "scheduler", "butler", "browser"] }
```

### Feature Flags

| Flag | Description | In `full`? |
|------|-------------|:----------:|
| `core` | Basic LLM integration, providers, config | Yes |
| `rag` | 5-tier RAG, SQLite, vector DBs | Yes |
| `tools` | Unified tool system, MCP protocol | Yes |
| `security` | RBAC, guardrails, PII detection | Yes |
| `analytics` | OpenTelemetry, metrics, eval | Yes |
| `multi-agent` | 5-role orchestration | Yes |
| `advanced-memory` | Entity memory, episodic, plans | Yes |
| `workflows` | DAG-based event workflows | Yes |
| `prompt-signatures` | DSPy-style prompt templates | Yes |
| `a2a` | Agent-to-agent protocol | Yes |
| `async-runtime` | reqwest + tokio async providers | Yes |
| `egui-widgets` | Pre-built chat UI components | Yes |
| `autonomous` | Autonomous agent loop | No |
| `scheduler` | Cron scheduler (requires `autonomous`) | No |
| `butler` | Environment auto-detection (requires `autonomous`) | No |
| `browser` | CDP browser automation (requires `autonomous`) | No |
| `containers` | Docker execution via bollard | No |
| `audio` | Speech STT/TTS | No |
| `distributed` | CRDTs, DHT, MapReduce | Yes |
| `distributed-network` | QUIC/TLS 1.3 networking | No |
| `p2p` | P2P with STUN/UPnP/ICE | No |
| `hitl` | Human-in-the-loop approval | No |
| `webrtc` | WebRTC for voice | No |
| `devtools` | Agent debugging tools | No |

## Quick Start

### Basic Usage

```rust
use ai_assistant::{AiAssistant, AiResponse};

fn main() {
    // Create assistant with default system prompt
    let mut assistant = AiAssistant::new();

    // Or with custom system prompt
    let mut assistant = AiAssistant::with_system_prompt(
        "You are a helpful coding assistant. Be concise and accurate."
    );

    // Fetch available models from all providers
    assistant.fetch_models();

    // Poll until models are loaded
    while assistant.is_fetching_models {
        std::thread::sleep(std::time::Duration::from_millis(100));
        assistant.poll_models();
    }

    println!("Found {} models", assistant.available_models.len());

    // Select a model
    if let Some(model) = assistant.available_models.first() {
        assistant.config.selected_model = model.name.clone();
        assistant.config.provider = model.provider.clone();
        println!("Using: {} ({})", model.name, model.provider.display_name());
    }

    // Send a message
    assistant.send_message("What is Rust?".to_string(), "");

    // Poll for streaming response
    loop {
        if let Some(response) = assistant.poll_response() {
            match response {
                AiResponse::Chunk(text) => print!("{}", text),
                AiResponse::Complete(_) => {
                    println!();
                    break;
                }
                AiResponse::Error(e) => {
                    eprintln!("\nError: {}", e);
                    break;
                }
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
```

### Synchronous Generation

For simpler use cases where you don't need streaming:

```rust
use ai_assistant::AiAssistant;

let mut assistant = AiAssistant::new();
// ... configure model ...

let response = assistant.generate_sync(
    "Explain ownership in Rust".to_string(),
    ""  // optional knowledge context
)?;

println!("Response: {}", response);
```

### With Knowledge Context

You can inject domain knowledge into the system prompt:

```rust
let knowledge = r#"
# Product Documentation
- ProductX supports REST and GraphQL APIs
- Rate limit: 100 requests/minute
- Authentication: Bearer token required
"#;

assistant.send_message(
    "How do I authenticate with ProductX?".to_string(),
    knowledge
);
```

## egui Widgets

Enable the `egui-widgets` feature for pre-built UI components:

```toml
ai_assistant = { path = "...", features = ["egui-widgets"] }
```

### Available Widgets

```rust
use ai_assistant::widgets::*;

// In your egui update function:
fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
    let colors = ChatColors::default();

    // Connection status indicator
    connection_status(ui, self.assistant.is_fetching_models, self.assistant.available_models.len());

    // Model selector dropdown - returns Some(model) when selection changes
    if let Some(model) = model_selector(ui, &mut self.selected_model, &self.assistant.available_models) {
        self.assistant.config.selected_model = model.name;
        self.assistant.config.provider = model.provider;
    }

    // Context usage progress bar
    let usage = self.assistant.calculate_context_usage(&self.knowledge);
    context_usage_bar(ui, &usage, 120.0);

    // Display chat messages
    egui::ScrollArea::vertical().show(ui, |ui| {
        for msg in self.assistant.get_display_messages() {
            chat_message(ui, msg, &colors, 400.0);
        }

        // Show streaming response while generating
        if self.assistant.is_generating {
            streaming_response(ui, &self.assistant.current_response, &colors, 400.0);
        }
    });

    // Chat input with send button
    if let Some(text) = chat_input(ui, &mut self.input, self.assistant.is_generating, "Ask something...") {
        self.assistant.send_message(text, &self.knowledge);
    }
}
```

### Widget Reference

| Widget | Description |
|--------|-------------|
| `chat_message()` | Render a single chat message bubble (user/assistant/system) |
| `streaming_response()` | Show the current streaming response with spinner |
| `thinking_indicator()` | Simple "Thinking..." indicator |
| `error_message()` | Display an error in a styled box |
| `model_selector()` | Dropdown for model selection |
| `context_usage_bar()` | Progress bar showing context window usage |
| `session_list()` | Sidebar list of saved sessions |
| `chat_input()` | Single-line input with send button |
| `chat_input_multiline()` | Multi-line input with Ctrl+Enter to send |
| `suggestions()` | Horizontal row of suggestion buttons |
| `welcome_screen()` | Centered welcome message with suggestions |
| `connection_status()` | Model count or "Searching..." indicator |
| `rag_controls()` | Checkboxes for enabling/disabling RAG features |
| `rag_status_compact()` | Compact RAG status (chunks, archived messages) |
| `rag_status_panel()` | Detailed RAG status grid |
| `context_full_hint()` | Hint when context is full with RAG enable buttons |
| `NotesManager` | Complete notes management widget (struct) |
| `notes_buttons()` | Legacy: notes control buttons |
| `session_notes_editor()` | Legacy: session notes editor popup |
| `global_notes_editor()` | Legacy: global notes editor popup |
| `knowledge_notes_editor()` | Legacy: knowledge notes editor popup |

## Session Management

Save and restore conversation sessions:

```rust
use std::path::Path;

// Auto-save current session
assistant.save_current_session();

// Create new session (saves current first)
assistant.new_session();

// Save all sessions to file
assistant.save_sessions_to_file(Path::new("ai_sessions.json"))?;

// Load sessions from file
assistant.load_sessions_from_file(Path::new("ai_sessions.json"))?;

// List available sessions
for session in assistant.get_sessions() {
    println!("{}: {} ({} messages)",
        session.id, session.name, session.messages.len());
}

// Load a specific session
assistant.load_session("session_1234567890");

// Delete a session
assistant.delete_session("session_1234567890");
```

## Context Usage Tracking

Monitor how much of the model's context window is being used:

```rust
let usage = assistant.calculate_context_usage(&knowledge_context);

println!("Context usage: {:.1}%", usage.usage_percent);
println!("Tokens: {} / {}", usage.total_tokens, usage.max_tokens);
println!("  System: {} tokens", usage.system_tokens);
println!("  Knowledge: {} tokens", usage.knowledge_tokens);
println!("  Conversation: {} tokens", usage.conversation_tokens);

if usage.is_critical {
    println!("Warning: Context almost full, old messages may be summarized");
} else if usage.is_warning {
    println!("Note: Context getting full");
}
```

### Automatic Summarization

When context usage exceeds 70%, old messages are automatically summarized:

```rust
// Check if summarization should happen
assistant.summarize_old_messages(&knowledge);

// After response completes, start background summarization
assistant.start_background_summarization();

// Poll for summarization completion (in update loop)
assistant.poll_summarization();
```

## Configuration

```rust
use ai_assistant::{AiConfig, AiProvider};

let mut config = AiConfig::default();

// Generation settings
config.temperature = 0.8;           // 0.0 - 2.0
config.max_history_messages = 30;   // Messages to keep in context

// Provider URLs
config.ollama_url = "http://localhost:11434".to_string();
config.lm_studio_url = "http://localhost:1234".to_string();
config.text_gen_webui_url = "http://localhost:5000".to_string();
config.kobold_url = "http://localhost:5001".to_string();
config.local_ai_url = "http://localhost:8080".to_string();

// Custom OpenAI-compatible endpoint
config.provider = AiProvider::OpenAICompatible {
    base_url: "http://my-server:8000".to_string()
};

assistant.load_config(config);
```

## Supported Providers

| Provider | API Type | Default URL | Streaming |
|----------|----------|-------------|-----------|
| Ollama | Native | `http://localhost:11434` | Yes |
| LM Studio | OpenAI-compatible | `http://localhost:1234` | Yes |
| text-generation-webui | OpenAI-compatible | `http://localhost:5000` | Yes |
| Kobold.cpp | Native | `http://localhost:5001` | No* |
| LocalAI | OpenAI-compatible | `http://localhost:8080` | Yes |
| OpenAI (cloud) | Native | `https://api.openai.com` | Yes |
| Anthropic (cloud) | Native | `https://api.anthropic.com` | Yes |
| Google Gemini (cloud) | Native | `https://generativelanguage.googleapis.com` | Yes |
| Mistral AI (cloud) | OpenAI-compatible | `https://api.mistral.ai` | Yes |
| HuggingFace (cloud) | Native | `https://api-inference.huggingface.co` | Yes |
| AWS Bedrock (cloud) | SigV4 | Region-based | Yes |
| DeepSeek (cloud) | OpenAI-compatible | `https://api.deepseek.com` | Yes |
| Groq (cloud) | OpenAI-compatible | `https://api.groq.com/openai` | Yes |
| Together AI (cloud) | OpenAI-compatible | `https://api.together.xyz` | Yes |
| Custom | OpenAI-compatible | User-defined | Yes |

*Kobold.cpp falls back to non-streaming responses.

Cloud providers require API keys. Set them via `AiConfig.api_key` or environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.).

## Known Model Context Sizes

The library automatically detects context window sizes for popular models:

| Model Family | Context Size |
|--------------|--------------|
| Llama 3.2, 3.1 | 128K tokens |
| Llama 2 | 4K tokens |
| Qwen 2.5 (32B+) | 128K tokens |
| Qwen 2.5 (smaller) | 32K tokens |
| Mistral, Mixtral | 32K tokens |
| Phi-3 | 128K tokens |
| Gemma 2 | 8K tokens |
| DeepSeek | 32K tokens |
| CodeLlama | 16K tokens |
| GPT-4o, GPT-4o-mini | 128K tokens |
| GPT-4 | 8K tokens |
| GPT-4-turbo, o1, o3 | 128K-200K tokens |
| Claude (all models) | 200K tokens |
| Others | 8K tokens (default) |

## User Preferences

The library can extract and remember user preferences:

```rust
// Manually set preferences
assistant.preferences.response_style = ResponseStyle::Concise;
assistant.preferences.interests.push("rust".to_string());

// Load preferences
assistant.load_preferences(saved_preferences);

// Custom preference extraction
assistant.extract_preferences_with(|messages, prefs| {
    for msg in messages {
        if msg.content.contains("be brief") {
            prefs.response_style = ResponseStyle::Concise;
        }
    }
});
```

## RAG (Retrieval-Augmented Generation)

Enable the `rag` feature for SQLite-based knowledge storage and retrieval:

```toml
ai_assistant = { path = "...", features = ["rag"] }
```

### Document Registration (Recommended)

The preferred way to use RAG is to register documents. The crate handles initialization and indexing automatically:

```rust
use ai_assistant::AiAssistant;
use std::path::Path;

let mut assistant = AiAssistant::new();

// Set RAG database path (lazy initialization - database created on first use)
assistant.set_rag_path(Path::new("./ai_data.db"));

// Register documents for automatic indexing
// Documents are indexed when first needed (on RAG query or manual process)
let guide = std::fs::read_to_string("docs/guide.md")?;
assistant.register_knowledge_document("User Guide", &guide);

let faq = std::fs::read_to_string("docs/faq.md")?;
assistant.register_knowledge_document("FAQ", &faq);

// Documents are indexed automatically when you use build_rag_context()
// The crate handles change detection - unchanged documents are skipped
let (knowledge_context, conversation_context) = assistant.build_rag_context("How do I configure X?");

// Or manually process pending documents if you want to control timing
let results = assistant.process_pending_documents();
for (source, chunks) in results {
    if chunks > 0 {
        println!("Indexed '{}': {} chunks", source, chunks);
    } else {
        println!("'{}' up-to-date (skipped)", source);
    }
}
```

### Direct Indexing (Alternative)

You can also use explicit initialization and indexing:

```rust
use ai_assistant::AiAssistant;
use std::path::Path;

let mut assistant = AiAssistant::new();

// Initialize RAG database explicitly
assistant.init_rag(Path::new("./ai_data.db"))?;

// Index a markdown document (splits into chunks automatically)
let content = std::fs::read_to_string("docs/guide.md")?;
let chunks = assistant.index_knowledge_document("User Guide", &content)?;
println!("Indexed {} chunks", chunks);

// Get knowledge base stats
let (chunk_count, total_tokens) = assistant.get_knowledge_stats()?;
println!("Knowledge base: {} chunks, {} tokens", chunk_count, total_tokens);
```

### Document Management

```rust
// Get list of registered sources
let sources = assistant.get_registered_sources();

// Check if there are pending documents to index
if assistant.has_pending_documents() {
    println!("Documents pending indexing");
}

// Unregister a document (removes from pending, keeps in database)
assistant.unregister_knowledge_document("Old Guide");

// Delete a document (removes from pending and database)
assistant.delete_knowledge_document("Outdated FAQ")?;

// Check if database is initialized
if assistant.is_rag_initialized() {
    println!("RAG database is open");
}

// Check if RAG is available (path set or initialized)
if assistant.has_rag() {
    println!("RAG is configured");
}
```

### RAG-Enhanced Queries

Build context from relevant knowledge chunks:

```rust
// Enable knowledge RAG
assistant.set_knowledge_rag_enabled(true);

// Build context based on user's query
// This automatically processes any pending documents first
let (knowledge_context, conversation_context) = assistant.build_rag_context("How do I configure X?");

// Use in message
assistant.send_message("How do I configure X?".to_string(), &knowledge_context);
```

### Conversation RAG

Store and retrieve conversation history for "infinite" context:

```rust
// Enable conversation RAG
assistant.set_conversation_rag_enabled(true);

// Store messages in RAG database
let msg = ChatMessage::user("Hello!");
assistant.store_message_in_rag(&msg, true)?;  // true = in current context

// Archive old messages (moves them out of active context but keeps searchable)
assistant.archive_messages_to_rag(4)?;  // Archive oldest 4 messages

// Get stats
let (total, archived, archived_tokens) = assistant.get_conversation_rag_stats()?;
println!("{} total messages, {} archived ({} tokens)", total, archived, archived_tokens);
```

### Encrypted Knowledge Packages (KPKG)

Knowledge packages allow distributing encrypted knowledge bases. Packages are AES-256-GCM encrypted ZIP archives that are decrypted entirely in memory.

```rust
use ai_assistant::{
    KpkgBuilder, KpkgReader, AppKeyProvider, ExamplePair,
    RagDbKpkgExt,  // trait for RAG integration
};

// Create a professional knowledge package
let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
    .name("Star Citizen Guide")
    .description("Comprehensive guide for Star Citizen")
    .version("1.0.0")
    // AI Configuration
    .system_prompt("You are an expert Star Citizen guide. Be helpful and accurate.")
    .persona("Veteran pilot with extensive experience")
    // Few-shot examples
    .add_example(
        "What ship should I buy for cargo?",
        "For cargo hauling, the Hull series offers the best capacity-to-price ratio..."
    )
    .add_example_with_category(
        "How do I quantum travel?",
        "Press B to spool your quantum drive, then hold B to initiate travel.",
        "controls"
    )
    // Package metadata
    .author("Your Name")
    .language("en")
    .license("CC-BY-4.0")
    .add_tag("gaming")
    .add_tag("guide")
    .url("https://example.com/guide")
    // RAG configuration
    .chunk_size(512)
    .top_k(5)
    .min_relevance(0.3)
    .priority_boost(10)
    // Add documents
    .add_document("ships/aurora.md", "# Aurora\n\nThe Aurora is a starter ship...", Some(5))
    .add_document("mechanics/quantum.md", "# Quantum Travel\n\n...", Some(8))
    // Add timestamps
    .with_current_timestamps()
    .build()?;

// Save the package
std::fs::write("guide.kpkg", &encrypted)?;

// Read a package
let data = std::fs::read("guide.kpkg")?;
let reader = KpkgReader::<AppKeyProvider>::with_app_key();

// Read just the manifest (for inspection)
let manifest = reader.read_manifest_only(&data)?;
println!("Package: {}", manifest.name);
if let Some(ref sys) = manifest.system_prompt {
    println!("System prompt: {}", sys);
}

// Read documents with manifest
let (docs, manifest) = reader.read_with_manifest(&data)?;
println!("Documents: {}", docs.len());
for example in &manifest.examples {
    println!("Example: {} -> {}", example.input, example.output);
}

// Index into RAG with extended result
let rag_db = assistant.rag_db().unwrap();
let result = rag_db.index_kpkg_ext(&data)?;

println!("Indexed {} docs, {} chunks", result.documents_indexed(), result.chunks_created());

// Use manifest data
if let Some(prompt) = result.build_effective_system_prompt() {
    assistant.config.system_prompt = prompt;
}

// Format examples for few-shot learning
let examples_text = result.format_examples_for_prompt();
```

#### KPKG CLI Tool

The `kpkg_tool` binary provides command-line operations:

```bash
# Create a package with professional options
kpkg_tool create -i ./knowledge -o guide.kpkg \
    -n "My Guide" \
    -d "A helpful guide" \
    -a "Author Name" \
    -l "en" \
    --license "MIT" \
    --system-prompt "You are a helpful assistant." \
    --persona "Expert in the field" \
    --examples examples.json \
    -t "guide" -t "documentation" \
    --chunk-size 512 \
    --top-k 5

# Inspect package manifest
kpkg_tool inspect guide.kpkg
kpkg_tool inspect guide.kpkg --json

# List documents
kpkg_tool list guide.kpkg

# Extract contents
kpkg_tool extract -i guide.kpkg -o ./extracted
```

#### Example File Format (examples.json)

```json
[
    {
        "input": "What is Rust?",
        "output": "Rust is a systems programming language focused on safety."
    },
    {
        "input": "How do I compile?",
        "output": "Use `cargo build` for debug or `cargo build --release` for optimized builds.",
        "category": "compilation"
    }
]
```

## Multi-User Support

The RAG system supports multiple users with isolated data:

```rust
use ai_assistant::{AiAssistant, DEFAULT_USER_ID};

let mut assistant = AiAssistant::new();
assistant.init_rag(Path::new("./ai_data.db"))?;

// Use default user (for single-user apps)
// assistant.user_id is already set to DEFAULT_USER_ID ("default")

// Or set a specific user
assistant.set_user_id("user_123");

// Ensure user exists in database (creates if needed)
let global_notes = assistant.ensure_user()?;

// All RAG operations now use this user_id:
// - Knowledge notes
// - Session notes
// - Global notes
// - Conversation history
```

### User Data Isolation

Each user has isolated:
- **Global notes**: Persistent notes across all sessions
- **Session notes**: Notes specific to each chat session
- **Knowledge notes**: Notes about specific knowledge documents
- **Conversation history**: Messages stored per user

```rust
// Global notes (stored in database, per user)
let notes = assistant.get_rag_global_notes();
assistant.set_rag_global_notes("I prefer concise answers")?;

// Session notes (per user, per session)
let session_notes = assistant.get_rag_session_notes();
assistant.set_rag_session_notes("Discussing ship upgrades")?;

// Knowledge notes (per user, per document)
let guide_notes = assistant.get_knowledge_notes("CCU Guide");
assistant.set_knowledge_notes("CCU Guide", "Focus on Warbond CCUs")?;
```

### Notes with Messages

Include notes in the system prompt:

```rust
// Get session-specific notes
let session_notes = assistant.get_rag_session_notes();

// Get global notes
let global_notes = assistant.get_rag_global_notes();

// Get notes for all knowledge sources
let knowledge_notes = assistant.build_knowledge_notes_context();

// Send message with all notes included
assistant.send_message_with_notes(
    "What ship should I get?".to_string(),
    &knowledge_context,
    &session_notes,
    &knowledge_notes,
);
```

## Notes Manager Widget

The `egui-widgets` feature includes a complete notes management widget:

```rust
use ai_assistant::widgets::{NotesManager, NotesManagerConfig};

struct MyApp {
    notes_manager: NotesManager,
    // ...
}

impl MyApp {
    fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Get current notes values
        let session_notes = self.assistant.get_rag_session_notes();
        let global_notes = self.assistant.get_rag_global_notes();

        // Render notes buttons
        let (open_session, open_global, open_knowledge) =
            self.notes_manager.render_buttons(ui, &session_notes, &global_notes);

        // Open editors when buttons clicked
        if open_session {
            self.notes_manager.open_session_notes(&session_notes);
        }
        if open_global {
            self.notes_manager.open_global_notes(&global_notes);
        }
        if open_knowledge {
            self.notes_manager.open_knowledge_notes("", "");
        }

        // Render active editor and handle saves
        let sources = self.assistant.get_knowledge_sources();
        let response = self.notes_manager.render_editor(
            ctx,
            &sources,
            |source| self.assistant.get_knowledge_notes(source).unwrap_or_default(),
        );

        // Apply saved notes
        if let Some(notes) = response.session_saved {
            let _ = self.assistant.set_rag_session_notes(&notes);
        }
        if let Some(notes) = response.global_saved {
            let _ = self.assistant.set_rag_global_notes(&notes);
        }
        if let Some((source, notes)) = response.knowledge_saved {
            let _ = self.assistant.set_knowledge_notes(&source, &notes);
        }
    }
}
```

### Notes Widget Configuration

```rust
let config = NotesManagerConfig {
    session_enabled: true,
    global_enabled: true,
    knowledge_enabled: true,
    session_label: "Session Notes".to_string(),
    global_label: "Global Notes".to_string(),
    knowledge_label: "Guide Notes".to_string(),
    session_tooltip: "Notes for this conversation".to_string(),
    global_tooltip: "Notes for all conversations".to_string(),
    knowledge_tooltip: "Notes for knowledge guides".to_string(),
    session_title: "Edit Session Notes".to_string(),
    global_title: "Edit Global Notes".to_string(),
    knowledge_title: "Edit Knowledge Notes".to_string(),
};

let notes_manager = NotesManager::with_config(config);
```

## API Reference

### Core Types

- `AiAssistant` - Main assistant state and logic
- `AiConfig` - Configuration for providers and generation
- `AiProvider` - Enum of supported providers
- `ChatMessage` - A single conversation message
- `AiResponse` - Response variants (Chunk, Complete, Error, ModelsLoaded)
- `ModelInfo` - Information about an available model
- `ChatSession` - A saved conversation session
- `UserPreferences` - Learned user preferences
- `ContextUsage` - Context window usage statistics

### RAG Types (with `rag` feature)

- `RagDb` - SQLite database for knowledge and conversation storage
- `RagConfig` - Configuration for RAG retrieval
- `KnowledgeChunk` - A chunk of indexed knowledge
- `StoredMessage` - A conversation message in the database
- `User` - User information with global notes
- `DEFAULT_USER_ID` - Constant for single-user applications ("default")

### Event Types
- `AiEvent` - Event variants (MessageSent, ResponseComplete, ProviderFailed, etc.)
- `EventBus` - Event dispatcher with handler registration
- `EventHandler` - Trait for event processing
- `CollectingHandler` - Handler that collects events into a vector

### Server Types
- `ServerConfig` - Server configuration (host, port, body size, timeout)
- `AiServer` - HTTP server wrapping an AiAssistant
- `ServerHandle` - Handle to a running background server

### Async Types (with `async-runtime` feature)
- `AsyncHttpClient` - Trait for async HTTP operations
- `ReqwestClient` - Async client implementation using reqwest

### Cloud Provider Types
- `AiProvider::OpenAI` - OpenAI cloud provider variant
- `AiProvider::Anthropic` - Anthropic cloud provider variant

### Queue Types
- `RequestQueue` - Thread-safe priority queue
- `QueuedRequest` - A queued request with priority
- `RequestPriority` - Low, Normal, High

### Unified Tool Types (with `tools` feature)
- `ToolDef` - Tool definition with parameter schemas
- `ToolBuilder` - Fluent builder for tool definitions
- `ToolRegistry` - Registry for tool execution and validation
- `ToolCall` - Parsed tool invocation with typed accessors
- `ToolOutput` / `ToolError` - Tool execution results

### Widget Types (with `egui-widgets` feature)

- `NotesManager` - Complete notes management widget
- `NotesManagerConfig` - Configuration for NotesManager
- `NotesManagerResponse` - Response from notes widget (saved notes)
- `NotesEditorType` - Enum of notes editor types (Session, Global, Knowledge)
- `RagWidgetConfig` - Configuration for RAG control widgets
- `RagStatus` - RAG status information for display
- `ChatColors` - Color scheme for chat UI

### Key Methods

```rust
impl AiAssistant {
    // Creation
    fn new() -> Self;
    fn with_system_prompt(prompt: &str) -> Self;

    // Model discovery
    fn fetch_models(&mut self);
    fn poll_models(&mut self) -> bool;

    // Generation
    fn send_message(&mut self, msg: String, knowledge: &str);
    fn send_message_with_notes(&mut self, msg: String, knowledge: &str, session_notes: &str, knowledge_notes: &str);
    fn poll_response(&mut self) -> Option<AiResponse>;
    fn generate_sync(&mut self, msg: String, knowledge: &str) -> Result<String>;

    // Session management
    fn new_session(&mut self);
    fn save_current_session(&mut self);
    fn load_session(&mut self, id: &str);
    fn delete_session(&mut self, id: &str);
    fn save_sessions_to_file(&self, path: &Path) -> Result<()>;
    fn load_sessions_from_file(&mut self, path: &Path) -> Result<()>;

    // Context management
    fn calculate_context_usage(&self, knowledge: &str) -> ContextUsage;
    fn summarize_old_messages(&mut self, knowledge: &str);
    fn start_background_summarization(&mut self);
    fn poll_summarization(&mut self);

    // Utilities
    fn clear_conversation(&mut self);
    fn get_display_messages(&self) -> &[ChatMessage];
    fn message_count(&self) -> usize;

    // RAG (with `rag` feature)
    fn set_rag_path(&mut self, db_path: &Path);  // Lazy initialization
    fn init_rag(&mut self, db_path: &Path) -> Result<()>;  // Explicit init
    fn has_rag(&self) -> bool;
    fn is_rag_initialized(&self) -> bool;
    fn set_user_id(&mut self, user_id: &str);
    fn get_user_id(&self) -> &str;
    fn ensure_user(&mut self) -> Result<String>;

    // Document registration (recommended approach)
    fn register_knowledge_document(&mut self, source: &str, content: &str);
    fn unregister_knowledge_document(&mut self, source: &str);
    fn delete_knowledge_document(&mut self, source: &str) -> Result<()>;
    fn get_registered_sources(&self) -> &[String];
    fn has_pending_documents(&self) -> bool;
    fn process_pending_documents(&mut self) -> Vec<(String, usize)>;

    // Direct indexing (alternative)
    fn index_knowledge_document(&mut self, source: &str, content: &str) -> Result<usize>;
    fn clear_knowledge(&mut self) -> Result<()>;
    fn get_knowledge_stats(&mut self) -> Result<(usize, usize)>;
    fn build_rag_context(&mut self, query: &str) -> (String, String);
    fn store_message_in_rag(&mut self, msg: &ChatMessage, in_context: bool) -> Result<()>;
    fn archive_messages_to_rag(&mut self, count: usize) -> Result<()>;
    fn get_conversation_rag_stats(&mut self) -> Result<(usize, usize, usize)>;
    fn set_knowledge_rag_enabled(&mut self, enabled: bool) -> bool;
    fn set_conversation_rag_enabled(&mut self, enabled: bool) -> bool;

    // Notes (with `rag` feature)
    fn get_rag_global_notes(&mut self) -> String;
    fn set_rag_global_notes(&mut self, notes: &str) -> Result<()>;
    fn get_rag_session_notes(&mut self) -> String;
    fn set_rag_session_notes(&mut self, notes: &str) -> Result<()>;
    fn get_knowledge_notes(&mut self, source: &str) -> Option<String>;
    fn set_knowledge_notes(&mut self, source: &str, notes: &str) -> Result<()>;
    fn get_knowledge_sources(&mut self) -> Vec<String>;
    fn build_knowledge_notes_context(&mut self) -> String;
}
```

## New Modules (v0.1+)

### Entity Extraction

Extract named entities and facts from text:

```rust
use ai_assistant::{EntityExtractor, EntityExtractorConfig, EntityType};

let extractor = EntityExtractor::new(EntityExtractorConfig::default());
let entities = extractor.extract("Contact user@example.com or visit https://example.com for v2.0 release");

for entity in entities {
    println!("{:?}: {} (confidence: {:.2})", entity.entity_type, entity.text, entity.confidence);
}
// Output:
// Email: user@example.com (confidence: 0.95)
// Url: https://example.com (confidence: 0.95)
// Version: v2.0 (confidence: 0.90)
```

Supported entity types: `Email`, `Url`, `Phone`, `Version`, `ProgrammingLanguage`, `Money`, `Percentage`, `Date`, `Time`, `Organization`, `Person`, `Location`, `Custom`.

### Fact Tracking

Extract and reinforce facts from conversations:

```rust
use ai_assistant::{FactExtractor, FactExtractorConfig, FactStore};

let extractor = FactExtractor::new(FactExtractorConfig::default());
let facts = extractor.extract("I prefer Rust over Python. My goal is to learn systems programming.");

let mut store = FactStore::new();
for fact in facts {
    store.add_fact(fact);  // Deduplicates and reinforces existing facts automatically
}

println!("Stored {} facts", store.len());
```

### Quality Analysis

Evaluate response quality:

```rust
use ai_assistant::{QualityAnalyzer, QualityConfig};

let analyzer = QualityAnalyzer::new(QualityConfig::default());
let score = analyzer.analyze(
    "What is Rust?",
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    None,
);

println!("Overall: {:.0}%", score.overall * 100.0);
println!("Relevance: {:.0}%", score.relevance * 100.0);
println!("Coherence: {:.0}%", score.coherence * 100.0);
println!("Fluency: {:.0}%", score.fluency * 100.0);
println!("Issues: {:?}", score.issues);
```

### Language Detection

Detect language from text:

```rust
use ai_assistant::LanguageDetector;

let detector = LanguageDetector::new();

let result = detector.detect("Bonjour, comment allez-vous?");
println!("{} ({:.0}% confidence)", result.name, result.confidence * 100.0);
// Output: French (85% confidence)

let result = detector.detect("Hola, cómo estás?");
println!("{}", result.name);  // Spanish
```

Supported languages: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean.

### Export/Import

Export conversations to multiple formats:

```rust
use ai_assistant::{ConversationExporter, ExportFormat, ExportOptions};

let exporter = ConversationExporter::new(ExportOptions {
    format: ExportFormat::Markdown,
    include_metadata: true,
    redact_pii: true,  // Redact emails, phones, etc.
    ..Default::default()
});

let markdown = exporter.export(&messages, Some(&metadata))?;
std::fs::write("conversation.md", markdown)?;

// Also supports: JSON, CSV, HTML
let json_exporter = ConversationExporter::new(ExportOptions {
    format: ExportFormat::Json,
    ..Default::default()
});
```

### Benchmarking

Run performance benchmarks:

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
    println!("{}: mean={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        result.name,
        result.stats.mean_ms,
        result.stats.p95_ms,
        result.stats.p99_ms
    );
}

// Compare two benchmark runs
let comparison = compare_results(&baseline, &current);
println!("Regression: {}", comparison.has_regression);
```

Built-in benchmarks: token estimation, entity extraction, quality analysis, language detection, sentiment analysis, topic detection.

### Provider Plugins

Auto-discover and use LLM providers:

```rust
use ai_assistant::{discover_providers, DiscoveryConfig, create_registry_with_discovery};

// Discover available providers on the network
let config = DiscoveryConfig::default();
let providers = discover_providers(&config);

for provider in &providers {
    println!("Found: {}", provider.name());
    if provider.is_available() {
        let models = provider.list_models()?;
        println!("  Models: {:?}", models.iter().map(|m| &m.name).collect::<Vec<_>>());
    }
}

// Or create a registry with auto-discovery
let registry = create_registry_with_discovery(&config);
```

### Retry with Circuit Breaker

Automatic retry with exponential backoff:

```rust
use ai_assistant::{RetryConfig, CircuitBreaker, ResilientExecutor};
use std::time::Duration;

// Circuit breaker for cascading failure protection
let breaker = CircuitBreaker::new(5, Duration::from_secs(30));

// Resilient executor: retry + circuit breaker combined
let mut executor = ResilientExecutor::new(
    RetryConfig::default(),
    5,                           // failure threshold
    Duration::from_secs(30),     // recovery timeout
);

let result = executor.execute(|| api_call())?;
```

### Model Profiles

Pre-defined generation settings for different use cases:

```rust
use ai_assistant::{ProfileManager, ProfileApplicator};

let manager = ProfileManager::new();

// Available profiles: balanced, creative, precise, coding, conversational, concise, detailed
let profile = manager.get("coding").unwrap();
println!("Temperature: {}", profile.temperature);

// Generate provider-specific parameters from a profile
let params = ProfileApplicator::to_openai_params(profile);
println!("OpenAI params: {}", params);
```

### Prompt Templates

Reusable prompt templates with variable substitution:

```rust
use ai_assistant::{TemplateManager, BuiltinTemplates};

let manager = TemplateManager::with_builtins();

// Available templates: code_review, translation, explain, bug_fix, summarize, api_docs, refactor
let prompt = manager.render("code_review", &[
    ("code", "fn main() { ... }"),
    ("language", "Rust"),
])?;
```

### Streaming Metrics

Track real-time generation metrics:

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

### Function Calling

OpenAI-compatible function calling:

```rust
use ai_assistant::{FunctionBuilder, ParameterProperty};

// Build a function definition for OpenAI-compatible function calling
let function = FunctionBuilder::new("get_weather")
    .description("Get current weather")
    .param("location", ParameterProperty::string("City name"), true)
    .param("unit", ParameterProperty::string("Temperature unit"), false)
    .build();

// Serialize to OpenAI format
let json = serde_json::to_value(&function).unwrap();
```

### Vision Support

Multimodal support for vision models:

```rust
use ai_assistant::{VisionMessage, ImageInput};
use std::path::Path;

let image = ImageInput::from_file(Path::new("photo.jpg"))?;
let message = VisionMessage::user("What's in this image?", vec![image]);

let openai_format = message.to_openai_format();
```

### Conversation Memory

Long-term memory with decay:

```rust
use ai_assistant::{MemoryManager, MemoryConfig, MemoryType};

let mut memory = MemoryManager::new(MemoryConfig::default());

// Remember facts (with decay over time)
memory.remember_fact("User prefers Rust", 0.8);
memory.remember_preference("Concise explanations");
memory.remember_goal("Learn systems programming");

// Recall relevant memories
let memories = memory.recall("What programming language?");

// Build context for prompts
let context = memory.build_context("Tell me about Rust", 500);
```

### Model Routing

Intelligent model selection based on task:

```rust
use ai_assistant::{ModelRouter, ModelRequirements, TaskType};

let router = ModelRouter::new();

// Auto-detect task type
let task = ModelRouter::detect_task_type("Write code to sort an array");
assert_eq!(task, TaskType::Coding);

// Select best model
let requirements = ModelRequirements::for_task(TaskType::Coding)
    .with_min_context(8000);

let best = router.select_best(&available_models, &requirements);
```

### Cost Estimation

Track API usage costs:

```rust
use ai_assistant::{CostEstimator, CostTracker, BudgetManager};

let estimator = CostEstimator::new();
let mut tracker = CostTracker::new();

// Estimate cost
let estimate = estimator.estimate("gpt-4", "openai", 1000, 500);
println!("Cost: {}", estimate.format());  // $0.0450 USD

// Track spending
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

### Response Caching

Cache responses with TTL:

```rust
use ai_assistant::{ResponseCache, CacheConfig};

let mut cache = ResponseCache::new(CacheConfig {
    max_entries: 1000,
    default_ttl: Duration::from_secs(3600),
    fuzzy_matching: true,
    similarity_threshold: 0.85,
    ..Default::default()
});

// Cache a response
cache.put("What is Rust?", "llama-3", "Rust is...", 50, Some("factual"));

// Get from cache (with fuzzy matching)
if let Some(response) = cache.get("What's Rust?", "llama-3") {
    println!("Cached: {}", response.content);
}

println!("Hit rate: {:.1}%", cache.stats().hit_rate() * 100.0);
```

### Structured Output

JSON output with schema validation:

```rust
use ai_assistant::{JsonSchema, SchemaProperty, SchemaValidator, SchemaBuilder};

// Define schema
let schema = JsonSchema::new("sentiment")
    .with_property("sentiment", SchemaProperty::string()
        .with_enum(vec!["positive".into(), "negative".into(), "neutral".into()]))
    .with_property("confidence", SchemaProperty::number()
        .with_minimum(0.0)
        .with_maximum(1.0))
    .with_required("sentiment")
    .with_required("confidence");

// Generate prompt
let prompt = format!("{}\n{}", user_question, schema.to_prompt());

// Validate response
let validation = SchemaValidator::validate(&json_response, &schema);
if !validation.valid {
    println!("Validation errors: {:?}", validation.errors);
}

// Pre-built schemas
let sentiment = SchemaBuilder::sentiment_analysis();
let entities = SchemaBuilder::entity_extraction();
```

### Agent Framework

Multi-step task execution:

```rust
use ai_assistant::{ReactAgent, AgentConfig, AgentTool, create_builtin_agent_tools};

let mut agent = ReactAgent::new(AgentConfig {
    max_steps: 10,
    verbose: true,
    ..Default::default()
});

// Add tools
for tool in create_builtin_agent_tools() {
    agent.add_tool(tool);
}

// Execute step
let observation = agent.execute_step(
    "I need to calculate".to_string(),
    "calculator".to_string(),
    "2 + 2 * 3".to_string(),
)?;

// Check state
if agent.should_stop() {
    println!("Agent finished");
}
```

### Embedding Cache

Cache embedding vectors:

```rust
use ai_assistant::{SharedEmbeddingCache, EmbeddingCacheConfig, cosine_similarity};

let cache = SharedEmbeddingCache::new(EmbeddingCacheConfig {
    max_entries: 10000,
    ttl: Duration::from_secs(86400),
    ..Default::default()
});

// Cache embeddings
cache.set("hello world", "text-embedding", vec![0.1, 0.2, 0.3]);

// Retrieve
if let Some(embedding) = cache.get("hello world", "text-embedding") {
    let similarity = cosine_similarity(&embedding, &other_embedding);
}

// Stats
println!("Hit rate: {:.1}%", cache.stats().unwrap().hit_rate() * 100.0);
```

### Diff Viewer

Compare text and responses:

```rust
use ai_assistant::{diff, compare_responses};

// Line-by-line diff
let result = diff("old text\nline 2", "new text\nline 2\nline 3");

println!("Additions: {}, Deletions: {}", result.additions, result.deletions);
println!("{}", result.to_unified("old.txt", "new.txt"));

// Response comparison
let comparison = compare_responses(
    "The quick brown fox",
    "The fast brown fox jumps"
);

println!("Similarity: {:.1}%", comparison.similarity * 100.0);
println!("Common phrases: {:?}", comparison.common_phrases);
```

### Cloud Providers

Connect to OpenAI and Anthropic APIs natively:

```rust
use ai_assistant::{AiAssistant, AiConfig, AiProvider};

let mut config = AiConfig::default();
config.provider = AiProvider::OpenAI;
config.api_key = "sk-...".to_string(); // or set OPENAI_API_KEY env var
config.selected_model = "gpt-4o".to_string();

let mut assistant = AiAssistant::new();
assistant.load_config(config);
assistant.send_message("Hello!".to_string(), "");
```

For Anthropic:

```rust
config.provider = AiProvider::Anthropic;
config.api_key = "sk-ant-...".to_string(); // or set ANTHROPIC_API_KEY env var
config.selected_model = "claude-sonnet-4-5-20250929".to_string();
```

### Event System

Monitor assistant lifecycle events:

```rust
use ai_assistant::events::{EventBus, AiEvent, CollectingHandler};
use std::sync::Arc;

let mut bus = EventBus::new();

// Collect all events
let collector = Arc::new(CollectingHandler::new());
bus.add_handler(collector.clone());

// Or use a callback
bus.on(|event: &AiEvent| {
    println!("[{}] {}", event.category(), event.name());
});

// Emit events
bus.emit(AiEvent::MessageSent {
    content_length: 5,
    has_knowledge: false,
});

// Check collected events
assert_eq!(collector.len(), 1);
```

### Request Queue

Thread-safe priority queue for multi-threaded usage:

```rust
use ai_assistant::request_queue::{RequestQueue, QueuedRequest, RequestPriority};

let queue = RequestQueue::new(100);

// Enqueue with priority
queue.enqueue(
    QueuedRequest::new("Hello")
        .with_priority(RequestPriority::Normal)
        .with_session("session_1")
);

// Dequeue highest priority first
if let Some(request) = queue.try_dequeue() {
    println!("Processing: {}", request.message);
}

// Stats
let stats = queue.stats();
println!("Pending: {}, Processed: {}", stats.pending, stats.total_processed);
```

### Unified Tool System

Define and execute tools with validation (feature `tools`):

```rust
use ai_assistant::unified_tools::{ToolBuilder, ParamSchema, parse_tool_calls};

// Define a tool with the builder
let tool = ToolBuilder::new("get_weather", "Get current weather for a city")
    .required_string("city", "City name")
    .optional_string("unit", "Temperature unit")
    .build();

// Parse tool calls from multiple formats (bracket, XML, OpenAI JSON)
let calls = parse_tool_calls(r#"[TOOL:calculate(expression="2+2*3")]"#);
```

### Async Providers

Async model fetching and generation (feature `async-runtime`):

```rust
use ai_assistant::async_providers::{ReqwestClient, fetch_models_async, block_on_async};
use ai_assistant::AiConfig;

let client = ReqwestClient::new();
let config = AiConfig::default();

// Blocking bridge for sync code
let models = block_on_async(
    fetch_models_async(&client, &config)
).unwrap();

println!("Found {} models", models.len());
```

### Embedded HTTP Server

Expose the assistant as a REST API with OpenAI-compatible endpoints:

```rust
use ai_assistant::server::{ServerConfig, AiServer, ServerEnrichmentConfig};

let config = ServerConfig {
    port: 8090,
    enrichment: ServerEnrichmentConfig {
        enable_guardrails: true,
        enable_rag: true,
        ..Default::default()
    },
    ..Default::default()
};
let server = AiServer::new(config);
let handle = server.start_background().unwrap();
println!("Server at {}", handle.url());

// Native endpoints:
// GET  /health, /models, /config, /metrics, /sessions, /openapi.json
// POST /chat, /chat/stream, /config
// GET  /ws (WebSocket)
//
// OpenAI-compatible endpoints:
// POST /v1/chat/completions  (drop-in for any OpenAI-compatible tool)
// GET  /v1/models
```

## Database Schema

The RAG feature uses SQLite with FTS5 for full-text search. Tables:

| Table | Description |
|-------|-------------|
| `users` | User info with global_notes, created/updated timestamps |
| `knowledge_chunks` | Indexed document chunks (shared across users) |
| `knowledge_fts` | FTS5 virtual table for knowledge search |
| `knowledge_sources` | Tracks indexed documents with content hash for change detection |
| `knowledge_notes` | Per-user notes for each knowledge source |
| `session_notes` | Per-user notes for each chat session |
| `conversation_messages` | Per-user conversation history |
| `conversation_fts` | FTS5 virtual table for conversation search |

The `knowledge_sources` table tracks:
- Source name and content hash (for change detection)
- Chunk count and total tokens per document
- Indexed/updated timestamps

This enables smart re-indexing: documents are only re-indexed when their content has changed.

Migrations are automatic when opening an existing database with schema changes.

## Microservice Architecture (v30)

The crate supports three deployment tiers:

| Tier | Binary | Features | Use Case |
|------|--------|----------|----------|
| Embedded | Library (cargo dep) | `full` | Integrate into your Rust app |
| Standalone | `ai_assistant_standalone` | `full,server-axum` | Single-binary server + REPL |
| Cluster | `ai_cluster_node` + `ai_proxy` | `server-cluster` | Horizontal scaling with CRDT sync |

### Quick Start

```bash
# Standalone server
cargo run --features "full,server-axum" --bin ai_assistant_standalone -- --port 8090

# 3-node cluster with Docker
docker compose up

# Lightweight API proxy
cargo run --features "server-axum" --bin ai_proxy -- --backends "localhost:8091,localhost:8092"
```

### Competitive Advantages

| Metric | ai_assistant | Python (LangChain) |
|--------|-------------|-------------------|
| Binary size | ~5 MB | 300+ MB |
| RAM (idle) | ~10 MB | 200+ MB |
| Cold start | <100 ms | 5-10 s |
| GC pauses | None | Unpredictable |
| Runtime deps | None | Python + pip |
| Modules | 313 | ~50 |
| Tests | 6,696 | Varies |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed comparisons.

## Contributing

This project is not currently accepting external contributions. A CLA (Contributor License Agreement) will be implemented before opening contributions.

## Author

**Orlando José Luque Moraira** — orlando.luque@gmail.com

## License

Licensed under [PolyForm Noncommercial 1.0.0](LICENSE).

This license permits personal, academic, and research use. **Commercial use requires a separate license** — contact orlando.luque@gmail.com for inquiries.

See [LICENSE](LICENSE) for the full text.
