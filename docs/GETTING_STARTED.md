# Getting Started

This guide walks you through downloading, running, and using `ai_assistant` — from pre-built binaries to using it as a Rust library or as an HTTP server backend from any language.

## Prerequisites

- **A local LLM provider** (at least one):
  - [Ollama](https://ollama.com) — recommended, one-click install
  - [LM Studio](https://lmstudio.ai) — GUI-based, easy to use
  - Or a cloud API key (OpenAI, Anthropic, etc.)

- **For pre-built binaries**: Windows 10/11 (x86_64)
- **For building from source**: [Rust 1.82+](https://rustup.rs)

### Quick Ollama Setup

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2:1b    # Small, fast model (~1.3 GB)
ollama pull llama3.2:3b    # Better quality (~2 GB)
```

> **Windows firewall**: When you first start the server or Ollama, Windows may show a firewall dialog. Click **Allow access** for private networks.

---

## Option A: Pre-Built Binaries (No Rust Needed)

The fastest way to try `ai_assistant`.

1. Download the latest release from [GitHub Releases](https://github.com/OrlandoLuque/ai_assistant/releases)
2. Extract the zip file
3. Open a terminal in the extracted directory
4. Run:

```bash
ai_assistant_cli.exe
```

That's it — the CLI auto-detects your local LLM and starts a chat session.

---

## Option B: Install from crates.io

```bash
# Install the CLI (compiles from source, takes a few minutes)
cargo install ai_assistant --bin ai_assistant_cli --features "full,butler"

# Install the HTTP server
cargo install ai_assistant --bin ai_assistant_server --features "full"

# Install the knowledge package tool
cargo install ai_assistant --bin kpkg_tool --features "rag"
```

---

## Option C: Clone & Build

```bash
git clone https://github.com/OrlandoLuque/ai_assistant.git
cd ai_assistant

# Build the CLI
cargo build --release --bin ai_assistant_cli --features "full,butler"

# Build the server
cargo build --release --bin ai_assistant_server --features "full"

# Build the GUI
cargo build --release --bin ai_gui --features "gui"

# Build kpkg_tool
cargo build --release --bin kpkg_tool --features "rag"

# Build the cluster node (heavy deps: QUIC, TLS)
cargo build --release --bin ai_cluster_node --features "full,server-cluster"
```

Binaries are in `target/release/`.

---

## Included Binaries

| Binary | Features | Description |
|--------|----------|-------------|
| `ai_assistant_cli` | `full,butler` | Interactive REPL with auto-detection |
| `ai_gui` | `gui` | Desktop GUI (WIP) |
| `ai_assistant_server` | `full` | HTTP API server (OpenAI-compatible) |
| `kpkg_tool` | `rag` | Knowledge package manager |
| `ai_cluster_node` | `full,server-cluster` | Distributed cluster node (QUIC mesh) |

---

## The CLI (`ai_assistant_cli`)

The CLI is the quickest way to start chatting. It auto-detects local LLM providers using the **Butler** system.

```bash
# Auto-detect everything
ai_assistant_cli

# Force a specific provider and model
ai_assistant_cli --provider ollama --model llama3.2:1b

# Skip auto-detection
ai_assistant_cli --no-butler
```

<!-- SCREENSHOT: docs/screenshots/cli_startup.png -->
<!-- SCREENSHOT: docs/screenshots/cli_chat.png -->

### REPL Commands

| Command | Description |
|---------|-------------|
| `/help`, `/h`, `/?` | Show help |
| `/models` | List available models |
| `/model <name>` | Switch to a different model |
| `/config` | Show current configuration |
| `/history` | Show conversation history |
| `/clear` | Clear conversation history |
| `/save <path>` | Save session to a file |
| `/load <path>` | Load session from a file |
| `/template <name>` | Load a prompt template |
| `/cost` | Show token usage and cost |
| `/exit`, `/quit`, `/q` | Exit |

---

## The Desktop GUI (`ai_gui`)

Graphical interface with chat, model scanning, and `.kpkg` file support.

```bash
ai_gui
```

<!-- SCREENSHOT: docs/screenshots/gui_main.png -->

> **Note**: The GUI is functional but still under active development.

---

## The HTTP Server (`ai_assistant_server`)

Runs an HTTP server with REST API endpoints. **OpenAI-compatible** — works as a drop-in replacement in any tool that supports custom endpoints (Continue.dev, Cursor, LangChain, etc.).

```bash
# Start with defaults (127.0.0.1:8090)
ai_assistant_server

# Custom port with API key authentication
ai_assistant_server --port 3000 --api-key mysecretkey

# With TLS
ai_assistant_server --tls-cert cert.pem --tls-key key.pem
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List available models |
| `POST` | `/chat` | Send a message (JSON body) |
| `POST` | `/chat/stream` | SSE streaming responses |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/config` | View current config |
| `POST` | `/config` | Update config |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/openapi.json` | OpenAPI 3.0 spec |
| `GET` | `/sessions` | List active sessions |

### Quick Test

```bash
# Health check
curl http://localhost:8090/health

# List models
curl http://localhost:8090/models

# Send a message
curl -X POST http://localhost:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# With system prompt and temperature
curl -X POST http://localhost:8090/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is Rust?"}
    ],
    "temperature": 0.3
  }'
```

---

## Using the Server from Other Languages

The server speaks JSON over HTTP, so you can use it from **any language**. Below are interactive chat clients and code snippets in multiple languages. Full runnable scripts are in [`examples/clients/`](../examples/clients/).

### Python

```python
import requests

BASE = "http://localhost:8090"

# Simple chat
r = requests.post(f"{BASE}/chat", json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain ownership in Rust in one sentence."},
    ],
    "temperature": 0.3,
})
print(r.json()["response"])

# SSE streaming
r = requests.post(f"{BASE}/chat/stream", json={
    "messages": [{"role": "user", "content": "Write a haiku about code."}],
}, stream=True)
for line in r.iter_lines(decode_unicode=True):
    if line.startswith("data: ") and line[6:] != "[DONE]":
        import json
        token = json.loads(line[6:]).get("token", "")
        print(token, end="", flush=True)
```

**OpenAI SDK compatibility** — use the standard `openai` package pointed at our server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8090/v1", api_key="not-needed")

# Works exactly like the OpenAI API
response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming also works
for chunk in client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Count to 5."}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

See: [`examples/clients/python_chat.py`](../examples/clients/python_chat.py), [`examples/clients/openai_compat.py`](../examples/clients/openai_compat.py)

### Node.js

```javascript
const BASE = "http://localhost:8090";

// Simple chat
const r = await fetch(`${BASE}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [
      { role: "system", content: "Reply in one sentence." },
      { role: "user", content: "What is WebAssembly?" },
    ],
    temperature: 0.5,
  }),
});
const data = await r.json();
console.log(data.response);

// SSE streaming
const stream = await fetch(`${BASE}/chat/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Tell me a joke." }],
  }),
});
const reader = stream.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ") && line.slice(6) !== "[DONE]") {
      const { token } = JSON.parse(line.slice(6));
      process.stdout.write(token || "");
    }
  }
}
```

See: [`examples/clients/node_chat.mjs`](../examples/clients/node_chat.mjs)

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "io"
)

func main() {
    body, _ := json.Marshal(map[string]interface{}{
        "messages": []map[string]string{
            {"role": "system", "content": "You are a Go expert."},
            {"role": "user", "content": "What is a goroutine?"},
        },
        "temperature": 0.5,
    })

    resp, err := http.Post("http://localhost:8090/chat", "application/json", bytes.NewReader(body))
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    data, _ := io.ReadAll(resp.Body)
    fmt.Println(string(data))
}
```

See: [`examples/clients/go_chat.go`](../examples/clients/go_chat.go)

### C# (.NET)

```csharp
using System.Net.Http.Json;

var client = new HttpClient { BaseAddress = new Uri("http://localhost:8090") };

var response = await client.PostAsJsonAsync("/chat", new {
    messages = new[] {
        new { role = "system", content = "You are a .NET expert." },
        new { role = "user", content = "What is LINQ?" },
    },
    temperature = 0.5,
});

var json = await response.Content.ReadAsStringAsync();
Console.WriteLine(json);
```

See: [`examples/clients/csharp_chat.cs`](../examples/clients/csharp_chat.cs)

### Java

```java
var client = HttpClient.newHttpClient();
var body = """
    {
      "messages": [{"role": "user", "content": "What is a record in Java?"}],
      "temperature": 0.5
    }
    """;

var request = HttpRequest.newBuilder(URI.create("http://localhost:8090/chat"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(body))
    .build();

var response = client.send(request, HttpResponse.BodyHandlers.ofString());
System.out.println(response.body());
```

See: [`examples/clients/java_chat.java`](../examples/clients/java_chat.java)

### curl

See [`examples/clients/curl_examples.sh`](../examples/clients/curl_examples.sh) for a comprehensive set of curl one-liners covering every endpoint.

---

## Using the Rust Library Directly

`ai_assistant` is a full-featured Rust crate. Add it to your project:

```toml
# Cargo.toml
[dependencies]
# From crates.io (stable release):
ai_assistant = { version = "0.1", features = ["full"] }

# Or from git (latest development version):
# ai_assistant = { git = "https://github.com/OrlandoLuque/ai_assistant", features = ["full"] }
```

### Basic Chat

```rust
use ai_assistant::{AiAssistant, AiConfig, AiProvider};

fn main() {
    let config = AiConfig {
        provider: AiProvider::Ollama,
        selected_model: "llama3.2:1b".to_string(),
        ollama_url: "http://localhost:11434".to_string(),
        ..Default::default()
    };

    let mut assistant = AiAssistant::new();
    assistant.load_config(config);
    assistant.set_system_prompt("You are a helpful coding assistant.");

    // Discover available models
    assistant.fetch_models();
    for model in &assistant.available_models {
        println!("  - {} ({})", model.name, model.provider.display_name());
    }

    // Send a message
    assistant.send_message_simple("What is the Rust ownership model?".to_string());

    // Read the conversation
    for msg in &assistant.conversation {
        println!("[{}] {}", msg.role, msg.content);
    }
}
```

### Butler Auto-Detection

```rust
use ai_assistant::butler::Butler;
use ai_assistant::AiAssistant;

fn main() {
    // Butler scans for local LLM providers (Ollama, LM Studio, cloud API keys)
    let mut butler = Butler::new();
    let report = butler.scan();

    println!("Found {} LLM providers:", report.llm_providers.len());
    for provider in &report.llm_providers {
        println!("  - {}: {} ({})",
            provider.name, provider.url,
            if provider.available { "online" } else { "offline" });
    }

    // Create an assistant with the best detected provider
    let mut assistant = AiAssistant::with_system_prompt("You are helpful.");
    // ... configure from report ...
}
```

### Multi-Agent Orchestration

```rust
use ai_assistant::{Agent, AgentOrchestrator, AgentRole, AgentTask, OrchestrationStrategy};

fn main() {
    let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Parallel);

    // Register agents with different specializations
    let researcher = Agent::new("r1", "Researcher", AgentRole::Researcher)
        .with_capability("web_search")
        .with_model("llama3");

    let analyst = Agent::new("a1", "Analyst", AgentRole::Analyst)
        .with_capability("data_analysis")
        .with_model("llama3");

    orchestrator.register_agent(researcher);
    orchestrator.register_agent(analyst);

    // Create tasks with dependencies
    let task1 = AgentTask::new("t1", "Research async runtimes in Rust")
        .with_priority(10);
    let task2 = AgentTask::new("t2", "Analyze and summarize findings")
        .depends_on("t1");

    orchestrator.add_task(task1);
    orchestrator.add_task(task2);

    let status = orchestrator.get_status();
    println!("Agents: {}, Tasks: {}", status.total_agents, status.total_tasks);
}
```

### RAG Pipeline

```rust
use ai_assistant::{
    KnowledgeGraphBuilder, KnowledgeGraphConfig,
    RagPipeline, RagPipelineConfig,
};

fn main() {
    // Build a knowledge graph from documents
    let config = KnowledgeGraphConfig::default();
    let mut graph = KnowledgeGraphBuilder::new(config).build();

    // Add documents
    graph.add_document("rust-book", "Rust uses ownership with borrowing...");
    graph.add_document("async-guide", "Tokio is an async runtime for Rust...");

    // Configure the RAG pipeline (5 tiers: Self-RAG, CRAG, Graph RAG, RAPTOR, Agentic)
    let rag_config = RagPipelineConfig {
        chunk_size: 512,
        chunk_overlap: 64,
        ..Default::default()
    };

    let pipeline = RagPipeline::new(rag_config);
    println!("RAG pipeline ready with {} tiers", pipeline.tier_count());
}
```

### Streaming with Backpressure

```rust
use std::time::Duration;
use ai_assistant::{BackpressureStream, StreamingConfig};

fn main() {
    let config = StreamingConfig {
        buffer_size: 4096,
        high_water_mark: 3072,
        low_water_mark: 1024,
        backpressure_timeout: Duration::from_secs(5),
        chunk_size: 64,
        auto_chunk: true,
    };

    let stream = BackpressureStream::<String>::new(config);
    let producer = stream.producer();
    let consumer = stream.consumer();

    // Producer thread sends tokens
    std::thread::spawn(move || {
        producer.send("Hello ".to_string()).unwrap();
        producer.send("streaming ".to_string()).unwrap();
        producer.send("world!".to_string()).unwrap();
        producer.finish();
    });

    // Consumer reads tokens as they arrive
    while let Some(chunk) = consumer.recv() {
        print!("{}", chunk);
    }

    let metrics = stream.metrics();
    println!("\nChunks: {}, Bytes: {}", metrics.chunks_produced, metrics.bytes_produced);
}
```

### Guardrails Pipeline

```rust
use ai_assistant::{GuardrailPipeline, GuardrailRule, GuardrailAction};

fn main() {
    let mut pipeline = GuardrailPipeline::new();

    // Add content safety rules
    pipeline.add_rule(GuardrailRule::new("no-pii", "Block PII in outputs")
        .with_action(GuardrailAction::Block)
        .with_pattern(r"\b\d{3}-\d{2}-\d{4}\b"));  // SSN pattern

    pipeline.add_rule(GuardrailRule::new("max-length", "Limit response length")
        .with_action(GuardrailAction::Truncate)
        .with_max_tokens(500));

    // Check content before sending to user
    let result = pipeline.check("The answer is 42.");
    println!("Passed: {}, Violations: {}", result.passed, result.violations.len());
}
```

> **More examples**: See the [`examples/`](../examples/) directory for 50+ Rust examples covering every feature: autonomous agents, browser automation, distributed networks, voice agents, MCP protocol, prompt signatures, and more.

---

## Knowledge Packages (`kpkg_tool`)

Create encrypted knowledge packages for RAG:

```bash
# Create a .kpkg from a folder of documents
kpkg_tool create --input ./my-docs --output knowledge.kpkg --name "My Knowledge Base"

# List contents
kpkg_tool list knowledge.kpkg

# Inspect metadata
kpkg_tool inspect knowledge.kpkg

# Extract back to files
kpkg_tool extract --input knowledge.kpkg --output ./extracted
```

Load `.kpkg` files in the CLI or GUI for context-aware conversations.

---

## Distributed Cluster (`ai_cluster_node`)

Run multiple nodes that synchronize via QUIC mesh networking with CRDTs. For advanced users who want distributed AI deployments.

```bash
# Start the first node (seed)
ai_cluster_node --node-id node1 --port 8091 --quic-port 9001

# Start a second node and join the cluster
ai_cluster_node --node-id node2 --port 8092 --quic-port 9002 \
  --bootstrap-peers 192.168.1.10:9001

# With join token for authentication
ai_cluster_node --node-id node3 --port 8093 --quic-port 9003 \
  --bootstrap-peers 192.168.1.10:9001 --join-token <TOKEN>
```

Each node runs a full HTTP server + QUIC mesh. State syncs automatically via CRDTs.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--node-id <ID>` | Unique node identifier |
| `--host <HOST>` | HTTP bind address (default: 127.0.0.1) |
| `--port <PORT>` | HTTP port |
| `--quic-port <PORT>` | QUIC mesh port |
| `--bootstrap-peers <ADDR>` | Comma-separated peer addresses |
| `--join-token <TOKEN>` | Authentication token for joining |
| `--data-dir <PATH>` | Data persistence directory |
| `--api-key <KEY>` | API key for HTTP auth |
| `--tls-cert <PATH>` | TLS certificate |
| `--tls-key <PATH>` | TLS private key |
| `--enable-p2p` | Enable P2P knowledge sharing |
| `--dry-run` | Validate config and exit |

---

## FreshContext Mode

FreshContext is an alternative context strategy that maximizes tokens available for knowledge retrieval:

```rust
use ai_assistant::{AiAssistant, ContextMode};

let mut assistant = AiAssistant::new();
// ... configure provider and model ...

// Enable RAG for knowledge retrieval
assistant.init_rag(&std::path::PathBuf::from("knowledge.db")).unwrap();

// Switch to FreshContext — only last message sent to LLM
assistant.set_context_mode(ContextMode::FreshContext);

// Check configuration health
let status = assistant.fresh_context_status(false);
for w in &status.warnings {
    println!("Warning: {}", w);
}
println!("Effectiveness: {:?}", status.effectiveness);
```

### Adding Memory to FreshContext

```rust
use ai_assistant::memory::MemoryConfig;

// Enable memory for session awareness
assistant.enable_memory(MemoryConfig::default());

// Now FreshContext auto-injects memory context alongside RAG results
assistant.send_message("What is Rust?".to_string(), "");
```

---

## MCP Knowledge Tools

Register knowledge tools on an MCP server to let external agents query your documents:

```rust
use ai_assistant::mcp_protocol::McpServer;
use ai_assistant::mcp_protocol::knowledge_tools::register_knowledge_tools;

let mut server = McpServer::new("my-server", "1.0.0");
register_knowledge_tools(&mut server, "knowledge.db".into(), None);
// Available: search_knowledge, list_knowledge_sources
// With graph: also query_graph, get_entity
```

---

## Feature Flags

`ai_assistant` uses Cargo feature flags to control what gets compiled:

| Feature | What it enables |
|---------|----------------|
| `full` | All lightweight features (recommended starting point) |
| `butler` | Auto-detection of local LLM providers |
| `gui` | Desktop GUI with egui |
| `rag` | RAG pipeline + knowledge packages |
| `multi-agent` | Multi-agent orchestration |
| `autonomous` | Autonomous agent system |
| `scheduler` | Cron scheduler |
| `browser` | CDP browser automation |
| `audio` | Speech STT/TTS |
| `server-tls` | HTTPS via rustls |
| `server-cluster` | Distributed cluster (QUIC mesh) |
| `distributed-network` | CRDTs, DHT, MapReduce |
| `p2p` | Peer-to-peer knowledge sharing |
| `containers` | Docker sandbox execution |

---

## Next Steps

- [**API Reference**](API_REFERENCE.md) — Full HTTP API documentation
- [**Guide**](../GUIDE.md) — Comprehensive library guide
- [**Examples**](../examples/) — 50+ Rust examples
- [**Client Examples**](../examples/clients/) — Python, Node.js, Go, C#, Java, curl
- [**Feature Matrix**](feature_matrix.html) — Interactive feature comparison
- [**crates.io**](https://crates.io/crates/ai_assistant) — Package page
