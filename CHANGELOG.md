# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - v30 (2026-04-09)

### Added
- `ContextBudgetConfig` struct: centralizes all hardcoded allocator values (15 configurable fields)
- `ScoringMode` enum: 4 dynamic scoring modes (Static, Heuristic, LlmEnhanced, Hybrid)
- Intent-based context scoring: maps 16 intent types to per-source score boosts
- Knowledge graph as separate `ContextItem` (extracted from `build_rag_context()`, prevents double-counting)
- `StrategyBandit` wired into production: UCB1 arm selection with utilization reward
- `LlmEnhancerCompressor` bridge: adapts `LlmEnhancer` → `LlmCompressor` with fallback
- `context_scoring_mode` in `RagFeatures`: per-tier scoring mode override
- `arm_to_strategy()` for bandit arm → `OverflowStrategy` conversion
- CI: `FEATURES_STD` / `FEATURES_NETWORK` env vars for standardized feature sets
- CI: Feature-matrix expanded from 19 to 36 combinations
- CI: `cargo audit` security scan job
- CI: Integration tests (`cargo test --test '*'`)
- CI: Binary compilation verification (5 binaries)
- 82 new tests (context_budget: 16 new, total 34)

### Changed
- `build_allocated_context()` uses `ContextBudgetConfig` instead of hardcoded values
- Graph context extracted from `build_rag_context()` to standalone `build_graph_context_string()`
- RAG tier defaults: Enhanced=Heuristic, Thorough/Agentic/Graph=Hybrid(0.6)
- CI coverage aligned with `FEATURES_STD`
- Release pipeline updated: `needs: [check, test, clippy, fmt, binaries]`

### Stats
- 360+ source modules
- 7,469 passing tests (from 7,387 in v73)
- 60 Cargo feature flags
- 0 clippy warnings

## [Unreleased] - v29 (2026-03-06)

### Added
- OpenAI-compatible API: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`
- Full enrichment pipeline: 7 sub-configs, 52 configurable fields
- Selective guardrail pipeline: individual guard toggles, rate limiting, pattern blocking
- Budget manager: daily/monthly/per-request cost limits with HTTP 429
- Output guardrails: configurable PII redaction (per-type toggles) and toxicity filtering
- Butler Advisor: 30 optimization recommendations across 6 categories
- Advanced routing: Thompson Sampling, UCB1, NFA/DFA pipeline, 10 MCP routing tools
- Routing enhancements: composite rewards, per-query preferences, private arms, context-aware routing
- 5 new benchmark suites: LiveCodeBench, AiderPolyglot, TerminalBench, APPS, CodeContests
- RAG tier expansion: 20 → 28 features (discourse chunking, dedup, cascade reranking, etc.)
- 12 MCP tools: 6 config management + 6 evaluation tools
- Unified BPE tokenizer with model-aware routing (GPT, Claude, Gemini, Mistral, DeepSeek)
- Emoticon/emoji detection and sentiment analysis

### Changed
- Token estimation unified across 7 modules → central `crate::context::estimate_tokens`
- `concepts.html` rendering fix for unescaped HTML in code blocks
- `framework_comparison.html` new "Documentation, DX & Economics" category

### Stats
- 220+ source modules
- 6,565+ passing tests (from 6,401 in v28)
- 20+ Cargo feature flags
- 0 clippy warnings

## [0.1.0] - 2026-02-19

### Added

#### Core
- Multi-provider LLM support: Ollama, LM Studio, Kobold, LocalAI, OpenAI, Anthropic, Google Gemini, Mistral AI, HuggingFace Inference, AWS Bedrock
- OpenAI-compatible presets: Groq, Together AI, Fireworks, DeepSeek, vLLM
- Provider auto-discovery with failover and API key rotation
- Context window management with auto-truncation
- Session persistence with journal compaction and snapshots
- Adaptive thinking and response quality analysis

#### RAG & Knowledge
- 5-tier RAG: Self-RAG, CRAG, Graph RAG, RAPTOR, auto-selection
- Vector DB backends: InMemory, Qdrant, LanceDB, Pinecone, Chroma, Milvus, pgvector
- Document parsing: PDF, EPUB, DOCX, ODT, HTML, TXT, CSV, EML, PPTX, XLSX, image metadata
- Knowledge graph with entity/relation extraction
- Embedding-based semantic chunking
- Encrypted knowledge packages (.kpkg) with AES-256-GCM
- Query expansion, citations, and reranking

#### Multi-Agent & Autonomous
- 5-role multi-agent orchestration (Coordinator, Researcher, Analyst, Writer, Reviewer)
- Autonomous agent with 5 autonomy levels and policy-based sandbox
- Task board with undo, priorities, and listener callbacks
- Cron scheduler with event-driven triggers (FileChange, FeedUpdate)
- Butler environment auto-detection
- Chrome DevTools Protocol browser automation
- Distributed agent execution across nodes

#### Security
- RBAC with MFA, CIDR ranges, time windows, and usage limits
- Constitutional AI guardrails and bias detection (8 dimensions)
- Toxicity detection (9 categories) and injection detection (6 types)
- PII detection with 4 redaction strategies
- AES-256-GCM content encryption

#### Streaming & API
- SSE streaming with aggregation and chunking
- WebSocket (RFC 6455) with handshake from scratch
- Resumable streaming with checkpoint/replay
- Stream compression (Deflate, Gzip)
- MCP protocol (2025-03-26 spec) with tool annotations and pagination

#### Distributed Computing
- CRDTs (5 types), DHT (Kademlia), MapReduce with consistent hashing
- QUIC/TLS 1.3 transport with mutual TLS and node security
- Phi-accrual failure detection and Merkle sync
- P2P networking with STUN/UPnP/NAT-PMP and ICE

#### Analytics & Observability
- Prometheus-compatible metrics and flow analysis
- OpenTelemetry integration for traces, spans, and metrics
- Conversation analytics and engagement tracking
- LLM-as-judge evaluation

#### Infrastructure
- Cloud connectors (S3, Google Drive)
- Code sandbox for safe agent execution
- AWS SigV4 authentication for Bedrock
- Binary integrity verification
- WASM support (web-sys, js-sys, wasm-bindgen)
- egui chat widgets

### Stats
- 190+ source modules
- 2010+ passing tests
- 20+ Cargo feature flags
- Zero external service requirements for core functionality
