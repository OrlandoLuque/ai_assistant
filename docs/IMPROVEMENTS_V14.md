# v14 Improvements Changelog

> Date: 2026-02-26

v14 focuses on **test depth for the three weakest modules (tools.rs, assistant.rs, knowledge_graph.rs), expanding the benchmark suite, and completing the example coverage for all feature flags**. No new user-facing features were added; v14 continues the code quality and completeness work from v12/v13.

---

## Summary Metrics

| Metric | v13 | v14 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,390 | 5,509 | +119 |
| Examples | 43 | 47 | +4 |
| Benchmarks | 16 | 28 | +12 |
| Source files | ~285 | ~285 | 0 |
| Feature flags (real, excl. default/full) | 45 | 45 | 0 |
| Compiler warnings | 0 | 0 | 0 |

---

## Phase 1: tools.rs Test Coverage Boost (14 → 55)

**Problem**: `tools.rs` had 1,433 LOC with only 14 tests — the worst test-to-LOC ratio in the entire codebase (9.8 per 1K LOC). Major public APIs had zero coverage including ToolDefinition builder, ToolCall accessors, ToolResult variants, ProviderRegistry, ToolChain, ToolValidator, and ApprovalGate.

**Fix**: Added 41 new tests covering all previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| ToolDefinition builder | 5 | `new()`, parameter handling, required fields, display |
| ToolCall accessors | 6 | accessors, display, defaults, argument extraction |
| ToolResult variants | 4 | success, error, display, serialization |
| ToolRegistry advanced | 7 | batch add, duplicates, overwrite, categories, lookup |
| ProviderRegistry | 5 | register, list, get, unknown, duplicate handling |
| ToolChain advanced | 5 | multi-step, argument sources (FromInput/FromPrevious/Literal), output transforms |
| ToolValidator advanced | 4 | pattern, enum, nested, optional fields |
| ApprovalGate | 3 | auto-approve, deny, custom logic |
| Edge cases | 2 | empty names, very long descriptions |

**Files changed**: `src/tools.rs` (+544 lines)

---

## Phase 2: assistant.rs Test Coverage Boost (101 → 147)

**Problem**: `assistant.rs` is the core module (5,758 LOC) but had a relatively low test density (17.5 per 1K LOC). Configuration, context management, session handling, and RAG integration lacked targeted tests.

**Fix**: Added 46 new tests covering previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| Knowledge context advanced | 5 | format, empty, multi-note combinations |
| Conversation management | 6 | history, multi-turn, clear, stats tracking |
| Session management | 5 | save/restore, multiple sessions, notes |
| Context/model sizing | 5 | token limits, overflow, model switching |
| Notes management | 4 | session notes CRUD, persistence across calls |
| Fallback providers advanced | 4 | chain, recovery, empty fallback |
| RAG feature-gated | 5 | search, context building, toggle |
| Metrics | 3 | token counting, timing, accuracy tracking |
| Adaptive thinking | 2 | budget, toggle, auto-adjustment |
| Constructor/default state | 7 | `new()`, field defaults, builder patterns |

**Files changed**: `src/assistant.rs` (+532 lines)

---

## Phase 3: knowledge_graph.rs Test Coverage Boost (48 → 80)

**Problem**: `knowledge_graph.rs` had 3,353 LOC with 48 tests (14.3 per 1K LOC). Entity CRUD, relation management, graph traversal, and the builder pattern were under-tested.

**Fix**: Added 32 new tests covering all previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| Entity CRUD advanced | 6 | update, delete, list by type, properties, duplicate handling |
| Relation management | 5 | remove, bidirectional, query by type, confidence filtering |
| Chunk operations | 4 | add, retrieve, search, delete |
| GraphQuery builder | 5 | filters, limits, ordering, multiple conditions |
| KnowledgeGraph high-level | 4 | add/query entities, clear, stats |
| Graph algorithms | 4 | path finding, connected components, cycles |
| Builder pattern | 4 | KnowledgeGraphConfig options, defaults, validation |

**Files changed**: `src/knowledge_graph.rs` (+676 lines)

---

## Phase 4: New Benchmarks (+12)

**Problem**: Only 16 benchmarks covering basic operations. No benchmarks for RAG, embeddings, vector search, knowledge graph, tool parsing, or PII detection — the most compute-intensive subsystems.

**Fix**: Added 12 new criterion benchmarks:

| Benchmark | What it Measures |
|----------|-----------------|
| `smart_chunker_1k_chars` | Adaptive document chunking speed |
| `rag_fts_search_100_docs` | FTS5 search latency over 100 indexed documents |
| `rag_build_context_20_chunks` | KnowledgeUsage context building from chunks |
| `cosine_similarity_1536d` | Cosine similarity on OpenAI-sized (1536d) embeddings |
| `hnsw_search_1k_vectors_128d` | HNSW approximate nearest neighbor (1K vectors, 128d) |
| `knowledge_graph_query_relations_100` | Graph relation traversal (100 nodes, 200 edges) |
| `tool_call_parsing_markdown` | Tool call extraction from markdown LLM output |
| `pii_detection_1k_chars` | PII scanning throughput (emails, SSNs, credit cards) |
| `embedding_generate_128d` | Local embedding generation (128 dimensions) |
| `json_schema_to_prompt` | Schema-to-prompt serialization |
| `embedding_batch_5x128d` | Batch embedding generation (5 texts) |
| `rag_index_document_2k_chars` | Document indexing with chunking |

**Files changed**: `benches/core_benchmarks.rs` (+338 lines)

---

## Phase 5: Four Feature Examples (advanced-streaming, integrity-check, vector-pgvector, cloud-connectors)

**Problem**: 4 feature flags had no example code: `advanced-streaming`, `integrity-check`, `vector-pgvector`, `cloud-connectors`.

**New files**:

- **`examples/advanced_streaming_demo.rs`** (`advanced-streaming` feature): SSE events/writer, WebSocket frames (RFC 6455), AI protocol messages, streaming compression (gzip/deflate), resumable streaming with checkpoints
- **`examples/integrity_check_demo.rs`** (no feature gate): SHA256 hashing, IntegrityConfig builder, IntegrityResult variants, IntegrityChecker verification, startup flow
- **`examples/vector_pgvector_demo.rs`** (`vector-pgvector` feature): PgVectorConfig, SQL generation for schema/CRUD/search, HNSW indexing, cosine/L2 distance, vector formatting/parsing
- **`examples/cloud_connectors_demo.rs`** (`cloud-connectors` feature): S3 client/requests (+ MinIO), Google Drive, Azure Blob Storage, GCS, CloudStorage trait, unified StorageConnector

**Files changed**: 4 new `examples/*.rs`, `Cargo.toml` (4 `[[example]]` entries)

---

## Phase 6: Documentation Updates

- `docs/TESTING.md`: Updated test count 5,390 → 5,509, benchmarks 16 → 28, added v14 history row
- `docs/feature_matrix.html`: v19 → v20, updated all counts
- `docs/framework_comparison.html`: v16 → v17, updated all counts and history
- `docs/IMPROVEMENTS_V14.md`: This file
- `docs/backups/`: Timestamped HTML backups created before editing

---

## Verification

All checks pass:

```
cargo test --lib --features "full,...,devtools,vector-pgvector,cloud-connectors"  → 5509 passed, 0 failed
cargo clippy --features "full,...,devtools" -- -W clippy::all                     → 0 warnings
cargo build --examples --features "full,...,devtools,vector-pgvector,cloud-connectors" → 47 examples compile
cargo bench --features full --no-run                                              → 28 benchmarks compile
cargo bench --features full --bench core_benchmarks -- --test                     → 28 benchmarks pass
```
