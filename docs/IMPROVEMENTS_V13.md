# v13 Improvements Changelog

> Date: 2026-02-25

v13 focuses on **RAG test depth, production safety (.unwrap() elimination), module-level documentation, and feature examples**. No new user-facing features were added; instead, v13 continues the code quality and completeness work from v12.

---

## Summary Metrics

| Metric | v12 | v13 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,350 | 5,390 | +40 |
| Examples | 37 | 43 | +6 |
| Production `.unwrap()` (non-Mutex) | 13 | 0 | -13 |
| Files >1K LOC without `//!` docs | 6 | 0 | -6 |
| Source files | ~285 | ~285 | 0 |
| Feature flags (real, excl. default/full) | 45 | 45 | 0 |
| Benchmarks | 16 | 16 | 0 |
| Compiler warnings | 0 | 0 | 0 |

---

## Phase 1: rag.rs Test Coverage Boost (40 → 80)

**Problem**: `rag.rs` had 3,350 LOC with only 40 tests. Thirty-one public APIs had zero test coverage, including multi-user management, conversation storage/retrieval, knowledge notes, hybrid search, export/import, and source priority.

**Fix**: Added 40 new tests covering all previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| Multi-user management | 5 | `get_or_create_user`, `get/set_user_global_notes`, `list_users` |
| Conversation storage/retrieval | 8 | `store_message`, `mark_messages_out_of_context`, `search_conversation`, `get_recent_archived_messages`, `get_conversation_stats`, `clear_session_history`, default wrappers |
| Knowledge notes (per-user) | 6 | `set/get/delete_knowledge_notes`, `get_all_knowledge_notes`, user isolation, default wrappers |
| Session notes | 3 | `set/get/delete_session_notes` |
| Hybrid/semantic search | 5 | `HybridRagConfig::default`, BM25-only fallback, combined score calculation, `open_with_hybrid`, `set_semantic_enabled` |
| Source priority | 3 | `set/get_source_priority`, default zero, search ordering by priority |
| Export/import | 4 | Empty DB export, replace mode, merge mode, file round-trip |
| Reindex checks | 3 | `needs_reindex`, `is_document_indexed`, `get_source_info` |
| Filtered/auto search | 3 | `search_knowledge_filtered` (single/multi source), `search_knowledge_auto` BM25 fallback |

**Files changed**: `src/rag.rs` (+538 lines)

---

## Phase 2: Production .unwrap() → .expect()

**Problem**: 13 production `.unwrap()` calls across 6 files violated the project's zero-unwrap-in-production rule. All were logically safe (guarded by preceding checks) but lacked explicit documentation.

**Fix**: Replaced each with `.expect("reason")` explaining why the unwrap is safe:

| File | Count | Pattern |
|------|-------|---------|
| `src/hnsw.rs` | 3 | `entry_point.unwrap()` → `.expect("entry_point guaranteed by early return")` |
| `src/models.rs` | 4 | `.as_ref().unwrap()` in `min_by` → `.expect("guaranteed by filter")` |
| `src/cost_integration.rs` | 1 | `.last().unwrap()` → `.expect("just pushed entry")` |
| `src/agent_graph.rs` | 2 | `.first()/.last().unwrap()` → `.expect("non-empty: checked above")` |
| `src/rag.rs` | 1 | `.last_mut().unwrap()` → `.expect("non-empty: checked on line above")` |
| `src/resumable_streaming.rs` | 2 | `.keys().next().unwrap()` → `.expect("non-empty: while loop guarantees")` |

Note: ~12 `Mutex::lock().unwrap()` calls were left as-is (idiomatic Rust for non-poisoned mutexes).

**Files changed**: `src/hnsw.rs`, `src/models.rs`, `src/cost_integration.rs`, `src/agent_graph.rs`, `src/rag.rs`, `src/resumable_streaming.rs`

---

## Phase 3: Module-Level `//!` Documentation

**Problem**: Six files over 1,000 LOC each had `//` header comments but no `//!` inner doc comments, meaning their documentation didn't appear in `cargo doc` output.

**Fix**: Converted existing `//` headers to `//!` and expanded with module purpose, key types, and feature flag requirements.

| File | LOC | Key Types Documented |
|------|-----|---------------------|
| `src/server.rs` | 3,812 | `ServerConfig`, `AiServer`, `ServerMetrics`, endpoint table |
| `src/opentelemetry_integration.rs` | 3,317 | `OtelTracer`, `OtelSpan`, `OtelMetrics`, `OtelConfig` |
| `src/unified_tools.rs` | 2,107 | `ToolRegistry`, `ToolCall`, `ToolResult`, `ToolSchema`, `ToolError` |
| `src/cloud_connectors.rs` | 1,878 | `CloudStorage`, `S3Client`, `GoogleDriveClient`, `StorageConnector` |
| `src/document_pipeline.rs` | 1,140 | `DocumentPipeline`, `CreateRequest`, `ConversionResult` |
| `src/container_sandbox.rs` | 1,042 | `ContainerSandbox`, `ExecutionBackend`, `ContainerSandboxConfig` |

**Files changed**: All 6 listed above

---

## Phase 4: Three Feature Examples (security, analytics, vision)

**New files**:

- **`examples/security_demo.rs`** (`security` feature): PII detection/redaction, content moderation, prompt injection detection, input sanitization, rate limiting, RBAC access control, audit logging
- **`examples/analytics_demo.rs`** (`analytics` feature): Conversation analytics with reports, sentiment analysis, latency tracking with P50/P95/P99, response quality analysis, response comparison ranking
- **`examples/vision_demo.rs`** (`vision` feature): Image input creation (URL/bytes/base64), vision messages with detail levels, model capability checks, image preprocessing, batch processing, token estimation

**Files changed**: 3 new `examples/*.rs`, `Cargo.toml` (3 `[[example]]` entries)

---

## Phase 5: Three Feature Examples (adapters, eval, code-sandbox)

**New files**:

- **`examples/adapters_demo.rs`** (`adapters` feature): OpenAI/Anthropic/HuggingFace API adapters, request builders, model presets, config variants, local provider discovery
- **`examples/eval_demo.rs`** (`eval` feature): Evaluation samples, text quality/relevance/safety evaluators, eval suite with weighted metrics, benchmarking with percentiles, A/B testing
- **`examples/code_sandbox_demo.rs`** (`code-sandbox` feature): Language support, sandbox configuration, dangerous command detection, environment sanitization, execution result handling

**Files changed**: 3 new `examples/*.rs`, `Cargo.toml` (3 `[[example]]` entries)

---

## Phase 6: Documentation Updates

- `docs/TESTING.md`: Updated test count 5,350 → 5,390, added v13 history row
- `docs/feature_matrix.html`: v18 → v19, updated all counts
- `docs/framework_comparison.html`: v15 → v16, updated all counts and history
- `docs/IMPROVEMENTS_V13.md`: This file
- `docs/backups/`: Timestamped HTML backups created before editing

---

## Verification

All checks pass:

```
cargo test --lib --features "full,...,devtools,vector-pgvector,cloud-connectors"  → 5390 passed, 0 failed
cargo clippy --features "full,...,devtools" -- -W clippy::all                     → 0 warnings
cargo build --examples --features "full,...,devtools"                              → 43 examples compile
cargo bench --features full --no-run                                              → 16 benchmarks compile
```
