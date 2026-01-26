# Testing Guide

## Overview

The project has two testing layers:

| Layer | Tests | Run Command |
|-------|-------|-------------|
| Unit tests (`#[test]`) | 867 | `cargo test -p ai_assistant --lib` |
| Integration tests | 21 | `cargo test -p ai_assistant --test integration_tests` |
| Test harness (CLI) | 420 | `cargo run --bin ai_test_harness -- --all` |

**Total: 1,308 tests**

## Quick Start

```bash
# Run all unit + integration tests
cargo test -p ai_assistant

# Run the CLI test harness (all categories)
cargo run --bin ai_test_harness -- --all

# Run a specific harness category
cargo run --bin ai_test_harness -- --category=security

# List all harness categories
cargo run --bin ai_test_harness -- --list

# CI mode (no ANSI colors)
cargo run --bin ai_test_harness -- --all --no-color
```

## Unit Tests (`cargo test`)

Unit tests live inside each source file in `crates/ai_assistant/src/` using `#[cfg(test)]` modules. They test individual functions and types in isolation.

### Covered Modules

**Core**:
- `config.rs` - AiProvider, AiConfig defaults, URL resolution
- `models.rs` - ModelInfo, format_size
- `messages.rs` - ChatMessage roles, AiResponse variants
- `session.rs` - ChatSession, ChatSessionStore, UserPreferences
- `context.rs` - estimate_tokens, ContextUsage, model context sizes

**Security**:
- `security/sanitization.rs` - Input cleaning, control chars, length limits
- `security/rate_limiting.rs` - Rate limit enforcement, window management
- `injection_detection.rs` - Injection pattern matching, sensitivity levels
- `pii_detection.rs` - PII recognition (emails, phones, SSNs, credit cards)
- `content_moderation.rs` - Moderation rules, blocked terms, risk scoring

**Analysis**:
- `confidence_scoring.rs` - Confidence calculations
- `hallucination_detection.rs` - Claim verification against sources
- `entities.rs` - Entity extraction (names, orgs, locations)
- `quality.rs` - Response quality scoring

**RAG & Memory**:
- `rag_advanced.rs` - SmartChunker strategies
- `embeddings.rs` - LocalEmbedder, cosine similarity
- `embedding_cache.rs` - Cache operations
- `memory.rs` - MemoryStore, memory entries
- `query_expansion.rs` - Query expansion with synonyms/keywords

**Infrastructure**:
- `cost.rs` - CostTracker, cost estimation
- `token_budget.rs` - Budget management, daily limits
- `priority_queue.rs` - Priority ordering, max size
- `distributed_rate_limit.rs` - Distributed rate limiting
- `export.rs` - Conversation export (JSON, Markdown)

**Formatting & Templates**:
- `formatting.rs` - Response parsing, code block extraction
- `templates.rs` - PromptTemplate, variable rendering
- `diff.rs` - Text diff computation
- `streaming.rs` - StreamBuffer operations
- `cache_compression.rs` - Compress/decompress roundtrips

## CLI Test Harness (`ai_test_harness`)

The test harness is a standalone binary that tests the `ai_assistant` crate's public API through comprehensive functional tests. It includes:

### Category Types

| Type | Categories | Tests | Description |
|------|-----------|-------|-------------|
| Unit | 77 | 289 | Individual module functionality |
| Integration (2-module) | 10 | 14 | Cross-module data flow |
| Chain (3-4 module) | 10 | 10 | Multi-step processing |
| Pipeline (5-6 module) | 8 | 9 | End-to-end workflows |
| Stress & Edge-case | 13 | 98 | Empty inputs, unicode, large data, errors, boundaries, concurrency, memory, regression, performance, fuzzing, api_contracts, serialization, chaos |

### All 108 Categories

**Core**: core, session, context, security, analysis, formatting, templates, export, streaming, memory, tools, cost, embeddings, llm

**Extended**: additional, decision_trees, rate_limiter, topic_summarizer, chunking, structured_output, batch, fallback, prompt_chaining, few_shot, token_budget, quantization, i18n, agent, task_decomposition, document_parsing, conversation_analytics, vision, self_consistency, answer_extraction, cot_parsing, translation_analysis, response_ranking, output_validation, priority_queue, conversation_compaction, query_expansion, smart_suggestions, html_extraction, table_extraction, entity_enrichment, conversation_flow, memory_pinning, advanced_guardrails, agent_memory, api_key_rotation, caching, citations, content_versioning, context_window, conversation_templates, crawl_policy, data_anonymization, intent, latency_metrics, message_queue, request_coalescing, content_encryption, access_control, auto_model_selection, cache_compression, conflict_resolution, connection_pool, content_moderation, conversation_control, distributed_rate_limit, embedding_cache, entities, evaluation, fine_tuning, forecasting, health_check, keepalive

**Integration (2 modules)**: integration_entity_anonymize, integration_intent_template, integration_versioning_merge, integration_embedding_similarity, integration_facts_context, integration_cache_compression, integration_expansion_ranking, integration_health_keepalive, integration_moderation_citations, integration_latency_selection

**Chain (3-4 modules)**: chain_entity_anon_cache_compress, chain_intent_template_context_budget, chain_chunker_entities_embed_similarity, chain_facts_memory_context_compact, chain_moderation_version_merge_export, chain_latency_health_select_cost, chain_analytics_topics_compact_export, chain_access_priority_ratelimit, chain_expansion_chunk_embed_rank, chain_intent_entity_citation_validate

**Pipeline (5-6 modules)**: pipeline_rag, pipeline_content_safety, pipeline_session_lifecycle, pipeline_request_processing, pipeline_knowledge_ingestion, pipeline_query_to_response, pipeline_multi_format_export, pipeline_guardrails

**Stress & Edge-case**: stress_empty_inputs, stress_unicode, stress_large_inputs, stress_error_paths, stress_boundaries, stress_concurrency, stress_memory, stress_regression, stress_performance, stress_fuzzing, stress_api_contracts, stress_serialization, stress_chaos

### RustRover Configuration

A run configuration is available at `.idea/runConfigurations/AiTestHarness_Debug.xml` for running the harness with `--all` flag and `RUST_BACKTRACE=short`.

## Adding New Tests

### Unit Test Pattern

```rust
// At the end of your module file:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        let result = my_function("input");
        assert_eq!(result, expected_value);
    }
}
```

### Harness Test Pattern

```rust
fn tests_my_category() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ My Category")));
    let mut results = Vec::new();

    results.push(run_test("test description", || {
        // test logic here
        assert_test!(condition, "failure message");
        Ok(())
    }));

    CategoryResult { name: "my_category".to_string(), results }
}
```

Then register in `all_categories()`:
```rust
("my_category", tests_my_category),
```

## CI Integration

```bash
# Run all tests with exit code reporting
cargo test -p ai_assistant --lib 2>&1 && \
cargo run --bin ai_test_harness -- --all --no-color
```

The harness exits with code 1 if any test fails, making it suitable for CI pipelines.
