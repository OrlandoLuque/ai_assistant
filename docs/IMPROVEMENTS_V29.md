# v29 Improvements Changelog

> Date: 2026-03-05

v29 covers **6 commits** since v28: advanced routing documentation, Butler Advisor, benchmark suites, MCP/tokenizer tooling, OpenAI-compatible API, and full enrichment config expansion (52 configurable fields).

---

## Summary Metrics

| Metric | v28 | v29 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 6,401 | 6,565 | +164 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 0 | 0 |
| Lines added | — | ~2,300 | — |

---

## 1. Advanced Routing Documentation & Concepts (commit `32d2a29`)

**Theme**: Comprehensive documentation for v27+v28 advanced routing system.

### What was added

- **`docs/concepts.html`** — 338 KB interactive reference with 163 concepts, dark/light mode toggle, search, and cross-references across all modules
- Updated `AGENT_SYSTEM_DESIGN.md`, `CONCEPTS.md`, `GUIDE.md`, `TESTING.md`
- HTML backups for `feature_matrix.html` and `framework_comparison.html`

### v27 Recap (code in previous commits)

- Thompson Sampling, UCB1, epsilon-greedy bandit algorithms
- NFA/DFA compilation (powerset + Hopcroft minimization)
- Feature-matching routing rules
- Closed-loop pipeline with bandit synthesizer
- 10 MCP runtime tools for routing management
- ~6,500 lines in `advanced_routing.rs`

### v28 Recap (code in previous commits)

- `RewardPolicy` — composite quality+latency+cost scoring
- `RoutingPreferences` — per-query overrides, arm exclusion/boosting
- `ArmVisibility` — private arms for distributed sharing
- `BanditBootstrapper` — auto-benchmark via eval-suite
- `RoutingContext` — agent-level budget-aware preferences
- `FeatureImportance` — feature attribution tracking
- ~1,400 additional lines

---

## 2. Butler Advisor (commit `9e4eb8e`)

**Theme**: AI-powered optimization guidance system.

### New types

- **`ButlerAdvisor`** — analyzes environment + config state
- **`AdvisorRecommendation`** — actionable recommendation with category, priority, description, impact estimate
- **`AdvisorCategory`** — 6 categories: Efficiency, Quality, Cost, Security, Scalability, Observability
- **`AdvisorPriority`** — Critical, High, Medium, Low

### Capabilities

- 30 actionable recommendations across 6 categories
- Environment analysis: detects GPU availability, memory constraints, provider configs
- Priority ranking with estimated impact percentages
- Per-category filtering and sorting

### Tests (+18)

`test_advisor_new`, `test_advisor_scan_default`, `test_advisor_categories`, `test_advisor_recommendations`, `test_advisor_priority_sorting`, `test_advisor_filter_by_category`, `test_advisor_efficiency_*`, `test_advisor_security_*`, `test_advisor_cost_*`, `test_advisor_quality_*`, `test_advisor_scalability_*`, `test_advisor_observability_*`, `test_cached_result`, `test_butler_scan`

### Files changed

- `src/butler.rs` (+993 lines)
- Updated: `AGENT_SYSTEM_DESIGN.md`, `CONCEPTS.md`, `GUIDE.md`

---

## 3. Benchmark Suites + RAG Tier Expansion (commit `ffba4ec`)

**Theme**: Evaluation framework expansion and RAG feature coverage.

### Eval Suite Enhancements

5 new `BenchmarkSuiteType` variants:

| Suite | Description |
|-------|-------------|
| `LiveCodeBench` | Live coding tasks with real-time evaluation |
| `AiderPolyglot` | Multi-language coding assessment |
| `TerminalBench` | Command-line and shell tasks |
| `APPS` | Algorithm and programming problems |
| `CodeContests` | Competitive programming challenges |

Plus: 3 `AnswerFormat` variants, 4 helper functions, 2 filters, heuristic scoring, format-specific prompt enrichment.

### RAG Tier Expansion (20 → 28 features)

8 new RAG features added across tiers:

| Feature | Description |
|---------|-------------|
| `discourse_chunking` | Discourse-aware text segmentation |
| `deduplication` | Near-duplicate detection and removal |
| `diversity_mmr` | Maximal Marginal Relevance diversity |
| `cascade_reranking` | Multi-stage reranking pipeline |
| `web_search_augmentation` | Web search result integration |
| `memory_augmented` | Memory-enhanced retrieval |
| `entity_extraction` | Named entity recognition in chunks |
| `multi_layer_graph` | Graph-based multi-layer retrieval |

All tier mappings updated, 5 new `RagRequirement` variants added.

### Documentation fixes

- `concepts.html`: Fixed rendering bug (unescaped HTML tags in `<code>` elements)
- `framework_comparison.html`: New "Documentation, DX & Economics" category (6 rows: docs quality, ease of use, test coverage, performance, license, cost model)

### Tests (+44)

32 eval suite tests + 12 RAG tier tests.

---

## 4. MCP Config/Eval Tools + Emoticon Detection + BPE Tokenizer (commit `a42a38b`)

**Theme**: MCP tooling completeness, text analysis, token estimation unification.

### MCP Config Tools (6 tools)

| Tool | Description |
|------|-------------|
| `get_provider_config` | Read provider configuration |
| `set_provider_config` | Update provider configuration |
| `list_providers_config` | List all configured providers |
| `validate_config` | Validate configuration consistency |
| `get_provider_url` | Get provider endpoint URL |
| `can_write_provider_config` | Check write permission gate |

### MCP Eval Tools (6 tools)

| Tool | Description |
|------|-------------|
| `list_eval_suites` | List available benchmark suites |
| `create_eval_dataset` | Create evaluation dataset |
| `run_eval_suite` | Execute benchmark suite |
| `get_eval_result` | Retrieve evaluation results |
| `compare_eval_results` | Compare results across runs |
| `generate_eval_report` | Generate formatted report |

### Emoticon & Emoji Detection

- `EmojiClassifier` with 50+ patterns covering emoticons, kaomoji, and Unicode emoji
- Sentiment scoring and density analysis
- Integration with text analysis pipeline

### Token Estimation Unification

- Replaced 7 local `estimate_tokens` copies across the crate with central `crate::context::estimate_tokens`
- Model-aware BPE routing: GPT, Claude, Gemini, Mistral, DeepSeek
- Expanded `ProviderTokenCounter` with new models: o4, claude4, gemini, mistral, deepseek, command-r

### Tests (+37)

17 config MCP tests, 8 eval MCP tests, 12 tokenizer tests.

### Files changed

`analysis.rs`, `config_file.rs`, `eval_suite/mod.rs`, `token_counter.rs`, and 9 files updated for `estimate_tokens` unification.

---

## 5. OpenAI-Compatible API (commit `0bdf179`)

**Theme**: Drop-in OpenAI API compatibility for ecosystem integration.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Non-streaming chat completion |
| POST | `/v1/chat/completions` (stream=true) | SSE streaming chat completion |
| GET | `/v1/models` | List available models |

All also available at `/api/v1/` prefix.

### Request format

```json
{
  "model": "llama3",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

### Response format

**Non-streaming:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
}
```

**Streaming:** `chat.completion.chunk` SSE events with `delta.role` / `delta.content`.

### Compatibility

Any OpenAI-compatible tool can use as a drop-in: Open WebUI, LangChain, LiteLLM, Cursor, Continue, etc.

### Tests (+16)

Format validation, streaming, routing, error responses.

---

## 6. Basic Enrichment Pipeline (commit `304cb85`)

**Theme**: Wire guardrails, RAG, and PII redaction into OpenAI endpoints.

### EnrichmentConfig (6 fields)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_rag` | bool | false | Enable RAG context retrieval |
| `enable_guardrails` | bool | false | Enable input guardrails |
| `enable_memory` | bool | false | Enable conversation memory |
| `block_on_input_violation` | bool | true | Hard-reject unsafe input |
| `redact_output_pii` | bool | true | Automatic PII masking |
| `guardrail_threshold` | f32 | 0.8 | Confidence cutoff |

### Pipeline flow

1. **Input guardrails** — AttackGuard, PiiGuard, ToxicityGuard, ContentLengthGuard
2. **RAG context** — knowledge retrieval appended to system prompt
3. **LLM generation** — provider call
4. **Output guardrails** — PII redaction on response text

### Tests (+15)

Config defaults, serialization, pipeline init, attack blocking, clean passthrough, disabled passthrough, error format, stream blocking, RAG graceful, PII redaction.

---

## 7. Full Enrichment Config Expansion (this commit)

**Theme**: Expose ALL configurable capabilities through 7 nested sub-configs (52 total fields).

### 7 Sub-Config Structs

#### `GuardrailEnrichmentConfig` (18 fields)

| Field | Type | Default |
|-------|------|---------|
| `attack_guard` | bool | true |
| `pii_guard` | bool | true |
| `toxicity_guard` | bool | true |
| `content_length_guard` | bool | true |
| `rate_limit_guard` | bool | false |
| `rate_limit_max_requests` | usize | 60 |
| `rate_limit_window_secs` | u64 | 60 |
| `blocked_patterns` | Vec\<String\> | [] |
| `output_pii_guard` | bool | true |
| `output_pii_action` | String | "redact" |
| `output_pii_redact_char` | char | '*' |
| `output_pii_check_emails` | bool | true |
| `output_pii_check_phones` | bool | true |
| `output_pii_check_ssns` | bool | true |
| `output_pii_check_credit_cards` | bool | true |
| `output_pii_check_ip_addresses` | bool | true |
| `output_toxicity_guard` | bool | false |
| `output_toxicity_threshold` | f64 | 0.7 |

#### `RagEnrichmentConfig` (8 fields)

| Field | Type | Default |
|-------|------|---------|
| `knowledge_rag` | bool | true |
| `conversation_rag` | bool | false |
| `max_knowledge_tokens` | usize | 2000 |
| `max_conversation_tokens` | usize | 1500 |
| `top_k_chunks` | usize | 5 |
| `min_relevance_score` | f32 | 0.1 |
| `dynamic_context` | bool | false |
| `auto_store_messages` | bool | false |

#### `ContextEnrichmentConfig` (5 fields)

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | false |
| `total_budget` | usize | 4096 |
| `response_reserve` | usize | 1024 |
| `overflow_detection` | bool | true |
| `hybrid_compaction` | bool | false |

#### `CompactionEnrichmentConfig` (6 fields)

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | false |
| `max_messages` | usize | 50 |
| `target_messages` | usize | 20 |
| `preserve_recent` | usize | 10 |
| `preserve_first` | usize | 2 |
| `min_importance` | f64 | 0.8 |

#### `ModelSelectionEnrichmentConfig` (5 fields)

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | false |
| `optimize_cost` | bool | false |
| `optimize_speed` | bool | false |
| `min_quality` | f64 | 0.7 |
| `enable_learning` | bool | true |

#### `CostEnrichmentConfig` (5 fields)

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | false |
| `daily_limit` | Option\<f64\> | None |
| `monthly_limit` | Option\<f64\> | None |
| `per_request_limit` | Option\<f64\> | None |
| `warning_threshold` | f64 | 0.8 |

#### `ThinkingEnrichmentConfig` (7 fields)

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | false |
| `min_depth` | ThinkingDepth | Trivial |
| `max_depth` | ThinkingDepth | Expert |
| `inject_cot_instructions` | bool | true |
| `parse_thinking_tags` | bool | true |
| `strip_thinking_from_response` | bool | true |
| `adjust_temperature` | bool | true |

### Expanded EnrichmentConfig

6 top-level fields (unchanged) + 7 `#[serde(default)]` sub-config fields → **52 total configurable fields**.

### Selective Guard Pipeline

`init_guardrail_pipeline()` rewritten to conditionally add each guard:

- Individual toggle for each guard type
- `RateLimitGuard` with configurable max requests and window
- `PatternGuard` for custom blocked patterns
- `OutputPiiGuard` with configurable action (block/redact), redact character, and per-type toggles
- `OutputToxicityGuard` with configurable threshold

### Budget Manager

`init_budget_manager()` creates `BudgetManager` when `cost.enabled`:

- `Arc<Mutex<BudgetManager>>` on `ServerConfig`
- Daily, monthly, and per-request limits
- Pre-check in both handlers returns HTTP 429 when budget exceeded

### Handler Updates

Both non-streaming and streaming handlers enhanced:

1. **Cost pre-check** — return 429 if budget exceeded
2. **RAG sub-config** — apply `rag.*` fields to `AiAssistant.rag_config`
3. **Compaction** — apply `compaction.*` via `set_compaction_config()`
4. **Adaptive thinking** — apply `thinking.*` to `adaptive_thinking` config
5. **Output guardrails** — use `OutputPiiGuard::new(config).redact()` with full sub-config

### Backward Compatibility

All 46 new fields use `#[serde(default)]`. Empty JSON `{}` deserializes to all features disabled — identical behavior to v28.

### New exports in `lib.rs`

```rust
pub use server::{
    EnrichmentConfig as ServerEnrichmentConfig,
    GuardrailEnrichmentConfig, RagEnrichmentConfig, ContextEnrichmentConfig,
    CompactionEnrichmentConfig, ModelSelectionEnrichmentConfig,
    CostEnrichmentConfig, ThinkingEnrichmentConfig,
};
```

### Tests (+22)

**Sub-config defaults (7):** `test_guardrail_sub_config_defaults`, `test_rag_sub_config_defaults`, `test_context_sub_config_defaults`, `test_compaction_sub_config_defaults`, `test_model_selection_sub_config_defaults`, `test_cost_sub_config_defaults`, `test_thinking_sub_config_defaults`

**Backward compatibility (3):** `test_enrichment_empty_json_backward_compat`, `test_enrichment_sub_configs_in_server_config`, `test_enrichment_old_6field_backward_compat`

**Pipeline customization (7):** `test_selective_guard_disable`, `test_rate_limit_guard_added`, `test_pattern_guard_added`, `test_output_pii_guard_config`, `test_output_toxicity_guard_config`, `test_all_guards_disabled`, `test_pii_action_block_mode`

**Budget manager (2):** `test_budget_manager_created_with_limits`, `test_budget_manager_not_created_when_disabled`

**Integration (3):** `test_full_enrichment_config_no_panic`, `test_thinking_enrichment_defaults`, `test_enrichment_config_52_fields`

---

## Example Configuration

```json
{
  "enrichment": {
    "enable_rag": true,
    "enable_guardrails": true,
    "enable_memory": false,
    "block_on_input_violation": true,
    "redact_output_pii": true,
    "guardrail_threshold": 0.8,

    "guardrails": {
      "attack_guard": true,
      "pii_guard": true,
      "toxicity_guard": true,
      "content_length_guard": true,
      "rate_limit_guard": true,
      "rate_limit_max_requests": 100,
      "rate_limit_window_secs": 60,
      "blocked_patterns": ["DROP TABLE", "rm -rf"],
      "output_pii_guard": true,
      "output_pii_action": "redact",
      "output_pii_redact_char": "*",
      "output_pii_check_emails": true,
      "output_pii_check_phones": true,
      "output_pii_check_ssns": true,
      "output_pii_check_credit_cards": true,
      "output_pii_check_ip_addresses": true,
      "output_toxicity_guard": false,
      "output_toxicity_threshold": 0.7
    },

    "rag": {
      "knowledge_rag": true,
      "conversation_rag": false,
      "max_knowledge_tokens": 2000,
      "max_conversation_tokens": 1500,
      "top_k_chunks": 5,
      "min_relevance_score": 0.1,
      "dynamic_context": false,
      "auto_store_messages": false
    },

    "context": {
      "enabled": false,
      "total_budget": 4096,
      "response_reserve": 1024,
      "overflow_detection": true,
      "hybrid_compaction": false
    },

    "compaction": {
      "enabled": false,
      "max_messages": 50,
      "target_messages": 20,
      "preserve_recent": 10,
      "preserve_first": 2,
      "min_importance": 0.8
    },

    "model_selection": {
      "enabled": false,
      "optimize_cost": false,
      "optimize_speed": false,
      "min_quality": 0.7,
      "enable_learning": true
    },

    "cost": {
      "enabled": true,
      "daily_limit": 10.0,
      "monthly_limit": 100.0,
      "per_request_limit": 0.50,
      "warning_threshold": 0.8
    },

    "thinking": {
      "enabled": true,
      "min_depth": "trivial",
      "max_depth": "expert",
      "inject_cot_instructions": true,
      "parse_thinking_tags": true,
      "strip_thinking_from_response": true,
      "adjust_temperature": true
    }
  }
}
```

---

## Deferred (documented as future work)

- `tools` / `function_call` — tool-use via OpenAI format
- HITL approval gates in enrichment pipeline
- MCP client integration in enrichment pipeline
- Session persistence / conversation history
- `structured_output` / `response_format`
- Full `ContextComposer` integration

---

## Changes to Existing Files

| File | Changes |
|------|---------|
| `src/server.rs` | 7 sub-configs, expanded EnrichmentConfig, pipeline rewrite, handler updates, 22 tests (+945 lines) |
| `src/lib.rs` | 8 new type exports (+7 lines) |
| `docs/IMPROVEMENTS_V29.md` | **New** — this changelog |

---

## Verification

```bash
# Compile check
cargo check --features full

# Clippy
cargo clippy --features full -- -D warnings

# Run new tests
cargo test --features full --lib -- server::tests

# Full regression
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite" --lib
```

Expected: 6,565 tests passed, 2 pre-existing failures (byte serialization), 0 clippy warnings.
