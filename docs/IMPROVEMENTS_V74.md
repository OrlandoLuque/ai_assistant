# Improvements V74 — CI Improvements + ContextBudgetAllocator Full Integration

## Workstream A: CI Improvements

| # | Item | File | Estado |
|---|------|------|--------|
| 1 | Standardize feature sets with `FEATURES_STD` / `FEATURES_NETWORK` env vars | ci.yml | HECHO |
| 2 | Expand feature-matrix (19 → 36 combinations) | ci.yml | HECHO |
| 3 | `audit` job (cargo audit via rustsec/audit-check) | ci.yml | HECHO |
| 4 | Integration tests in `test` job (`cargo test --test '*'`) | ci.yml | HECHO |
| 5 | `binaries` job (ai_cli, ai_assistant_cli, ai_assistant_server, kpkg_tool, ai_test_harness) | ci.yml | HECHO |
| 6 | Fix coverage features (aligned with FEATURES_STD) | ci.yml | HECHO |
| 7 | Release needs updated: `[check, test, clippy, fmt, binaries]` | ci.yml | HECHO |

## Workstream B: ContextBudgetAllocator Full Integration

| # | Item | File | Estado |
|---|------|------|--------|
| B1 | `ContextBudgetConfig` struct + `ScoringMode` enum + Default + validation | context_budget.rs | HECHO |
| B2 | `adjust_score_for_intent()` heuristic scoring (intent → source boost mapping) | context_budget.rs | HECHO |
| B3 | `ScoringMode` in `RagFeatures` + tier defaults (Enhanced=Heuristic, Thorough+=Hybrid) | rag_tiers.rs | HECHO |
| B4 | Wire `context_budget_config` into `AiAssistant` + builder | assistant.rs | HECHO |
| B5 | Replace all hardcoded scores/limits in `build_allocated_context()` + intent in `send_message` | assistant.rs | HECHO |
| B6 | Knowledge graph as separate `ContextItem` (extracted from `build_rag_context()`) | assistant.rs | HECHO |
| B7 | `StrategyBandit` wired into production (UCB1 arm selection + utilization reward) | assistant.rs, context_budget.rs | HECHO |
| B8 | `LlmEnhancerCompressor` bridge (LlmEnhancer → LlmCompressor adapter) | context_budget.rs | HECHO |
| B9 | Re-exports: `ContextBudgetConfig`, `ScoringMode`, `LlmEnhancerCompressor` | lib.rs | HECHO |
| B10 | 16 new tests (config validation, scoring modes, bandit, compressor, graph) | context_budget.rs | HECHO |

## New Types

### `ScoringMode` (4 variants)
- `Static` — Use base scores from config (default, zero cost)
- `Heuristic` — Adjust scores using IntentClassifier + intent-to-source boost mapping (zero cost)
- `LlmEnhanced` — LLM classifies query and returns per-source weights (1 LLM call)
- `Hybrid { confidence_threshold }` — Heuristic first, LLM when confidence < threshold

### `ContextBudgetConfig` (15 fields)
Centralizes all previously-hardcoded values:
- 6 per-source base scores (rag, memory, procedural, reference, graph, notes)
- `scoring_mode` — which dynamic scoring method to use
- Token limits (memory_max, procedural_max, procedural_max_items)
- Response reserve, compression thresholds, overflow strategy
- `enable_strategy_learning` — activates StrategyBandit

### `LlmEnhancerCompressor`
Adapter that bridges `LlmEnhancer` trait (V68) to `LlmCompressor` trait for context compression. Falls back to extractive compression on LLM failure.

## Intent-to-Source Boost Mapping

| Intent | RAG | Memory | Procedural | Graph |
|--------|-----|--------|-----------|-------|
| Question | +0.05 | +0.10 | 0 | +0.05 |
| CodeRequest | +0.10 | 0 | +0.10 | 0 |
| Explanation | +0.10 | 0 | 0 | +0.10 |
| Comparison | +0.10 | 0 | 0 | +0.15 |
| Command | 0 | 0 | +0.15 | 0 |
| Greeting/Farewell/Thanks | -0.20 | -0.20 | -0.20 | -0.20 |
| Chitchat | -0.10 | -0.10 | -0.10 | -0.10 |

## StrategyBandit Production Wiring

- `strategy_bandit: Option<StrategyBandit>` in `AiAssistant`
- Activated when `enable_strategy_learning = true`
- UCB1 selects between 6 arms: score_truncation, extractive_light/medium, llm_light/medium/aggressive
- `arm_to_strategy()` maps arm name → `OverflowStrategy` (LLM arms require compressor_model)
- Reward = allocation utilization ratio (0.0–1.0)
- Persistence via `StorageContext.save_json/load_json`

## Graph Context Refactor

- Graph entities were previously appended inline inside `build_rag_context()` (double-counted with RAG)
- Extracted to `build_graph_context_string()` — standalone method
- Added as separate `ContextItem` with `ContextSourceType::Graph` in `build_allocated_context()`
- Gets its own score (default 0.85) and responds to intent-based boosts independently

## Test Count

- Before: 7,387 (V73)
- After: 7,469 (+82)
- New context_budget tests: 34 total (was 18)

## Files Modified

| File | Changes |
|------|---------|
| `src/context_budget.rs` | +ContextBudgetConfig, +ScoringMode, +intent_source_boost, +LlmEnhancerCompressor, +arm_to_strategy, +16 tests |
| `src/assistant.rs` | +context_budget_config field, +strategy_bandit field, +build_graph_context_string, refactored build_allocated_context, updated 6 call sites |
| `src/rag_tiers.rs` | +context_scoring_mode in RagFeatures, tier defaults |
| `src/lib.rs` | +3 re-exports |
| `.github/workflows/ci.yml` | Complete rewrite: env vars, expanded matrix, audit, binaries, integration tests |
| `docs/modus-operandi.md` | Updated V74 entry, test count |
