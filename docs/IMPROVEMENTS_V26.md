# v26 Improvements Changelog

> Date: 2026-03-01

v26 extends the `eval-suite` feature with **per-subtask model routing** and a **configuration search engine** — enabling automatic optimization of which model handles each subtask (reasoning, coding, formatting, etc.) with statistical significance testing, evolution tracking, and cost/quality trade-off analysis.

---

## Summary Metrics

| Metric | v25 | v26 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 6,061 | 6,133 | +72 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 3 | +3 |
| New error variants | 0 | 2 (`SearchFailed`, `InvalidAgentConfig`) | +2 |
| Lines added | 0 | ~5,950 | +5,950 |

---

## New Module: `agent_config.rs` (795 lines, 16 tests)

**Purpose**: Per-subtask model routing configuration, multi-model generator dispatch, and search dimension definitions.

### Key Types

- **`EvalAgentConfig`** — Agent workflow config with per-subtask model assignments
  - `default_model: ModelIdentifier` — fallback model for untagged subtasks
  - `subtask_models: HashMap<String, ModelIdentifier>` — model per subtask (e.g., "CodeGeneration" → gpt-4)
  - `subtask_temperatures`, `subtask_cot`, `subtask_max_tokens`, `subtask_templates` — per-subtask overrides
  - `cost_per_call: HashMap<String, f64>` — cost estimation per model for optimization
  - Builder pattern: `with_subtask_model()`, `with_temperature()`, `with_chain_of_thought()`, etc.
  - Helpers: `model_for_subtask()`, `temperature_for_subtask()`, `cot_for_subtask()`, `cost_for_model()`
  - `diff(&self, other) -> Vec<String>` — human-readable config comparison

- **`SearchDimension`** — Dimension that can be varied during search
  - Variants: `SubtaskModel`, `Temperature`, `SubtaskTemperature`, `ChainOfThought`, `SubtaskChainOfThought`, `RagLevel`, `MaxTokens`, `SubtaskMaxTokens`
  - Methods: `name()`, `variant_count()`, `variant_label()`, `apply_variant()`

- **`MultiModelGenerator`** — Routes prompts to different LLM generators based on model identity
  - `new(default_generator)` — create with fallback generator
  - `register_model(model_key, generator)` — register model-specific generator
  - `generate(model_key, prompt)` — dispatch to correct generator

- **`ConfigMeasurement`** — Snapshot of quality/cost/latency for one configuration
  - `per_problem_scores()` — extract per-problem scores for t-test

---

## New Module: `config_search.rs` (1,594 lines, 17 tests including 1 `#[ignore]` integration test)

**Purpose**: Systematic configuration search engine using one-at-a-time coordinate descent with statistical significance testing, evolution tracking, and cost optimization.

### Algorithm

One-at-a-time coordinate descent (NOT grid search — O(sum(V_i)) instead of O(V^D)):

1. Measure baseline configuration
2. For each search dimension, sweep all variant values
3. Use Welch's t-test for statistical significance (quality-focused objectives)
4. Adopt improvement only if statistically significant
5. If `adaptive_priority` enabled, re-sweep highest-variance dimensions
6. Track evolution (quality/cost/latency) at each iteration
7. Convergence detection + budget guard

### Key Types

- **`ConfigSearchConfig`** — Search parameters
  - `confidence_level` (default 0.95), `min_samples` (default 5), `max_evaluations` (default 100)
  - `adaptive_priority` (default true), `objective: SearchObjective`

- **`SearchObjective`** — Optimization target
  - `MaxQuality` — maximize accuracy regardless of cost
  - `MinCost { min_quality }` — minimize cost above quality threshold
  - `CostEfficiency` — maximize quality/cost ratio
  - `Weighted { quality_weight, cost_weight, latency_weight }` — custom weighting

- **`ConfigSearchEngine`** — Core engine
  - `search(baseline, dimensions, dataset, subtask_tags)` — run full optimization
  - `measure(config, dataset, subtask_tags)` — evaluate single configuration
  - Own run loop (not using `BenchmarkSuiteRunner` — needs per-subtask routing context)
  - Objective-aware `is_better()` comparison (t-test for quality, deterministic for cost)

- **`ConfigSearchResult`** — Complete search results
  - `baseline` / `best` — before/after measurements
  - `evolution: Vec<EvolutionSnapshot>` — quality/cost/latency over iterations
  - `dimension_variance: HashMap<String, f64>` — impact per dimension
  - `search_cost: SearchCost` — cost of the search process itself
  - `converged`, `stopped_by_budget` — termination flags
  - `top_dimensions(n)`, `improvements()`, `summary()` — analysis helpers

- **`EvolutionSnapshot`** — Tracks optimization progress per iteration
  - Includes `objective_score` for the active `SearchObjective`

- **`SearchCost`** — Tracks total cost of the search process

### Integration Test

`test_integration_ollama_real_models` (`#[ignore]`):
- Configurable via `OLLAMA_URL` and `EVAL_MODELS` environment variables
- Raw HTTP POST to Ollama `/api/generate` (no external dependencies)
- 8 problems: multiple-choice (math, reasoning, knowledge, calculation) + numeric + code + free text
- Tests per-subtask model routing with real model responses
- Prints evolution curve and summary to stdout

---

## Changes to Existing Files

### `src/error.rs` (+30 lines)
- Added 2 new `EvalSuiteError` variants:
  - `SearchFailed { reason }` — configuration search failed (recoverable)
  - `InvalidAgentConfig { field, reason }` — invalid agent configuration (not recoverable)
- Added Display, suggestion(), is_recoverable() arms
- Updated existing error test

### `src/eval_suite/mod.rs` (+12 lines)
- Added `mod agent_config;` and `mod config_search;`
- Added `pub use` re-exports for all new public types

### `src/eval_suite/report.rs` (+80 lines)
- Added `config_search: Option<ConfigSearchResult>` field to `EvalSuiteReport`
- Added `with_config_search(ConfigSearchResult)` method to `ReportBuilder`
- Added key findings generation from search results (quality improvement %, cost change %, most impactful dimensions)
- Added `test_report_with_config_search` test

### `src/lib.rs` (+6 lines)
- Added new types to `pub use eval_suite::{...}` block:
  `ConfigMeasurement`, `ConfigSearchConfig`, `ConfigSearchEngine`, `ConfigSearchResult`,
  `EvalAgentConfig`, `EvolutionSnapshot`, `MultiModelGenerator`, `SearchCost`,
  `SearchDimension`, `SearchIteration`, `SearchObjective`

---

## Design Decisions

1. **Own run loop in `measure()` vs reusing `BenchmarkSuiteRunner`**: The runner's generator callback takes only `&str` prompt with no subtask context. Per-subtask routing requires knowing which problem maps to which subtask. Solution: `measure()` implements its own loop with subtask-aware model dispatch.

2. **Coordinate descent vs grid search**: Grid search is O(V^D) — combinatorial explosion. One-at-a-time is O(sum(V_i)) — linear. Isolates each dimension's effect. Second adaptive pass catches first-order interactions.

3. **Objective-aware `is_better()`**: MaxQuality/Weighted require t-test significance. MinCost/CostEfficiency use deterministic comparison (cost comes from `cost_per_call`).

4. **Synthetic "routed" ModelIdentifier**: When per-subtask routing is active, `BenchmarkRunResult.model_id` uses `{ name: config.name, provider: "routed" }`. Individual `ProblemResult.model_id` stores the actual model used.

5. **Own welch_t_test copy**: Follows existing pattern — comparison.rs and ablation.rs each have their own copies.

---

## Test Summary (72 new tests)

| Module | Tests | Description |
|--------|-------|-------------|
| `agent_config.rs` | 16 | Config defaults, builder, serialization, subtask routing, temperatures, CoT, costs, dimensions, multi-model dispatch, diff |
| `config_search.rs` | 16 + 1 ignored | Config defaults, objectives, measure, subtask routing, search (single/multiple/adaptive/budget/significance/evolution/cost), result helpers, error handling + Ollama integration |
| `report.rs` | +1 | Config search integration in report builder |
| `feature_combos.rs` | 32 | Cross-feature integration: decision tree (3), embeddings+VDB+neural (8), KG+RAG pipeline (11), episodic/procedural memory (2), multi-agent (2), RAG+memory combos (2), RAG+agent combo (1), full-stack (3) |
| `feature_combos.rs` (heavy) | 7 | Multi-document realistic workflows: KB construction, cross-doc RAG comparison, multi-agent analysis, RAG+evaluation, knowledge evolution with memory, fact verification pipeline, end-to-end document QA (all features) |

---

## New Module: `feature_combos.rs` (~3,500 lines, 39 tests)

**Purpose**: Comprehensive cross-feature integration tests proving that all crate functionalities compose correctly.

### Feature-gated test modules

| Module | Feature gates | Tests |
|--------|---------------|-------|
| `tree_tests` | none | 3 |
| `embedding_tests` | `embeddings` | 8 |
| `rag_tests` | `rag` | 11 |
| `memory_tests` | `advanced-memory` | 2 |
| `agent_tests` | `multi-agent` | 2 |
| `rag_memory_tests` | `rag` + `advanced-memory` | 2 |
| `rag_agent_tests` | `rag` + `multi-agent` | 1 |
| `full_stack_tests` | all 4 features | 3 |
| `heavy_integration_tests` | all 4 features | 7 |

### Heavy Integration Tests (Urban Sustainability domain)

Three ~300-word documents (Barcelona Solar, Copenhagen Transit, Singapore Green Buildings) with 20+ entities across the knowledge graph. Tests exercise realistic multi-document workflows:

1. **Multi-document KB construction** — DocumentParser + KG + VectorDB + Embeddings
2. **Cross-document RAG comparison** — RAG tiers + KG + VDB + DecisionTree + ContextComposer
3. **Multi-agent document analysis** — AgentOrchestrator + SharedContext + KG + DocumentParser
4. **RAG-augmented generation with evaluation** — RAG + ContextComposer + LlmJudge + BenchmarkSuiteRunner
5. **Knowledge evolution with memory** — EpisodicStore + ProceduralStore + KG + VDB
6. **Fact verification pipeline** — DocumentParser + KG + RAG + DecisionTree (6 claims: 3 true, 3 false)
7. **End-to-end document QA pipeline** — ALL components (DocumentParser, KG, VDB, RAG, ContextComposer, DecisionTree, MultiModelGenerator, EpisodicStore, ProceduralStore, AgentOrchestrator, SharedContext, LlmJudge)

---

## Verification

- `cargo check --features full,eval-suite` — clean, 0 warnings
- `cargo test --features full,...,eval-suite --lib` — 6,133 passed, 0 failed, 1 ignored
- `cargo clippy --features full,eval-suite` — 0 warnings
