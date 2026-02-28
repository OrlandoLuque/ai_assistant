# v25 Improvements Changelog

> Date: 2026-02-28

v25 adds the **`eval-suite` feature**: a comprehensive evaluation and benchmarking system for running standard AI benchmarks against real LLMs, comparing models, measuring technique impact, and generating reports.

---

## Summary Metrics

| Metric | v24 | v25 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,963 | 6,061 | +98 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 1 (`eval-suite`) | +1 |
| New source files | 0 | 8 | +8 |
| New error types | 0 | 1 (`EvalSuiteError`) | +1 |
| Lines added | 0 | ~3,961 | +3,961 |

---

## New Feature: `eval-suite`

**Purpose**: Run standard benchmark suites (HumanEval, MMLU, GSM8K, SWE-bench, etc.) against real LLMs via a callback-based API, compare models, measure technique impact, and generate comprehensive reports.

**Feature flag**: `eval-suite` (depends on `eval`, NOT in `full` — requires network/LLM access)

### Architecture

Uses the same callback pattern as `LlmJudgeEvaluator`:
```rust
let runner = BenchmarkSuiteRunner::new(|prompt: &str| {
    // Call any LLM provider here
    Ok("response".to_string())
});
```

### Module Structure (8 files)

| File | Purpose | Lines | Tests |
|------|---------|-------|-------|
| `mod.rs` | Module re-exports | ~50 | — |
| `dataset.rs` | Benchmark dataset loading (JSONL/JSON), problem definitions | ~530 | 16 |
| `scoring.rs` | Pass@k, accuracy, ELO, answer extraction, DefaultScorer | ~500 | 22 |
| `runner.rs` | Benchmark execution engine with cost tracking | ~500 | 14 |
| `comparison.rs` | Multi-model comparison matrix with Welch's t-test | ~500 | 10 |
| `ablation.rs` | A/B technique impact studies with Cohen's d | ~430 | 11 |
| `subtask.rs` | Per-subtask analysis + optimal model routing | ~370 | 10 |
| `report.rs` | Report generation + JSON export | ~370 | 11 |

### Key Types

- `BenchmarkSuiteType` — HumanEval, MBPP, SWE-bench, MMLU, GSM8K, ARC, AgentBench, TaskBench, GAIA, Custom
- `BenchmarkProblem` — problem with id, prompt, answer format, reference solution, metadata
- `BenchmarkDataset` — collection of problems with from_jsonl(), from_json(), filter(), sample()
- `AnswerFormat` — Code, MultipleChoice, Numeric, FreeText, AgentTrajectory
- `BenchmarkSuiteRunner` — orchestrates LLM calls via generator callback
- `RunConfig` — samples_per_problem, temperature, max_tokens, timeout, chain_of_thought
- `ModelIdentifier` — name + provider + variant for cost tracking
- `ProblemResult` — responses, scores, latencies, token counts, cost per problem
- `BenchmarkRunResult` — aggregated results for an entire dataset
- `ComparisonMatrix` — model × metric with significance, ELO, cost-effectiveness
- `AblationStudy` / `AblationResult` — control vs treatment with Cohen's d + recommendations
- `SubtaskAnalysis` — per-subtask performance with optimal model routing
- `EvalSuiteReport` / `ReportBuilder` — comprehensive reports with cost breakdowns

### Scoring

- Multiple-choice: letter extraction from various response patterns
- Numeric: tolerance-based with partial credit
- Code: heuristic Jaccard similarity against reference
- Free text: exact/fuzzy/containment matching
- Pass@k: unbiased estimator from Codex paper
- ELO: pairwise model ranking

### Error Type

`EvalSuiteError` with 8 variants: DatasetLoadFailed, InvalidProblem, GenerationFailed, ScoringFailed, NoResults, InsufficientData, ReportFailed, Timeout.

---

## Verification

- `cargo check --features full,eval-suite` — clean, 0 warnings
- `cargo test --features full,...,eval-suite --lib` — 6,061 passed, 0 failed
- `cargo clippy --features full,eval-suite` — 0 warnings
