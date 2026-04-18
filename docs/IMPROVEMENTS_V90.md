# IMPROVEMENTS V90 — Dataset hallucination/faithfulness benchmarks (0.2.22)

## Context

The anti-hallucination pipeline landed in V81–V88 and was wired end-to-end in
V89, but until now the only numbers we had came from *self-tests* —
detectability on in-repo fixtures. To get comparable numbers against published
models and prior art, we needed a way to run the pipeline on *community*
benchmarks: TruthfulQA, HaluEval, FActScore, RAGAS, FEVER. V90 adds that
harness, with zero new dependencies and zero redistributed data — every
dataset is fetched on explicit user action, never vendored.

## Scope

* New module `src/eval_benchmarks/` with a `BenchmarkLoader` trait, on-disk
  cache, HTTP downloader with size caps + atomic writes, a runner, a post-hoc
  threshold calibrator, and text/JSON report renderers.
* Five loaders: `truthfulqa`, `halueval_qa`, `factscore`, `ragas_wikiqa`,
  `fever` (the latter is opt-in behind `--accept-license`).
* CLI: `ai_cli benchmark <list|info|download|run|calibrate>`.
* HTTP server: `GET /benchmarks` and `GET /benchmarks/<name>` (read-only).
* MCP: `list_benchmarks` and `get_benchmark` tools, read-only + idempotent.
* Example: `examples/eval_benchmarks_demo.rs` exercises the full pipeline
  without network or LLM, using the in-tree TruthfulQA fixture and a mock
  generator, so the calibration curve is reproducible in CI.

## Commits (chronological)

| Step | Commit | Subject |
|------|--------|---------|
| V90.1 | `34c8da2` | eval_benchmarks module skeleton — types + BenchmarkLoader trait |
| V90.2 | `3fe854f` | cache + HTTP downloader (atomic .part-rename, 200MB cap) |
| V90.3+4 | `9135b43` | fixtures + TruthfulQA loader (hand-rolled CSV) |
| V90.5 | `c0ae474` | HaluEval QA loader |
| V90.6 | `c90a73f` | FActScore loader (bool-or-string label tolerance) |
| V90.7 | `ea1e4aa` | RAGAS WikiEval via HF datasets-server JSON API |
| V90.8 | `e160834` | FEVER loader (claim-only, fetch-only, opt-in) |
| V90.10 | `7f9915e` | runner + calibration + report |
| V90.11 | `34a9ff1` | cmd_benchmark in ai_cli |
| V90.12 | `9b11622` | server + MCP read-only endpoints |
| V90.13 | `1b8d963` | eval_benchmarks_demo example |

V90.9 (self-consistency module) was skipped — `src/self_consistency.rs`
already provides `ConsistencyChecker` + `VotingConsistency`, so there was
nothing left to build.

## Design notes

### Zero new dependencies

* HTTP: existing `ureq`, no `reqwest`.
* CSV: hand-rolled parser in `loaders/truthfulqa.rs` — handles quoted fields,
  embedded commas, doubled-quote escapes, CRLF. Avoids pulling `csv`.
* RAGAS: HuggingFace `datasets-server` JSON API, not Parquet, so no `parquet`.
* Cache root: `$CARGO_TARGET_DIR/eval_benchmarks` (or `./target/...`), so we
  never needed the `dirs` crate.

### Scoring by sample type

`runner.rs` dispatches per `SampleType` — QA uses Jaccard vs. correct-vs-
incorrect references; HallucinationPair grades the margin between the two
references; AtomicClaims echoes the FActScore precision formula; Supports/
Refutes/NEI uses a keyword label predictor; ContextualQA takes the min of
faithfulness and similarity. Every scorer produces a scalar in [0, 1] plus
a `details` bag so external tools can re-score without re-running the model.

### Post-hoc calibration

`calibration.rs` re-partitions existing scores across a 21-point grid
(0.00–1.00 step 0.05) and picks the threshold that maximises Accuracy or F1.
No second model run needed. This is what `ai_cli benchmark calibrate` wires
into the CLI, and what the demo illustrates end-to-end.

### Opt-in for restrictive licenses

FEVER is CC-BY-SA 3.0. We don't redistribute the data; we only *fetch* on
explicit user action. The loader reports `requires_opt_in() = true` and the
CLI refuses to download unless the caller passes `--accept-license`.

## Quick start

```bash
# List what's registered and see licensing terms.
ai_cli benchmark list
ai_cli benchmark info truthfulqa

# Fetch (caches under target/eval_benchmarks/<loader>/).
ai_cli benchmark download truthfulqa
ai_cli benchmark download fever --accept-license

# Run with a local model.
ai_cli benchmark run truthfulqa \
    --provider ollama --model mistral:7b-instruct --limit 50

# Sweep the correctness threshold to see where this model peaks.
ai_cli benchmark calibrate truthfulqa \
    --provider ollama --model mistral:7b-instruct \
    --limit 200 --objective f1 --json
```

The `--json` flag on `run` / `calibrate` emits a stable machine-readable form
suitable for CI dashboards or diffing across runs.

### HTTP

```
GET /benchmarks            → { "total": N, "benchmarks": [...] }
GET /benchmarks/<name>     → metadata or 404 { "error": "Unknown benchmark: <name>" }
```

### MCP

Tools `list_benchmarks` and `get_benchmark` (annotated read-only + idempotent)
are registered via `mcp_protocol::register_benchmark_tools(&mut server)`.

## Testing

* 59 unit tests in `src/eval_benchmarks/` (types, cache, http, loaders, runner,
  calibration, report).
* 2 MCP tool tests (`list_benchmarks`, `get_benchmark` via the `tools/call`
  wire format).
* 1 HTTP route test (`/benchmarks` + `/benchmarks/<name>` + 404 case).
* `examples/eval_benchmarks_demo.rs` runs the full pipeline end-to-end
  against the in-tree fixture; verifiable manually with:

  ```
  cargo run --example eval_benchmarks_demo --features eval
  ```

## Feature gating

Everything lives behind `feature = "eval"` — already in the workspace. The
CLI prints a helpful error when the feature is missing. The HTTP routes and
MCP tool registration are `#[cfg(feature = "eval")]` so builds without the
feature compile clean.

## Version bump

`0.2.21 → 0.2.22` (patch-level; no API breakage, no new deps).
