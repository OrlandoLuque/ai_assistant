# V126 — Phase C.5: performance budgets active

**Date**: 2026-05-06
**Version**: 0.2.73
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.5
**Tasks**: #335 (V126 C.5 — performance budgets active)

## Why

The CI `benchmark` job has been running criterion benches since
V108, but with `continue-on-error: true` and `alert-threshold: 200%`
on the `github-action-benchmark` step. That meant a 1.99× slowdown
on a hot path (intent classifier, guardrails, token counter) would
land silently — the alert would fire informationally but the PR
would still merge green. The plan calls for *active* budgets: a
hard CI gate with explicit ceilings per benchmark, so a regression
in user-perceived latency cannot ship without an explicit budget
bump in the same PR.

## What changed

### `bench_budget.toml`

A new top-level file declaring per-bench `max_ns` ceilings. Format:

```toml
[budgets]
"intent_classification"     = { max_ns = 250_000, note = "every-request, classifies user intent" }
"guardrail_check_input"     = { max_ns = 500_000, note = "every-request, safety pre-filter" }
# ... 15 budgets covering hot-path benches
```

Methodology:

- **Scope**: opt-in. Only benches listed in `[budgets]` are gated;
  the other ~30 criterion benches in `core_benchmarks.rs` and
  `vision_benchmarks.rs` continue to run informationally. Adding a
  new entry is the explicit signal "this bench protects a user-
  visible latency contract".
- **Ceiling**: `budget = observed_max * 1.5`. The 50 % slack
  absorbs runner jitter on GH-hosted ubuntu-latest (single vCPU
  shared host) without letting a real 2× regression slip through.
- **Trade-offs**: when a feature deliberately trades speed for
  correctness or security (e.g. heavier prompt-injection scan), the
  same PR bumps the budget and explains why in `note = `. The
  reviewer can then question the trade-off explicitly instead of
  waiting for a production p99 alert.

The first 15 budgets cover three categories:

| Category | Benches | Why |
|---|---|---|
| Per-request safety | intent_classification, guardrail_check_input, attack_detect_clean / adversarial, pii_detection_1k_chars, rate_limiter_check | Run on every user message; latency hits time-to-first-token directly. |
| Per-request context | bpe_token_count_200_words, context_window_trim_50_msgs | Run on every message and on compaction. |
| Per-RAG-query | cosine_similarity_384d / 1536d, hnsw_search_1k_vectors_128d, rag_fts_search_100_docs, rag_build_context_20_chunks | Run per RAG-augmented query; sit on the critical path of the answer. |
| Crypto / compression | request_signing_hmac_sha256, gzip_compress_1kb_json | Per-request when middleware is active. |

### `scripts/check_bench_budget.py`

Python 3.11+ checker (uses stdlib `tomllib`, available on
ubuntu-latest by default):

- Reads `bench_budget.toml` and the bencher-format `output.txt` the
  benchmark step already produces (filtered from `bench_full.log`).
- Cross-checks each measured benchmark against its budget. Benches
  without a budget entry are skipped (opt-in gate).
- Prints both PASS and FAIL lines so the CI log shows the full
  measured-vs-budget table, useful when debugging a near-miss.
- Exit codes:
  - `0` — all measured benches within budget.
  - `1` — one or more measured benches over budget (gate fails).
  - `2` — config error (`bench_budget.toml` missing or empty).
- If the bench harness produced no measurements at all, the script
  warns loudly but exits `0` — a broken harness is a CI bug
  separate from a budget regression, surfaced upstream by the
  benchmark step's `head/tail` log dumps.

The PASS/FAIL output is plain ASCII (no unicode glyphs) so the GH
Actions log renders cleanly on every runner regardless of locale.

### `.github/workflows/ci.yml` (benchmark job)

- `continue-on-error: false` — the job now blocks merges on
  failure (was `true` before V126).
- New step `Check bench budget (V126 / C.5)` runs immediately after
  the `Run benchmarks` step, calling `python3 scripts/check_bench_budget.py`.
  Gated on `steps.bench.outputs.have_output == 'true'` so a
  scheduled-but-skipped bench run doesn't false-fail the gate.
- `github-action-benchmark` step:
  - `alert-threshold: '125%'` (was `200%`). Still informational
    (`fail-on-alert: false`); the *real* gate is the budget step
    above. The alert now flags a 25 % regression instead of waiting
    for a 2× blow-up.
  - `comment-on-alert: true` and `auto-push: false` unchanged.

## Compatibility

- Pure addition: no `Cargo.toml` deps changed, no library code
  changed, no test count changed.
- The CI step uses Python 3 from the ubuntu-latest runner (3.12 by
  default since April 2024) — no extra setup-python action needed.
  If a future runner downgrades to <3.11, the script's import
  guard prints a clear actionable error and exits with code `2`.

## What's next

- V127 / C.6 — Feature-flag deprecation policy (`#[deprecated]`
  convention, per-feature CHANGELOG, `experimental_*` canary prefix).
