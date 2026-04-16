# IMPROVEMENTS V81 — Anti-Hallucination Orchestrator + Foundation (0.2.13)

## Context

V81 is the first phase of the Anti-Hallucination Framework (V81-V88). It
establishes the foundation: an orchestrator pipeline, per-claim confidence
scoring, calibrated abstention, auto-temperature for factual queries,
and two new guardrail guards.

These features build on 5 existing modules (~5400 LOC) that already
provide partial detection. V81 extends them — it does NOT duplicate.

---

## Changes

### New Module

**`src/anti_hallucination.rs`** (~580 lines) — Central orchestrator.

| Type | Description |
|------|-------------|
| `UngroundedClaimStrategy` | 7 strategies: Omit, Mark, Warn, Footnote, VerifyThenMark, VerifyThenOmit, Ask |
| `AntiHallucinationConfig` | Full config: strategy, thresholds, toggles, temperature, format |
| `AntiHallucinationPipeline` | Process responses: abstention check, claim decomposition, strategy application |
| `AntiHallucinationResult` | Output: processed text, claims, confidence, abstention status |
| `ProcessedClaim` | Per-claim: text, grounded flag, confidence, source IDs, action taken |
| `is_factual_query()` | Heuristic: factual vs creative query detection |

Preset configurations:
- `AntiHallucinationConfig::production()` — balanced defaults
- `AntiHallucinationConfig::strict()` — high thresholds, verify strategies
- `AntiHallucinationConfig::permissive()` — minimal intervention

### Extended Modules

**`src/confidence_scoring.rs`** — 2 new methods:
- `score_per_claim(claims)` — per-claim confidence using `Claim` structs
- `score_texts(texts)` — simpler API with raw string slices

**`src/adaptive_thinking.rs`** — Auto-temperature for factual queries:
- `AdaptiveThinkingConfig.auto_temperature_factual` — enable/disable (default: off)
- `AdaptiveThinkingConfig.factual_temperature` — temperature for factual queries (default: 0.3)
- `QueryClassifier::is_factual_query()` — keyword + signal-based detection
- Integration in `build_strategy()` — factual queries get forced low temperature

**`src/guardrail_pipeline.rs`** — 2 new guards:
- `AbstentionGuard` — blocks responses when confidence < threshold (PostReceive)
  - Configurable threshold, custom abstention message
  - Lightweight linguistic confidence estimation
- `AttributionGuard` — warns on ungrounded claim patterns (PostReceive)
  - Detects "studies show", "research has shown", "experts say", etc.
  - Configurable mark format and severity

**`src/rag_tiers.rs`** — 3 new fields in `RagFeatures`:
- `calibrated_abstention` — enabled at Thorough+ tier
- `mandatory_attribution` — enabled at Enhanced+ tier
- `auto_temperature` — enabled at Enhanced+ tier

Updated `all()`, `none()`, `enabled_count()`, `enabled_features()`,
and tier mappings (Enhanced, Thorough, Agentic, Graph, Full).

### Wiring

- `src/lib.rs` — `pub mod anti_hallucination;` under `#[cfg(feature = "eval")]`

---

## Test Summary

| Module | New Tests | Total |
|--------|-----------|-------|
| `anti_hallucination.rs` | 28 | 28 |
| `confidence_scoring.rs` | 5 | 15 |
| `adaptive_thinking.rs` | 7 | 54 |
| `guardrail_pipeline.rs` | 17 | 88 |
| `rag_tiers.rs` | 10 | 57 |
| **Total** | **67** | **242** |

---

## Version

- **0.2.12 → 0.2.13**
- Feature flags: No new flags (anti-hallucination modules under existing `eval` flag)
