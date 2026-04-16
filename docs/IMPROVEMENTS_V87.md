# IMPROVEMENTS V87 — Quality Gates & RAG Tier Integration (0.2.19)

## Context

V87 adds configurable quality gates for LLM output validation and
feature group helpers in RagFeatures for easy mode switching.

---

## Changes

### New Module

**`src/quality_gates.rs`** (~400 lines) — Quality gate system.

| Type | Description |
|------|-------------|
| `QualityMetric` | Faithfulness, Confidence, GroundingRatio, ConsistencyScore, CitationCoverage |
| `GateAction` | Fail (blocks), Warn (allows), Log (silent) |
| `QualityGate` | name, metric, threshold (0.0-1.0), action |
| `GateCheckResult` | per-gate pass/fail with actual vs threshold |
| `QualityScores` | Optional scores for all 5 metrics, overall(), badge_color() |
| `QualityGateRunner` | Runs gates against scores, production_defaults(), strict() |
| `QualityGateResult` | passed, gate_results, failing_gates, warnings, summary() |

### Extended Modules

**`src/rag_tiers.rs`** — Feature group helpers:
- `enable_verification_mode()` — enables 7 anti-hallucination fields
- `enable_research_mode()` — enables 4 research-related fields
- `enable_academic_mode()` — combines research + verification

### Wiring

- `src/lib.rs` — `pub mod quality_gates;` under `#[cfg(feature = "eval")]`

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `quality_gates.rs` | 21 |
| `rag_tiers.rs` | 4 |
| **Total** | **25** |

---

## Version

- **0.2.18 → 0.2.19**
