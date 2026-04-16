# IMPROVEMENTS V82 — Faithfulness & Grounded Generation (0.2.14)

## Context

V82 adds claim-level faithfulness scoring (NLI) and grounded generation
to the anti-hallucination framework. Every claim in a response can now be
checked against retrieved context, and every sentence can be anchored to
a source chunk.

---

## Changes

### New Module

**`src/faithfulness.rs`** (~380 lines) — NLI-based faithfulness scoring.

| Type | Description |
|------|-------------|
| `NliVerdict` | Entailed, Contradicted, Neutral |
| `DecompositionMethod` | SentenceSplit, LlmDecomposition |
| `NliMethod` | WordOverlap (zero-cost), LlmNli (1 call) |
| `FaithfulnessConfig` | Method selection, thresholds, strategy |
| `AtomicClaim` | Single verifiable assertion with position |
| `ClaimFaithfulness` | Per-claim verdict + confidence + supporting chunks |
| `FaithfulnessReport` | Overall score, counts, processed text |
| `FaithfulnessScorer` | Main scorer: decompose + evaluate + apply strategy |

### Extended Modules

**`src/anti_hallucination.rs`** — Grounded Generation:
- `ChunkAnchorMethod` enum (PostHoc, Prompted)
- `GroundedGenerationConfig` — enable/disable, strategy, similarity threshold
- `GroundedGenerator` — anchor sentences to sources, process unanchored
- `GroundedGenerationResult` — grounding ratio, per-sentence anchoring

**`src/hallucination_detection.rs`**:
- `extract_claims()` made public (was private)
- `decompose_atomic(text)` — finer-grained atomic claim decomposition
  for faithfulness NLI evaluation

**`src/citations.rs`**:
- `SourceType::AcademicPaper` — new variant for academic papers
- `anchor_to_sources(sentences, sources, min_similarity)` — anchor
  sentences to best-matching source by word overlap

**`src/evaluation.rs`**:
- `MetricType::Faithfulness` — faithfulness score metric
- `MetricType::GroundingRatio` — grounding ratio metric
- `FaithfulnessEvaluator` implementing `Evaluator` trait — evaluates
  faithfulness of response against context

**`src/rag_tiers.rs`** — 2 new fields in `RagFeatures`:
- `faithfulness_scoring` — enabled at Thorough+ tier
- `grounded_generation` — enabled at Thorough+ tier

### Wiring

- `src/lib.rs` — `pub mod faithfulness;` under `#[cfg(feature = "eval")]`

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `faithfulness.rs` | 22 |
| `anti_hallucination.rs` | 8 (GroundedGeneration) |
| `hallucination_detection.rs` | 5 |
| `citations.rs` | 4 |
| `evaluation.rs` | 6 |
| `rag_tiers.rs` | 5 |
| **Total** | **50** |

---

## Version

- **0.2.13 → 0.2.14**
- Total RagFeatures fields: 43 (38 base + 5 anti-hallucination)
