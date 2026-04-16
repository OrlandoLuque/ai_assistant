# IMPROVEMENTS V83 — Verification Pipeline (0.2.15)

## Context

V83 adds the Chain-of-Verification (CoVe) pipeline, search-integrated fact
verification, and self-consistency divergence metrics to the anti-hallucination
framework. Claims in LLM responses can now be automatically verified against
RAG chunks or web search results.

---

## Changes

### New Module

**`src/chain_of_verification.rs`** (~490 lines) — CoVe pipeline.

| Type | Description |
|------|-------------|
| `VerificationSource` | RagOnly, WebSearchOnly, RagThenWeb, Both |
| `CorrectionMode` | Replace, Annotate, Footnote |
| `ClaimVerificationStatus` | Supported, Contradicted, Unverifiable, PartiallySupported |
| `CoVeConfig` | Source, mode, thresholds, LLM call budget, presets |
| `VerifiedClaimResult` | Per-claim verdict + confidence + evidence + correction |
| `VerificationEvidence` | Source, content, relevance, supports flag |
| `CoVeResult` | Overall accuracy, corrections count, per-claim results |
| `ChainOfVerification` | Main pipeline: extract claims + verify + correct |

### Extended Modules

**`src/fact_verification.rs`** — Search-integrated verification:
- `SearchVerifiedFact` — result with search source provenance
- `SearchFactSource` — per-source snippet + relevance + support flag
- `verify_with_search(claim, search_results)` — verify against web search
- `verify_with_rag(claim, rag_chunks)` — verify against RAG chunks

**`src/self_consistency.rs`** — Divergence metrics:
- `ConsistencyRecommendation` enum (High, Medium, Low, Abstain)
- `DivergenceMetrics` — entropy, max_group_ratio, distinct_groups,
  effective_distinct, recommendation
- `ConsistencyResult::measure_divergence()` — compute divergence from groups

**`src/web_search.rs`**:
- `search_for_claim(engine, claim, max_results)` — keyword extraction +
  stopword filtering + relevance scoring for claim verification

**`src/rag_tiers.rs`** — 2 new fields in `RagFeatures`:
- `chain_of_verification` — enabled at Agentic+ tier
- `fact_check_search` — enabled at Agentic+ tier
- Updated `is_feature_enabled()` with all missing fields (V62-V83)

### Wiring

- `src/lib.rs` — `pub mod chain_of_verification;` under `#[cfg(feature = "eval")]`

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `chain_of_verification.rs` | 26 |
| `fact_verification.rs` | 5 |
| `self_consistency.rs` | 5 |
| `web_search.rs` | 3 |
| `rag_tiers.rs` | 6 |
| **Total** | **45** |

---

## Version

- **0.2.14 → 0.2.15**
- Total RagFeatures fields: 45 (38 base + 5 anti-hallucination + 2 verification)
