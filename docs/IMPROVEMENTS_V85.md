# IMPROVEMENTS V85 — Paper Metadata & Agent Roles (0.2.17)

## Context

V85 adds paper metadata extraction from plain text and research-focused
agent roles / knowledge graph entity types. All modules gated behind
`research` feature.

---

## Changes

### New Module

**`src/paper_metadata.rs`** (~400 lines) — Paper metadata extraction.

| Type | Description |
|------|-------------|
| `SectionType` | Abstract, Introduction, RelatedWork, Methodology, Results, Discussion, Conclusion, References, Appendix, Other |
| `PaperSection` | title, content, level, section_type, word_count() |
| `PaperMetadata` | title, authors, abstract, keywords, DOI, year, sections, references_raw, page_count, extraction_confidence |
| `ExtractionConfig` | extract_sections, extract_references, max_text_length, min_confidence |
| `PaperMetadataExtractor` | Heuristic extraction: headings, DOI patterns, keywords lines, numbered sections, ALL CAPS headings |

### Extended Modules

**`src/multi_agent.rs`** — 3 new `AgentRole` variants:
- `ResearchAssistant` — searches papers, filters, summarizes
- `PeerReviewer` — critiques drafts, verifies claims
- `WritingCoach` — improves academic writing style

**`src/knowledge_graph.rs`** — 2 new `EntityType` variants:
- `Paper` — academic papers (aliases: article, publication, preprint)
- `Author` — paper authors (aliases: researcher, scientist)
- Updated `as_str()`, `from_str()`, `all()` (7 → 9 variants)

### Wiring

- `src/lib.rs` — `pub mod paper_metadata;` under `#[cfg(feature = "research")]`

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `paper_metadata.rs` | 20 |
| `knowledge_graph.rs` | 2 updated |
| **Total** | **20** |

---

## Version

- **0.2.16 → 0.2.17**
