# IMPROVEMENTS V86 — Literature Review Pipeline + MCP Tools (0.2.18)

## Context

V86 adds the literature review pipeline (search → filter → synthesize)
and MCP tool definitions for research operations. Builds on V84 (academic
search) and V85 (paper metadata) modules.

---

## Changes

### New Modules

**`src/literature_review.rs`** (~600 lines) — Literature review pipeline.

| Type | Description |
|------|-------------|
| `SearchDepth` | Quick (20), Standard (50), Deep (100) |
| `SynthesisStyle` | Narrative, Systematic, Annotated, Comparative |
| `BibliographyFormat` | BibTeX, Apa, Mla, Chicago, Ieee |
| `LiteratureReviewConfig` | max_papers, search_depth, synthesis_style, bibliography_format, year_range, fields_of_study |
| `ReviewSection` | heading, content, paper_ids, word_count() |
| `LiteratureReview` | sections, bibliography, bibtex, statistics, to_markdown() |
| `LiteratureReviewPipeline` | execute(query) → LiteratureReview |

**`src/mcp_research_tools.rs`** (~300 lines) — MCP tool definitions.

| Tool | Category | Description |
|------|----------|-------------|
| `search_papers` | Search | Search academic databases |
| `get_paper_metadata` | Metadata | Get paper details by ID |
| `import_bibtex` | Bibliography | Parse .bib content |
| `export_bibtex` | Bibliography | Export citations as BibTeX |
| `literature_review` | Review | Generate literature review |
| `extract_paper_metadata` | Metadata | Extract metadata from text |

### Wiring

- `src/lib.rs` — `pub mod literature_review;` and `pub mod mcp_research_tools;` under `#[cfg(feature = "research")]`

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `literature_review.rs` | 20 |
| `mcp_research_tools.rs` | 11 |
| **Total** | **31** |

---

## Version

- **0.2.17 → 0.2.18**
