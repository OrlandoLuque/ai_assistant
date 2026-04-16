# IMPROVEMENTS V84 — Academic APIs & BibTeX (0.2.16)

## Context

V84 adds academic literature search through a unified provider interface
(arXiv, Semantic Scholar, PubMed) and BibTeX parsing/generation with
LaTeX injection sanitization. All modules are gated behind the `research`
feature flag.

---

## Changes

### New Modules

**`src/academic_search.rs`** (~800 lines) — Academic search APIs.

| Type | Description |
|------|-------------|
| `AcademicSource` | ArXiv, SemanticScholar, PubMed, CrossRef (stub), OpenAlex (stub) |
| `Author` | Name, affiliation, provider-specific ID |
| `AcademicPaper` | Full metadata: title, authors, abstract, year, venue, DOI, URL, PDF URL, citation count, fields of study, keywords, external IDs |
| `SortField` | Relevance, Date, Citations |
| `AcademicSearchConfig` | max_results, year_range, fields_of_study, sort_by, timeout |
| `AcademicSearchError` | Network, Parse, RateLimit, NoResults, InvalidQuery, ProviderUnavailable |
| `AcademicSearchProvider` | Trait: search_papers, get_paper, get_citations, get_references |
| `ArxivProvider` | Atom/XML API (3s rate limit) |
| `SemanticScholarProvider` | REST/JSON API (optional API key from `SEMANTIC_SCHOLAR_API_KEY`) |
| `PubMedProvider` | E-utilities XML (optional API key from `NCBI_API_KEY`) |
| `AcademicSearchEngine` | Multi-provider aggregation with DOI-based dedup |

**`src/bibtex.rs`** (~500 lines) — BibTeX parser and generator.

| Type | Description |
|------|-------------|
| `BibEntryType` | Article, Book, InProceedings, InCollection, Thesis, TechReport, Misc, Unpublished, Online |
| `BibEntry` | entry_type, cite_key, fields (HashMap) |
| `BibParser` | Parse `.bib` files with brace nesting, quoted values, bare numbers |
| `BibGenerator` | Generate BibTeX from entries or `AcademicPaper` structs |
| `BibParseError` | FileTooLarge, TooManyEntries, SyntaxError |
| `sanitize_latex()` | Strip dangerous LaTeX commands (20 patterns) |
| `latex_to_unicode()` | Convert accent commands to Unicode (27 patterns) |

### Extended Modules

**`src/citations.rs`** — Academic paper fields on `Source`:
- `doi: Option<String>` — Digital Object Identifier
- `venue: Option<String>` — journal or conference name
- `citation_count: Option<u32>` — citation count

**`src/web_search.rs`** — Academic search adapter:
- `AcademicSearchAdapter` — implements `SearchProvider` by wrapping academic providers
- `paper_to_result()` — converts `AcademicPaper` to `SearchResult`
- `default_providers()` — creates adapters for all 3 providers

### Wiring

- `src/lib.rs` — `pub mod academic_search;` and `pub mod bibtex;` under `#[cfg(feature = "research")]`
- `Cargo.toml` — `research = []` feature flag, included in `full`

---

## Security

| Vector | Mitigation |
|--------|------------|
| SSRF via query | URL-encode all parameters |
| API key leak | Keys from env vars only, never logged |
| Rate limit abuse | Per-provider throttling (arXiv 3s, S2 100/5min, PubMed 3/s) |
| BibTeX injection | 20 dangerous LaTeX commands stripped (sanitize_latex) |
| BibTeX bomb | Max 10MB file, 10K entries, 10K chars per field |

---

## Test Summary

| Module | New Tests |
|--------|-----------|
| `academic_search.rs` | 26 |
| `bibtex.rs` | 23 |
| `web_search.rs` | 3 |
| `citations.rs` | 2 |
| **Total** | **54** |

---

## Version

- **0.2.15 → 0.2.16**
- New feature flag: `research`
