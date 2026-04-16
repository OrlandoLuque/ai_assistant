# Academic Research Guide

## Why This Matters

Searching for academic papers is tedious. You open arXiv, Semantic Scholar, and
PubMed in separate tabs. You copy-paste titles into a spreadsheet. You manually
cross-reference citations. You format BibTeX by hand, fixing broken entries one
by one. And when you finally ask an LLM to summarize the literature, you have
no way to know whether it invented a citation that does not exist.

`ai_assistant` eliminates all of that. One command searches three academic APIs
simultaneously, generates a structured literature review, exports valid BibTeX,
and -- critically -- **verifies its own output against the source papers** using
the anti-hallucination pipeline. No other framework does this.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Feature Flag and Configuration](#2-feature-flag-and-configuration)
3. [Academic Search Providers](#3-academic-search-providers)
4. [The AcademicPaper Data Model](#4-the-academicpaper-data-model)
5. [CLI Commands](#5-cli-commands)
6. [Literature Review Pipeline](#6-literature-review-pipeline)
7. [BibTeX Support](#7-bibtex-support)
8. [Paper Metadata Extraction](#8-paper-metadata-extraction)
9. [Anti-Hallucination Integration](#9-anti-hallucination-integration)
10. [Research Agent Roles](#10-research-agent-roles)
11. [Server and API Integration](#11-server-and-api-integration)
12. [MCP Tools](#12-mcp-tools)
13. [Security](#13-security)
14. [Comparison vs LangChain and LlamaIndex](#14-comparison-vs-langchain-and-llamaindex)
15. [Cross-References](#15-cross-references)

---

## 1. Quick Start

**What**: Search academic papers and generate a literature review in one command.

**Why**: A researcher preparing a related-work section should not have to
manually query three different APIs, deduplicate results, and format references.

**How**:

```bash
# Search across all providers, filter to recent papers
cargo run --bin ai_cli --features "full,research" -- research \
    "transformer attention mechanisms" \
    --providers arxiv,scholar,pubmed \
    --max-results 20 \
    --year-range 2020-2026

# Generate a full literature review with BibTeX and anti-hallucination checks
cargo run --bin ai_cli --features "full,research" -- research \
    "reinforcement learning from human feedback" \
    --review \
    --format systematic \
    --bibtex \
    --faithfulness \
    --quality-gates
```

The second command does everything: searches papers, filters by relevance,
synthesizes a structured review, generates a BibTeX bibliography, scores the
output for faithfulness, and applies quality gates to reject hallucinated
citations. Copy, paste, run.

---

## 2. Feature Flag and Configuration

**What**: The `research` feature flag gates all academic search functionality.
It is included in the `full` meta-feature.

**Why**: Not every user needs academic search. Feature flags keep compile times
fast and binary sizes small for users who only need chat or RAG.

**How**:

In `Cargo.toml`:

```toml
[dependencies]
ai_assistant = { version = "0.2", features = ["research"] }
# Or include everything:
ai_assistant = { version = "0.2", features = ["full", "research"] }
```

All research types are gated with `#[cfg(feature = "research")]`.

### TOML Configuration

```toml
[research]
default_providers = ["arxiv", "semantic_scholar"]
default_max_results = 20
default_bibliography_format = "bibtex"
# API keys via env vars (recommended) or config:
# semantic_scholar_api_key = "..."
# ncbi_api_key = "..."
```

API keys should be set via environment variables for security:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
export NCBI_API_KEY="your-key-here"
```

---

## 3. Academic Search Providers

**What**: Three academic search providers are available today, with two more
reserved for Batch 2. All implement the `AcademicSearchProvider` trait.

**Why**: No single database covers all of science. arXiv dominates computer
science preprints, PubMed covers biomedical literature, and Semantic Scholar
provides citation graphs that neither of the others offer. Querying all three
from a unified interface eliminates the "tab-switching" problem.

**How**:

| Provider | API | Auth | Rate Limit | Best For |
|----------|-----|------|------------|----------|
| **arXiv** | Atom/XML | None | 3s between requests | CS, Physics, Math preprints |
| **Semantic Scholar** | REST/JSON | Optional API key | 100 req/5min (free), 1/s with key | CS papers with citation graph |
| **PubMed** | E-utilities XML | Optional API key | 3 req/s (10 with key) | Biomedical, life sciences |
| CrossRef | *(reserved Batch 2)* | | | DOI resolution, metadata |
| OpenAlex | *(reserved Batch 2)* | | | Open scholarly metadata |

### Provider Selection

```bash
# Search only arXiv (no API key needed)
cargo run --bin ai_cli --features "full,research" -- research \
    "diffusion models image generation" \
    --providers arxiv

# Search only PubMed for biomedical topics
cargo run --bin ai_cli --features "full,research" -- research \
    "CRISPR gene editing" \
    --providers pubmed \
    --max-results 10

# Search all available providers
cargo run --bin ai_cli --features "full,research" -- research \
    "large language models safety alignment" \
    --providers arxiv,scholar,pubmed \
    --max-results 20
```

### API Keys

- **arXiv**: No key required. Respects 3-second rate limit automatically.
- **Semantic Scholar**: Works without a key (100 requests per 5 minutes). With
  `SEMANTIC_SCHOLAR_API_KEY`, rate limit improves to 1 request per second with
  higher quotas.
- **PubMed**: Works without a key (3 requests per second). With `NCBI_API_KEY`,
  rate limit increases to 10 requests per second.

---

## 4. The AcademicPaper Data Model

**What**: The `AcademicPaper` struct is the unified representation of a paper
regardless of which provider returned it.

**Why**: arXiv returns Atom/XML, Semantic Scholar returns JSON, PubMed returns
E-utilities XML. The `AcademicPaper` struct normalizes all of these into one
consistent type with optional fields for data that not every provider supplies.

**How**:

```rust
pub struct AcademicPaper {
    pub id: String,
    pub title: String,
    pub authors: Vec<Author>,
    pub abstract_text: Option<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub doi: Option<String>,
    pub url: Option<String>,
    pub pdf_url: Option<String>,
    pub citation_count: Option<u32>,
    pub fields_of_study: Vec<String>,
    pub keywords: Vec<String>,
    pub source: AcademicSource,
    pub external_ids: HashMap<String, String>,
}
```

Key design decisions:

- **`citation_count`** is `Option<u32>` because arXiv does not provide citation
  counts, while Semantic Scholar does. This lets the pipeline sort by citation
  impact when available without panicking when it is not.
- **`external_ids`** is a `HashMap<String, String>` that holds cross-references
  like `{"arxiv": "2301.12345", "doi": "10.1234/foo"}`. This enables
  deduplication across providers.
- **`source`** tracks which `AcademicSource` (arXiv, SemanticScholar, PubMed)
  returned this paper, so provenance is always known.

---

## 5. CLI Commands

**What**: The `ai_cli` binary exposes the `research` subcommand for all
academic search and literature review operations.

**Why**: Researchers and developers should be able to search papers, generate
reviews, and export BibTeX without writing Rust code. The CLI makes the entire
pipeline accessible from a terminal.

**How**:

### Basic Search

```bash
cargo run --bin ai_cli --features "full,research" -- research \
    "transformer attention mechanisms" \
    --providers arxiv,scholar,pubmed \
    --max-results 20 \
    --year-range 2020-2026
```

This searches all three providers for papers about transformer attention
mechanisms published between 2020 and 2026, returning up to 20 results.

### Literature Review with BibTeX

```bash
cargo run --bin ai_cli --features "full,research" -- research \
    "RLHF reinforcement learning" \
    --review \
    --format systematic \
    --bibtex
```

This runs the full literature review pipeline: search, filter, synthesize a
systematic review, and export citations in BibTeX format.

### Single-Provider Search

```bash
cargo run --bin ai_cli --features "full,research" -- research \
    "CRISPR gene editing" \
    --providers pubmed \
    --max-results 10
```

### Verified Review (with Anti-Hallucination)

```bash
cargo run --bin ai_cli --features "full,research" -- research \
    "reinforcement learning from human feedback" \
    --review \
    --format systematic \
    --bibtex \
    --faithfulness \
    --quality-gates
```

See [Section 9: Anti-Hallucination Integration](#9-anti-hallucination-integration)
for details on what `--faithfulness` and `--quality-gates` do.

---

## 6. Literature Review Pipeline

**What**: An automated pipeline that takes a research query and produces a
structured literature review with proper citations.

**Why**: Writing a literature review is one of the most time-consuming parts of
academic work. The typical process -- search, read abstracts, group by theme,
write prose, format citations -- takes days. This pipeline reduces it to
minutes while maintaining academic standards.

**How**:

The pipeline has five stages:

1. **Search** -- Query all configured providers in parallel.
2. **Filter** -- Remove duplicates (using `external_ids` cross-reference),
   apply year range, sort by citation count when available.
3. **Synthesize** -- Use a local or cloud LLM to write a structured review
   from the collected abstracts.
4. **Cite** -- Attach inline citations to claims in the review text.
5. **Export** -- Generate a bibliography in the requested format.

### Search Depth

| Depth | Papers Retrieved | Use Case |
|-------|-----------------|----------|
| `Quick` | 20 papers | Fast survey, checking what exists |
| `Standard` | 50 papers | Typical related-work section |
| `Deep` | 100 papers | Comprehensive survey paper |

### Synthesis Styles

| Style | Description |
|-------|-------------|
| `Narrative` | Flowing prose organized by theme, suitable for introduction sections |
| `Systematic` | Structured with explicit inclusion/exclusion criteria, tables, and methodology |
| `Annotated` | Per-paper summaries with critical commentary |
| `Comparative` | Side-by-side comparison of approaches, methods, and results |

### Bibliography Formats

| Format | Output |
|--------|--------|
| `BibTeX` | Standard `.bib` file compatible with LaTeX |
| `Apa` | APA 7th edition formatted references |
| `Mla` | MLA 9th edition formatted references |
| `Chicago` | Chicago Manual of Style references |
| `Ieee` | IEEE citation format |

---

## 7. BibTeX Support

**What**: Full BibTeX parsing, generation, and security-hardened processing.

**Why**: BibTeX is the standard bibliography format in academic publishing.
Every researcher using LaTeX needs `.bib` files. But BibTeX files from the
internet are untrusted input -- they can contain LaTeX injection attacks.

**How**:

### Entry Types

Nine BibTeX entry types are supported:

| Type | Example |
|------|---------|
| `Article` | Journal paper |
| `Book` | Published book |
| `InProceedings` | Conference paper |
| `InCollection` | Chapter in an edited book |
| `Thesis` | Master's or PhD thesis |
| `TechReport` | Technical report |
| `Misc` | Datasets, software, other |
| `Unpublished` | Preprints, working papers |
| `Online` | Web resources |

### BibParser

Parse existing `.bib` files into structured `Vec<BibEntry>`:

```rust
use ai_assistant::research::BibParser;

let bib_content = std::fs::read_to_string("references.bib")?;
let entries = BibParser::parse(&bib_content)?;
println!("Parsed {} entries", entries.len());
```

### BibGenerator

Generate `.bib` content from entries, or directly from `AcademicPaper` search
results:

```rust
use ai_assistant::research::BibGenerator;

// From search results
let papers: Vec<AcademicPaper> = search_results;
let bib_output = BibGenerator::from_papers(&papers);
std::fs::write("output.bib", bib_output)?;
```

### Security: LaTeX Injection Protection

BibTeX fields are sanitized against known LaTeX injection vectors:

- `\input{...}` -- reads arbitrary files
- `\include{...}` -- reads arbitrary files
- `\write18{...}` -- executes shell commands
- `\immediate\write18{...}` -- immediate shell execution
- `\openout` -- writes to arbitrary files
- `\csname` -- constructs arbitrary control sequences

All of these are stripped or escaped during parsing. This is not optional --
it happens automatically whenever a `.bib` file is loaded.

### Limits

| Limit | Value | Reason |
|-------|-------|--------|
| Max file size | 10 MB | Prevent memory exhaustion from malicious files |
| Max entries | 10,000 | Prevent CPU exhaustion during parsing |
| Max field length | 10,000 chars | Prevent individual field abuse |

---

## 8. Paper Metadata Extraction

**What**: Extract structured metadata from PDF files -- title, authors,
sections, references.

**Why**: Many papers exist only as PDFs. Extracting metadata programmatically
enables indexing, citation extraction, and integration with the RAG pipeline
without manual data entry.

**How**:

The extraction system detects the following sections in academic PDFs:

| Section | Detection |
|---------|-----------|
| `Abstract` | Heading match or first paragraph heuristic |
| `Introduction` | Heading match |
| `RelatedWork` | "Related Work" or "Background" heading |
| `Methodology` | "Method", "Methodology", "Approach" heading |
| `Results` | "Results", "Experiments", "Evaluation" heading |
| `Discussion` | Heading match |
| `Conclusion` | Heading match |
| `References` | Heading match, reference list parsing |
| `Appendix` | Heading match |

### MCP Tool

```
extract_paper_metadata -- extract metadata from a PDF file path
```

This is available as an MCP tool so that IDE integrations and external agents
can extract paper metadata without using the CLI.

---

## 9. Anti-Hallucination Integration

This is the most important section of this guide. **Every other research tool
trusts its own LLM output blindly. `ai_assistant` does not.**

### The Problem

When an LLM generates a literature review, it can:

1. **Invent citations** -- papers that do not exist, with plausible-sounding
   titles and author names.
2. **Misattribute claims** -- assign a finding to the wrong paper.
3. **Fabricate statistics** -- cite numbers (p-values, accuracy scores) that
   do not appear in the source.
4. **Hallucinate venues** -- claim a paper appeared at NeurIPS when it was
   actually at a workshop.

In an academic context, any of these is a career-ending mistake. A retracted
paper due to fabricated citations is not a minor issue.

### The Solution: Three-Layer Verification

`ai_assistant` applies three verification mechanisms to literature review
output:

#### Layer 1: FaithfulnessScorer

The `FaithfulnessScorer` decomposes the generated review into individual claims
and checks each claim against the source material (paper abstracts and
metadata). Each claim receives a faithfulness score between 0.0 and 1.0.

- Claims that cannot be traced to any source paper are flagged.
- Claims that contradict source material are flagged.
- The overall review receives an aggregate faithfulness score.

#### Layer 2: Quality Gates

Quality gates enforce minimum standards on the review output:

- **Citation accuracy gate**: Every inline citation must correspond to a real
  paper in the search results. Made-up citations fail this gate.
- **Claim-source alignment gate**: Claims attributed to specific papers must
  be supported by those papers' abstracts.
- **Completeness gate**: The review must cover a minimum percentage of the
  retrieved papers (prevents the LLM from ignoring most results and writing
  about only one or two).

If any gate fails, the review is rejected and regenerated with stricter
constraints.

#### Layer 3: CoVe (Chain-of-Verification)

CoVe takes each factual claim from the review and generates verification
questions:

- "Did paper X report an accuracy of Y%?" -- checked against the abstract.
- "Was paper X published at venue Z?" -- checked against the metadata.
- "Did authors A, B, and C collaborate on paper X?" -- checked against the
  author list.

Claims that fail verification are revised or removed from the final output.

### CLI Usage

```bash
cargo run --bin ai_cli --features "full,research" -- research \
    "reinforcement learning from human feedback" \
    --review \
    --format systematic \
    --bibtex \
    --faithfulness \
    --quality-gates
```

- `--faithfulness` enables the FaithfulnessScorer (Layer 1) and CoVe (Layer 3).
- `--quality-gates` enables the quality gate checks (Layer 2).

Both flags can be used independently, but using them together provides the
strongest verification.

### Why This Matters

No other framework does this:

| Framework | Academic Search | Literature Review | BibTeX | Anti-Hallucination Verification |
|-----------|:-:|:-:|:-:|:-:|
| LangChain | No | No | No | No |
| LlamaIndex | No | No | No | No |
| **ai_assistant** | **Yes** (3 providers) | **Yes** (4 styles) | **Yes** (parse + generate) | **Yes** (3-layer verification) |

When a researcher uses `ai_assistant` to generate a literature review, they
get a review that has been checked against its own sources. This is not
perfection -- no automated system is -- but it is a **qualitative leap** over
tools that generate text and hope for the best.

### Cross-Reference

For the full anti-hallucination system documentation (beyond research), see
`docs/GUIDE_ANTI_HALLUCINATION.md`.

---

## 10. Research Agent Roles

**What**: Three specialized agent roles for research workflows, available when
both `research` and `multi-agent` features are enabled.

**Why**: Research is not a single task. Searching papers, critiquing a draft,
and improving prose are distinct skills that benefit from specialized agents
with different system prompts and behaviors.

**How**:

| Role | Responsibility |
|------|---------------|
| `ResearchAssistant` | Searches papers across providers, filters results by relevance and quality, summarizes findings |
| `PeerReviewer` | Critiques draft text, verifies claims against sources, identifies gaps in argumentation |
| `WritingCoach` | Improves academic writing style, suggests restructuring, fixes clarity issues |

These roles can be composed in a multi-agent pipeline:

1. `ResearchAssistant` searches and summarizes.
2. `PeerReviewer` checks the summary for accuracy.
3. `WritingCoach` polishes the final text.

The orchestration system handles passing context between agents automatically.

---

## 11. Server and API Integration

**What**: Four REST endpoints expose the research pipeline over HTTP.

**Why**: Not every consumer is a Rust application. Python scripts, web
frontends, and other services need HTTP access to the research pipeline.

**How**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/research/search` | Search academic papers across providers |
| `POST` | `/api/v1/research/review` | Generate a literature review from a query |
| `POST` | `/api/v1/research/bibtex/import` | Import a `.bib` file into the RAG knowledge base |
| `POST` | `/api/v1/research/bibtex/export` | Export collected citations as a `.bib` file |

### Example: Search via API

```bash
curl -X POST http://localhost:8080/api/v1/research/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer attention mechanisms",
    "providers": ["arxiv", "semantic_scholar"],
    "max_results": 20,
    "year_range": [2020, 2026]
  }'
```

### Example: Generate Review via API

```bash
curl -X POST http://localhost:8080/api/v1/research/review \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RLHF reinforcement learning",
    "format": "systematic",
    "bibtex": true,
    "faithfulness": true,
    "quality_gates": true
  }'
```

### Example: Import BibTeX into RAG

```bash
curl -X POST http://localhost:8080/api/v1/research/bibtex/import \
  -H "Content-Type: application/octet-stream" \
  --data-binary @references.bib
```

This parses the `.bib` file (with LaTeX injection sanitization), converts
entries to documents, and indexes them in the RAG vector database. After
import, paper content is available for retrieval-augmented generation.

---

## 12. MCP Tools

**What**: Six MCP (Model Context Protocol) tools for research operations.

**Why**: MCP enables IDE integrations (VS Code, JetBrains, etc.) and external
agents to use the research pipeline through a standardized protocol. A
researcher working in their IDE can search papers and import citations without
switching to a terminal.

**How**:

| Tool | Description |
|------|-------------|
| `search_papers` | Search academic APIs with query, provider, and filter parameters |
| `get_paper_metadata` | Retrieve detailed metadata for a specific paper by ID |
| `import_bibtex` | Parse a `.bib` file and import entries into the RAG knowledge base |
| `export_bibtex` | Export collected citations as a `.bib` file |
| `literature_review` | Run the full literature review pipeline (search + filter + synthesize + cite + export) |
| `extract_paper_metadata` | Extract metadata from a local PDF file |

All six tools are registered automatically when the `research` feature is
enabled and the MCP server is running.

---

## 13. Security

**What**: Security measures specific to the research pipeline.

**Why**: The research pipeline handles untrusted input from three external APIs
and processes user-uploaded `.bib` files. Every external data source is a
potential attack vector.

**How**:

| Measure | What It Protects Against |
|---------|------------------------|
| URL-encode all query parameters | Injection via search queries |
| API keys from env vars, never logged | Credential leakage in logs |
| Response Content-Type validation | Response spoofing / SSRF |
| BibTeX sanitization | LaTeX injection (`\write18`, `\input`, etc.) |
| File size limit (10 MB) | Memory exhaustion from oversized files |
| Entry count limit (10,000) | CPU exhaustion during parsing |
| Field length limit (10,000 chars) | Individual field abuse |
| Per-provider rate limiting | Respecting API terms of service, avoiding bans |

### API Key Handling

API keys are loaded exclusively from environment variables:

```bash
export SEMANTIC_SCHOLAR_API_KEY="sk-..."
export NCBI_API_KEY="..."
```

Keys are never written to log files, never included in error messages, and
never serialized to disk. The configuration file supports key fields as a
fallback, but environment variables are the recommended approach.

---

## 14. Comparison vs LangChain and LlamaIndex

This section is for developers evaluating frameworks.

### Academic Search

- **LangChain**: No built-in academic search. You would need to write custom
  tool wrappers for each API (arXiv, Semantic Scholar, PubMed), handle
  authentication, implement rate limiting, and normalize the response formats
  yourself. **No equivalent.**
- **LlamaIndex**: No built-in academic search. Same situation as LangChain.
  **No equivalent.**
- **ai_assistant**: Three providers with unified `AcademicSearchProvider` trait,
  automatic rate limiting, response normalization, and deduplication across
  providers. Works out of the box.

### BibTeX

- **LangChain**: No BibTeX support. **No equivalent.**
- **LlamaIndex**: No BibTeX support. **No equivalent.**
- **ai_assistant**: Full BibTeX parsing and generation with security
  sanitization. Parse `.bib` files, generate `.bib` from search results,
  import into RAG.

### Literature Review Pipeline

- **LangChain**: You could build a chain that calls an LLM with paper
  abstracts, but there is no built-in pipeline for search, filter, synthesize,
  cite, and export. You would build it from scratch. **No equivalent.**
- **LlamaIndex**: Similar situation. You could use the query engine over
  indexed papers, but there is no structured literature review pipeline.
  **No equivalent.**
- **ai_assistant**: Complete five-stage pipeline (search, filter, synthesize,
  cite, export) with configurable depth, synthesis style, and bibliography
  format.

### Anti-Hallucination Verification of Research Output

- **LangChain**: No built-in verification of generated research content.
  **No equivalent.**
- **LlamaIndex**: No built-in verification of generated research content.
  **No equivalent.**
- **ai_assistant**: Three-layer verification (FaithfulnessScorer, quality
  gates, CoVe) applied specifically to literature review output. Citations are
  checked against real search results. Claims are verified against source
  abstracts.

### Summary

| Capability | LangChain | LlamaIndex | ai_assistant |
|------------|-----------|------------|--------------|
| Academic paper search | Manual wrappers needed | Manual wrappers needed | Built-in, 3 providers |
| BibTeX parse/generate | Not available | Not available | Built-in with security |
| Literature review pipeline | Build from scratch | Build from scratch | Built-in, 5 stages |
| Anti-hallucination for research | Not available | Not available | 3-layer verification |
| Research agent roles | Generic agents only | Generic agents only | 3 specialized roles |
| MCP integration | Not available | Not available | 6 research tools |

---

## 15. Cross-References

- **docs/CONCEPTS.md** -- Sections #274 and #275 cover the research module
  architecture and provider design.
- **docs/USE_CASES.md** -- Use case #11 ("Academic literature review with
  BibTeX export") provides a condensed scenario walkthrough.
- **docs/GUIDE_ANTI_HALLUCINATION.md** -- Companion guide covering the full
  anti-hallucination system beyond research (FaithfulnessScorer, CoVe, quality
  gates for all LLM output).
- **docs/IMPROVEMENTS_V84.md through V88.md** -- Implementation history of the
  research features across multiple development batches.

---

## Appendix: Complete CLI Reference

```
ai_cli research <query> [OPTIONS]

ARGUMENTS:
  <query>              The research topic or search query

OPTIONS:
  --providers <LIST>   Comma-separated list: arxiv, scholar, pubmed
                       Default: arxiv,scholar
  --max-results <N>    Maximum papers to retrieve per provider
                       Default: 20
  --year-range <RANGE> Filter by publication year (e.g., 2020-2026)
  --review             Generate a literature review (not just search)
  --format <STYLE>     Review style: narrative, systematic, annotated, comparative
                       Default: narrative
  --bibtex             Export citations in BibTeX format
  --faithfulness       Enable FaithfulnessScorer + CoVe verification
  --quality-gates      Enable quality gate checks on review output
```

---

*Feature flag: `research` | Included in: `full` | Gate: `#[cfg(feature = "research")]`*

*This guide covers ai_assistant's academic research capabilities. For the
complete feature reference, see docs/GUIDE.md.*
