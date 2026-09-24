# What this library does, and where each thing is wired

Inventory, not history. The CHANGELOG says what changed; this says what exists
**now**, which surfaces reach it, and — the column that matters — **when
somebody last checked**.

This file is deliberately incomplete. See "Why this is not a full table".

---

## How to read it

Wiring columns take four values, because a tick cannot tell "finished here" from
"here, doing half of it":

| value | means |
|---|---|
| `si` | reachable from that surface and doing what it says |
| `parcial` | reachable and doing less than the capability offers — NOTES say what is missing |
| `no` | exists in the crate, not exposed here. A gap unless a note says otherwise |
| `n/a` | makes no sense on that surface |

The surfaces: **API** (the crate), **CLI** (`ai_cli` and friends), **MCP**
(`ai_mcp_server`), **GUI** (`ai_gui`, `ai_gui-pro`), **srv** (`server.rs`,
`server_axum`).

**`verificado`** is a date or `—`. A row without a date is a row nobody has
checked against the code; treat it as a question, not as information. A row with
a date was true then, and is worth what the date is worth.

---

## Why this is not a full table

The crate is 95 features, 41 `[[bin]]` entries and 339 `pub mod` in `lib.rs`,
187 of them behind a feature. One row per module would be unreadable; one row
per subsystem, filled in from memory, would be **wrong**.

That is not a guess. Writing the equivalent table for `ai_portable_kit` — a
crate of four modules — from memory and then checking it produced **two wrong
claims out of the first five**, and between them three rows that said "missing"
about things that existed.

So this file grows by verified rows only. An empty file is honest; a full one
written from recall is not, and it is worse than nothing because it gets cited.

Adding a row costs one grep. Do that rather than remember.

---

## Retrieval and RAG

Checked in depth on 2026-09-23/24 while fixing N95, N99, N100 and N104. The
findings are what most of these notes are.

| capability | lives in | API | CLI | MCP | GUI | srv | estado | verificado | notes |
|---|---|---|---|---|---|---|---|---|---|
| Keyword search (BM25 over FTS5) | `RagDb::search_knowledge` | si | si | si | parcial | si | hecho | 2026-09-24 | CLI `ai_cli.rs:3622`; MCP tool in `knowledge_tools.rs`; server via `AiAssistant` enrichment (`assistant/rag.rs:590`). The GUI only names the MCP tool in a list |
| Keyword search with filters | `RagDb::search_knowledge_filtered` | si | no | no | no | si | parcial | 2026-09-24 | only `assistant/rag.rs:819` |
| **Hybrid: BM25 + semantic re-scoring** | `RagDb::search_knowledge_hybrid` | si | **no** | **no** | **no** | **no** | **parcial** | 2026-09-24 | **nothing in the crate calls it.** Reachable only by an outside consumer. See below |
| Semantic **retrieval** over the knowledge base | — | **no** | no | no | no | no | **no** | 2026-09-24 | `knowledge_chunks` has no vector column. N100 |
| Semantic retrieval over a vector index | `rag_pipeline::VectorDbRetrieval` | si | no | no | no | no | parcial | 2026-09-24 | real, but a **separate store** the knowledge base does not populate, with different chunk ids. N103 |
| Embeddings | `embeddings::LocalEmbedder` | si | — | — | — | — | parcial | 2026-09-24 | TF-IDF + hashing. A paraphrase scores **0.0**. N104 |
| Neural embeddings | `neural_embeddings::DenseEmbedder` | si | — | — | — | — | parcial | 2026-09-24 | only with `api_url`; otherwise falls back to TF-IDF, and now says so (`is_neural()`) |
| Rank fusion (RRF) | `rank_fusion` | si | n/a | n/a | n/a | n/a | hecho | 2026-09-24 | one implementation; was three. Normalised by default |
| Weighted fusion | `RagPipeline::weighted_fusion` | si | n/a | n/a | n/a | n/a | hecho | 2026-09-24 | when `fusion_rrf` is off |
| Reranking with an LLM | `RagPipeline::llm_rerank` | si | — | — | — | — | hecho | 2026-09-24 | reorders, does not re-score. Reports when it could not |
| Cross-encoder reranking | `reranker::CrossEncoderReranker` | si | n/a | n/a | n/a | n/a | **parcial** | 2026-09-24 | its `default_scorer` is **Jaccard**, so it reranks semantic results by literal word overlap. N96 |
| Diversity (MMR) | `reranker::DiversityReranker` | si | n/a | n/a | n/a | n/a | parcial | 2026-09-24 | exists; the pipeline never calls it |
| Cascade reranking | `reranker::CascadeReranker` | si | n/a | n/a | n/a | n/a | parcial | 2026-09-24 | same: exists, uncalled |
| Sentence-window expansion | `RagPipeline::apply_sentence_window` | si | — | — | — | — | hecho | 2026-09-24 | neighbours inherit the hit's relevance |
| Parent-document retrieval | `RagPipeline::apply_parent_document` | si | — | — | — | — | hecho | 2026-09-24 | parent inherits from its best child |
| Tier presets | `rag_tiers::RagTier` | si | si | — | si | si | **parcial** | 2026-09-24 | 30 of 46 `RagFeatures` flags gate nothing. Ratcheted |
| Autocut (cut where the score drops) | `topic_matcher::autocut_scores` | si | n/a | n/a | si | n/a | **no** | 2026-09-24 | declared `true` by five tiers, read by none. The GUI draws a checkbox that toggles a value nothing reads |

### Nothing this library ships calls its own hybrid search

`search_knowledge_hybrid` is public, documented, configurable
(`HybridRagConfig`), and **called from nowhere in the crate**. Every surface —
CLI, MCP, GUI, server, and `AiAssistant`'s own enrichment — goes through the
plain `search_knowledge`, which is BM25 and nothing else.

So `semantic_enabled`, `semantic_weight` and `bm25_weight` are reachable only by
somebody consuming the library directly and calling that method by name. Turning
them on through any shipped surface is not possible, because no shipped surface
gets there.

That is the third time this week the same shape has turned up: `RrfFusion::fuse`
(nobody), `rag_methods::LlmReranker` (nobody), and now this. The pattern is a
capability built, tested, documented and never connected — invisible because
every test passes and every doc is accurate about the part that exists.

### The one-line summary of that table

`RagTier::Semantic` is documented as "Keyword + semantic search, better recall".
With the built-in pieces it is **keyword search, scored twice by keywords** —
and on every surface this library ships, it is keyword search scored **once**,
because nothing reaches the hybrid path at all.

Three layers of the same thing, each invisible on its own:

1. the embedder is TF-IDF, so "semantic" means word overlap (N104);
2. the knowledge base stores no vectors, so semantic cannot retrieve, only
   re-score (N100);
3. and nothing calls the method that does the re-scoring.

---

## Evaluation and benchmarks

Checked 2026-09-24. Everything here is behind the `eval` feature.

| capability | lives in | API | CLI | MCP | GUI | srv | estado | verificado | notes |
|---|---|---|---|---|---|---|---|---|---|
| Download a benchmark dataset | `eval_benchmarks::http` | si | si | no | no | parcial | hecho | 2026-09-24 | size cap, atomic write via `.part`, fallback URL list |
| Cache datasets on disk | `eval_benchmarks::cache` | si | si | no | no | parcial | hecho | 2026-09-24 | short-circuits the download when the size matches |
| Run a model against a benchmark | `eval_benchmarks::runner` | si | si | no | no | parcial | hecho | 2026-09-24 | `ai_cli benchmark run --provider X --model Y` |
| Sweep the correctness threshold | `eval_benchmarks::calibration` | si | si | no | no | no | hecho | 2026-09-24 | `ai_cli benchmark calibrate --objective accuracy\|f1` |
| TruthfulQA | `loaders::truthfulqa` | si | si | no | no | — | hecho | 2026-09-24 | factuality, closed book |
| FEVER | `loaders::fever` | si | si | no | no | — | hecho | 2026-09-24 | fact verification |
| HaluEval | `loaders::halueval` | si | si | no | no | — | hecho | 2026-09-24 | hallucination detection |
| FactScore | `loaders::factscore` | si | si | no | no | — | hecho | 2026-09-24 | atomic-fact precision |
| RAGAS | `loaders::ragas` | si | si | no | no | — | hecho | 2026-09-24 | RAG faithfulness / relevance |
| **Retrieval quality (recall@k, MRR, nDCG)** | — | **no** | no | no | no | no | **no** | 2026-09-24 | **not one occurrence of any of the three in the crate.** N102 |
| Agentic / tool-use benchmarks | — | no | no | no | no | no | **no** | 2026-09-24 | own harness categories exist (`agentic_code`, `agentic_rust`); no public benchmark |
| Prompt-injection / jailbreak suite | — | no | no | no | no | no | **no** | 2026-09-24 | the guardrails exist; nothing measures them against a public corpus |
| Long-context suite | — | no | no | no | no | no | **no** | 2026-09-24 | FreshContext and the budget allocator are unmeasured |

### The shape of what is there

The machinery the author asked for — "a binary that downloads whatever and runs
it and checks" — **exists**, since V90, and is more careful than it needed to be:
a size cap against download bombs, atomic writes, a fallback URL list, and an
explicit `--accept-license`.

What it covers is **hallucination and faithfulness**. Five loaders, all of that
family. Nothing for retrieval, agents, injection or long context.

Vendored into the repo: five fixtures of 468–963 bytes each, two or three rows
apiece. No third-party dataset is checked in, which keeps the licence question
where it belongs — at `download --accept-license`.

Adding a family is a **loader plus its metrics**, not a subsystem.

## Everything else

Not checked. The subsystems are listed in `CLAUDE.md` — multi-provider LLM,
multi-agent, autonomous agent, distributed, security, streaming, FreshContext,
MCP, WASM, research, anti-hallucination, vision — and each deserves a row here
once somebody has looked.

Counts, which are facts and are regenerable:

| | |
|---|---|
| features declared in `Cargo.toml` | 95 |
| `[[bin]]` entries | 41 |
| `pub mod` in `lib.rs` | 339 |
| of those, behind a `#[cfg(feature)]` | 187 |

Measured 2026-09-24 — and the binary count came out as 40 first, because the
regex assumed `name` follows `[[bin]]` immediately and one entry puts `path`
first. `scripts/check_binaries_documented.py`, which CI runs, says 41. Prefer
the existing checker to a fresh regex; if you must write one, make it disagree
with something you can verify.

`docs/BINARIES.md` is the binary inventory; `docs/README.md` indexes the 198
files in `docs/`.

## How to add a row

1. Find where it lives: `grep -rn "fn <thing>" src/`.
2. For each surface, grep for a call: the CLI in `src/bin/ai_cli.rs`, MCP in the
   tool registry, the GUI in `widgets.rs`, the server in `server.rs` /
   `server_axum.rs`.
3. A capability that exists and is reachable from nowhere is `no` on every
   surface — and that is a finding, not a blank.
4. Put today's date in `verificado`. Without it the row is a rumour.

Two traps, both of which caught me while writing this:

* **Grep the crate, not one file.** Twice this week a "missing" conclusion came
  from searching a single file while the implementation sat behind a feature
  gate in another one.
* **A checkbox is not wiring.** Fourteen `RagFeatures` flags are read only by
  `widgets.rs`, which draws a checkbox for each. The user can tick it; nothing
  reads the value. That is `no`, not `si`.
