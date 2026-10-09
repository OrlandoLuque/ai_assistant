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
| Diversity (MMR), again | `rag_methods::MmrScorer` | si | n/a | n/a | n/a | n/a | **parcial** | 2026-09-24 | **a second MMR.** 14 references, all 14 inside `rag_methods.rs` — definition, its own `select`, eight tests. N109 |
| Reranking with a real cross-encoder | `rerank_service::HttpReranker` | si | no | no | no | no | **parcial** | 2026-09-25 | speaks Jina-shaped `/v1/rerank`, so llama.cpp `--rerank`, Jina, Cohere and TEI. **Nothing in the pipeline calls it yet** — that is N96. Never falls back to a heuristic; validates every returned index; reports what it could not score |
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

## Tabular data

Added 2026-09-26. Behind the `tabular` feature, which defaults to
`tabular-sqlite`. Enabling `tabular` with no engine is a `compile_error!`.

| capability | lives in | API | CLI | MCP | GUI | srv | estado | verificado | notes |
|---|---|---|---|---|---|---|---|---|---|
| Load a CSV as a queryable table | `tabular::sqlite_engine` | si | no | no | no | no | **parcial** | 2026-09-26 | in-memory SQLite; quoted fields with commas and newlines survive |
| Read-only SQL over loaded tables | `TableEngine::query` | si | no | no | no | no | **parcial** | 2026-09-26 | four independent layers; `PRAGMA` deliberately excluded |
| Describe a table's columns | `TableEngine::describe` | si | no | no | no | no | **parcial** | 2026-09-26 | the tool a model needs *before* `query`: one that does not know the column names invents SQL |
| Get the RIGHT number from a column with a missing marker | `MixedColumns` (default) | si | no | no | no | no | **parcial** | 2026-09-27 | `10, N/A, 30` gives `SUM 40`, `AVG 20`, `COUNT 2`. Before: `40.0`, **13.33**, **3** — two of three silently wrong |
| Keep the marker text queryable instead | `LoadOptions::keeping_mixed_as_text` | si | no | no | no | no | **parcial** | 2026-09-27 | the opt-out, for when the marker itself is data. Carries its own, different warning |
| Say what counts as a missing value | `LoadOptions::treating_as_missing` | si | no | no | no | no | **parcial** | 2026-09-26 | for markers that must not even be *reported*, like a `-` you know means zero rows |
| Warn when an answer cannot be trusted | `QueryResult::warnings` | si | no | no | no | no | **parcial** | 2026-09-26 | required field. A mixed numeric column makes `AVG` and `COUNT` silently wrong; the result cannot come back without saying so |
| Parquet, lazy evaluation, out-of-core | `tabular::polars_engine` | si | no | no | no | no | **parcial** | 2026-09-26 | **+49.3 MiB** of binary and 397 crates, measured. SQLite cannot read Parquet — that is the whole reason |
| The two engines agree | `tabular::both_engines_agree` tests | si | n/a | n/a | n/a | n/a | hecho | 2026-09-27 | same data for six queries, **same error variant** for nine refusals, the same permissions, and since V355 the same right answer over a mixed column — by two different mechanisms. Compiled only when both features are on |
| `list_tables` as an MCP tool | `mcp_protocol::table_tools` | si | — | **si** | no | no | hecho | 2026-09-27 | names the engine too: the two dialects differ, and a model that knows which one it is talking to writes SQL that works first time |
| `describe_table` as an MCP tool | `mcp_protocol::table_tools` | si | — | **si** | no | no | hecho | 2026-09-27 | reports `mixed_numeric` and `non_numbers_nullified`, which is what stops a model trusting an `AVG` it should not |
| `query_table` as an MCP tool | `mcp_protocol::table_tools` | si | — | **si** | no | no | hecho | 2026-09-27 | echoes `sql_executed`, `row_count`, `truncated`, `warnings`, `trustworthy`. Rows capped at 200 and the cap is announced |
| Choose which files a model may query | `ai_mcp_server --table NAME=PATH` | si | **si** | n/a | no | no | hecho | 2026-09-27 | **there is deliberately no `load_table` tool** — the host picks the files, so no prompt can name a path |
| Pick the engine a file needs | `tabular::engine_for` | si | si | n/a | no | no | hecho | 2026-09-27 | Parquet → Polars, else SQLite. Without Polars a `.parquet` gets `NeedsEngine` **naming the feature**, not a parse error inside a CSV reader |
| One engine for a mixed set of files | `ai_mcp_server` | — | si | n/a | no | no | hecho | 2026-09-27 | if anything is Parquet, Polars reads them all — it reads CSV too. One engine per file would stop a query JOINing across two files |
| Extract tables from inside text | `table_extraction` | si | — | — | — | — | hecho | 2026-09-26 | a different job: finds tables in prose. Does not query them |

### Two things the type system enforces

`QueryResult` carries `sql_executed`, `row_count`, `truncated` and `warnings` as
**required fields**. No engine can return a result without saying what it ran, how
much came back, whether the limit cut it short, and whether there is a reason to
distrust it. Not a rule in a comment — a struct that will not compile otherwise.

### Where the two engines do not agree

Held to the same answers by a test module, so what survives is documented rather
than discovered by a user:

- **`SELECT SUM(x), COUNT(x)` errors on Polars and works on SQLite** — both output
  columns would be called `x`. The portable form is `SELECT SUM(x) AS s, COUNT(x)
  AS n`, and that is what generated SQL should use. Not papered over by rewriting
  the caller's SQL: aliasing somebody's columns changes the shape of their result.
- **Aggregate column names and type names differ** (`SUM(x)` against `x`,
  `INTEGER` against `i64`). `ColumnInfo::declared_type` is for a human or a
  prompt, never for comparison.

### What is NOT wired
**MCP is wired since V356** — `ai_mcp_server --table ventas=ventas.csv` serves the
three tools, and a client sees them over real stdio JSON-RPC. What is still missing:

- **no CLI** for ad-hoc querying outside MCP (`ai_cli table query` does not exist);
- **no GUI** surface, so the SQL a model ran has no place a human reads it yet;
- ~~no automatic engine choice~~ — **done in V358**: `tabular::engine_for(path)` picks
  Polars for Parquet and SQLite otherwise, and a build without Polars asked for a
  `.parquet` gets `TableError::NeedsEngine` naming the feature instead of a parse
  failure inside a CSV reader.

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
| **Retrieval quality (recall@k, MRR, nDCG)** | `retrieval_metrics` | si | **si** | no | no | no | **hecho** | 2026-10-01 | `ai_cli retrieval score <file.json> [--k N] [--json]`. Also `precision_at_k`, MAP. The scoring half. **Two precisions**: `precision_at_k` divides by what came back, `precision_at_k_strict` by `k` -- the latter is the TREC/BEIR definition and the only one comparable with a paper (N136). Producing the run is `retrieval_eval`, the row below |
| **Judged corpus -> scorable run** | `retrieval_eval` | si | **si** | no | no | no | **parcial** | 2026-10-01 | `ai_cli retrieval corpus <dir>` inspects a BEIR-layout corpus (BEIR/NanoBEIR/MIRACL/MessIRve) and reports unjudged queries and qrels pointing outside the collection -- both cap the scores invisibly. `run_corpus` turns a `Retriever` into a `RunFile` and **refuses** a run where no returned id belongs to the corpus, because an id-space mismatch would otherwise read as `recall@10 = 0.000`. `retrieval run <dir>` indexes the corpus with the crate's OWN retriever (`sqlite::SqliteRetriever`, lexical or hybrid) and scores it; `dedup_by_document` collapses chunks back to documents, which otherwise inflates precision and suppresses recall at once. **`--split test|dev|train`** picks the judgements AND therefore the query set, because a BEIR directory ships one `queries.jsonl` holding every split: on NFCorpus that is 323 judged of 3,237, and the unjudged ones were always dropped from the averages, so narrowing is free -- verified, nDCG@10 = 0.2979 either way. **What is still missing is the ~30 hand-judged queries over this project's docs, and N105 before any retriever comparison means anything.** N102 |
| Agentic / tool-use benchmarks | — | no | no | no | no | no | **no** | 2026-09-24 | own harness categories exist (`agentic_code`, `agentic_rust`); no public benchmark |
| Prompt-injection / jailbreak suite | — | no | no | no | no | no | **no** | 2026-09-24 | the guardrails exist; nothing measures them against a public corpus |
| Long-context suite | — | no | no | no | no | no | **no** | 2026-09-24 | FreshContext and the budget allocator are unmeasured |

### Families this library does not cover

Researched 2026-09-24. Every row is `no` on purpose: naming the absence is the
point of the file. The named benchmarks are what a loader would target, chosen
because each measures something the five loaders above cannot.

| family | what it measures that we cannot | candidate datasets | task |
|---|---|---|---|
| **Retrieval quality** | whether the right chunk came back at all | NanoBEIR (50 queries x 13 subsets), MIRACL-es, MessIRve | N102 |
| **Reranking quality** | whether reordering helped or hurt | MTEB reranking split (MAP, MRR@k) | N102 |
| **RAG end to end** | multi-hop and **aggregation** questions, which no single passage answers | CRAG (Meta, NeurIPS 2024), MultiHop-RAG (2-4 documents), ARES | N108 |
| **Word-level faithfulness** | *which span* was unsupported, not whether the answer was | RAGTruth | N108 |
| **Intrinsic hallucination** | inconsistency against a given source at scale | HaluBench (15K samples), FaithBench | N108 |
| **Closed-book factuality** | what the model knows **without** context — separates "did not know" from "retrieval failed" | SimpleQA | N108 |
| **Tool calling** | correct function, correct arguments, correct types | BFCL (v4, April 2026: Agentic 40% / Multi-Turn 30% / Live 10% / Non-Live 10% + hallucination) | N108 |
| **Agentic task completion** | the task finished, not the call was well formed | tau2-bench (dual control), GAIA 2 | N108 |
| **Prompt injection** | whether the guardrails hold against untrusted tool output | AgentDojo (97 tasks, 629 security cases), AgentDyn, AgentDrift | N108 |
| **Long context** | whether a bigger window is a **usable** window | RULER (13 tasks, 4K-128K), LongBench v2, multi-needle NIAH | N108 |

Two of these are not optional in the ordinary sense:

- **Prompt injection.** V346 found the guardrail pipeline ignoring
  `GuardAction::Block` when the score sat below its threshold — high-risk
  injections passed with `passed == true`. That is fixed. Nothing stops the same
  shape returning in another guard, because nothing measures it.
- **Long context.** `FreshContext` exists to maximise how much knowledge fits,
  and `ContextBudgetAllocator` decides what goes in. Neither has ever been
  measured against a benchmark that distinguishes a window that holds text from
  a window the model can still use. Published RULER results put the gap at
  30-60 points past 200K on frontier models; on the models this project runs it
  will be wider and it will start earlier.

RULER is worth calling out separately: it is **synthetic and generated, not
downloaded**. The tasks are constructed at whatever length you ask for, so it
needs no dataset, no licence acceptance and no cache — which makes it the
cheapest row in this table to turn into a `si`.

And on needles: the **single**-needle test overstates usable context by 15-40
points against the multi-needle one. If we measure this, we measure multi-needle;
the easy version would tell us what we want to hear.

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
