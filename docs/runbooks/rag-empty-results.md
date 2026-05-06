# Runbook: RAG returns empty results

**Severity**: P2 — answers degrade silently to no-context.
**Owner**: RAG platform.
**Last reviewed**: 2026-05-06 (V130).

The index opens cleanly (so this is *not*
[`vector-db-corruption`](vector-db-corruption.md)) but queries that
should hit return zero chunks. The user-visible effect is that the
LLM "doesn't know" things you indexed.

## 1. Symptoms

* `ai_cli rag query "<obvious phrase>"` → 0 hits.
* Faithfulness scoring drops; CoVe accuracy drops.
* `rag_query_total` counter increments but `rag_hits_total` flat.
* Specific to embedding-side problems: hit count > 0 but with
  similarity scores all <0.10 (the cut-off threshold filtered them).

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | Embedding model changed; query embeddings live in a different space than indexed embeddings | high |
| 2 | Similarity threshold too high in config | high |
| 3 | Filter-then-rank applied a metadata filter that excludes everything (e.g. `user_id` mismatch after a rename) | medium |
| 4 | Index is empty (ingestion never ran or wrote to a different path) | medium |
| 5 | Reranker dropped all results (reranker model returning constant scores) | low |
| 6 | Wrong tenant — multi-tenant deployments isolated by `user_id` | medium |

## 3. Diagnose

```bash
# Sanity: is anything in there at all?
ai_cli rag stats

# Pick a specific known chunk and try to retrieve it by its own
# text. If this fails, embedding mismatch or threshold is the
# cause.
ai_cli rag query "<verbatim phrase from a known chunk>" \
       --top-k 5 --min-similarity 0.0

# Print the raw similarity scores even when below threshold:
ai_cli rag query "..." --top-k 10 --raw

# What embedding model is configured for queries vs ingestion?
ai_cli rag info | grep -i model
ai_cli rag info | grep -i dimension
```

Compare ingestion-time vs query-time:

```bash
# The crate writes the embedding model id into the index metadata.
ai_cli rag info --json | jq '.embedding_model, .embedding_dimension'

# If your config provider is now a different model, queries will
# embed into a different vector space → all near-zero similarities.
```

## 4. Mitigate

**A. Embedding-model mismatch:**
- Switch the runtime config back to the model that was used to
  build the index.
- If you genuinely want to switch model, you must re-embed:
  ```bash
  ai_cli rag reembed --target-model <new-model>
  ```
  This walks every chunk and re-embeds. Plan for the duration —
  large corpora can take hours.

**B. Threshold too high:**
- Drop `[rag].min_similarity` from e.g. 0.7 to 0.3 in your config.
  Default in this crate is 0.0 (no cutoff); deployments that raised
  it for noise should re-tune after the model change that prompted
  the bump.

**C. Filter excludes everything:**
- `ai_cli rag query "..." --no-filter` — does it return anything?
  If yes, the filter is the culprit.
- Common case: `user_id` rename / tenant migration. Either re-tag
  the affected chunks (`ai_cli rag retag --from <old> --to <new>`)
  or update the filter to include both during the transition.

**D. Index empty:**
- `ai_cli rag stats` will show `0 chunks`. Re-run ingestion and
  monitor `ingest_chunks_written_total`.

**E. Reranker stuck:**
- Run with `--no-rerank` and see if the underlying retrieval is
  fine. If so, the reranker is the issue — restart it or roll back
  to the previous reranker model.

**F. Tenant isolation bug:**
- Verify the caller's `user_id` matches the chunks' `user_id`. If
  a deploy changed the way `user_id` is propagated through the
  request path, the symptom looks identical to "wrong filter".

## 5. Resolve

* **Pin the embedding model** in your config layer (config_layered)
  with a hash so a silent upgrade is caught at startup, not at the
  first user query.
* **Index-version metadata at startup** — the crate writes a
  `model_id` and `dimension` into the index header. Add a startup
  assertion that the *current* config matches; refuse to serve if
  not.
* **Bake a smoke query into the readiness probe** that retrieves a
  pinned, known chunk. If it returns 0 hits, the readiness probe
  fails and the load balancer routes around the bad instance.
* **Alert on `(rag_hits_total / rag_query_total) < 0.5`** — a sharp
  drop in hit ratio is a leading indicator of all six causes above.

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Detection | metric / quality drop / customer report |
| Cause | from §2 |
| Affected tenants | scope |
| Fix | from §4 |
| Customer impact | answers without context, duration |
| Action items | owner + due date |
