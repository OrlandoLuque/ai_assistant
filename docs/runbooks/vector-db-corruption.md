# Runbook: Vector DB corruption

**Severity**: P1 — RAG-dependent flows return wrong answers or fail.
**Owner**: RAG platform.
**Last reviewed**: 2026-05-06 (V130).

The crate supports several vector DB backends (HNSW in-memory + on
disk, SQLite-WAL via `rag` feature, LanceDB, pgvector, Qdrant,
Weaviate, Elasticsearch). Corruption symptoms differ; this runbook
covers the diagnostic workflow first, backend-specific recovery
second.

## 1. Symptoms

* RAG queries return zero hits or absurd hits for any input.
* `ai_cli rag query "<known phrase>"` panics with
  `Header invalid` / `IO error: bad magic` / `SQLite: malformed`.
* `ai_logs --component=rag` shows recurring `corrupted index` /
  `unable to deserialize` errors on startup.
* Embedding count from `rag stats` ≠ count from your ingestion log.

If the symptom is "zero results but the index opens cleanly", that's
[`rag-empty-results`](rag-empty-results.md) — different runbook.

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | Process killed mid-write (no `fsync` between batches) | high |
| 2 | Disk full → partial WAL truncation | high |
| 3 | Backend version upgrade without migration | medium |
| 4 | Dimension mismatch after re-embedding (768 vs 1024) | medium |
| 5 | Hardware: bit-flip / disk failure | low |
| 6 | Manual edit to data dir (yes, it happens) | low |

## 3. Diagnose

```bash
# Stop ingestion immediately. Reads are usually fine; writes can
# make a recoverable corruption unrecoverable.
ai_cli rag ingest --pause

# Identify the backend.
ai_cli rag info | head -20

# HNSW (in-memory + persisted):
file <data_dir>/hnsw.bin                         # expect data
ai_cli rag verify --strict                       # walks every node

# SQLite (rag feature):
sqlite3 <data_dir>/rag.db 'PRAGMA integrity_check;'
sqlite3 <data_dir>/rag.db 'PRAGMA quick_check;'

# LanceDB:
ai_cli rag verify --backend lancedb --table <name>

# Pgvector / Qdrant / Weaviate / Elastic — use the backend's own
# health endpoint; this crate does not own that infrastructure.
```

Capture the stack trace if a panic. The `error_taxonomy`
classification (V113) tells you whether it is `Recoverable`,
`Degraded`, or `Fatal`.

## 4. Mitigate

**A. If you have a recent `secure_backup` snapshot (V128):**

```bash
ai_backup verify  --input <snapshot>.zip.enc \
                  --passphrase-env AI_BACKUP_PASS
ai_backup restore --input <snapshot>.zip.enc \
                  --output  <data_dir_replacement> \
                  --passphrase-env AI_BACKUP_PASS

# Atomically swap (Linux):
mv <data_dir> <data_dir>.broken && mv <data_dir_replacement> <data_dir>
```

Restart the service. RAG should work; you'll have lost ingestions
since the snapshot.

**B. If no backup but the source documents are still on disk:**

Re-index from source. This is the standard "the truth is the
source documents" pattern.

```bash
mv <data_dir> <data_dir>.broken
ai_cli rag init
ai_cli rag ingest --source documents/ --recursive
```

This can take hours for large corpora. While re-indexing, route RAG
calls to a degraded path (web search, no-context generation) by
toggling the feature flag in your config layer.

**C. If neither of the above:**

The data is gone. Restore from your last backup of the *source
documents* (this is why source documents must themselves be backed
up). If you do not back up source documents, fix that *before* you
restart traffic.

## 5. Resolve

Schedule, do not rush:

* **Always-on `secure_backup` snapshots** of the RAG data dir on a
  cadence smaller than your re-index time. Daily for medium corpora,
  hourly for high-velocity ingestion.
* **Disk space alert** at 80% of the partition holding the data dir.
* **Atomic-write ingestion**: use `ai_cli rag ingest --batch-size N
  --fsync-every-batch` so a kill mid-batch loses ≤N records, not the
  whole index.
* **Dimension assertion** at startup — the crate already does this
  (panics on mismatch); make sure your deployment surfaces the panic
  early in the rollout (canary).

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Backend | hnsw / sqlite / lancedb / ... |
| Detection | first symptom → first responder |
| Records lost | rough count or "all since `<timestamp>`" |
| Root cause | one sentence |
| Recovery path | A / B / C above |
| Time to recover | start of recovery → traffic restored |
| Action items | each with owner + due date |

Update this runbook if a new failure mode was found.
