# v32 Improvements Changelog

> Date: 2026-03-07

v32 is the **Scalability Guardrails** release: runtime monitoring for data structures that degrade beyond their optimal operating range, with actionable `log::warn!` recommendations for switching backends, adding limits, or running maintenance.

---

## Summary Metrics

| Metric | v31 | v32 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | ~7,060 | ~7,086 | ~+26 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 1 | +1 |
| Lines added | — | ~640 | — |

---

## New Files

| File | LOC | Description |
|------|-----|-------------|
| `src/scalability_monitor.rs` | ~500 | Scalability monitoring: types, thresholds, check logic, recommendations |

---

## Phase 1: Core Module `scalability_monitor.rs`

New module behind `analytics` feature flag (already in `full`).

### Data Model

- **`Subsystem`** enum: 12 monitored subsystems (VectorDbInMemory, VectorDbHnsw, EmbeddingCache, KnowledgeGraph, MultiLayerGraph, SessionStore, FactStore, ResponseCache, EpisodicMemory, EntityStore, CrdtOrSetTombstones, DhtStorage).
- **`WarningSeverity`**: Info (60-79%), Warning (80-94%), Critical (95%+).
- **`ScalabilityAction`** enum: SwitchBackend, EnableFeature, AddSizeLimit, RunMaintenance, ReduceConfig, Custom.
- **`ScalabilityWarning`**: subsystem, severity, current_size, optimal_max, utilization_pct, message, recommendations.
- **`ScalabilitySnapshot`**: point-in-time snapshot of all subsystem sizes for batch audit.

### Default Thresholds

| Subsystem | Optimal Max | Type | Key Recommendation |
|-----------|-------------|------|--------------------|
| VectorDb (InMemory) | 10,000 | Bounded | Switch to LanceDB (10K-10M) or Qdrant (10M+) |
| HNSW Index | 100,000 | Unbounded | Run compaction; switch to Qdrant |
| Embedding Cache | 10,000 | Bounded | Enable persistent cache; reduce TTL |
| Knowledge Graph | 100,000 | Unbounded | Reduce traversal depth; avoid full-graph algorithms |
| Multi-Layer Graph | 50,000 | Unbounded | Archive old session layers |
| Session Store | 1,000 | Unbounded | Add max_sessions + TTL |
| Fact Store | 10,000 | Unbounded | Add max_facts with LRU eviction |
| Response Cache | 1,000 | Bounded | Switch to Redis |
| Episodic Memory | 1,000 | Bounded | Archive old episodes to disk |
| Entity Store | 5,000 | Unbounded | Add max_entities limit |
| ORSet Tombstones | 10,000 | Unbounded | Implement tombstone compaction |
| DHT Storage | 50,000 | Unbounded | Enable TTL cleanup task |

### Core Functions

- **`check_scalability(subsystem, current_size)`**: On-access check with `thread_local!` cooldown (60s per subsystem). Emits `log::warn!` with message + first recommendation.
- **`check_scalability_no_cooldown(subsystem, current_size)`**: Same logic without cooldown, for testing and batch audits.
- **`audit_snapshot(snapshot)`**: Batch audit from a `ScalabilitySnapshot`, returns all warnings.
- **`format_action(action)`**: Human-readable formatting for `ScalabilityAction`.

### Example Warning Output

```
WARN [scalability] VectorDb (InMemory): 8200 entries (82% of 10000 optimal limit)
  — Switch from InMemory to LanceDB (Disk-backed index, optimal for 10K-10M vectors)
```

### Tests: 26

---

## Phase 2: lib.rs Registration

- Added `#[cfg(feature = "analytics")] pub mod scalability_monitor;`
- Re-exported: `ScalabilityAction`, `ScalabilitySnapshot`, `ScalabilityWarning`, `Subsystem`, `ScalabilityWarningSeverity` (aliased to avoid collision with existing `WarningSeverity`).

---

## Phase 3: On-Access Checks in 11 Modules

Each check is 3-5 lines behind `#[cfg(feature = "analytics")]`, added after the mutating operation:

| Module | Method | Subsystem Checked |
|--------|--------|-------------------|
| `vector_db.rs` | `InMemoryVectorDb::insert()` | VectorDbInMemory |
| `hnsw.rs` | `HnswIndex::insert()` | VectorDbHnsw |
| `knowledge_graph.rs` | `get_or_create_entity()` | KnowledgeGraph |
| `multi_layer_graph.rs` | `SessionGraph::add_entity()` | MultiLayerGraph |
| `session.rs` | `ChatSessionStore::save_session()` | SessionStore |
| `entities.rs` | `FactStore::add_fact()` | FactStore |
| `distributed.rs` | `ORSet::merge()` | CrdtOrSetTombstones |
| `distributed.rs` | `Dht::put()` | DhtStorage |
| `caching.rs` | `ResponseCache::put()` | ResponseCache |
| `advanced_memory/manager.rs` | `add_episode()` | EpisodicMemory |
| `advanced_memory/manager.rs` | `add_entity()` | EntityStore |
| `embedding_cache.rs` | `EmbeddingCache::set()` | EmbeddingCache |

---

## Key Design Decisions

1. **`thread_local!` cooldown** — prevents log spam (max 1 warning per subsystem per 60s) with zero cross-thread synchronization cost.
2. **Feature-gated behind `analytics`** — already in `full`, zero overhead when disabled.
3. **Typed `ScalabilityAction` enum** — recommendations are structured data, not free-text.
4. **No new dependencies** — uses only `log`, `serde`, `std`.
5. **Thresholds based on actual backend capabilities** — InMemory vs LanceDB vs Qdrant optimal ranges documented in vector_db.rs `BackendInfo`.
6. **On-access checks** — triggered during mutating operations, not periodic polling. Near-zero cost (one comparison + cooldown check).
