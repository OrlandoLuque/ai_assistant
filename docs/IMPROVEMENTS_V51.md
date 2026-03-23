# V51 — Store Limits Hardening

**Estado**: COMPLETADO
**Fecha**: 2026-03-21

---

## Resumen

V51 adds bounded growth and eviction policies to the remaining unbounded in-memory stores,
completing the audit started in V48.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **A2AServer.tasks** — max 10K, evicts completed/failed/canceled tasks oldest-first | HECHO |
| 2 | **A2AServer.push_configs** — max 5K with overflow eviction | HECHO |
| 3 | **FactStore (consolidation)** — max 5K, evicts lowest-confidence facts | HECHO |
| 4 | **FactStore (entities)** — enforces 10K limit, evicts lowest-reinforcement, rebuilds indices | HECHO |
| 5 | **LWWMap** — optional max_entries with oldest-timestamp eviction (zero overhead when unconfigured) | HECHO |
| 6 | **ClusterState.sessions** — uses LWWMap::with_max_entries(10K) | HECHO |
| 7 | **8 unit tests** | HECHO |

## Test count

- **Before**: 6,920 lib tests
- **After**: 6,928 lib tests (+8)
- **0 failures**
