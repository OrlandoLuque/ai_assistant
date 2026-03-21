# V48 — Cache Policies Audit

**Estado**: COMPLETADO
**Fecha**: 2026-03-21

---

## Resumen

V48 audits every in-memory store in the codebase (57 stores identified) and ensures bounded
growth with proper eviction policies for the 9 most critical ones. The generic `BoundedCache<K, V>`
provides the foundation: max entries, max bytes, LRU eviction, TTL, pinning, and invalidation
callbacks. Stores converted from FIFO to LRU include SearchCache and ResponseCache. EmbeddingCache
switched from entry-count to memory-based limits. DHT storage gained pinned keys and invalidation
callbacks alongside its existing entry/byte limits.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **BoundedCache<K, V>** — generic cache with max_entries, max_bytes, LRU, TTL, pinning, invalidation callbacks | HECHO |
| 2 | **DHT storage** — max_entries + max_bytes + LRU + pinned keys + invalidation callbacks | HECHO |
| 3 | **EntityStore** — max 5,000 entities with LRU eviction | HECHO |
| 4 | **Context cache** — cap 500 entries with LRU eviction | HECHO |
| 5 | **SearchCache** — converted from FIFO to LRU | HECHO |
| 6 | **ResponseCache** — converted from FIFO to LRU | HECHO |
| 7 | **EmbeddingCache** — memory-based limit (max bytes) instead of entry count | HECHO |
| 8 | **CompressedCache** — max 5,000 entries with LRU eviction | HECHO |
| 9 | **Audit of 48 remaining stores** — verified adequate limits or lifecycle-bounded | HECHO |
| 10 | **6 unit tests** — BoundedCache eviction, pinning, TTL, byte limits, invalidation, metrics | HECHO |
| 11 | **Concept 196** — Cache Policies: Bounded Growth for Every Store | HECHO |
| 12 | **GUIDE section 161** — V48 usage guide | HECHO |

## Test count

- **Before**: 6,887 lib tests
- **After**: 6,893 lib tests (+6)
- **0 failures**

## Documentation updated

- `docs/CONCEPTS.md` — concept 196
- `docs/GUIDE.md` — section 161
- `docs/TESTING.md` — test count updated to 6,901 (includes V49), V48 row in history
- `docs/modus-operandi.md` — V48 line, test count, Latest updated
- `docs/IMPROVEMENTS_V48.md` — this file
