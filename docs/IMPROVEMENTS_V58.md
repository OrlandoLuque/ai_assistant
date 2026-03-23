# V58 — StorageContext Persistence

**Estado**: COMPLETADO
**Fecha**: 2026-03-23

---

## Resumen

V58 adds a unified persistence coordinator (StorageContext) and wires
persistence for StrategyBandit and RagTierStore so they survive restarts.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **StorageContext** — unified persistence coordinator | HECHO |
| 2 | **StorageConfig** — data_dir, auto_save_interval, schema_version | HECHO |
| 3 | **save_json/load_json** — atomic writes (temp + rename) for crash safety | HECHO |
| 4 | **DirtyFlags** — per-subsystem change tracking with AtomicBool | HECHO |
| 5 | **drain_writes()** — flush only dirty subsystems (shutdown/error paths) | HECHO |
| 6 | **StrategyBandit save/load** — persist learned preferences | HECHO |
| 7 | **RagTierStore save/load_custom** — persist user-created tiers | HECHO |
| 8 | **7 tests** (StorageContext) | HECHO |

## Test count

- **Before**: 6,985
- **After**: 6,992 (+7)
