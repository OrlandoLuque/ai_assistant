# V57 — Agent Interruption & Safety

**Estado**: COMPLETADO
**Fecha**: 2026-03-23

---

## Resumen

V57 adds tool safety classification, cross-platform file rollback, and
knowledge auto-reindexing.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **ToolSafetyProfile** — read_only, destructive_reversible, destructive_irreversible, long_running | HECHO |
| 2 | **SnapshotStore** — cross-platform file backup/rollback | HECHO |
| 3 | **FileSnapshot** — per-operation snapshots (write, delete, rename, copy, append, chmod, create) | HECHO |
| 4 | **rollback()** — restore individual snapshots | HECHO |
| 5 | **rollback_iteration()** — LIFO compensation of entire iteration (Saga pattern) | HECHO |
| 6 | **RollbackStrategy** — Snapshot (default) vs Git (5 modes: Commit/Stash/Branch/Worktree/Tag) | HECHO |
| 7 | **GitRollbackConfig** — auto_squash, auto_cleanup, prefixes, max commits | HECHO |
| 8 | **ToolCallRecord** — saga tracking with compensation flag | HECHO |
| 9 | **KnowledgeWatcher** — auto-detect when indexed documents change on disk | HECHO |
| 10 | **WatcherConfig** — poll_interval, extension filter, exclude paths | HECHO |
| 11 | **17 tests** (10 tool_safety + 7 knowledge_watcher) | HECHO |

## Test count

- **Before**: 6,968
- **After**: 6,985 (+17)
