# V45 — Procedure Import/Export/Defaults + Multi-User Isolation

**Estado**: COMPLETADO
**Fecha**: 2026-03-18

---

## Resumen

V45 añade import/export versionado de procedures, 6 defaults builtin, y un modelo
formal de aislamiento multi-usuario con clasificación de datos (Private/Shared/Replicated).

---

## Part A: Procedure Import/Export/Defaults — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| A.1 | **Default trait** for Procedure con `#[serde(default)]` en campos opcionales | HECHO |
| A.2 | **6 default procedures**: code review, pre-commit, deploy, bug investigation, documentation, test writing | HECHO |
| A.3 | **ProcedureExport** — formato versionado (v1) con timestamp, source, user_id | HECHO |
| A.4 | **ProcedureImportOptions** — merge/replace, skip_duplicates, reset_confidence | HECHO |
| A.5 | **ProceduralStore**: export(), export_to_file(), import(), import_from_file(), load_defaults() | HECHO |
| A.6 | **AiAssistant API**: load_default_procedures(), export/import_procedures(), export/import_to/from_file() | HECHO |
| A.7 | **9 tests**: default trait, defaults not empty, export/import roundtrip, merge skip, replace all, reset confidence, empty export, load defaults no overwrite, file roundtrip | HECHO |

## Part B: Multi-User Isolation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| B.1 | **UserScope enum**: Private, Shared, Replicated | HECHO |
| B.2 | **classify_data_scope()** — clasifica tipos de datos en su scope correcto | HECHO |
| B.3 | **SqliteSessionStore**: list_sessions_for_user(), search_messages_for_user() | HECHO |
| B.4 | **Concept 192**: Multi-User Isolation model (Private/Shared/Replicated) | HECHO |
| B.5 | **Concept 193**: Procedure Import/Export documentation | HECHO |
| B.6 | **2 tests**: user_scope_classification, sqlite_sessions_user_filtering | HECHO |

## Modelo de aislamiento multi-usuario

| Scope | Datos | Mecanismo |
|-------|-------|-----------|
| **Private** | Conversations, memories, procedures, preferences, notes | Per-AiAssistant instance |
| **Shared** | Knowledge base (RAG), knowledge graph, templates, guardrails | Global per-deployment |
| **Replicated** | Rate limits, active nodes, cluster config | P2P CRDTs (infra only) |

**Regla P2P**: NUNCA replicar datos personales via CRDTs — solo infraestructura.
