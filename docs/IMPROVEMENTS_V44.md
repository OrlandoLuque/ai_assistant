# V44 — Procedural Memory Integration

**Estado**: COMPLETADO
**Fecha**: 2026-03-18

---

## Resumen

V44 cablea el `ProceduralStore` existente (que ya tenía procedures, confianza, evolución
y persistencia) al pipeline de conversación del asistente. Ahora las procedures se
inyectan automáticamente como `--- WORKFLOW GUIDELINES ---` en el system prompt cuando
las keywords de la condición matchean el mensaje del usuario. Los outcomes se trackean
automáticamente para evolucionar la confianza de cada procedure.

---

## Completado

### Fase 1: ProceduralStore Enhancements

| # | Tarea | Estado |
|---|-------|--------|
| 1.1 | **remove(id)** — eliminar procedure por ID | HECHO |
| 1.2 | **find_relevant()** — búsqueda con min match ratio (30%), min confidence (0.1), max results, skip empty condition/steps | HECHO |
| 1.3 | **6 unit tests** — remove, min match ratio, min confidence, empty condition, empty steps, max results | HECHO |

### Fase 2: Context Building

| # | Tarea | Estado |
|---|-------|--------|
| 2.1 | **build_procedural_context()** — formatea procedures matching como `--- WORKFLOW GUIDELINES ---` | HECHO |
| 2.2 | **Token budget** — respeta límite (500 tokens default) via `estimate_tokens()` | HECHO |
| 2.3 | **Active tracking** — guarda IDs de procedures inyectadas en `active_procedure_ids` | HECHO |

### Fase 3: Wire into send_message

| # | Tarea | Estado |
|---|-------|--------|
| 3.1 | **send_message** — procedural context después de memory context, antes de resolved references | HECHO |
| 3.2 | **send_message_with_notes** — misma inyección | HECHO |
| 3.3 | **generate_sync** — misma inyección | HECHO |
| 3.4 | **send_message_cancellable** — misma inyección | HECHO |
| 3.5 | **send_message_cancellable_with_notes** — misma inyección | HECHO |

### Fase 4: CRUD API

| # | Tarea | Estado |
|---|-------|--------|
| 4.1 | **enable/disable/has_procedural_memory** | HECHO |
| 4.2 | **add/list/remove/find_procedures** | HECHO |
| 4.3 | **record_procedure_outcome** — feedback explícito | HECHO |
| 4.4 | **save/load_procedures** — persistencia a disco | HECHO |
| 4.5 | **procedural_store()** — acceso read-only | HECHO |

### Fase 5: Outcome Tracking

| # | Tarea | Estado |
|---|-------|--------|
| 5.1 | **Auto-tracking en poll_response** — actualiza confianza tras cada respuesta | HECHO |
| 5.2 | **ProcedureEvolver feedback** — registra feedback para evolución | HECHO |
| 5.3 | **Limpieza** — active_procedure_ids se limpia tras cada turno | HECHO |

### Fase 6: Tests

| # | Test | Estado |
|---|------|--------|
| 6.1 | procedural_remove | HECHO |
| 6.2 | find_relevant_min_match_ratio | HECHO |
| 6.3 | find_relevant_min_confidence | HECHO |
| 6.4 | find_relevant_empty_condition_skipped | HECHO |
| 6.5 | find_relevant_empty_steps_skipped | HECHO |
| 6.6 | find_relevant_max_results | HECHO |
| 6.7 | assistant_procedural_crud | HECHO |
| 6.8 | assistant_procedural_persistence | HECHO |
| 6.9 | assistant_procedural_context_formatting | HECHO |
| 6.10 | assistant_procedural_context_no_match | HECHO |

---

## Arquitectura

```
Usuario envía mensaje
        │
        ▼
┌─ Knowledge Augmentation ────────────────┐
│  1. Memory Context (FreshContext mode)   │
│  2. --- WORKFLOW GUIDELINES ---          │  ← NUEVO (V44)
│     Procedures matching user message     │
│     (top 5, ≥30% keyword match,          │
│      ≥0.1 confidence, ≤500 tokens)       │
│  3. --- RESOLVED REFERENCES ---          │
└─────────────────────────────────────────┘
        │
        ▼
    System Prompt → LLM
        │
        ▼
┌─ Post-processing ──────────────────────┐
│  Track procedure outcomes               │  ← NUEVO (V44)
│  Update confidence via ProcedureEvolver  │
│  Track lists in response                 │
│  Update working memory                   │
└─────────────────────────────────────────┘
```

---

## Tipos exportados

Todos bajo `#[cfg(feature = "advanced-memory")]`:
- `Procedure` — definición de un procedimiento (ya existía, ahora usado)
- `ProceduralStore` — almacén con find_relevant + remove (mejorado)
- `ProcedureEvolver` — evolución de confianza (ya existía, ahora cableado)
- `ProcedureFeedback`, `FeedbackOutcome` — tracking de resultados
