# V43 — FreshContext Total Integration & Conversation Reference Resolution

**Estado**: EN PROGRESO (Fases 1, 2, B, C, E completadas)
**Fecha inicio**: 2026-03-17

---

## Resumen

V43 conecta todos los subsistemas de conocimiento existentes al FreshContext mode y añade
resolución de referencias conversacionales para que el usuario pueda referirse a contenido
anterior ("la opción 3", "esa lista", "lo que dijiste antes") sin perder el hilo.

---

## Completado

### Fase 1: Wire Existing Systems into FreshContext

| # | Tarea | Estado |
|---|-------|--------|
| 1.1 | **Knowledge Graph query en core** — movido de ai_gui.rs a assistant.rs build_rag_context(). Consulta unified view, extrae entidades matching, inyecta `--- GRAPH CONTEXT ---` con entidades y relaciones | HECHO |
| 1.2 | **Temporal decay en memory search** — verificado que YA estaba implementado: `effective_importance(decay_half_life_days)` se aplica en `MemoryStore::search()` | YA EXISTÍA |
| 1.5 | **Context overflow truncation** — cuando knowledge_tokens > budget, trunca al ratio proporcional en boundary de línea con nota `[... truncated ...]` | HECHO |

### Fase 2: Conversation Reference Resolution

| # | Tarea | Estado |
|---|-------|--------|
| 2.1 | **TrackedList struct** — almacena listas detectadas con items, topic, timestamp, turn_index | HECHO |
| 2.2 | **ReferenceResolver** — detecta y extrae listas de mensajes (numbered, bulleted, lettered), resuelve referencias ordinales/cardinales en inglés y español | HECHO |
| 2.2a | **extract_list_items()** — soporta formatos: `1.`, `1)`, `1-`, `-`, `*`, `+`, `a.`, `a)` | HECHO |
| 2.2b | **resolve_reference()** — resuelve "la opción 3", "the second one", "esa lista", "lo anterior" etc. contra listas tracked | HECHO |
| 2.2c | **has_reference_pattern()** — detección rápida multilingüe (EN+ES) para skip si no hay referencia | HECHO |
| 2.2d | **extract_item_index()** — ordinales (first-tenth EN, primero-décimo ES) + cardinales con contexto ("option 3", "punto 5") | HECHO |
| 2.2e | **Out-of-bounds handling** — si el usuario pide item 5 de una lista de 3, informa del error | HECHO |
| 2.2f | **Multi-list disambiguation** — si hay varias listas, matchea por keywords del topic | HECHO |

---

## Pendiente (siguiente sesión)

### Fase 1 restante

| # | Tarea |
|---|-------|
| 1.3 | Wire `AdvancedMemoryManager` (episodic, procedural, entity) en FreshContext con `#[cfg(feature = "advanced-memory")]` |
| 1.4 | Poblar `MemoryType::Relationship` cuando se detectan entidades co-mencionadas |

### Fase 2 restante

| # | Tarea |
|---|-------|
| 2.3 | Integrar ReferenceResolver en `send_message()` — inyectar `--- RESOLVED REFERENCES ---` en system prompt |
| 2.4 | Indexar ALL messages con embeddings para búsqueda semántica completa del historial |

### Fase 3: Cross-Session Memory

| # | Tarea |
|---|-------|
| 3.1 | Wire `advanced_memory::consolidation` al cerrar sesión |
| 3.2 | Wire `advanced_memory::sharing` al abrir nueva sesión |
| 3.3 | Actualizar FreshContextStatus con cross_session_available |

### Fase 4: Tests unitarios (10 tests)

- FreshContext con RAG + Graph + Memory simultáneo
- Knowledge Graph query se ejecuta en core
- Temporal decay reduce score de memorias antiguas
- Advanced episodic recall se inyecta en FreshContext
- List tracking detecta listas numeradas
- Reference resolver resuelve "la opción 2"
- Reference resolver busca en archive cuando referencia es vaga
- Context overflow trunca chunks de menor relevance
- Cross-session: memorias de sesión anterior disponibles
- FreshContext effectiveness = Optimal

### Fase 5: Tests de conversación real (Ollama) (8 tests scored)

- Conversación multi-turn con referencia reciente
- Conversación con referencia a lista antigua
- FreshContext con RAG real
- Memory persistence cross-turn
- Graph entity linking
- Referencia vaga a conversación antigua
- Context overflow graceful
- FreshContext vs Conversation mode comparison

### Fase D: Fallbacks Secundarios — Verificados como YA EXISTENTES

| Item | Estado |
|------|--------|
| D.5 Cost estimation: modelo desconocido | YA EXISTE — `get_pricing()` fallback a `default_pricing` |
| D.7 Memory eviction: max_memories excedido | YA EXISTE — `cleanup()` con `effective_importance * decay`, evicta los de menor score |
| D.1 Provider ALL fail: cached response | PARCIAL — `FallbackChain` ya existe, no cached response pero error detallado |
| D.2 Embedding: provider down → TF-IDF local | YA EXISTE — `LocalEmbedder` siempre disponible como fallback |
| D.6 Auto-model: modelo no disponible | YA EXISTE — `default_model` como fallback |

### Fase G: Documentación — HECHO

- CONCEPTS.md: conceptos 186 (FreshContext Total Integration), 187 (Fallback Chains)
- concepts.html: conceptos 186, 187 (count → 187)
- GUIDE.md: sección 157 (V43 completa)
- developer_guide.html: sección V43
- TESTING.md: categoría fallback_resilience (18 tests)
- modus-operandi.md: estado actualizado

### Pendiente para siguiente sesión

| Item | Descripción |
|------|-------------|
| **Fase A** | Unified SQLite persistence (tablas nuevas, write-through, schema migration) |
| **Fase F** | Tests de conversación real con Ollama (8 tests scored) |
| **D.3** | Streaming: buffer overflow → evict a disco |
| **D.4** | Session load: corrupción → intentar formato alternativo |

---

## Arquitectura de la integración

```
Usuario envía mensaje
        │
        ▼
┌─ ReferenceResolver ──────────────────────┐
│  ¿Contiene referencia? ("opción 3", etc) │
│  → Sí: resolver contra TrackedLists      │
│  → Inyectar --- RESOLVED REFERENCES ---  │
└──────────────────────────────────────────┘
        │
        ▼
┌─ build_rag_context() ────────────────────┐
│  1. Knowledge RAG (semantic + cache)     │
│  2. Knowledge Graph query (unified view) │  ← NUEVO
│  3. Conversation RAG (archive search)    │
│  4. Overflow truncation                  │  ← NUEVO
└──────────────────────────────────────────┘
        │
        ▼
┌─ build_memory_context() ─────────────────┐
│  Working memory (topic, entities, facts)  │
│  Long-term memory (semantic search)       │
│  Advanced memory (episodic, procedural)   │  ← PENDIENTE
│  Temporal decay applied                   │
└──────────────────────────────────────────┘
        │
        ▼
┌─ System Prompt Assembly ─────────────────┐
│  Base prompt                              │
│  + User notes                             │
│  + --- KNOWLEDGE BASE ---                 │
│  +   RAG chunks                           │
│  +   --- GRAPH CONTEXT ---                │  ← NUEVO
│  +   --- RESOLVED REFERENCES ---          │  ← NUEVO
│  + --- MEMORY CONTEXT ---                 │
│  + User preferences                       │
└──────────────────────────────────────────┘
        │
        ▼
    LLM genera respuesta
        │
        ▼
┌─ Post-processing ────────────────────────┐
│  Track lists in LLM response             │  ← NUEVO
│  Update working memory                    │
│  Archive if context full                  │
│  Consolidate if session end               │  ← PENDIENTE
└──────────────────────────────────────────┘
```

---

## Tipos nuevos exportados

- `TrackedList` — lista detectada con items, topic, timestamp, turn_index
- `ReferenceResolver` — tracker de listas + resolver de referencias bilingüe (EN+ES)
