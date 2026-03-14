# V37 — FreshContext Mode + MCP Knowledge Tools + Memory Integration + Advisor API

**Tesis**: Implementar un modo alternativo de composicion de contexto (`FreshContext`)
que maximiza el presupuesto de tokens para conocimiento, complementado con herramientas
MCP para busqueda de conocimiento, integracion de memoria en la libreria, y una API
programatica de diagnostico (`FreshContextStatus`) accesible tanto desde GUIs como
desde codigo de usuario.

**Estado**: HECHO
**Fecha**: 2026-03-14
**LOC nuevas**: ~920 (assistant.rs: ~570, mcp knowledge_tools.rs: ~250, tests MCP: ~170,
GUI changes: ~100, lib.rs/mod.rs: ~10)

---

## Resumen de cambios

### 1. FreshContext Mode (ContextMode enum)

**Archivo principal**: `src/assistant.rs`

Nuevo enum que controla como se compone el contexto para cada mensaje:

```rust
pub enum ContextMode {
    Conversation,  // Default: historial completo en contexto
    FreshContext,   // Solo mensaje actual + conocimiento fresco
}
```

**Comportamiento en FreshContext**:
- El historial se sigue guardando en `self.conversation` (para GUI y persistencia)
- Solo el ultimo mensaje se envia al LLM: `vec![self.conversation.last()...]`
- `calculate_available_knowledge_tokens()` devuelve `conversation_tokens = 0`
  → ~54% mas tokens disponibles para conocimiento en una ventana de 8K
- No se ejecuta compactacion automatica (innecesaria sin historial en contexto)

**Metodos modificados** (5 variantes):
- `send_message()`, `send_message_with_notes()`
- `send_message_cancellable()`, `send_message_cancellable_with_notes()`
- `generate_sync()`

### 2. MCP Knowledge Tools

**Archivo nuevo**: `src/mcp_protocol/knowledge_tools.rs` (~250 LOC)
**Feature gate**: `#[cfg(feature = "rag")]`

4 herramientas MCP registrables via `register_knowledge_tools()`:

| Tool | Descripcion | Parametros | Requiere |
|------|-------------|------------|----------|
| `search_knowledge` | Busqueda BM25 sobre chunks indexados | `query` (req), `max_results`, `max_tokens` | RagDb |
| `list_knowledge_sources` | Lista fuentes indexadas | — | RagDb |
| `query_graph` | Busca entidades y relaciones en el grafo | `query` (req) | KnowledgeGraph |
| `get_entity` | Busca entidad por nombre | `name` (req) | KnowledgeGraph |

**Patron lazy-open**: `Arc<Mutex<Option<RagDb>>>` — la conexion SQLite se abre en el
primer uso del handler, no al registrar la tool. Resuelve que `rusqlite::Connection`
es `!Send` mientras los handlers MCP requieren `Send + Sync + 'static`.

**Todas las tools**: `read_only_hint: true`, `destructive_hint: false`, `idempotent_hint: true`.

**Nuevo metodo en KnowledgeGraph** (`src/knowledge_graph.rs`):
```rust
pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>>
```

### 3. Memory Integration

**Campo nuevo en AiAssistant**: `pub memory_manager: Option<MemoryManager>`

**API publica**:
- `enable_memory(config: MemoryConfig)` — activa el sistema de memoria
- `disable_memory()` — desactiva y descarta memorias
- `has_memory() -> bool`
- `memory_manager() -> Option<&MemoryManager>` / `memory_manager_mut()`
- `build_memory_context(query, max_tokens) -> String`

**Integracion con FreshContext**: En modo FreshContext, las 5 variantes de send_message
construyen contexto de memoria automaticamente y lo inyectan:
```
--- MEMORY CONTEXT ---
Current context: ...
Relevant memories:
- The project uses Rust
```

**Procesamiento en poll_response**: Al recibir respuesta del LLM, tanto el mensaje
del usuario como la respuesta del asistente se procesan en `MemoryManager::process_message()`
para aprendizaje automatico de hechos, preferencias y entidades.

### 4. FreshContext Advisor API

**3 tipos nuevos** en `src/assistant.rs`, exportados desde `lib.rs`:

```rust
pub enum FreshContextWarning {
    NoRag,              // "FreshContext sin RAG es casi inutil"
    NoSourcesIndexed,   // "Sin fuentes indexadas"
    NoGraph,            // "Sin grafo pierde entidades/relaciones"
    NoMemory,           // "Sin memoria pierde contexto de sesion"
    SmallBudget(usize), // "Presupuesto de tokens muy pequeno"
}

pub enum FreshContextEffectiveness {
    Optimal,     // RAG + Graph + Memory
    Good,        // RAG + (Graph o Memory)
    Limited,     // Solo RAG
    Ineffective, // Sin RAG
}

pub struct FreshContextStatus {
    pub mode, rag_available, sources_indexed, graph_available,
    pub memory_available, available_knowledge_tokens,
    pub warnings: Vec<FreshContextWarning>,
    pub effectiveness: FreshContextEffectiveness,
}
```

**Metodo**: `fresh_context_status(has_graph: bool) -> FreshContextStatus`

`has_graph` se pasa como parametro porque `KnowledgeGraph` vive en la GUI,
no en `AiAssistant`.

**FreshContextWarning implementa `Display`** para uso directo en `println!()`:
```rust
let status = assistant.fresh_context_status(false);
for w in &status.warnings { println!("{}", w); }
// "Knowledge graph not active — entity and relation context unavailable"
```

**Uso programatico** (sin GUI): Los usuarios de la libreria pueden consultar
el estado directamente sin depender de la GUI.

### 5. GUI Integration

**Ambos binarios** (`ai_gui.rs`, `ai_gui-pro.rs`):

- **Toggle FreshContext**: Selector `Conversation` / `Fresh` en sidebar de knowledge
- **Avisos dinamicos**: Reemplazo de warnings hardcodeados por llamadas al advisor API
  con colores segun severidad:
  - Naranja (220, 120, 50): NoRag, NoSourcesIndexed (critico)
  - Amarillo (200, 150, 50): SmallBudget (precaucion)
  - Amarillo tenue (160, 160, 100): NoGraph, NoMemory (informativo)
- **Badge de efectividad**: Color-coded (verde/amarillo/rojo) segun nivel
- **Contexto enriquecido en FreshContext**:
  - `conv_ctx` archivado (RAG sobre mensajes pasados) se incluye automaticamente
  - Chunks del knowledge graph se incluyen si el grafo esta activo

### 6. Mejoras de calidad

- 5 `.unwrap()` → `.expect("message was just pushed")` en ramas FreshContext
- Todos los tests nuevos usan patrones consistentes con el resto del proyecto
- `MemoryConfig::default()` para inicializacion sin boilerplate

---

## Tests nuevos: 20

### En `src/assistant.rs` (16 tests):

| Test | Que verifica |
|------|-------------|
| `test_context_mode_default_is_conversation` | Default = Conversation |
| `test_context_mode_set_and_get` | Getter/setter funciona |
| `test_fresh_context_preserves_conversation_history` | Historial no se borra al cambiar modo |
| `test_fresh_context_only_last_message` | Solo ultimo mensaje en FreshContext |
| `test_fresh_context_more_tokens_available` | Mas tokens disponibles en FreshContext |
| `test_memory_disabled_by_default` | Memoria None por defecto |
| `test_memory_enable_disable` | Enable/disable funciona |
| `test_build_memory_context_empty_when_disabled` | Sin memoria → string vacio |
| `test_build_memory_context_with_memories` | Con hechos → contexto no vacio |
| `test_fresh_context_status_ineffective_without_rag` | Sin RAG = Ineffective + 3 warnings |
| `test_fresh_context_status_limited_with_rag_only` | Solo RAG = Limited |
| `test_fresh_context_status_good_with_rag_and_memory` | RAG + Memory = Good |
| `test_fresh_context_status_optimal` | RAG + Graph + Memory = Optimal, 0 warnings |
| `test_fresh_context_warning_display` | Display impl muestra texto correcto |

### En `src/mcp_protocol/tests.rs` (4 tests):

| Test | Que verifica |
|------|-------------|
| `test_knowledge_tools_registration_without_graph` | 2 tools registradas sin grafo, 0 con grafo |
| `test_knowledge_tools_search_invocation` | tools/call busca en DB con documentos indexados |
| `test_knowledge_tools_annotations` | readOnlyHint=true, destructiveHint=false |
| `test_knowledge_tools_list_sources` | list_knowledge_sources devuelve fuentes indexadas |

---

## Archivos modificados

| Archivo | Cambios |
|---------|---------|
| `src/assistant.rs` | +570 LOC: ContextMode, 3 tipos advisor, MemoryManager field, 7 metodos, memoria en 5 send variants + poll_response, 16 tests |
| `src/mcp_protocol/knowledge_tools.rs` | NUEVO: ~250 LOC, 4 MCP tools con lazy-open RagDb |
| `src/mcp_protocol/mod.rs` | +2 LOC: modulo knowledge_tools |
| `src/mcp_protocol/tests.rs` | +172 LOC: 4 tests MCP knowledge tools |
| `src/knowledge_graph.rs` | +5 LOC: get_entity_by_name wrapper |
| `src/lib.rs` | +4 LOC: export FreshContextStatus/Warning/Effectiveness |
| `src/bin/ai_gui.rs` | +100 LOC: toggle FreshContext, advisor API, context enrichment |
| `src/bin/ai_gui-pro.rs` | +100 LOC: idem ai_gui.rs |

---

## Estadisticas del proyecto actualizadas

| Metrica | Valor |
|---------|-------|
| **LOC totales** | ~385K |
| **Archivos .rs** | 315 |
| **Tests** | 4,892+ (lib) |
| **Binarios** | 7 |
| **Feature flags** | 22+ |

---

## Proximos pasos (planificado, no implementado)

- **Model Library 3 tabs**: Tres vistas (Recommended/All/Installed) con delete, sort, filter
- **Advanced Memory en FreshContext**: Integrar EpisodicStore/ProceduralStore
  (feature `advanced-memory`) para recall semantico con embeddings
- **FreshContext conv_ctx forzado**: Opcion para forzar inclusion de conv_ctx
  archivado independientemente del setting `conversation_rag_enabled`
