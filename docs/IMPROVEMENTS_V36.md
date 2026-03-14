# V36 — GUI Pro: Full-Feature Desktop Application + GUI Enhancements

**Tesis**: Crear un segundo binario GUI (`ai_gui-pro`) que exponga TODAS las
funcionalidades de la libreria a traves de una interfaz de escritorio completa
con 18 paneles, sidebar colapsable por categorias, y todos los modulos cableados.
Ademas, mejoras al GUI base (`ai_gui`) con soporte de grafos de conocimiento
mejorado, presupuesto de contexto visible, y reconstruccion de grafos.

**Estado**: HECHO
**Fecha**: 2026-03-14
**LOC nuevas**: ~4600 (ai_gui-pro.rs: 4359, cambios en ai_gui.rs: ~90, Cargo.toml: ~25)

---

## Resumen de cambios

### 1. Nuevo binario: `ai_gui-pro` (4359 LOC)

**Archivo**: `src/bin/ai_gui-pro.rs`
**Feature flag**: `gui-pro` (incluye `gui` + `full` + 14 features opcionales)
**Comando**: `cargo run --bin ai_gui-pro --features gui-pro`

Aplicacion de escritorio completa con sidebar colapsable organizada en 6 categorias
y 18 paneles:

| Categoria | Paneles | Funcionalidad |
|-----------|---------|---------------|
| **CHAT** | Chat, Sessions | Chat principal con streaming, gestion de sesiones |
| **AGENTS** | Agent Pool, Tools/MCP, Automation | Multi-agente con roles, registro de tools MCP, scheduler cron, browser CDP, sandbox de codigo, workflows |
| **KNOWLEDGE** | RAG Sources, Memory, Cloud Storage | RAG con grafos de conocimiento, memoria (episodica/procedural/entidad), S3/GDrive/Azure/GCS |
| **GENERATE** | Media, Audio | Generacion de imagen/video (DALL-E/SD/Flux/Runway/Sora), STT/TTS con seleccion de voz |
| **OPTIMIZE** | Prompt Lab, Evaluation | Firmas de prompt con optimizacion, suite de evaluacion con metricas |
| **SYSTEM** | Security, Analytics, DevTools, Cluster, Butler, Settings | RBAC/PII/guardrails/audit, metricas de sesion, debugger de agentes, cluster distribuido, Butler advisor, configuracion expandida (7 secciones) |

#### Arquitectura del GUI Pro

- **`PanelStates`**: Struct que agrupa ~50 campos de estado de todos los paneles
- **`SidebarCategory`** (6 variantes): CHAT, AGENTS, KNOWLEDGE, GENERATE, OPTIMIZE, SYSTEM
- **`SidebarPanel`** (18 variantes): Un panel por funcionalidad
- **Sidebar colapsable**: Solo CHAT expandido por defecto, las demas categorias colapsadas
- **Patron de renderizado**: Cada panel es un metodo `render_panel_*(&mut self, ui)` en `AiGuiApp`
- **Async con mpsc**: Butler scan, model fetching, ollama pull/delete — mismos patrones que ai_gui

#### Paneles implementados con detalle

| Panel | Sub-tabs | Widgets principales |
|-------|----------|---------------------|
| Agent Pool | — | Crear/eliminar agentes con selector de rol (Researcher/Coder/Analyst/Writer/Critic) |
| Tools/MCP | — | Registro de tools buscable por categoria, vista JSON de schemas |
| Automation | Scheduler, Browser, Sandbox, Workflows | Cron con validacion en vivo (`CronExpression::parse`), CDP URL+acciones, sandbox multi-lenguaje (`CodeSandbox::execute`), editor de workflows |
| Memory | Episodic, Procedural, Entity | Timeline, procedure cards, entidades del knowledge graph |
| Cloud Storage | — | Selector de proveedor (S3/GDrive/Azure/GCS), config, file browser |
| Media | — | Selector de modelo (DALL-E 3/SD XL/Flux/Runway/Sora), prompt, galeria |
| Audio | STT, TTS | File picker + transcripcion, entrada de texto + selector de voz |
| Prompt Lab | — | Editor de firmas, input/output fields, optimizador con metricas |
| Evaluation | — | Suite de tests editables, lista de capacidades |
| Security | Overview, PII, Guardrails, RBAC, Audit Log | Toggles por categoria, editor de reglas, log con filtros |
| Analytics | — | Metricas de sesion, stats de respuesta, uso de tokens |
| DevTools | — | Grabacion de eventos, breakpoints, log de eventos |
| Cluster | — | Bootstrap/join, lista de nodos con indicadores de estado |
| Settings | 7 secciones | Provider, Generation, Security, RAG, Agents, Memory, UI |

### 2. Feature flag `gui-pro` en Cargo.toml

```toml
gui-pro = [
    "gui", "full",
    "scheduler", "browser", "code-sandbox", "hitl", "devtools",
    "audio", "media-generation", "constrained-decoding",
    "prompt-signatures", "cloud-connectors", "integrity-check", "workflows",
]
```

Incluye 14 features opcionales ademas de `gui` + `full`, garantizando que todos
los modulos estan disponibles cuando el GUI Pro compila.

### 3. Mejoras al GUI base (`ai_gui.rs`)

#### 3.1 Re-indexacion de grafos de conocimiento

**Problema**: Al activar el knowledge graph despues de que los documentos ya
estaban cargados e indexados, el grafo quedaba vacio porque `pending_graph_docs`
ya habia sido consumido durante la indexacion inicial.

**Solucion**: Nuevo metodo `reindex_existing_sources_into_graph()` que re-lee
los archivos fuente desde disco (tanto `.kpkg` como archivos de texto) y los
indexa en el grafo.

```rust
fn reindex_existing_sources_into_graph(&mut self) {
    // Re-read files from disk and push to pending_graph_docs
    // Then call index_pending_graph_docs()
}
```

- Se invoca automaticamente al activar el toggle de Knowledge Graph si hay fuentes ya indexadas
- Soporta archivos `.kpkg` (desempaqueta y procesa cada documento interno)
- Soporta archivos de texto plano

#### 3.2 Boton "Rebuild Graph"

Nuevo boton visible cuando el grafo esta habilitado:
1. Limpia el grafo existente (`kg.clear()`)
2. Limpia la visualizacion del grafo
3. Re-indexa todas las fuentes cargadas via `reindex_existing_sources_into_graph()`
4. Muestra toast de confirmacion

#### 3.3 Informacion de presupuesto de contexto

Muestra en la sidebar de knowledge:
- Tokens disponibles para conocimiento (calculado con `calculate_available_knowledge_tokens`)
- Tokens usados por la conversacion actual (estimacion rough: `content.len() / 4`)
- Ultima consulta: tokens usados y numero de fuentes (via `last_knowledge_usage`)

#### 3.4 Campos de configuracion ampliados

`GuiSettings` extendido con:
- `history_depth: usize` — profundidad del historial de conversacion
- `timeout_secs: u32` — timeout de peticiones al proveedor

### 4. Rename del binario

- `ai_gui_pro` → `ai_gui-pro` (nombre con guion para consistencia con convenciones Rust/Cargo)
- Actualizado en `Cargo.toml`: `name = "ai_gui-pro"`, `path = "src/bin/ai_gui-pro.rs"`

### 5. Plan de cableado completo (documentacion)

**Archivo**: `docs/GUI_FULL_WIRING_PLAN.md`

Documento de planificacion con 5 iteraciones que cubre:
- Arquitectura (archivo unico vs modulos — decision: archivo unico con marcadores de seccion)
- Diseno de sidebar (20+ items reducidos a 16 en 6 categorias con colapso)
- Patron de paneles (metodos en AiGuiApp con PanelStates)
- Acciones diferidas para operaciones cross-panel
- Inicializacion async con canales mpsc
- Especificaciones detalladas de cada panel (campos de estado, widgets, acciones)
- Auto-deteccion de binarios ai_assistant (server en :3000, cluster en :4000)

---

## Estadisticas del proyecto actualizadas

| Metrica | Valor |
|---------|-------|
| **LOC totales** | ~383K |
| **Archivos .rs** | 314 |
| **Tests** | 7,066 |
| **Binarios** | 7 (cli, server, standalone, cluster_node, kpkg_tool, ai_gui, **ai_gui-pro**) |
| **Feature flags** | 22+ (incluye nuevo `gui-pro`) |

---

## Proximos pasos

- [x] **FreshContext mode**: Implementado en V37 — ContextMode enum, 5 send_message variants,
  memoria integrada, advisor API programatico
- [x] **MCP knowledge tools**: Implementado en V37 — 4 tools (search_knowledge, query_graph,
  list_knowledge_sources, get_entity) con lazy-open RagDb
- [ ] **Model Library 3 tabs**: Tres vistas (Recommended/All/Installed) con delete, sort, filter
