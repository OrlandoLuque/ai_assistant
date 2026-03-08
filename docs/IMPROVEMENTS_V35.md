# V35 — Agent System Overhaul: Naming, Wiring, Tools Consolidation, Memory, MCP

**Tesis**: Convertir el sistema de agentes de un conjunto de prototipos aislados en una
arquitectura integrada y coherente. Renombrar tipos para eliminar ambiguedades,
cablear AgentDefinition a ejecucion real, consolidar 7 frameworks de tools en 1,
integrar memoria episodica, y exponer gestion de agentes via MCP.

**Estado**: EN PROGRESO
**LOC estimadas**: ~5400
**Tests estimados**: ~135 nuevos

**Futuro (v36)**: Integracion de MCTS planner con AgentRuntime para planificacion
optimizada de secuencias de tool calls.

---

## Bloque A — Renames de tipos de agentes

Eliminar la confusion entre tipos que suenan igual pero hacen cosas distintas.

**NOTA**: Los renames A1-A7 (multi_agent, autonomous_loop) fueron revertidos por el formateador.
Solo los renames de agentic_loop (A3, A5, A7) se mantuvieron. Los demas tipos conservan
sus nombres originales. Se usa con aliases en lib.rs donde sea necesario.

| Fase | Actual | Nuevo | Modulo | Estado |
|------|--------|-------|--------|--------|
| A3 | `ToolCallingLoop` | `AgenticLoop` | agentic_loop.rs | **HECHO** (nombre final distinto al plan, pero descriptivo) |
| A5 | `AgentConfig/State/Status/Role` | `LoopConfig/LoopState/LoopStatus/LoopRole` | agentic_loop.rs | **HECHO** |
| A7 | `ConversationMessage` | `LoopMessage` | agentic_loop.rs | **HECHO** |
| A1,A2,A4,A6 | (multi_agent, autonomous_loop renames) | — | — | **REVERTIDO** por formateador — se mantienen nombres originales |

**LOC est.**: ~450 (grep+replace + actualizar re-exports en lib.rs, tests, examples)

**Dependencias**: Ninguna. Ejecutar primero.

---

## Bloque B — Renames en origen para eliminar aliases de lib.rs

Renombrar ~78 tipos en su modulo de origen para que no colisionen y no necesiten
alias en lib.rs. Filosofia: cada tipo tiene nombre unico en origen.

**NOTA**: La mayoria de renames fueron revertidos por el formateador automatico.
Solo ~12 renames sobrevivieron (agentic_loop, rag_tiers, rag_pipeline, rag_methods,
citations, auto_indexing). Los ~65+ restantes se mantienen con `as` aliases en lib.rs.
**PARCIALMENTE HECHO**.

| Fase | Dominio | Ejemplos | Aliases eliminados |
|------|---------|----------|--------------------|
| B1 | Cache | `CacheStats` -> `ResponseCacheStats` (caching), `BoundedCacheStats` (memory_mgmt), `PersistenceCacheStats` (persistence) | ~4 |
| B2 | Validation | `ValidationError` -> `DatasetValidationError` (fine_tuning), `CotValidationResult` (cot_parsing), `OutputValidationResult` (output_validation) | ~6 |
| B3 | Audit | `AuditEntry` -> `ProviderAuditEntry` (providers), `ServerAuditEntry` (server), `SandboxAuditEntry` (agent_sandbox) | ~6 |
| B4 | Search/RAG | `SearchResult` -> `ConversationSearchResult` (search), `WebSearchResult` (web_search), `EmbeddingSearchResult` (embeddings); `QueryExpander` -> `RagQueryExpander` (rag_methods) | ~8 |
| B5 | Knowledge Graph | `Entity` -> `KGEntity`, `Relation` -> `KGRelation`, `EntityExtractor` -> `KGEntityExtractor`, etc. | ~8 |
| B6 | Context/Composition | `CompactableMessage` -> `ComposerMessage`, `OverflowLevel` -> nombres unicos por modulo, `CompactionConfig` -> `ConvCompactionConfig` | ~10 |
| B7 | OpenTelemetry/Cost | `BudgetAlert` -> `OtelBudgetAlert`, `ModelPricing` -> `OtelModelPricing`, `BudgetCheckResult` -> `OtelBudgetCheckResult` | ~4 |
| B8 | Streaming | `Algorithm` -> `StreamCompressionAlgorithm`, `Level` -> `StreamCompressionLevel`, `CompressionConfig` -> `StreamCompressionConfig` | ~8 |
| B9 | Auto-model | `Requirements` -> `ModelRequirements`, `TaskType` -> `ModelTaskType`, `FallbackChain` -> `ModelFallbackChain`, `RoutingRule` -> `PipelineRoutingRule` | ~8 |
| B10 | Resto | `Language` -> `SandboxLanguage`, `PageContent` -> `BrowserPageContent`, `PlanStep` -> `TaskPlanStep`, `ResponseStyle` -> `RegenResponseStyle`, etc. | ~16 |

**LOC est.**: ~800 (mecanico: grep+replace por tipo, compilar tras cada fase)

**Dependencias**: Bloque A completado (para no renombrar dos veces).

---

## Bloque C — Container Backend Abstraction

Unificar los dos `ContainerExecutor` (container_tools.rs y container_executor.rs)
bajo un trait comun con feature flag base.

| Fase | Que |
|------|-----|
| C1 | Crear feature `container-base` en Cargo.toml (sin deps extra) |
| C2 | Nuevo `container_backend.rs`: trait `ContainerBackend` + tipos compartidos (`ContainerConfig`, `ExecResult`, `ContainerError`, `ContainerStatus`, `NetworkMode`) extraidos de container_executor.rs |
| C3 | `DockerCliBackend` en container_tools.rs: implementa `ContainerBackend` via `std::process::Command` |
| C4 | `BollardBackend` en container_executor.rs: implementa `ContainerBackend` via Bollard API nativa |
| C5 | Feature deps: `autonomous` depende de `container-base`; `containers` depende de `container-base` + bollard + tokio + futures |
| C6 | Eliminar tipos duplicados entre ambos modulos |
| C7 | Tests del trait con ambos backends |

**LOC est.**: ~200

**Dependencias**: Ninguna (independiente de A/B).

---

## Bloque D — Consolidar frameworks de tools

Reducir 7 frameworks de tools (tools.rs, tool_use.rs, tool_calling.rs,
unified_tools.rs, function_calling.rs + agentic_loop + mcp_client) a 1 set
canonico de tipos.

| Fase | Que |
|------|-----|
| D1 | Auditar: que tipos exporta cada modulo, quien los usa, dependencias cruzadas. Crear tabla de equivalencias campo a campo para detectar diferencias semanticas antes de migrar. |
| D2 | Definir set canonico en `unified_tools.rs`: `ToolCall`, `ToolResult`, `ToolRegistry`, `ToolParameter`, `ToolDefinition` |
| D3 | Migrar `tool_use.rs`: re-exportar tipos unificados o eliminar si redundante |
| D4 | Migrar `tool_calling.rs`: idem |
| D5 | Migrar `function_calling.rs`: idem |
| D6 | `tools.rs`: mantener traits de provider tools, usar tipos unificados |
| D7 | Limpiar lib.rs: una sola exportacion por tipo, sin aliases de tools |

**LOC est.**: ~1500 (refactor mas grande del plan — migracion incremental)

**Dependencias**: Bloques A+B completados (para no luchar con aliases durante migracion).

**Riesgo**: ALTO — afecta muchos modulos. Mitigacion: migracion modulo a modulo,
compilar+test tras cada fase. D1 (auditoria de equivalencias) es obligatorio antes
de empezar a migrar.

---

## Bloque E — Agent Wiring + AgentPool + Supervisor

Cablear AgentDefinition a ejecucion real, crear pool de agentes, guardrails
async, BestFit mejorado, supervisor event-driven.

### E1-E5: Cableado core

| Fase | Que | LOC |
|------|-----|-----|
| E1 | `impl From<ConversationMessage> for ChatMessage` + inverso | ~25 |
| E2 | `role_system_prompt(role: &str) -> String` — templates para 7 TeamRoles + Custom fallback | ~50 |
| E2b | `fn parse_team_role(s: &str) -> TeamRole` — mapea strings de AgentDefinition ("Analyst", "Manager", "Worker", "Expert", "Validator") a TeamRole enum. Unificar valores validos entre ambos sistemas. `Custom` como fallback para roles no reconocidos. | ~25 |
| E3 | `make_response_generator(config: AiConfig, model: Option<String>) -> Arc<dyn Fn...>` — factory de callbacks LLM. Clona AiConfig internamente (son strings, coste despreciable). | ~35 |
| E4 | `AiAssistant::create_agent_from_definition(def: &AgentDefinition) -> Result<AgentRuntime>` — valida definicion (rechaza si errores de severidad Error), construye runtime con system_prompt, model, tools, guardrails. Usa parse_team_role para mapear rol. | ~90 |
| E4b | `AgentProfile::from_definition(def: &AgentDefinition) -> AgentProfile` — convierte definicion en perfil para registrar en el orquestador (capabilities, rol, model). | ~30 |
| E5 | ToolRef -> ToolRegistry filtering: solo registra tools nombrados en la definicion, respeta approval flags. Warning con log si un ToolRef nombra un tool no registrado. | ~30 |

### E6-E7: Guardrails async paralelos

| Fase | Que | LOC |
|------|-----|-----|
| E6 | Trait `AsyncGuard` con `async fn check()` + `GuardrailPipeline::run_stage_async()`: guards sync secuenciales (instantaneos), guards async en paralelo con `join_all`. Timeout configurable **por guard individual** (default 5s). | ~80 |
| E7 | `GuardrailSpec` -> construir pipeline: `PiiGuard`, `PatternGuard`, `MaxTokensGuard` (sync) + guards async opcionales | ~55 |

### E8: Budget enforcement

| Fase | Que | LOC |
|------|-----|-----|
| E8 | En `AgentRuntime::run_iteration()`: si `total_cost >= budget` -> `IterationOutcome::Error("budget exceeded")`. Solo si `budget > 0.0`. | ~20 |

### E9: BestFit mejorado

| Fase | Que | LOC |
|------|-----|-----|
| E9a | Anadir `required_capabilities: Vec<String>` y `preferred_role: Option<TeamRole>` a `AgentTask` | ~15 |
| E9b | `fn score_agent_for_task(agent: &AgentProfile, task: &AgentTask) -> f64` — scoring multi-factor | ~40 |
| E9c | Factores: capability match exacto (0-100), role affinity (0-30), fallback substring en descripcion (0-10, solo si required_capabilities vacio) | incluido |
| E9d | Reemplazar closure inline de BestFit por `score_agent_for_task` | ~15 |
| E9e | Si todos los scores son 0.0 (todo vacio), log warning y fallback a round-robin | ~10 |
| E9f | Tests: scoring con capabilities exactas, role affinity, fallback substring, empates, todo-vacio-warning | ~70 |

### E10: AgentPool

| Fase | Que | LOC |
|------|-----|-----|
| E10a | `AgentPool { max_agents, active: HashMap<String, JoinHandle>, queue: BinaryHeap<PendingTask>, response_generator_factory, supervisor_config }`. Cola por prioridad (BinaryHeap), no FIFO. Cada agente corre en `std::thread::spawn`. | ~100 |
| E10b | `response_generator_factory: Arc<dyn Fn(Option<&str>) -> ResponseGenerator>` — campo del pool, inyectado por AiAssistant al crear el pool. Permite crear callbacks LLM para cualquier agente (incluido supervisor). | ~20 |
| E10c | `IterationHook` callback: `Arc<dyn Fn(&str, usize, &IterationOutcome) + Send + Sync>` — el pool lo pasa al AgentRuntime al crearlo. AgentRuntime lo invoca al final de cada iteracion. | ~30 |
| E10d | `AgentPool::on_iteration_complete()` — evaluacion de umbrales numericos (sin LLM) invocada por el IterationHook. | ~60 |
| E10e | `SupervisorConfig { enabled, idle_streak_threshold, budget_warning_percent, iteration_warning_margin, supervisor_profile }` | ~20 |
| E10f | `AgentPool::trigger_supervisor()` — instancia AgentRuntime temporal con rol Coordinator, contexto del problema, tools MCP de gestion. Usa response_generator_factory para obtener callback LLM. Se destruye al completar. | ~50 |
| E10g | `TriggerReason` enum (StuckDetected, BudgetWarning, NearIterationLimit) + `SupervisorTrigger` struct | ~20 |
| E10h | `mpsc::Receiver<InterAgentMessage>` buzón de entrada en AgentRuntime — permite recibir mensajes de otros agentes (incluido supervisor). Se consulta entre iteraciones. | ~30 |
| E10i | Tests: supervisor se activa con idle_streak, budget warning, near iteration limit; supervisor no se activa si disabled; cola por prioridad; buzón de mensajes | ~90 |

### E11: Tests de integracion

| Fase | Que | LOC |
|------|-----|-----|
| E11 | Tests end-to-end: definicion JSON -> create_agent_from_definition -> AgentRuntime -> tool call -> resultado. Tests de AgentProfile::from_definition -> registrar en orquestador -> BestFit asigna correctamente. | ~90 |

**LOC total Bloque E**: ~1290

**Dependencias**: Bloque A (renames) completado.

---

## Bloque F — Casos puntuales de naming

| Fase | Que | LOC |
|------|-----|-----|
| F1 | `WebSearchManager` -> `WebSearchEngine` (consistencia con `SearchEngine`) | ~15 |
| F2 | `CustomPattern` (entities.rs) -> `EntityCustomPattern` (eliminar shadowing bug) | ~15 |
| F3 | `ChunkMetadata` (auto_indexing.rs) -> `IndexChunkMetadata` en origen (eliminar shadowing) | ~20 |

**LOC est.**: ~50

**Dependencias**: Bloque B completado.

---

## Bloque G — Memory Wiring + Tests Serios

Cablear la memoria episodica (ya implementada, 6495 LOC, 204 tests unitarios)
al sistema de agentes. Actualmente esta completamente huerfana.

Arquitectura de memoria compartida: **modelo canal** (Opcion B).
Agentes envian comandos via `mpsc::Sender<MemoryCommand>`, un background thread
gestiona el store. Elimina contención de write locks entre agentes concurrentes.

| Fase | Que | LOC |
|------|-----|-----|
| G1 | `MemoryCommand` enum: `AddEpisode(Episode)`, `Recall(RecallQuery, oneshot::Sender<Vec<Episode>>)`, `RecallByTags(Vec<String>, usize, oneshot::Sender<Vec<Episode>>)`, `Consolidate`, `Shutdown` | ~30 |
| G2 | `MemoryService`: background thread que consume `mpsc::Receiver<MemoryCommand>` y opera sobre `AdvancedMemoryManager`. Creado por AgentPool. | ~60 |
| G3 | `MemoryHandle`: wrapper sobre `mpsc::Sender<MemoryCommand>` con metodos ergonomicos: `add_episode()`, `recall()`, `recall_by_tags()`. Clonable, Send+Sync. | ~40 |
| G4 | Anadir `memory: Option<MemoryHandle>` a `AgentRuntime` (en vez de Arc<RwLock<...>>) | ~20 |
| G5 | En `run_iteration()`: tras cada iteracion, `memory.add_episode(...)` con contexto (non-blocking, solo envia al canal) | ~35 |
| G6 | En `run_iteration()`: antes de generar respuesta, `memory.recall(query)` — envia query + oneshot receiver, espera respuesta. Fallback a `recall_by_tags()` si no hay embedding model disponible. | ~45 |
| G7 | `AgentPool`: crea `MemoryService` al inicio, pasa `MemoryHandle` a cada AgentRuntime | ~25 |
| G8 | Auto-persistencia: `MemoryService` llama `save_compressed()` al recibir `Shutdown` | ~20 |
| G9 | `create_agent_from_definition()` respeta `MemorySpec` del `AgentDefinition` (tipo, max_episodes, consolidation). Configura el MemoryService con estos parametros. | ~30 |

### G10: Tests serios de memoria

| Test | Que valida | LOC |
|------|-----------|-----|
| `test_agent_records_episode_after_task` | Ejecutar tarea -> episodio grabado con contenido correcto | ~25 |
| `test_agent_recalls_relevant_episodes` | 10 episodios -> tarea similar -> relevantes aparecen en prompt | ~30 |
| `test_agent_ignores_irrelevant_episodes` | Episodios de "search" -> tarea de "write" -> no se inyectan | ~25 |
| `test_episodic_decay_over_time` | Episodios antiguos pierden relevancia vs recientes | ~20 |
| `test_pool_shared_memory_via_channel` | Agente A envia episodio -> Agente B puede recall (via MemoryService) | ~30 |
| `test_memory_persists_across_pool_restarts` | Pool shutdown -> save -> nuevo pool -> load -> episodios intactos | ~30 |
| `test_memory_compressed_roundtrip` | save_compressed -> load_compressed -> datos identicos (checksum) | ~20 |
| `test_consolidation_creates_procedures` | N episodios similares -> consolidar -> procedimiento generado | ~25 |
| `test_memory_respects_max_episodes` | Superar max -> eviction de los mas antiguos | ~15 |
| `test_definition_memory_spec_wired` | AgentDefinition con MemorySpec -> MemoryService configurado correctamente | ~20 |
| `test_agent_no_memory_if_not_configured` | Sin MemorySpec -> agente funciona con memory = None | ~15 |
| `test_memory_isolation_between_pools` | Dos pools con MemoryServices independientes -> episodios no se cruzan | ~20 |
| `test_concurrent_writes_no_contention` | 5 agentes escriben simultaneamente -> todos los episodios grabados, sin panic | ~30 |

**LOC total Bloque G**: ~630

**Dependencias**: Bloques A + E (AgentRuntime, AgentPool).

---

## Bloque H — MCP Agent Management Tools

Exponer gestion de agentes via MCP para que el supervisor (y el usuario)
puedan monitorizar y controlar el pool.

| Fase | Que | LOC |
|------|-----|-----|
| H1 | `register_mcp_agent_tools(server: &mut McpServer, pool: Arc<RwLock<AgentPool>>)` | ~250 |
| H2 | Tests: cada tool individual + test de supervisor usando tools para investigar agente atascado | ~150 |

### Tools MCP registrados

| Tool | Que hace | read_only | destructive |
|------|----------|-----------|-------------|
| `agent_pool_status` | Lista agentes activos, tarea, iteracion, coste | true | false |
| `agent_task_progress` | Detalle de agente: tools llamados, ultimo output | true | false |
| `agent_stop` | Para un agente por task_id | false | true |
| `agent_spawn` | Lanza agente con perfil + tarea + rol | false | false |
| `agent_simplify_task` | Reemplaza tarea actual por version simplificada | false | true |
| `agent_list_profiles` | Lista perfiles disponibles con capabilities | true | false |
| `agent_memory_recall` | Busca episodios relevantes en memoria compartida | true | false |
| `agent_send_message` | Envia InterAgentMessage a otro agente via su buzon (mpsc) | false | false |

**LOC total Bloque H**: ~400

**Dependencias**: Bloques E (AgentPool) + G (memory) + A (renames).

---

## Orden de ejecucion

```
A (renames agentes)
|
+--> B + F (renames origen + casos puntuales) [paralelo]
|
+--> C (container backend) [independiente, paralelo con B]
|
+--> D (consolidar tools) [tras B, el mas arriesgado]
|
+--> E (agent wiring + pool + supervisor)
     |
     +--> G (memory wiring con canales)
          |
          +--> H (MCP agent tools)
```

## Peligros y mitigaciones

| Peligro | Mitigacion |
|---------|------------|
| Renames rompen tests/examples | Compilar+test tras cada fase |
| Renames derivados olvidados (Builder, Result, Config) | A2b-A2e cubren todos los derivados de AutonomousAgent |
| Consolidar tools rompe APIs internas | Migracion incremental: re-export primero, eliminar duplicados despues. D1 audita equivalencias semanticas antes de migrar. |
| Diferencias semanticas entre frameworks de tools | D1 crea tabla de equivalencias campo a campo obligatoria antes de migrar |
| AiConfig lifetime en closure del response_generator | Clonar config (son strings, coste despreciable) |
| Modelo inexistente en AgentSpec | Fallback a modelo global del AiAssistant; error si tampoco existe |
| ToolRef nombra tool no registrado | Warning en validacion, skip con log en runtime |
| Roles AgentDefinition vs TeamRole no coinciden | E2b: parse_team_role() unifica valores + Custom como fallback |
| AsyncGuard necesita runtime tokio | `run_stage_async` requiere contexto async; para sync callers, usar `Handle::current().block_on()` solo si runtime ya existe |
| AgentRuntime no es Clone | Pool gestiona agentes via JoinHandle (std::thread::spawn) |
| Guards async lentos | Timeout configurable por guard individual (default 5s); si timeout -> pass con warning |
| Budget = 0 | Enforcement solo si `budget > 0.0` |
| Pool no puede hookear iteraciones | E10c: IterationHook callback inyectado por el pool al crear AgentRuntime |
| Supervisor no tiene acceso a LLM | E10b: response_generator_factory en AgentPool, inyectado por AiAssistant |
| Pool: cola FIFO ignora prioridad | E10a: BinaryHeap ordenado por prioridad de tarea |
| Ejecucion concurrente no definida | E10a: std::thread::spawn por agente |
| Falta AgentProfile::from_definition() | E4b lo cubre |
| Embeddings para memory recall | G6: fallback a recall_by_tags si no hay embedding model |
| Contención en memoria compartida | G1-G3: modelo canal (mpsc) — writers nunca bloquean, background thread unico gestiona el store |
| agent_send_message sin buzon de entrada | E10h: mpsc::Receiver en AgentRuntime, consultado entre iteraciones |
| BestFit con todo vacio (score 0 para todos) | E9e: log warning + fallback a round-robin |
| Backward compatibility | Metodos existentes NO cambian. Todo es metodos nuevos paralelos |
| 5400 LOC de cambios | Cada bloque compila+test independientemente; merge por bloque |

---

## Resumen

| Bloque | Que | LOC | Riesgo |
|--------|-----|-----|--------|
| A | Renames agentes (7 tipos + 4 derivados) | ~450 | Medio |
| B | Renames en origen (eliminar 78 aliases) | ~800 | Medio |
| C | Container backend abstraction | ~200 | Bajo |
| D | Consolidar 7->1 frameworks de tools | ~1500 | Alto |
| E | Agent wiring + guardrails + BestFit + Pool + Supervisor | ~1290 | Medio |
| F | Casos puntuales naming | ~50 | Bajo |
| G | Memory wiring con canales + tests serios (13 tests) | ~630 | Medio |
| H | MCP agent management tools (8 tools) | ~400 | Medio |
| **TOTAL** | | **~5320** | |
