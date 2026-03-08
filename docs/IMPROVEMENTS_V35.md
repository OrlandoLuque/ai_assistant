# V35 — Agent System Overhaul: Naming, Wiring, Tools Consolidation, Memory, MCP

**Tesis**: Convertir el sistema de agentes de un conjunto de prototipos aislados en una
arquitectura integrada y coherente. Renombrar tipos para eliminar ambiguedades,
cablear AgentDefinition a ejecucion real, consolidar 7 frameworks de tools en 1,
integrar memoria episodica, conectar MCTS planner, habilitar model selection flexible,
y exponer gestion de agentes via MCP.

**Estado**: EN PROGRESO
**LOC estimadas**: ~7720 total (~7120 restante)
**Tests estimados**: ~65 nuevos (~850 LOC de tests incluidas en las estimaciones de LOC)

**Nota**: MCTS planner y memory avanzada (originalmente diferidos a v36) incluidos
en esta version como Bloque I.

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

**LOC est.**: ~450 total, **~0 restante** (A3/A5/A7 HECHO, A1/A2/A4/A6 REVERTIDO y descartado)

**Dependencias**: Ninguna.

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

**LOC est.**: ~800 total, **~650 restante** (~12 renames sobrevivieron de ~78; ~65 pendientes)

**Dependencias**: Bloque A completado (para no renombrar dos veces).

---

## Bloque C — Container Backend Abstraction

Unificar los dos `ContainerExecutor` (container_tools.rs y container_executor.rs)
bajo un trait comun con feature flag base.

| Fase | Que |
|------|-----|
| C1 | Crear feature `container-base` en Cargo.toml (zero deps extra — solo traits y tipos compartidos, sin impl). Clave: como `autonomous` depende de `container-base`, cualquier dep aqui afecta a todos los tests de `autonomous`. |
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
| D1 | Auditar: que tipos exporta cada modulo, quien los usa, dependencias cruzadas. Crear tabla de equivalencias campo a campo para detectar diferencias semanticas antes de migrar. **Artefacto obligatorio**: seccion "D1 Audit" en este documento (o archivo separado `docs/tools_audit.md`) con la tabla completa. No proceder a D2 sin artefacto revisado. |
| D2 | Definir set canonico en `unified_tools.rs`: `ToolCall`, `ToolResult`, `ToolRegistry`, `ToolParameter`, `ToolDefinition` |
| D3 | Migrar `tool_use.rs`: re-exportar tipos unificados o eliminar si redundante |
| D4 | Migrar `tool_calling.rs`: idem |
| D5 | Migrar `function_calling.rs`: idem |
| D6 | `tools.rs`: mantener traits de provider tools, usar tipos unificados |
| D7 | Consolidar `CompactableMessage`: mantener en `conversation_compaction.rs`, eliminar duplicado de `context_composer.rs` (re-exportar). Idem con `ConversationCompactor` duplicado. |
| D8 | Limpiar lib.rs: una sola exportacion por tipo, sin aliases de tools |
| D9 | Tests de regresion: verificar que tool_calling, function_calling, tools y unified_tools siguen funcionando tras migracion. Compilar cada feature flag independientemente. | ~80 |

**LOC est.**: ~1580 (refactor mas grande del plan — migracion incremental + tests regresion)

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
| E6 | Trait `AsyncGuard` con `async fn check()` + `GuardrailPipeline::run_stage_async()`: guards sync secuenciales (instantaneos), guards async en paralelo con `join_all`. Timeout configurable **por guard individual** (default 5s). Si no hay tokio runtime disponible (contexto sync puro), los async guards se saltan con log warning — solo ejecutan sync guards. | ~80 |
| E7 | `GuardrailSpec` -> construir pipeline: `PiiGuard`, `PatternGuard`, `MaxTokensGuard` (sync) + guards async opcionales | ~55 |

### E8: Budget enforcement

| Fase | Que | LOC |
|------|-----|-----|
| E8 | En `AgentRuntime::run_iteration()`: si `total_cost >= budget` -> `IterationOutcome::Error("budget exceeded")`. Solo si `budget > 0.0`. | ~20 |

### E9: BestFit mejorado

| Fase | Que | LOC |
|------|-----|-----|
| E9a | Anadir `required_capabilities: Vec<String>` y `preferred_role: Option<TeamRole>` a `AgentTask` (multi_agent). Anadir los mismos campos a `PoolTask` (agent_wiring). `PoolTask::to_agent_task()` convierte para que BestFit pueda puntuar. | ~25 |
| E9b | `fn score_agent_for_task(agent: &AgentProfile, task: &AgentTask) -> f64` — scoring multi-factor | ~40 |
| E9c | Factores: capability match exacto (0-100), role affinity (0-30), fallback substring en descripcion (0-10, solo si required_capabilities vacio) | incluido |
| E9d | Reemplazar closure inline de BestFit por `score_agent_for_task` | ~15 |
| E9e | Si todos los scores son 0.0 (todo vacio), log warning y fallback a round-robin | ~10 |
| E9f | Tests: scoring con capabilities exactas, role affinity, fallback substring, empates, todo-vacio-warning | ~70 |

### E10: AgentPool

| Fase | Que | LOC |
|------|-----|-----|
| E10a | `AgentPool { max_agents, active: HashMap<String, JoinHandle>, queue: BinaryHeap<PendingTask>, response_generator_factory, supervisor_config }`. Cola por prioridad (BinaryHeap), no FIFO. `PendingTask` impl `Ord` por prioridad (desc) + sequence number (asc, tiebreaker FIFO para misma prioridad). Cada agente corre en `std::thread::spawn`. | ~105 |
| E10b | `response_generator_factory: Arc<dyn Fn(Option<&str>) -> ResponseGenerator>` — campo del pool, inyectado por AiAssistant al crear el pool. Permite crear callbacks LLM para cualquier agente (incluido supervisor). | ~20 |
| E10c | `IterationHook` callback: `Arc<dyn Fn(&str, usize, &IterationOutcome) + Send + Sync>` — el pool lo pasa al AgentRuntime al crearlo. AgentRuntime lo invoca al final de cada iteracion. | ~30 |
| E10d | `AgentPool::on_iteration_complete()` — evaluacion de umbrales numericos (sin LLM) invocada por el IterationHook. | ~60 |
| E10e | `SupervisorConfig { enabled, idle_streak_threshold, budget_warning_percent, iteration_warning_margin, supervisor_profile, min_interval }`. El `supervisor_profile` incluye campo `model` — puede ser un modelo local/barato (ej: llama3) ya que el supervisor hace decisiones simples (evaluar umbrales, diagnosticar). BestFit NO interviene en la seleccion del modelo del supervisor — es config fija. | ~25 |
| E10f | `AgentPool::trigger_supervisor()` — instancia AgentRuntime temporal con rol Coordinator, contexto del problema, tools MCP de gestion. Usa `response_generator_factory(supervisor_profile.model)` para obtener callback LLM. Se destruye al completar. **Coste**: 1 LLM call por activacion; `min_interval` evita activaciones excesivas. | ~50 |
| E10g | `TriggerReason` enum (StuckDetected, BudgetWarning, NearIterationLimit) + `SupervisorTrigger` struct | ~20 |
| E10h | `mpsc::Receiver<InterAgentMessage>` buzón de entrada en AgentRuntime — permite recibir mensajes de otros agentes (incluido supervisor). Se consulta entre iteraciones. | ~30 |
| E10i | `CancellationToken`: `Arc<AtomicBool>` compartido entre pool y agente. AgentRuntime verifica `token.load()` al inicio de cada iteracion — si true, retorna `IterationOutcome::Cancelled`. El MCP tool `agent_stop` setea este flag. | ~25 |
| E10j | `ResultCollector`: `mpsc::Sender<(String, Result<AgentResult, String>)>` — cada agente envia su resultado al pool al terminar. Pool tiene `mpsc::Receiver` para recoger resultados. `AgentPool::collect_results() -> Vec<(task_id, Result)>` drena el canal. | ~40 |
| E10k | **Panic recovery**: `std::thread::spawn` wrappea en `std::panic::catch_unwind(AssertUnwindSafe(|| { ... }))`. `AssertUnwindSafe` necesario porque AutonomousAgent tiene campos `Arc`/closures que no son `UnwindSafe`. Si panic, envia `Err("agent panicked: {reason}")` al ResultCollector, marca tarea como Failed, log error. Opcionalmente reencola la tarea (configurable via `retry_on_panic: bool`, default false). | ~35 |
| E10l | **Graceful shutdown**: `AgentPool::shutdown(timeout: Duration)` — setea CancellationToken de todos los agentes activos, espera JoinHandles con timeout. Si timeout expira, log warning (threads no se pueden force-kill en Rust, pero el agente dejara de iterar al detectar el token). Envia `Shutdown` al MemoryService. | ~40 |
| E10m | `SupervisorConfig::min_interval: Duration` (default 30s) — evita activaciones excesivas del supervisor. Cada activacion cuesta tokens LLM. | ~10 |
| E10n | Tests: supervisor se activa con idle_streak, budget warning, near iteration limit; supervisor no se activa si disabled; cola por prioridad; buzón de mensajes; cancellation token; panic recovery; result collection; graceful shutdown | ~120 |

### E12: Model selection flexible

Hoy el modelo se decide una vez (baked en la closure `response_generator`) y no se puede
cambiar. Faltan overrides en multiples puntos. Prioridad de modelo (mayor a menor):
**query/tarea > definicion agente > config global**.

| Fase | Que | LOC |
|------|-----|-----|
| E12a | **PoolTask model override**: anadir `model_override: Option<String>` a `PoolTask`. Si presente, tiene prioridad sobre `AgentDefinition.agent.model`. En `execute_task_sync()` y `spawn_agent()`: `let model = task.model_override.as_deref().or(def.agent.model.as_deref());` | ~15 |
| E12b | **AutonomousAgentBuilder.with_response_generator()**: ya existe (`response_generator` es campo del builder). El caller crea el generator con el modelo deseado via factory y lo pasa al builder. NO anadir `.with_model()` al builder — el builder no tiene acceso a la factory; la responsabilidad de elegir modelo es del caller (AgentPool, AiAssistant). Documentar este patron. | ~10 |
| E12c | **AiAssistant model override por query**: `send_message_with_model(msg, context, model: &str)` — nuevo metodo que clona config, setea `selected_model`, y genera con ese modelo. El metodo original `send_message()` sigue usando el modelo global. | ~30 |
| E12d | **HTTP API model override**: `POST /chat` y `POST /v1/chat/completions` aceptan campo `model` en el body JSON. Si presente, override sobre config global. En server.rs y server_axum.rs. **Nota**: `/v1/chat/completions` ya deberia aceptar `model` segun spec OpenAI — esto es necesario para compatibilidad real, no solo nice-to-have. | ~40 |
| E12e | **MCP tool `agent_spawn` model param**: el tool acepta parametro opcional `model` que se pasa como `PoolTask.model_override`. | ~10 |
| E12f | **Runtime model switch (entre iteraciones)**: `response_generator` pasa de `Arc<dyn Fn>` a `Arc<RwLock<Box<dyn Fn(&[LoopMessage]) -> String + Send + Sync>>>`. Nuevo metodo `AutonomousAgent::switch_model(factory, model)` que pide al factory una nueva closure y la reemplaza bajo write lock. Read lock (uncontended, ~20ns) en cada iteracion al llamar al generator. El supervisor puede llamar esto via MCP tool `agent_switch_model`. Switch solo ocurre entre iteraciones, nunca durante generacion. El agente necesita recibir la factory como campo adicional (`model_factory: Option<ResponseGeneratorFactory>`). | ~50 |
| E12g | Tests: cadena completa de prioridad (HTTP body > PoolTask.model_override > AgentDefinition.model > AiConfig.selected_model — 4 niveles); send_message_with_model; HTTP /v1/chat/completions con model (OpenAI compat); runtime switch_model via MCP tool end-to-end | ~50 |

### E11: Tests de integracion

| Fase | Que | LOC |
|------|-----|-----|
| E11 | Tests end-to-end: definicion JSON -> create_agent_from_definition -> AgentRuntime -> tool call -> resultado. Tests de AgentProfile::from_definition -> registrar en orquestador -> BestFit asigna correctamente. | ~90 |

**LOC total Bloque E**: ~1655

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

## Bloque G1 — Memory Wiring + Tests Serios

Cablear la memoria episodica (ya implementada, 6495 LOC, 204 tests unitarios)
al sistema de agentes. Actualmente esta completamente huerfana.

Arquitectura de memoria compartida: **modelo canal** (Opcion B).
Agentes envian comandos via `mpsc::Sender<MemoryCommand>`, un background thread
gestiona el store. Elimina contencion de write locks entre agentes concurrentes.

| Fase | Que | LOC |
|------|-----|-----|
| G1a | `MemoryCommand` enum con sub-enums: `MemoryCommand::Episodic(EpisodicCmd)` (`AddEpisode`, `Recall`, `RecallByTags`, `Consolidate`), `MemoryCommand::Entity(EntityCmd)` (usados por G2: `Add`, `Query`, `Update`, `Remove`, `ListTypes`, `Relate`), `MemoryCommand::Plan(PlanCmd)` (usados por G3: `Save`, `Load`, `List`, `UpdateStep`), `MemoryCommand::System(SystemCmd)` (`Shutdown`, `FlushToDisk`). Sub-enums mantienen el enum raiz manejable. | ~50 |
| G1b | `MemoryService`: **std::thread::spawn** (no tokio) que consume `std::sync::mpsc::Receiver<MemoryCommand>` y opera sobre `AdvancedMemoryManager`. Creado por AgentPool. Loop con `recv_timeout(flush_interval)` para combinar receive + flush periodico (auto-save). Sin dependencia async. | ~70 |
| G1c | `MemoryHandle`: wrapper sobre `std::sync::mpsc::Sender<MemoryCommand>` con metodos ergonomicos: `add_episode()`, `recall()`, `recall_by_tags()`. Para recall (request-response), usa `std::sync::mpsc::sync_channel(0)` (rendezvous) como oneshot — no requiere tokio. Clonable, Send+Sync. | ~45 |
| G1d | Anadir `memory: Option<MemoryHandle>` a `AutonomousAgent` (via builder `.with_memory()`) | ~20 |
| G1e | En `run_iteration()`: tras cada iteracion exitosa, `memory.add_episode(...)` con contexto (non-blocking, solo envia al canal). Importance: 1.0 si tool output nuevo, 0.5 si similar a episodio reciente (similar = >=50% tag overlap con alguno de los ultimos 10 episodios). | ~40 |
| G1f | En `run_iteration()`: antes de generar respuesta, `memory.recall(query)` — envia query + oneshot receiver, espera respuesta con **timeout 100ms** (si MemoryService ocupado, el agente continua sin context de memoria, log warning). Inyecta como mensaje de contexto (no modifica system prompt). Fallback a `recall_by_tags()` si no hay embedding model. | ~55 |
| G1g | `AgentPool`: crea `MemoryService` al inicio, pasa `MemoryHandle` a cada AgentRuntime spawneado | ~25 |
| G1h | Auto-persistencia: `MemoryService` llama `save_compressed()` al recibir `Shutdown` | ~20 |
| G1i | `create_agent_from_definition()` respeta `MemorySpec` del `AgentDefinition` (tipo, max_episodes, consolidation). Configura el MemoryService con estos parametros. | ~30 |

### G1 Tests serios de memoria

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

**LOC total Bloque G1**: ~690

**Dependencias**: Bloques A + E (AgentRuntime, AgentPool).

---

## Bloque G2 — EntityStore como Knowledge Store generico

El `EntityStore` de `advanced_memory/entity.rs` ya tiene la estructura correcta para
almacenar cualquier "cosa" que el agente necesite recordar: dispositivos IoT, ficheros,
pisos para comprar, ofertas de empleo, APIs, contactos, etc. El `EntityRecord` tiene
`entity_type: String` + `attributes: HashMap<String, Value>` + `relations`.

Lo que falta es convertirlo de store interno a store de uso general:

### G2.1: Persistencia propia del EntityStore

| Fase | Que | LOC |
|------|-----|-----|
| G2.1a | `EntityStore::save(path)` / `EntityStore::load(path)` — JSON, independiente del manager | ~40 |
| G2.1b | Persistencia incremental: `save_entity(id)` guarda una sola entidad (append-friendly) | ~30 |
| G2.1c | Auto-save configurable: guardar tras N cambios o T segundos. Implementado dentro del background thread de `MemoryService` (G2), no como timer independiente. MemoryService ya tiene el loop — solo anadir logica de flush periodico. | ~25 |

### G2.2: Query por atributos

| Fase | Que | LOC |
|------|-----|-----|
| G2.2a | `EntityQuery` struct: filtros por `entity_type`, rangos numericos, substring en strings, tags | ~60 |
| G2.2b | `EntityStore::query(q: &EntityQuery) -> Vec<&EntityRecord>` — evaluacion secuencial de filtros | ~50 |
| G2.2c | Operadores: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `contains`, `starts_with`, `in_list` | incluido |
| G2.2d | `EntityStore::count() -> usize`, `list_types() -> Vec<String>`, `list_by_type(t) -> Vec<&EntityRecord>` — metodos basicos de gestion | ~20 |

### G2.3: Indice vectorial para busqueda semantica

El EntityStore tiene `attributes` pero no embeddings. Para buscar "entidades relacionadas
con X" necesita busqueda semantica.

| Fase | Que | LOC |
|------|-----|-----|
| G2.3a | Anadir `embedding: Option<Vec<f32>>` a `EntityRecord` con `#[serde(default)]` para backward compat con datos serializados existentes | ~5 |
| G2.3b | `EntityStore::search_similar(query_embedding, top_k) -> Vec<(&EntityRecord, f64)>` — cosine similarity brute-force (suficiente para <10K entidades) | ~40 |
| G2.3c | `EntityStore::auto_embed(entity_id, embedder)` — genera embedding a partir de nombre+tipo+atributos concatenados | ~30 |
| G2.3d | ~~Indice HNSW~~ **DIFERIDO** — brute-force es <1ms para <10K entidades. HNSW solo justificado para >100K. Se puede anadir como feature flag futuro sin cambiar la API publica (search_similar sigue igual, cambia solo la implementacion interna). | 0 |

### G2.4: TTL y health check

| Fase | Que | LOC |
|------|-----|-----|
| G2.4a | Anadir `ttl: Option<Duration>` y `expires_at: Option<u64>` a `EntityRecord` con `#[serde(default)]` | ~10 |
| G2.4b | `EntityStore::evict_expired()` — elimina entidades cuyo TTL ha expirado | ~15 |
| G2.4c | `EntityStore::health_check(id) -> HealthStatus` — hook para verificar si un servicio/dispositivo sigue accesible (trait `HealthChecker`) | ~30 |

### G2.5: MCP tools para entidades

| Tool | Descripcion | read_only | destructive |
|------|-------------|-----------|-------------|
| `entity_add` | Crear entidad con tipo, atributos, relaciones | false | false |
| `entity_search` | Buscar por tipo, atributos, o semanticamente | true | false |
| `entity_update` | Actualizar atributos de una entidad | false | false |
| `entity_remove` | Eliminar entidad | false | true |
| `entity_list_types` | Listar tipos de entidad distintos | true | false |
| `entity_relate` | Crear relacion entre dos entidades | false | false |

LOC estimadas: ~120 (registro MCP + handlers)

**Routing**: Los MCP tools acceden al EntityStore a traves de `MemoryService` (nuevos
comandos: `AddEntity`, `QueryEntities`, `UpdateEntity`, `RemoveEntity`, `ListEntityTypes`,
`RelateEntities`). No acceden directamente al store para mantener consistencia con G1.

### G2.6: Tests

| Test | Que valida | LOC |
|------|-----------|-----|
| `test_entity_save_load_roundtrip` | Persistencia JSON ida y vuelta | ~20 |
| `test_entity_query_by_type` | Filtro por entity_type | ~15 |
| `test_entity_query_numeric_range` | price > 100000 AND price < 200000 | ~20 |
| `test_entity_query_substring` | Buscar "Madrid" en atributos string | ~15 |
| `test_entity_similar_search` | Busqueda semantica por embedding | ~25 |
| `test_entity_auto_embed` | Auto-generar embedding desde texto | ~20 |
| `test_entity_ttl_expiration` | Entidad con TTL expira y se evicta | ~15 |
| `test_entity_health_check` | Health check de servicio online/offline | ~15 |
| `test_entity_mcp_add_search` | MCP tool add -> search -> encontrado | ~20 |
| `test_entity_incremental_save` | Guardar entidad individual sin reescribir todo | ~15 |

**LOC total Bloque G2**: ~620

**Dependencias**: Bloque G1 (memory wiring). Feature: `advanced-memory` (ya en `full`).

---

## Bloque G3 — Persistencia de planificaciones

`task_planning.rs` ya tiene `TaskPlan`, `PlanStep`, `PlanBuilder`, `PlanSummary`
con `to_json()/from_json()`. Lo que falta es integrarlo con el sistema de agentes
y la persistencia automatica.

| Fase | Que | LOC |
|------|-----|-----|
| G3.1 | `TaskPlan::save(path)` / `TaskPlan::load(path)` — wrapper sobre to_json/from_json con I/O | ~20 |
| G3.2 | `PlanStore`: HashMap<String, TaskPlan> con save/load del conjunto completo | ~40 |
| G3.3 | Integrar con `MemoryService`: nuevo comando `SavePlan(TaskPlan)`, `LoadPlan(id)`, `ListPlans` | ~35 |
| G3.4 | `PlanHandle` (como MemoryHandle): wrapper ergonomico sobre sender | ~25 |
| G3.5 | Auto-persistencia: guardar plan tras cada cambio de estado (step done/blocked) | ~20 |
| G3.6 | Vincular plan a agente: `AgentRuntime` tiene `active_plan: Option<PlanHandle>` | ~15 |
| G3.7 | MCP tools: `plan_create`, `plan_status`, `plan_update_step`, `plan_list`. Routing via MemoryService (comandos `SavePlan`, `LoadPlan`, `ListPlans`, `UpdatePlanStep`). | ~80 |

### G3 Tests

| Test | Que valida | LOC |
|------|-----------|-----|
| `test_plan_save_load` | Persistencia ida y vuelta | ~15 |
| `test_plan_store_multiple` | Guardar/cargar multiples planes | ~15 |
| `test_plan_auto_save_on_step_change` | Cambiar status de step -> auto-guardado | ~20 |
| `test_plan_summary_accurate` | PlanSummary refleja estado real | ~10 |
| `test_plan_mcp_create_status` | MCP create -> status -> correcto | ~15 |

**LOC total Bloque G3**: ~310

**Dependencias**: Bloque G1 (MemoryService).

---

## Bloque H — MCP Agent Management Tools

Exponer gestion de agentes via MCP para que el supervisor (y el usuario)
puedan monitorizar y controlar el pool.

| Fase | Que | LOC |
|------|-----|-----|
| H1 | `register_mcp_agent_tools(server: &mut McpServer, pool: Arc<RwLock<AgentPool>>)`. Patron de registro: cada dominio tiene su funcion (`register_mcp_docker_tools` ya existe, G2.5 anade `register_mcp_entity_tools`, G3.7 anade `register_mcp_plan_tools`, H1 anade `register_mcp_agent_tools`). El binario llama a todas en secuencia. Composable, no monolitico. | ~250 |
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
| `agent_switch_model` | Cambia modelo de un agente en runtime (E12f). Parametros: agent_id, model. | false | false |

**LOC total Bloque H**: ~430

**Dependencias**: Bloques E (AgentPool) + G (memory) + A (renames).

---

## Bloque I — MCTS Planner Wiring en Agent Runtime

Conectar `mcts_planner.rs` (planificacion optimizada de tool calls) con el runtime
de agentes. La memoria ya queda cableada por el Bloque G (MemoryHandle via canales).
MCTS es el unico modulo que sigue aislado.

Originalmente diferido a V36, movido a V35 por decision del autor.

**NOTA**: Los sub-bloques I3/I4 originales (memory en agent/pool) se eliminaron porque
duplicaban G4-G7. La memoria se inyecta via `MemoryHandle` (G), no `Arc<RwLock<...>>`.

### I1 — AgentState para MCTS (~180 LOC)

Implementar `MctsState` sobre el estado de un agente para que MCTS pueda planificar
secuencias de tool calls antes de ejecutarlas.

| Fase | Que | LOC |
|------|-----|-----|
| I1a | `AgentPlanningState` impl `MctsState` — estado: conversation context (resumido como String), tools disponibles (Vec<String> con descripciones de ToolDef), actions completadas (Vec<String>), goal (String), memory_hints (Vec<String> de episodios relevantes). `available_actions()` retorna tools del registry menos los ya usados en esta simulacion. `apply_action()` NO ejecuta el tool — clona el estado y anade la accion a `completed`. `is_terminal()` true si completed.len() >= max_depth o no quedan acciones. `reward()` usa dos capas (ver I1b). | ~80 |
| I1b | `AgentRewardModel` con `SimulationStrategy` configurable. **Default (MetadataHistorico)**: Capa 1 (metadata): similitud entre `ToolDef.description` y goal (bag-of-words cosine, peso 0.6) + diversidad (penaliza repetir tool, peso 0.2) + progreso (ratio actions/max_depth, peso 0.1). Capa 2 (historico): si `MemoryHandle` disponible, bonus +0.1 si episodios pasados muestran exito del tool para goals similares (match por tags). Coste cero. **Alternativas configurables**: `SimulationStrategy::HeuristicOnly` — solo keyword match nombre tool vs goal (rapido, menos preciso, coste cero). `SimulationStrategy::Hybrid { llm_depth, simulate_fn }` — metadata+historico para la mayoria de nodos, LLM simulation (via `simulate_fn: Arc<dyn Fn(&str) -> String>` inyectado por el caller, desacoplado de MCTS) para los top-3 branches a profundidad <= llm_depth (mas preciso, coste medio: ~3-9 LLM calls por planning cycle). Pesos configurables via `RewardWeights`. | ~80 |
| I1c | `plan_next_actions(planner, state) -> Vec<String>` — funcion publica que ejecuta MCTS search y devuelve la secuencia optima de nombres de tools. Devuelve `Vec` vacio si MCTS no encuentra mejora sobre random (root_value < 0.1). | ~40 |

### I2 — MCTS hook en AutonomousAgent (~130 LOC)

| Fase | Que | LOC |
|------|-----|-----|
| I2a | Campo `planner: Option<MctsPlanner>` en `AutonomousAgent`. Metodo `.with_planner(planner)` en `AutonomousAgentBuilder`. | ~15 |
| I2b | En `run_iteration()`, si `self.planner.is_some()` y `iteration % plan_interval == 0`: construir `AgentPlanningState` desde estado actual, llamar `plan_next_actions()`, inyectar resultado como **mensaje al final de la conversacion** (no system prompt): `LoopMessage { role: System, content: "Planning hint: consider using [tool1, tool2] next for optimal progress" }`. El LLM NO esta obligado a seguir el plan — es una sugerencia. **Cleanup**: campo `planning_hint_idx: Option<usize>` en AutonomousAgent almacena indice del mensaje hint. Al inicio de la siguiente iteracion, si existe, se elimina de self.conversation antes de inyectar el nuevo (o de generar sin hint). | ~80 |
| I2c | `plan_interval` configurable (default: cada 3 iteraciones). `MctsPlanner::default()` usa `max_iterations: 100`, latencia esperada <10ms para <20 tools. | ~15 |
| I2d | Metricas integradas con OTel: `mcts_plans_generated` (counter), `mcts_plan_followed` (counter, incrementa si LLM elige la tool sugerida como primera accion). Registradas via `record_metric()` existente en `opentelemetry_integration.rs`. | ~25 |

### I3 — Tests (~120 LOC, 8 tests)

| Test | Que valida |
|------|-----------|
| `test_agent_planning_state_actions` | `available_actions()` retorna tools del registry menos los ya usados |
| `test_agent_planning_state_terminal` | `is_terminal()` true al max_depth |
| `test_agent_planning_state_apply` | `apply_action()` clona estado y anade accion sin side effects |
| `test_agent_reward_model_relevance` | Tool relevante al goal puntua > irrelevante |
| `test_agent_reward_model_diversity` | Repetir tool penaliza reward |
| `test_mcts_plan_next_actions` | `plan_next_actions()` retorna secuencia no vacia con tools relevantes |
| `test_mcts_plan_empty_tools` | Sin tools → `Vec` vacio |
| `test_mcts_plan_low_value_returns_empty` | root_value < 0.1 → `Vec` vacio (no vale la pena planificar) |

**LOC total Bloque I**: ~430

**NOTA**: Los sub-bloques I3/I4 originales (~320 LOC) se eliminaron porque
duplicaban G4-G7. LOC reducido de ~750 a ~430.

**Dependencias**: Bloques E (AgentPool, IterationHook) + G1 (memory wiring base).
Feature: `autonomous` + `devtools` (mcts_planner esta en devtools).

---

## Fase de documentacion

Actualizar toda la documentacion tras completar los bloques funcionales.
Incluye deuda documental acumulada de V30-V34 (detectada en auditoria).

### Deuda documental V30-V34 (catch-up)

| Doc | Que falta | Prioridad |
|-----|-----------|-----------|
| **GUIDE.md** | 4+ secciones nuevas: v31 (real streaming), v32 (scalability monitors), v33 (Docker integration tests), v34 (MCP Docker tools, `/docker` REPL). Actualizar seccion containers existente. | ALTA |
| **CONCEPTS.md** | 16+ secciones faltantes: v30 features (BPE tokenizer, emoji detection, benchmark suite, MCP config/eval, OpenAI-compat API, routing DAG), v31 (virtual models), v32 (scalability monitoring), v33-v34 (Docker integration patterns). | ALTA |
| **mcp_catalog.html** | 8 Docker MCP tools de v34 no listados (docker_list/create/start/stop/remove/exec/logs/status). Ultimo update fue v9. | ALTA |
| **AGENT_SYSTEM_DESIGN.md** | Faltan secciones de server/cluster architecture, virtual models, Docker integration patterns. | MEDIA |
| **framework_comparison.html** | Parado en v33, falta v34 (Docker CLI/MCP). | MEDIA |
| **TESTING.md** | Falta seccion Docker integration tests (24 tests v33), test count desactualizado (~6417 vs ~6657 actual). | MEDIA |
| **feature_matrix.html** | Minor — ya tiene v35 backup, solo necesita visibilidad de nuevos items. | BAJA |

### Documentacion V35 (nuevos bloques)

| Doc | Que anadir |
|-----|------------|
| **GUIDE.md** | Secciones: AgentPool, MCTS planning, EntityStore como knowledge store, MemoryService canales, supervisor event-driven, BestFit mejorado |
| **CONCEPTS.md** | Conceptos: pool de agentes, supervisor, cancellation token, MCTS planning para agentes, EntityStore generico, plan persistence, memory wiring canales |
| **AGENT_SYSTEM_DESIGN.md** | Seccion AgentPool architecture, supervisor triggers, MCTS integration, memory wiring |
| **feature_matrix.html** | Filas: AgentPool, MCTS wiring, EntityStore vector search, plan persistence, 22+ MCP tools nuevos |
| **framework_comparison.html** | Actualizar scores de agent orchestration, memory, tool frameworks |
| **mcp_catalog.html** | 22+ MCP tools nuevos: 8 Docker (v34) + 6 entity + 4 plan + 8 agent management |
| **TESTING.md** | Actualizar test count final, listar categorias nuevas |
| **examples/multi_agent.rs** | Actualizar con tipos renombrados y nueva API (AgentPool, etc.) |

### Politica

- **Backups HTML obligatorios** ANTES de modificar (docs/backups/)
- Verificar que framework_comparison refleja estado real vs competencia
- Verificar que feature_matrix marca correctamente HECHO/PENDIENTE

**Dependencias**: Todos los bloques funcionales completados.

---

## Orden de ejecucion

```
A (renames agentes) .............. PARCIALMENTE HECHO
|
+--> B + F (renames + casos puntuales) [paralelo] .. B PARCIALMENTE HECHO
|
+--> C (container backend) [independiente, paralelo con B]
|
+--> D (consolidar tools + CompactableMessage) [tras B, el mas arriesgado]
|
+--> E (agent wiring + pool + supervisor + shutdown)
     |
     +--> G1 (memory wiring con canales)
          |
          +--> G2 (EntityStore knowledge store) [paralelo con G3/I/H]
          |
          +--> G3 (plan persistence) [paralelo con G2/I/H]
          |
          +--> I (MCTS planner wiring) [paralelo con G2/G3/H]
          |
          +--> H (MCP agent tools) [paralelo con G2/G3/I]
               |
               +--> Documentacion (GUIDE, CONCEPTS, HTML, examples) [ultimo]
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
| MCTS overhead en cada iteracion | I2c: plan_interval configurable (default cada 3 iter), no cada paso |
| MCTS sin tools utiles | I1c: devuelve Vec vacio si root_value < 0.1, agente continua normal |
| MCTS simulacion imprecisa | I1b: dos capas (metadata ToolDef + historico MemoryHandle), sin LLM, coste cero. Precision suficiente para sugerir, no decidir |
| Memory recall lento con muchos episodios | I3b: top_k=3, fallback a recall_by_tags sin embeddings |
| Memory consolidation bloquea agente | I3d: consolidation en background, no bloquea iteracion |
| Agent thread panic | E10k: catch_unwind wrappea el thread, envia Err al ResultCollector, marca tarea Failed |
| Pool no recolecta resultados | E10j: ResultCollector via mpsc canal, pool drena resultados |
| Pool sin shutdown graceful | E10l: shutdown() setea CancellationToken + espera JoinHandles con timeout |
| Agent no para cuando se le pide | E10i: CancellationToken (AtomicBool) verificado cada iteracion |
| Supervisor se activa excesivamente (coste LLM) | E10e: min_interval configurable (default 30s) entre activaciones. Ademas, supervisor_profile puede usar modelo local/barato. |
| D sin tests de regresion | D9: tests verifican que cada feature flag compila y funciona tras migracion |
| CompactableMessage x2 sin bloque asignado | D7: consolidar en conversation_compaction, re-exportar desde context_composer |
| EntityRecord nuevo campo rompe JSON existente | G2.3a/G2.4a: #[serde(default)] en campos nuevos |
| G2/G3 MCP tools acceden a store sin coordinacion | G2.5/G3.7: routing via MemoryService, no acceso directo |
| Runtime model switch: data race | E12f: Arc<RwLock<ResponseGenerator>> — write lock solo durante switch, read lock en cada iteracion. Switch entre iteraciones, nunca durante generacion. |
| Model override confusion (quien gana?) | E12a: prioridad clara: query/tarea > definicion agente > config global. Documentado en GUIDE. |
| Recall bloqueante si MemoryService ocupado | G1f: timeout 100ms, si expira el agente continua sin memoria con log warning |
| BinaryHeap Ord incompleto | E10a: sequence number como tiebreaker FIFO para misma prioridad |
| MemoryCommand enum explota con G2/G3 | G1a: sub-enums (Episodic, Entity, Plan, System) mantienen manejable |
| PoolTask vs AgentTask desconectados | E9a: PoolTask::to_agent_task() convierte para BestFit |
| Examples desactualizados tras renames | Fase documentacion: actualizar examples/multi_agent.rs |
| ~7700 LOC de cambios | Cada bloque compila+test independientemente; merge por bloque |

---

## Resumen

| Bloque | Que | LOC total | LOC restante | Riesgo |
|--------|-----|-----------|--------------|--------|
| A | Renames agentes | ~450 | **~0** | — (HECHO) |
| B | Renames en origen (eliminar aliases lib.rs) | ~800 | **~650** | Medio |
| C | Container backend abstraction | ~200 | ~200 | Bajo |
| D | Consolidar tools + CompactableMessage + tests regresion | ~1580 | ~1580 | **Alto** |
| E | Agent wiring + pool + supervisor + shutdown + cancel + model selection | ~1655 | ~1655 | Medio-Alto |
| F | Casos puntuales naming | ~50 | ~50 | Bajo |
| G1 | Memory wiring con canales std (13 tests) | ~690 | ~690 | Medio |
| G2 | EntityStore: knowledge store + vector search brute-force + MCP | ~620 | ~620 | Medio |
| G3 | Persistencia de planificaciones (TaskPlan) + MCP | ~310 | ~310 | Bajo |
| H | MCP agent management tools (9 tools) | ~430 | ~430 | Medio |
| I | MCTS planner wiring (solo MCTS, memory via G) | ~435 | ~435 | Medio |
| — | Documentacion V30-V34 catch-up + V35 (GUIDE, CONCEPTS, HTML, examples) | ~500 | ~500 | Bajo |
| **TOTAL** | | **~7720** | **~7120** | |

---

## Auditoria de APIs y medios de comunicacion

Inventario completo de todas las superficies API y protocolos del proyecto,
incluyendo redundancias detectadas y protocolos ausentes evaluados.

### APIs expuestas

#### HTTP/REST (26 endpoints, 2 servidores)

**Servidor ligero (`server.rs` — bloqueante, `std::net::TcpListener`):**

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | Listar modelos |
| POST | `/chat` | Enviar mensaje (no-streaming) |
| GET | `/config` | Obtener configuracion |
| POST | `/config` | Actualizar configuracion |
| GET | `/metrics` | Metricas Prometheus |
| GET | `/sessions` | Listar sesiones |
| GET | `/sessions/{id}` | Detalle de sesion |
| DELETE | `/sessions/{id}` | Eliminar sesion |
| POST | `/v1/chat/completions` | Completions compatible OpenAI |
| GET | `/v1/models` | Modelos formato OpenAI |
| POST | `/chat/stream` | SSE streaming token-a-token |
| GET | `/openapi.json` | Especificacion OpenAPI 3.0 |

**Servidor produccion (`server_axum.rs` — async, Axum + Tower):**
Todos los anteriores MAS:

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ws` | WebSocket bidireccional (RFC 6455) |
| POST | `/mcp` | MCP JSON-RPC 2.0 |
| GET | `/admin/virtual-models` | Listar virtual models |
| POST | `/admin/virtual-models` | Crear virtual model |
| GET | `/admin/virtual-models/{name}` | Obtener virtual model |
| PUT | `/admin/virtual-models/{name}` | Actualizar virtual model |
| DELETE | `/admin/virtual-models/{name}` | Eliminar virtual model |
| GET | `/admin/models` | Listar todos los modelos |
| POST | `/admin/models/{name}/publish` | Publicar modelo |
| POST | `/admin/models/{name}/unpublish` | Despublicar modelo |

**Nota**: Los dos servidores son complementarios por diseno, NO redundantes.
`server.rs` es zero-dependency (HTTP manual); `server_axum.rs` es para produccion
(HTTP/2, WebSocket, Tower middleware). Ambos comparten `ServerConfig`, `AuthResult`,
`CorsConfig`.

#### MCP Tools (8 — Docker management)

Registrados en `POST /mcp` via JSON-RPC 2.0:

| Tool | Descripcion | read_only | destructive |
|------|-------------|-----------|-------------|
| `docker_list_containers` | Listar containers | true | false |
| `docker_create_container` | Crear container | false | false |
| `docker_start_container` | Iniciar container | false | false |
| `docker_stop_container` | Parar container | false | false |
| `docker_remove_container` | Eliminar container | false | true |
| `docker_exec` | Ejecutar comando | false | false |
| `docker_logs` | Obtener logs | true | false |
| `docker_container_status` | Estado del container | true | false |

#### CLI REPL (~20 comandos)

**Comandos estandar (`repl.rs` — ReplEngine):**
`/help`, `/exit`, `/quit`, `/models`, `/config`, `/clear`, `/save`, `/load`,
`/template`, `/model`, `/history`, `/cost`

**Subcomandos Docker (`ai_assistant_standalone`):**
`/docker list|create|start|stop|rm|exec|logs|status|cleanup|help`

#### A2A Protocol (Agent-to-Agent, protocolo Google)

- Discovery: `/.well-known/agent.json` (AgentCard)
- Transporte: JSON-RPC 2.0
- Lifecycle: Submitted -> Working -> InputRequired -> Completed/Failed/Canceled
- AgentDirectory para registro y busqueda por skills

#### Binarios (7)

| Binario | Funcion |
|---------|---------|
| `ai_assistant_standalone` | HTTP server + REPL + Docker |
| `ai_assistant_server` | HTTP server CLI (--host, --port, --tls) |
| `ai_assistant_cli` | REPL interactivo + Docker |
| `ai_cluster_node` | Nodo de cluster distribuido |
| `ai_proxy` | Proxy gateway |
| `ai_test_harness` | Harness de testing |
| `kpkg_tool` | Package management |

---

### Protocolos y medios de comunicacion (17 total)

| # | Protocolo | Modulo(s) | Descripcion |
|---|-----------|-----------|-------------|
| 1 | HTTP/HTTPS | server.rs, server_axum.rs | REST API, TLS (rustls), gzip, auth, rate limiting, CORS |
| 2 | WebSocket | server.rs (`/ws`) | Chat bidireccional, RFC 6455 (handshake SHA-1+base64 from scratch) |
| 3 | SSE | server.rs (`/chat/stream`) | Streaming token-a-token, `text/event-stream` |
| 4 | MCP JSON-RPC 2.0 | mcp_protocol/, mcp_client.rs | Model Context Protocol — tools, resources, prompts (v1+v2) |
| 5 | A2A | a2a_protocol.rs | Agent-to-Agent — AgentCards, skills discovery, task lifecycle |
| 6 | P2P / QUIC | p2p.rs, distributed_network.rs | QUIC/TLS 1.3 (Quinn), descubrimiento LAN, peer exchange, anti-entropy |
| 7 | Kademlia DHT | distributed.rs | Tabla hash distribuida |
| 8 | CRDTs | distributed.rs | GCounter, PNCounter, LWWRegister, ORSet |
| 9 | MapReduce | distributed.rs | Ejecucion distribuida map/reduce |
| 10 | ICE/STUN/NAT | p2p.rs | NAT traversal: STUN binding, UPnP, NAT-PMP |
| 11 | WebRTC | voice_agent.rs | Audio bidireccional en tiempo real (SDP, ICE) |
| 12 | CDP | browser_tools.rs | Chrome DevTools Protocol — automatizacion browser |
| 13 | Redis pub/sub | server_cluster.rs | Comunicacion inter-nodo en cluster |
| 14 | mpsc channels | agent_wiring.rs, autonomous_loop.rs | Comunicacion interna agentes/threads |
| 15 | Resumable streaming | resumable_streaming.rs | Streaming con reconexion automatica y checkpoint |
| 16 | Webhooks (outbound) | webhooks.rs | HMAC-SHA256 signing, retry, 8 tipos de evento |
| 17 | Plugin hooks | plugins.rs | on_request/on_response/on_event (colocated, no inter-plugin) |

---

### Redundancias detectadas

| Nivel | Que | Detalle | Accion |
|-------|-----|---------|--------|
| **CRITICA** | 3 frameworks de tools | `tools.rs`, `tool_calling.rs`, `unified_tools.rs` definen cada uno `ToolCall`, `ToolResult`, `ToolRegistry`. Ademas `function_calling.rs` duplica `ToolChoice`. | **Bloque D** de este plan |
| **ALTA** | CompactableMessage x2 | Definido identico en `conversation_compaction.rs` Y `context_composer.rs`. Tambien `ConversationCompactor` duplicado. | Consolidar: mantener en conversation_compaction, re-exportar desde context_composer |
| OK | 2 servidores HTTP | `server.rs` (bloqueante) vs `server_axum.rs` (async) | Complementarios por diseno |
| OK | MCP v1/v2 | Separacion limpia en `mcp_protocol/` | Bien estructurado |
| OK | CacheStats x3 | memory_management, caching, persistence — dominios distintos | Nombres ya diferenciados via lib.rs aliases |
| OK | ValidationResult x4 | output_validation, fine_tuning, cot_parsing, structured — dominios distintos | Nombres ya diferenciados via lib.rs aliases |
| OK | SearchResult x4 | widgets, web_search, search, embeddings — dominios distintos | Nombres ya diferenciados via lib.rs aliases |

---

### Protocolos ausentes — evaluacion

| Protocolo | Relevancia | Decision | Motivo |
|-----------|------------|----------|--------|
| **OTLP gRPC export** | Media | **PENDIENTE** | Tracer local funciona (Prometheus text + OTLP JSON), pero no envia a collectors (Jaeger/Datadog/Tempo). Falta `opentelemetry-otlp` + `tonic`. |
| **Webhook inbound** | Media | **PENDIENTE** | Outbound funciona (HMAC-SHA256), pero no hay receptor/validador de firmas entrantes. |
| **gRPC** | Baja | **DESCARTADO** | REST + OpenAPI es el estandar para APIs LLM. Sin .proto ni tonic/prost. |
| **GraphQL** | Baja | **DESCARTADO** | REST es adecuado; OpenAPI proporciona introspeccion. |
| **MQTT** | Baja | **DESCARTADO** | Protocolo IoT (pub/sub ligero para sensores/domotica). No encaja con el perfil del proyecto. Si se necesitara control IoT, se haria via MCP tool externo (50 lineas), no nativo en el crate. Redis pub/sub cubre las necesidades de pub/sub distribuido. |
| **Email/SMTP** | Baja | **DESCARTADO** | No es una libreria de notificaciones. Requeriria gestion de secrets SMTP. |
| **Unix sockets** | Baja | **DESCARTADO** | IPC via channels + locks es suficiente para proceso unico. |
| **Named pipes** | Baja | **DESCARTADO** | Windows-specific, no es core. |
| **Shared memory** | Baja | **DESCARTADO** | Arc<Mutex<T>> y canales mpsc cubren la concurrencia interna. |

### Protocolos cloud implementados

| Servicio | Protocolo | Auth | Modulo |
|----------|-----------|------|--------|
| Amazon S3 | HTTP REST | AWS SigV4 | cloud_connectors.rs, aws_auth.rs |
| Google Drive | HTTP REST | OAuth2 Bearer | cloud_connectors.rs |
| HuggingFace Hub | HTTP REST | API token | huggingface_connector.rs |

---

### Resumen numerico

| Categoria | Cuenta |
|-----------|--------|
| HTTP endpoints | 26 |
| WebSocket endpoints | 1 |
| SSE streams | 2 |
| MCP tools | 8 |
| CLI REPL commands | ~20 |
| A2A endpoints | 1 |
| Protocolos de comunicacion | 17 |
| Modulos publicos Rust (lib.rs) | ~233 |
| Feature flags | 33 |
| Binarios | 7 |
| Redundancias criticas | 2 (tools x3, CompactableMessage x2) |
| Protocolos pendientes | 2 (OTLP export, webhook inbound) |
| Protocolos descartados | 6 (gRPC, GraphQL, MQTT, email, unix sockets, named pipes) |
