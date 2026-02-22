# Plan de Mejoras para ai_assistant — v4

> Documento generado el 2026-02-22.
> Basado en completitud de v1 (39/39), v2 (22/22), v3 (21/21) con 2730+ tests, 0 failures.
> 220+ source files, ~210k+ LoC.
>
> **Planes anteriores**: v1 (providers, embeddings, MCP, documents, guardrails),
> v2 (async parity, vector DBs, evaluation, testing),
> v3 (containers, document pipeline, speech/audio, CI/CD maturity).

---

## Contexto

Tras v3, ai_assistant cubre: 15+ LLM providers, RAG 5 niveles, multi-agente,
agente autonomo, distribuido P2P/QUIC, seguridad RBAC, containers Docker,
generacion de documentos, speech STT/TTS, y CI/CD completo.

Analisis competitivo contra LangChain/LangGraph, LlamaIndex, Semantic Kernel,
CrewAI, DSPy, Haystack, Vercel AI SDK y AutoGen revela estas **brechas clave**:

1. **Sin motor de workflows event-driven** — El DAG executor actual es estatico.
   LangGraph 1.0, LlamaIndex Workflows y CrewAI Flows usan arquitecturas
   event-driven con checkpointing duradero y time-travel debugging.

2. **Sin optimizacion automatica de prompts** — DSPy define signatures
   declarativas y compila prompts optimos via busqueda bayesiana. Ningun
   framework Rust ofrece esto.

3. **Sin protocolo A2A** — Google/Linux Foundation impulsan Agent-to-Agent
   (150+ organizaciones). Es el estandar emergente para interoperabilidad
   entre agentes de distintos frameworks.

4. **Memoria limitada** — Falta memoria episodica, procedural, entidad
   persistente y consolidacion (episodica → semantica). CrewAI y la
   investigacion A-MEM muestran -90% uso de tokens con memoria rica.

5. **Sin evaluacion online** — La evaluacion actual es offline. LangSmith
   y TruLens ofrecen feedback functions en produccion en tiempo real.

6. **Guardrails no operan en streaming** — Los guardrails aplican solo a
   respuestas completas. NeMo v0.20 aplica safety rails en tiempo real
   durante la generacion.

---

## Decisiones de Diseno (propuestas)

| Decision | Eleccion |
|---|---|
| **Workflow engine** | Event-driven con typed events, checkpointing SQLite, time-travel |
| **Feature flag workflows** | `workflows` — NOT in `full` (nuevo modulo pesado) |
| **Prompt optimizer** | DSPy-style con signatures, BootstrapFewShot, busqueda metrica |
| **Feature flag optimizer** | `prompt-optimizer` — NOT in `full` |
| **A2A protocol** | JSON-RPC 2.0 sobre HTTPS, agent cards, task lifecycle |
| **Feature flag A2A** | `a2a` — NOT in `full` (requiere HTTP server) |
| **Memory avanzada** | Sobre persistence.rs existente + knowledge_graph.rs |
| **Feature flag memory** | `advanced-memory` — under `full` (ligero) |
| **Eval online** | Hooks en pipeline de generacion, metricas Prometheus |
| **Streaming guardrails** | Interceptor en streaming pipeline existente |

---

## Fase 1 — Workflow Engine con Checkpointing

### 1.1 EventWorkflow — Motor Event-Driven con Typed Events

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/event_workflow.rs`

Motor de workflows basado en eventos tipados, inspirado en LangGraph 1.0 y
LlamaIndex Workflows. Construye sobre el DAG executor existente pero anade
ejecucion reactiva y estado duradero.

**Tipos clave**:
- `Event<T>`: Evento tipado con payload, timestamp, source_node
- `EventBus`: Canal broadcast para eventos (tokio::broadcast o crossbeam)
- `WorkflowNode`: Nodo que recibe eventos, ejecuta logica, emite nuevos eventos
- `WorkflowGraph`: Grafo de nodos conectados por tipos de evento
- `WorkflowState`: Estado compartido mutable (HashMap<String, serde_json::Value>)
- `WorkflowRunner`: Ejecutor que procesa eventos y gestiona ciclo de vida

**Capacidades**:
- Nodos conectados por tipo de evento (no por aristas estaticas)
- Ejecucion paralela de nodos independientes
- Conditional branching basado en estado
- Human-in-the-loop: nodos que pausan esperando input
- Timeout por nodo y por workflow
- Error propagation con retry configurable por nodo

**Patron de uso**:
```rust
let mut wf = WorkflowGraph::new("my_pipeline");
wf.add_node("retrieve", retrieve_handler);
wf.add_node("rerank", rerank_handler);
wf.add_node("generate", generate_handler);
wf.connect::<RetrievalEvent>("retrieve", "rerank");
wf.connect::<RerankEvent>("rerank", "generate");
let result = wf.run(StartEvent::new(query)).await?;
```

---

### 1.2 WorkflowCheckpointer — Estado Duradero + Time-Travel

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En `src/event_workflow.rs`:

- `Checkpointer` trait: `save(workflow_id, step, state)` / `load(workflow_id, step)`
- `SqliteCheckpointer`: Persiste estado en SQLite (usa rusqlite existente)
- `InMemoryCheckpointer`: Para tests y uso ligero
- `CheckpointEntry`: workflow_id, step_number, node_name, state_snapshot, timestamp
- **Time-travel**: `replay_from(workflow_id, step_number)` — rebobina y re-ejecuta
- **Branching**: `fork_from(workflow_id, step_number)` — crea rama alternativa
- Auto-checkpoint configurable: cada N pasos, cada nodo, o manual

---

### 1.3 ErrorSnapshot — Captura de Estado Parcial en Fallos

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/event_workflow.rs`:

- `ErrorSnapshot`: Captura outputs de todos los nodos hasta el punto de fallo
- Incluye: ultimo evento procesado, estado del workflow, stack trace, nodo fallido
- Serializable para debugging post-mortem
- `WorkflowRunner::on_error(|snapshot| { ... })` callback

---

### 1.4 WorkflowNode Breakpoints

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

En `src/event_workflow.rs`:

- `WorkflowRunner::set_breakpoint(node_name)` — pausa ejecucion antes del nodo
- `WorkflowRunner::resume()` — continua tras breakpoint
- `WorkflowRunner::inspect_state()` — examina estado en breakpoint
- Util para debugging de pipelines complejos

---

## Fase 2 — Prompt Optimization (DSPy-style)

### 2.1 PromptSignature — Declaracion de Input/Output

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/prompt_optimizer.rs`

Sistema inspirado en DSPy donde se declaran signatures (que quieres) y el
sistema compila automaticamente los prompts optimos.

**Tipos clave**:
- `Signature`: Declaracion de campos input + output con descripciones
- `Field`: Nombre, tipo, descripcion, constraints (required, max_length, etc.)
- `CompiledPrompt`: Prompt optimizado con instrucciones + demos seleccionadas
- `Optimizer` trait: `optimize(signature, dataset, metric) -> CompiledPrompt`

**Patron de uso**:
```rust
let sig = Signature::new("question_answering")
    .input("context", "Relevant passages from documents")
    .input("question", "User question to answer")
    .output("answer", "Concise factual answer")
    .output("confidence", "Confidence score 0.0-1.0");

let compiled = BootstrapFewShot::new()
    .with_metric(exact_match)
    .with_max_demos(4)
    .optimize(&sig, &training_data, &provider)?;
```

---

### 2.2 BootstrapFewShot — Generacion Automatica de Demos

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En `src/prompt_optimizer.rs`:

- Ejecuta la signature sobre N ejemplos del dataset de entrenamiento
- Filtra por metrica (solo demos que pasan el umbral)
- Selecciona las K mejores demos por diversidad + calidad
- Genera prompt final con instrucciones + demos seleccionadas
- `BootstrapFewShot { max_demos, metric_threshold, teacher_provider }`

---

### 2.3 MetricSearch — Busqueda de Prompts Optimos

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/prompt_optimizer.rs`:

- `GridSearchOptimizer`: Prueba combinaciones de instrucciones + demos
- `RandomSearchOptimizer`: Muestreo aleatorio del espacio de prompts
- `BayesianOptimizer`: Optimizacion bayesiana (surrogate model simple basado en GP)
- Metricas soportadas: exact_match, f1_score, semantic_similarity, llm_judge
- Budget control: max_evaluations, max_cost, timeout
- Resultado: ranking de prompts con scores por metrica

---

### 2.4 SelfReflect — Auto-mejora Reflexiva

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/prompt_optimizer.rs`:

Inspirado en GEPA/SIMBA de DSPy: el LLM analiza sus propias ejecuciones
fallidas y genera reglas de mejora.

- `SelfReflector`: Recibe trazas de ejecucion (input, output, expected, metric_score)
- Agrupa fallos por patron (tematico, structural, etc.)
- Pide al LLM que identifique que salio mal y proponga mejoras
- Genera `ImprovementRule`: condicion + accion (ej: "cuando la pregunta es numerica, incluir paso de calculo")
- Las reglas se inyectan como instrucciones adicionales en el prompt compilado
- Iterativo: optimize → evaluate → reflect → re-optimize

---

## Fase 3 — Protocolo Agent-to-Agent (A2A)

### 3.1 A2A Core — Agent Cards + Task Lifecycle

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/a2a_protocol.rs`

Implementacion del protocolo A2A de Google (JSON-RPC 2.0 sobre HTTPS) para
interoperabilidad entre agentes de distintos frameworks.

**Tipos clave**:
- `AgentCard`: Descripcion publica del agente (name, description, skills, endpoint, auth)
- `A2ATask`: Tarea con lifecycle (submitted, working, input-required, completed, failed, canceled)
- `A2AMessage`: Mensaje con parts (TextPart, FilePart, DataPart)
- `TaskStatus`: Estado actual + historial de transiciones
- `A2AServer`: Endpoint HTTP que recibe y despacha tareas A2A
- `A2AClient`: Cliente para enviar tareas a agentes remotos

**Metodos JSON-RPC**:
- `tasks/send` — Enviar tarea a un agente
- `tasks/get` — Consultar estado de tarea
- `tasks/cancel` — Cancelar tarea en curso
- `tasks/sendSubscribe` — Streaming de actualizaciones via SSE
- `agent/authenticatedExtendedCard` — Obtener capabilities autenticadas

**Integracion**: Los agentes existentes (MultiAgentSession, AutonomousAgent)
se exponen automaticamente como agentes A2A via AgentCard.

---

### 3.2 A2A Discovery + Push Notifications

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/a2a_protocol.rs`:

- `AgentDirectory`: Registro local de agentes A2A conocidos
- Discovery via well-known URL (`/.well-known/agent.json`)
- Push notifications via webhook callbacks (task status changes)
- Integracion con distributed_network.rs para discovery en red P2P

---

## Fase 4 — Sistema de Memoria Avanzado

### 4.1 EpisodicMemory — Memoria de Eventos

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

Nuevo archivo: `src/advanced_memory.rs`

Memoria episodica: almacena experiencias concretas (que paso, cuando, resultado)
para que el agente aprenda de situaciones pasadas.

- `Episode`: contexto, accion, resultado, outcome (success/failure), timestamp
- `EpisodicStore`: almacen indexado por similitud semantica + temporal
- `recall(query, k)`: recupera episodios similares al contexto actual
- `consolidate()`: episodios frecuentes se abstraen en reglas (→ procedural)
- Persistencia via SQLite (persistence.rs existente)
- Window decay: episodios antiguos pierden peso gradualmente

---

### 4.2 ProceduralMemory — Memoria de Procedimientos

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/advanced_memory.rs`:

Memoria procedural: almacena HOW-TO knowledge (reglas, patrones, procedimientos
aprendidos).

- `Procedure`: condicion (when), pasos (steps), confianza (confidence), uso_count
- `ProceduralStore`: indexado por condicion de activacion
- `match_procedures(context)`: encuentra procedimientos aplicables
- Las reglas se generan por consolidacion de episodios o por SelfReflector (Fase 2.4)
- Formato compatible con guardrails (se pueden inyectar como instrucciones)

---

### 4.3 EntityMemory — Entidades Persistentes

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/advanced_memory.rs`:

Evoluciona el entity extraction existente (knowledge_graph.rs) en un store
persistente dedicado.

- `EntityRecord`: nombre, tipo, atributos, relaciones, ultima_mencion, frecuencia
- `EntityStore`: CRUD sobre entidades, indexado por nombre y tipo
- Auto-update: cuando el agente menciona una entidad, se actualiza automaticamente
- Merge: detecta entidades duplicadas y las fusiona
- Integracion con knowledge_graph.rs existente (lectura bidireccional)
- Persistencia SQLite

---

### 4.4 MemoryConsolidation — Episodica a Semantica

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio-Alto | **Estado**: HECHO

En `src/advanced_memory.rs`:

Proceso de consolidacion inspirado en A-MEM:

- `consolidate_episodes(threshold)`: agrupa episodios similares
- Genera abstracciones: "cuando X ocurre, generalmente Y funciona"
- Las abstracciones se guardan como ProceduralMemory
- Los episodios originales se marcan como consolidados (no se borran)
- Ejecutable manualmente o via scheduler (cron)

---

## Fase 5 — Evaluacion Online + Streaming Guardrails

### 5.1 OnlineEvaluator — Feedback en Produccion

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

Nuevo archivo: `src/online_eval.rs`

Evaluacion en tiempo real sobre respuestas en produccion (no solo offline).

- `FeedbackHook` trait: `evaluate(request, response) -> FeedbackScore`
- Hooks pre-registrados: latencia, costo, relevancia (embedding), toxicidad
- `OnlineEvaluator`: pipeline de hooks que se ejecuta post-respuesta
- `FeedbackScore`: dimension (str), score (f64), metadata
- Metricas agregadas exportables a Prometheus (prometheus.rs existente)
- Almacenamiento: feedback scores guardados junto a la conversacion
- Alertas: umbrales configurables, callback on_alert
- Sampling: evaluar N% de las respuestas (no todas, para reducir coste)

---

### 5.2 StreamingGuardrails — Safety en Tiempo Real

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/guardrail_pipeline.rs` (extension del modulo existente):

Aplica guardrails durante streaming, no solo sobre respuestas completas.

- `StreamingGuard` trait: `check_chunk(chunk, accumulated) -> GuardAction`
- GuardAction: Pass, Flag, Pause (acumula mas antes de decidir), Block (corta stream)
- Guards adaptados: PII detection incremental, toxicity sliding window, pattern matching
- Buffer configurable: acumula N tokens antes de evaluar (balanceo latencia/precision)
- Integracion con SSE y WebSocket streaming existentes
- Metricas: chunks evaluados, flags, blocks, latencia anadida

---

### 5.3 AgentFingerprint — Tracking de Ejecuciones

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

En `src/online_eval.rs`:

Identificacion unica de cada ejecucion de agente para trazabilidad.

- `ExecutionFingerprint`: hash(agent_id + task + timestamp + random)
- Se propaga por toda la cadena de ejecucion (workflow, tools, LLM calls)
- Permite correlacionar logs, metricas y feedback con una ejecucion especifica
- Compatible con OpenTelemetry trace_id existente

---

## Fase 6 — Quality of Life

### 6.1 ProviderRegistry Global

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

En `src/providers.rs` (extension):

Registro global de providers con sintaxis `"provider/model"`.

- `ProviderRegistry::resolve("openai/gpt-4o")` → AiConfig configurado
- `ProviderRegistry::resolve("ollama/llama3")` → AiConfig para Ollama
- `ProviderRegistry::list_models()` → todos los modelos disponibles
- Auto-discovery: detecta providers activos (Ollama, LM Studio via butler)
- Aliases: `"gpt-4o"` → `"openai/gpt-4o"`, `"claude"` → `"anthropic/claude-sonnet"`

---

### 6.2 Pipeline Serialization

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio | **Estado**: HECHO

En `src/event_workflow.rs` (extension):

Serializar/deserializar definiciones de workflow a JSON/YAML.

- `WorkflowGraph::to_json()` / `WorkflowGraph::from_json()`
- Formato portable: nodos, conexiones, configuracion
- Versionado de schemas para backwards compatibility
- Import/export para compartir pipelines entre instancias

---

### 6.3 PipelineAsTool — Exponer Pipelines como Tools

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

En `src/event_workflow.rs` (extension):

Permite usar un workflow completo como tool invocable por un agente.

- `WorkflowTool::from_workflow(graph, name, description)` → ToolDefinition
- El agente puede invocar pipelines complejos como si fueran herramientas
- Composicion recursiva: un workflow puede contener tools que son otros workflows

---

## Fase 7 — Grafos Multinivel: Mejoras al Sistema de 4 Capas

### Alimentacion de Capas

Cada capa tiene su fuente de datos distinta:

| Capa | Como se alimenta | Ejemplo |
|------|------------------|---------|
| **Knowledge** | Manual — el desarrollador carga knowledge packs (.kpkg), indexa documentos en RAG, cura bases de datos | Documentacion tecnica, manuales, datos verificados |
| **User** | Automatico — `BeliefExtractor` analiza mensajes del usuario y extrae preferencias, opiniones, hechos | "Prefiero Python" → UserBelief(Preference, "Python") |
| **Internet** | Automatico — cuando el agente busca en internet (web_search), los resultados se cachean con TTL | Busqueda web → datos con fecha de caducidad |
| **Session** | Automatico — entidades y relaciones mencionadas en la conversacion actual, se borran al cerrar sesion | "Estoy trabajando en el proyecto X" → entidad temporal |
| **Custom** (7.2) | Configurable — el desarrollador define la fuente: manual, API, feed, o extraccion automatica | Capa "Regulatory" alimentada por feed de normativas |

---

### 7.1 Prioridades Configurables por Capa

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Actualmente las prioridades estan hardcodeadas en `GraphLayer::priority()`:
Knowledge=100, User=80, Internet=50, Session=30. Deben ser configurables.

- `LayerConfig`: priority (u8), enabled (bool), weight_in_merge (f32), conflict_policy
- `MultiLayerGraphConfig`: HashMap<GraphLayer, LayerConfig> con defaults
- `MultiLayerGraph::with_config(config)` — constructor con prioridades custom
- `MultiLayerGraph::set_layer_priority(layer, priority)` — cambio en runtime
- `MultiLayerGraph::set_layer_enabled(layer, bool)` — activar/desactivar capas
- Backward compatible: `MultiLayerGraph::new()` mantiene defaults actuales

**Casos de uso por dominio**:
- App medica: Knowledge=100, User=30 (la evidencia medica manda sobre preferencias)
- Asistente personal: User=100, Knowledge=50 (las preferencias del usuario son lo primero)
- Herramienta de investigacion: Internet=90, Knowledge=80 (datos recientes importan mas)
- Compliance: Custom("Regulatory")=110, Knowledge=100 (normativa por encima de todo)

---

### 7.2 Capas Custom (Extensible)

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Permitir capas adicionales definidas por el usuario, mas alla de las 4 fijas.

- `GraphLayer` pasa de enum a trait object o enum extensible con variante `Custom(String)`
- `CustomLayer`: nombre, prioridad, persistencia (en memoria vs SQLite), TTL opcional
- `MultiLayerGraph::add_layer(name, config)` — registrar capa custom
- Ejemplos: capa "Regulatory" para normativas, capa "Team" para conocimiento de equipo
- Las capas custom participan en query merging, contradicciones y build_context

---

### 7.3 MultiLayerGraph Unificado — Vista Consolidada

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Actualmente cada capa (SessionGraph, UserGraph, InternetGraph, KnowledgeGraph)
tiene su propia estructura interna. Falta una vista unificada que combine
todas las entidades y relaciones en un solo grafo navegable.

- `UnifiedView`: grafo virtual que fusiona entidades de todas las capas
- `UnifiedEntity`: entidad con datos de todas las capas donde aparece,
  prioridad resuelta, conflictos marcados
- `UnifiedRelation`: relacion con fuente(s) y confianza agregada
- `query_unified(entity_name)` — busca en todas las capas, resuelve prioridad
- `traverse_unified(start, depth)` — traversal multi-hop sobre la vista unificada
- No duplica datos: es una vista de lectura sobre las capas existentes
- Cache invalidation: la vista se regenera cuando una capa cambia

---

### 7.4 Subgrafos Jerarquicos (Zoom In/Out)

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Nodos de un nivel pueden contener subgrafos completos, permitiendo
zoom in (detalle) y zoom out (resumen) del conocimiento.

- `GraphCluster`: grupo de entidades relacionadas con un nodo resumen
- `cluster_entities(similarity_threshold)` — agrupa entidades por proximidad semantica
- `expand_cluster(cluster_id)` — muestra las entidades internas
- `collapse_cluster(cluster_id)` — muestra solo el nodo resumen
- Jerarquia de N niveles: clusters de clusters
- Integrable con RAPTOR (summaries multinivel ya existentes)
- Export: DOT/Mermaid con clusters colapsables

---

### 7.5 Cross-Layer Inference

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio-Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Inferir relaciones nuevas cruzando informacion entre capas.

**Mecanica paso a paso**:
1. Seleccionar entidad — el agente pregunta sobre "proyecto actual"
2. Recoger datos de cada capa:
   - Session: "usuario trabaja en proyecto X"
   - User: "usuario prefiere Python, usa VS Code"
   - Knowledge: "Python es bueno para data analysis"
   - Internet: "proyecto X es de analisis de datos" (si se busco)
3. Buscar relaciones implicitas cruzando atributos entre capas:
   - Session(proyecto X) + User(prefiere Python) → "proyecto X probablemente usa Python"
   - User(prefiere Python) + Knowledge(Python para data) → "proyecto X es de data analysis"
4. Asignar confianza — inferencias tienen confianza < sus fuentes (0.3-0.5)
5. Validar — preguntar al usuario: "¿proyecto X usa Python?"
   - Confirma → se promueve a User layer con confianza alta
   - Niega → se descarta

**Tipos**:
- `infer_cross_layer(entity)` — busca relaciones implicitas entre capas
- `InferredRelation`: from, to, relation_type, confidence, evidence (capas fuente)
- Las inferencias se guardan en una capa virtual "Inferred" con confianza < source layers
- Configurable: activar/desactivar, umbral de confianza minimo
- Validacion: las inferencias se pueden confirmar (→ se mueven a Knowledge o User)
- Decay: inferencias no confirmadas pierden confianza con el tiempo

---

### 7.6 Intra-Layer Conflict Policies

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Las prioridades inter-capa (Knowledge > User > Internet > Session) ya existen.
Falta resolucion de conflictos DENTRO de una misma capa.

- `ConflictPolicy` enum:
  - `MostRecent` — gana el dato mas reciente (timestamp). Para datos que cambian rapido.
  - `HighestConfidence` — gana el dato con mayor score de confianza. Para fuentes curadas.
  - `SourceAuthority` — algunas fuentes pesan mas (ej: Wikipedia > blog random). Configurable.
  - `ExplicitOverInferred` — declaraciones explicitas superan a las inferidas. Para User layer.
  - `GenerateContradiction` — no resuelve, genera Contradiction para revision manual. Para safety-critical.
- `LayerConfig` incluye `conflict_policy: ConflictPolicy` (default por capa):
  - Knowledge: HighestConfidence
  - User: ExplicitOverInferred
  - Internet: MostRecent
  - Session: MostRecent
- Chaining: se pueden combinar (intentar HighestConfidence, fallback a MostRecent)
- `resolve_intra_layer(entity, attribute)` → valor ganador + justificacion

---

### 7.7 Graph Diff + Merge (Multi-instancia)

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Para escenarios distribuidos: calcular diferencias entre grafos y fusionarlos.

- `GraphDiff`: entidades/relaciones anadidas, modificadas, eliminadas
- `diff(graph_a, graph_b)` → GraphDiff
- `merge(graph_a, graph_b, strategy)` → grafo fusionado
- `MergeStrategy`: PriorityWins, MostRecent, HighestConfidence, Manual
- Compatible con CRDTs existentes (distributed.rs) para sync en red P2P
- Conflict resolution: genera Contradictions para conflictos no resolubles

---

## Resumen de Prioridades

| # | Mejora | Fase | Prioridad | Esfuerzo | Impacto | Estado |
|---|--------|------|-----------|----------|---------|--------|
| 1.1 | EventWorkflow engine | 1 | CRITICA | L | Muy Alto | HECHO |
| 1.2 | WorkflowCheckpointer (time-travel) | 1 | CRITICA | M | Muy Alto | HECHO |
| 1.3 | ErrorSnapshot | 1 | ALTA | S | Alto | HECHO |
| 1.4 | Workflow breakpoints | 1 | MEDIA | S | Medio | HECHO |
| 2.1 | PromptSignature | 2 | CRITICA | M | Muy Alto | HECHO |
| 2.2 | BootstrapFewShot | 2 | ALTA | M | Muy Alto | HECHO |
| 2.3 | MetricSearch (Bayesian) | 2 | ALTA | M | Alto | HECHO |
| 2.4 | SelfReflect (auto-mejora) | 2 | MEDIA-ALTA | M | Alto | HECHO |
| 3.1 | A2A core (agent cards + tasks) | 3 | ALTA | L | Muy Alto | HECHO |
| 3.2 | A2A discovery + push | 3 | MEDIA | S | Alto | HECHO |
| 4.1 | EpisodicMemory | 4 | ALTA | M | Muy Alto | HECHO |
| 4.2 | ProceduralMemory | 4 | ALTA | M | Alto | HECHO |
| 4.3 | EntityMemory persistente | 4 | MEDIA-ALTA | M | Alto | HECHO |
| 4.4 | MemoryConsolidation | 4 | MEDIA | S | Medio-Alto | HECHO |
| 5.1 | OnlineEvaluator | 5 | ALTA | M | Alto | HECHO |
| 5.2 | StreamingGuardrails | 5 | ALTA | M | Alto | HECHO |
| 5.3 | AgentFingerprint | 5 | MEDIA | S | Medio | HECHO |
| 6.1 | ProviderRegistry global | 6 | MEDIA | S | Medio | HECHO |
| 6.2 | Pipeline serialization | 6 | MEDIA | M | Medio | HECHO |
| 6.3 | PipelineAsTool | 6 | MEDIA | S | Medio | HECHO |
| 7.1 | Prioridades configurables | 7 | ALTA | S | Alto | HECHO |
| 7.2 | Capas custom (extensible) | 7 | ALTA | M | Alto | HECHO |
| 7.3 | Vista unificada MultiLayerGraph | 7 | ALTA | M | Muy Alto | HECHO |
| 7.4 | Subgrafos jerarquicos (zoom) | 7 | MEDIA-ALTA | M | Alto | HECHO |
| 7.5 | Cross-layer inference | 7 | MEDIA | M | Medio-Alto | HECHO |
| 7.6 | Intra-layer conflict policies | 7 | ALTA | S | Alto | HECHO |
| 7.7 | Graph diff + merge | 7 | MEDIA | M | Medio | HECHO |

---

## Fase 8 — Integraciones Pendientes (Gaps descubiertos)

### 8.1 KPKG → Knowledge Layer Bridge

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/assistant.rs` y `src/multi_layer_graph.rs`:

Actualmente los `.kpkg` se cargan SOLO en el RAG vector database. La capa
Knowledge del MultiLayerGraph existe pero nadie la alimenta. Crear el puente.

- Al cargar un kpkg via `index_kpkg()`, extraer entidades del contenido
- Insertar entidades en MultiLayerGraph con `GraphLayer::Knowledge` y `ConfidenceLevel::Verified`
- Inyectar system_prompt/persona/examples del manifest en el contexto de sesion
- `AiAssistant::load_kpkg_to_graph(path)` — metodo de conveniencia que hace ambas cosas

---

### 8.2 MCP Session Resources — Continuidad de Sesion

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En `src/mcp_protocol.rs` y `src/session.rs`:

Exponer sesiones guardadas via MCP para que una IA pueda leer datos de otra
sesion. Caso de uso principal: sesion se revienta, nueva sesion puede continuar
cogiendo lo importante de la anterior.

**Resources MCP**:
- `session://list` — lista sesiones guardadas con timestamps y nombres
- `session://{id}/messages` — mensajes completos de una sesion
- `session://{id}/summary` — resumen automatico de la conversacion
- `session://{id}/context` — entidades, relaciones y contexto extraidos
- `session://{id}/beliefs` — creencias del usuario de esa sesion

**Tools MCP**:
- `resume_session(id)` — carga sesion anterior y aplica su contexto a la actual
- `get_session_highlights(id)` — extrae lo mas importante (decisiones, conclusiones)

**Auto-recovery**: Al iniciar nueva sesion, si hay una sesion reciente que
termino abruptamente (sin close explicito), ofrecer automaticamente continuar.

**Reparacion de sesion**:
- `repair_session(id)` — intenta reconstruir una sesion corrupta
- Parsea lo que se pueda del fichero (JSON/bincode parcial)
- Recupera mensajes legibles, descarta los corruptos
- Reconstruye timestamps y metadata desde los mensajes recuperados
- Devuelve sesion parcial + informe de que se perdio

---

### 8.3 Context Overflow Prevention — Checkpoint Proactivo

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/context_window.rs` y `src/session.rs`:

Actualmente ContextWindow trunca y ConversationCompactor resume, pero no hay
deteccion proactiva de "voy a desbordar" para tomar acciones preventivas.

- `ContextOverflowDetector`: monitoriza uso de tokens en cada turno
- **Auto max_tokens**: sacar de `ModelCapabilityInfo.context_window` del modelo activo
  (actualmente hay que pasar max_tokens manualmente al ContextWindow)
- Umbrales configurables: warning (70%), critical (85%), emergency (95%)
- Al llegar a warning: compactar conversacion automaticamente
- Al llegar a critical: checkpoint sesion + extraer resumen ejecutivo
- Al llegar a emergency: crear nueva sesion automaticamente con el resumen
  del contexto anterior como system message, linkear con sesion previa
- `on_overflow(callback)` — hook para que el desarrollador implemente su logica
- Integracion con MCP session resources (8.2): la nueva sesion puede leer la anterior

---

### 8.4 ContextComposer — Composicion Unificada de Contexto

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto | **Estado**: HECHO

En nuevo `src/context_composer.rs` o extension de `src/context_window.rs`:

Actualmente cada sistema (graph, RAG, memory, conversacion) construye su
contexto por separado y no hay ensamblaje automatico. Falta un compositor
que respete el budget de tokens del modelo.

**Orden de composicion** (prioridad de token budget):
1. System prompt (siempre, reservado)
2. Graph context por prioridad de capas (si multi_layer_graph habilitado)
3. RAG chunks relevantes (si rag habilitado y hay query)
4. Memory context — working + long-term (si memory habilitado)
5. Mensajes recientes de conversacion (sliding window)
6. Ultimo prompt del usuario (siempre, reservado)

**Token budgeting**:
- Total = model.context_window - response_reserve
- Cada seccion tiene un % configurable (ej: graph 15%, RAG 25%, memory 10%, conv 45%)
- Si una seccion usa menos, el sobrante se redistribuye
- Seccion 1 y 6 son fijas (no se recortan)

**Integracion con graph (compactacion hibrida)**:

Cuando el graph de sesion esta activo, la compactacion cambia de rol:
- **Sin graph**: compactacion genera resumen largo (~500 tokens) intentando
  capturar todo — ineficiente, pierde estructura
- **Con graph**: el graph captura el QUE (entidades, decisiones, hechos) de
  forma estructurada. La compactacion se reduce a un mini-resumen del COMO
  (~50-100 tokens: flujo conversacional, tono, hilos abiertos)

Flujo al compactar con graph activo:
1. Extraer entidades de mensajes a compactar → Session graph
2. Entidades recurrentes (>3 menciones) → promover a User graph como beliefs
3. Decisiones confirmadas → User graph como Goal/Fact
4. Datos factuales verificados → Knowledge graph (si procede)
5. Generar mini-resumen SOLO del flujo conversacional (no hechos)
6. Reemplazar mensajes viejos por mini-resumen

Resultado: el graph "recuerda" lo factual, la compactacion solo mantiene
el hilo narrativo. Mucho menos tokens, mejor recall.

---

### 8.5 Capa de Sincronizacion P2P por Layer

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio-Alto | **Estado**: HECHO

En `src/multi_layer_graph.rs`:

Configurar que capas del grafo se sincronizan entre nodos P2P y cuales son privadas.

- `SyncPolicy` per layer: Shared, Private, ReadOnly
  - `Shared`: sync bidireccional entre nodos (Knowledge, Internet)
  - `Private`: nunca se sincroniza (User — datos personales)
  - `ReadOnly`: el nodo puede leer de otros pero no publica la suya (Session)
- Default (se aplican automaticamente al activar multi-layer + distributed):
  - Knowledge=Shared (datos curados, universales)
  - Internet=Shared (evita busquedas duplicadas entre nodos)
  - User=Private (datos personales, nunca salen del nodo)
  - Session=Private (efimero, local a la conversacion)
- Custom layers: Private por defecto, configurable al crear la capa

---

## Fase 9 — P2P/Distribuido Hardening

### 9.1 Read Repair

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/distributed_network.rs`:

Cuando se lee un dato y se detecta que una replica tiene version antigua,
repararla automaticamente.

- Al hacer `handle_get()` con quorum > 1, comparar versiones de las replicas
- Si alguna replica tiene version menor, enviar `Replicate` con la version correcta
- Reduce tiempo de convergencia (actualmente depende del ciclo de 30s)

---

### 9.2 Hinted Handoff

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto | **Estado**: HECHO

En `src/distributed_network.rs`:

Cola de reintentos para replicaciones fallidas.

- `HintedHandoffQueue`: HashMap<NodeId, Vec<PendingWrite>>
- Cuando un nodo de replica esta caido, guardar el write en la cola
- Cuando el nodo vuelve (PeerConnected event), vaciar la cola
- Maximo size configurable para evitar memoria infinita
- TTL por hint: descartar hints muy antiguos

---

### 9.3 Dead Node Ring Removal

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/distributed_network.rs`:

Automaticamente sacar nodos muertos del hash ring tras un periodo configurable.

- `dead_node_removal_timeout`: Duration (default: 5 minutos)
- Tras timeout, `ring.remove_node(dead_node_id)`
- Re-replication inmediata de keys afectados a nodos vivos
- Si el nodo vuelve antes del timeout, cancelar removal
- Si vuelve despues, se re-añade como nodo nuevo (probation)

---

### 9.4 Explicit Version Conflict Resolution

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

En `src/distributed_network.rs`:

Actualmente LWW implicito. Hacer explicito y configurable.

- `ConflictResolution` enum: LastWriteWins, HighestVersion, Merge(fn), Manual
- Aplicar en anti-entropy sync cuando dos nodos tienen mismo key con versiones distintas
- Log de conflictos resueltos para auditoria
- Default: LastWriteWins (comportamiento actual, pero ahora explicito)

---

### 9.5 Quorum Enforcement

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio-Alto | **Estado**: HECHO

En `src/distributed_network.rs`:

Actualmente `handle_get()` devuelve resultados parciales si el quorum no se cumple.
Hacer configurable: estricto (error si no hay quorum) o flexible (actual).

- `QuorumMode` enum:
  - `Strict`: error si < read_quorum → no devuelve nada
  - `Flexible`: devuelve lo que haya, sin indicar fiabilidad (comportamiento actual)
  - `Mixed`: devuelve lo que haya + marca de confianza segun nodos que respondieron
    - Si quorum=2 y responde 1 → dato con `QuorumConfidence::Low`
    - Si quorum=2 y responden 2 → dato con `QuorumConfidence::High`
    - El consumidor decide que hacer con datos de baja confianza
- Default: Flexible (backward compatible)
- Mixed: recomendado para la mayoria de aplicaciones (datos siempre disponibles, con contexto de fiabilidad)
- Strict: solo para aplicaciones que necesitan consistencia fuerte

---

## Resumen de Prioridades

| # | Mejora | Fase | Prioridad | Esfuerzo | Impacto | Estado |
|---|--------|------|-----------|----------|---------|--------|
| 1.1 | EventWorkflow engine | 1 | CRITICA | L | Muy Alto | HECHO |
| 1.2 | WorkflowCheckpointer (time-travel) | 1 | CRITICA | M | Muy Alto | HECHO |
| 1.3 | ErrorSnapshot | 1 | ALTA | S | Alto | HECHO |
| 1.4 | Workflow breakpoints | 1 | MEDIA | S | Medio | HECHO |
| 2.1 | PromptSignature | 2 | CRITICA | M | Muy Alto | HECHO |
| 2.2 | BootstrapFewShot | 2 | ALTA | M | Muy Alto | HECHO |
| 2.3 | MetricSearch (Bayesian) | 2 | ALTA | M | Alto | HECHO |
| 2.4 | SelfReflect (auto-mejora) | 2 | MEDIA-ALTA | M | Alto | HECHO |
| 3.1 | A2A core (agent cards + tasks) | 3 | ALTA | L | Muy Alto | HECHO |
| 3.2 | A2A discovery + push | 3 | MEDIA | S | Alto | HECHO |
| 4.1 | EpisodicMemory | 4 | ALTA | M | Muy Alto | HECHO |
| 4.2 | ProceduralMemory | 4 | ALTA | M | Alto | HECHO |
| 4.3 | EntityMemory persistente | 4 | MEDIA-ALTA | M | Alto | HECHO |
| 4.4 | MemoryConsolidation | 4 | MEDIA | S | Medio-Alto | HECHO |
| 5.1 | OnlineEvaluator | 5 | ALTA | M | Alto | HECHO |
| 5.2 | StreamingGuardrails | 5 | ALTA | M | Alto | HECHO |
| 5.3 | AgentFingerprint | 5 | MEDIA | S | Medio | HECHO |
| 6.1 | ProviderRegistry global | 6 | MEDIA | S | Medio | HECHO |
| 6.2 | Pipeline serialization | 6 | MEDIA | M | Medio | HECHO |
| 6.3 | PipelineAsTool | 6 | MEDIA | S | Medio | HECHO |
| 7.1 | Prioridades configurables | 7 | ALTA | S | Alto | HECHO |
| 7.2 | Capas custom (extensible) | 7 | ALTA | M | Alto | HECHO |
| 7.3 | Vista unificada MultiLayerGraph | 7 | ALTA | M | Muy Alto | HECHO |
| 7.4 | Subgrafos jerarquicos (zoom) | 7 | MEDIA-ALTA | M | Alto | HECHO |
| 7.5 | Cross-layer inference | 7 | MEDIA | M | Medio-Alto | HECHO |
| 7.6 | Intra-layer conflict policies | 7 | ALTA | S | Alto | HECHO |
| 7.7 | Graph diff + merge | 7 | MEDIA | M | Medio | HECHO |
| 8.1 | KPKG → Knowledge layer bridge | 8 | ALTA | S | Alto | HECHO |
| 8.2 | MCP session resources | 8 | ALTA | M | Muy Alto | HECHO |
| 8.3 | Context overflow prevention | 8 | ALTA | M | Alto | HECHO |
| 8.4 | ContextComposer (composicion unificada) | 8 | ALTA | M | Muy Alto | HECHO |
| 8.5 | Sync policy per layer (P2P) | 8 | MEDIA | S | Medio-Alto | HECHO |
| 9.1 | Read repair | 9 | ALTA | M | Alto | HECHO |
| 9.2 | Hinted handoff | 9 | ALTA | M | Alto | HECHO |
| 9.3 | Dead node ring removal | 9 | ALTA | S | Alto | HECHO |
| 9.4 | Version conflict resolution | 9 | MEDIA-ALTA | S | Alto | HECHO |
| 9.5 | Quorum enforcement | 9 | MEDIA | S | Medio-Alto | HECHO |

**Leyenda**: S = Small (1-2 dias), M = Medium (3-5 dias), L = Large (1-2 semanas)

**v4 completo**: 38/38 HECHO — 3401 tests, 0 failures, 1 pre-existing clippy warning

---

## Orden de Ejecucion

```
Fase 1 (workflows):
  1.1 -> 1.2 -> 1.3 -> 1.4
  (engine primero, checkpointing, error snapshots, breakpoints)

Fase 2 (prompt optimizer) — PARALELO con Fase 1:
  2.1 -> 2.2 -> 2.3 -> 2.4
  (signatures, bootstrap, metric search, self-reflect)

Fase 3 (A2A) — DESPUES de Fase 1 (usa workflow engine):
  3.1 -> 3.2
  (core protocol, discovery)

Fase 4 (memory) — PARALELO con Fases 1+2:
  4.1 -> 4.2 -> 4.3 -> 4.4
  (episodic, procedural, entity, consolidation)

Fase 5 (eval + guardrails) — PARALELO con Fases 1+2:
  5.1 -> 5.2 -> 5.3
  (online eval, streaming guardrails, fingerprinting)

Fase 6 (QoL) — DESPUES de Fase 1:
  6.1 -> 6.2 -> 6.3
  (provider registry, serialization, pipeline-as-tool)

Fase 7 (grafos multinivel) — PARALELO con Fases 1+2 (independiente):
  7.1 -> 7.6 -> 7.2 -> 7.3 -> 7.4 -> 7.5 -> 7.7
  (config, intra-policies, capas custom, vista unificada, subgrafos, inference, diff/merge)

Fase 8 (integraciones) — PARALELO con todo (independiente):
  8.1 -> 8.2 -> 8.3 -> 8.4 -> 8.5
  (kpkg bridge, MCP sessions, context overflow, context composer, P2P sync policy)

Fase 9 (P2P hardening) — PARALELO con todo (independiente):
  9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.5
  (read repair, hinted handoff, ring removal, version conflicts, quorum)
```

---

## Dependencias Entre Fases

```
Fase 1 (workflows) <-- requerida por --> Fase 3 (A2A usa workflow engine)
Fase 1 (workflows) <-- requerida por --> Fase 6.2, 6.3 (serialization, pipeline-as-tool)
Fase 2 (optimizer) <-- independiente --> puede hacerse en paralelo con todo
Fase 2.4 (reflect) <-- sinergia con --> Fase 4.2 (procedural memory consume reglas)
Fase 4 (memory)    <-- independiente --> puede hacerse en paralelo
Fase 5 (eval)      <-- independiente --> puede hacerse en paralelo
Fase 6 (QoL)       <-- depende de   --> Fase 1
Fase 7 (grafos)    <-- independiente --> puede hacerse en paralelo
Fase 7.4 (zoom)    <-- sinergia con --> RAPTOR existente (rag_methods.rs)
Fase 7.7 (diff)    <-- sinergia con --> CRDTs existentes (distributed.rs)
Fase 8 (integr.)   <-- independiente --> puede hacerse en paralelo
Fase 8.1 (kpkg)    <-- sinergia con --> Fase 7.1 (layer config)
Fase 8.3 (overflow) <-- sinergia con --> Fase 8.2 (MCP session) y Fase 8.4 (composer)
Fase 8.4 (composer) <-- sinergia con --> multi_layer_graph, RAG, memory existentes
Fase 8.5 (P2P sync)<-- sinergia con --> Fase 7.7 (diff/merge) y Fase 9
Fase 9 (P2P hard.) <-- independiente --> puede hacerse en paralelo
```

---

## Cambios en Cargo.toml (estimados)

```toml
# New features (NOT in `full` — opt-in):
workflows = ["dep:tokio"]
prompt-optimizer = []
a2a = ["dep:tokio"]

# New features (in `full` — lightweight):
advanced-memory = []

# Fase 7 no requiere feature flag nuevo — extiende multi_layer_graph.rs existente
# (ya incluido en `full` via core)

# Fases 8-9 no requieren features nuevos — extienden modulos existentes

# No new external dependencies expected — todo construye sobre:
# - rusqlite (ya presente, para checkpointing)
# - serde/serde_json (ya presente, para serialization)
# - tokio (ya presente, para async workflows)
# - ureq (ya presente, para A2A HTTP client)
```
