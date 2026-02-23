# Plan de Mejoras para ai_assistant — v6

> **Estado: 34/34 items COMPLETE**

> Documento generado el 2026-02-23.
> Basado en completitud de v1 (39/39), v2 (22/22), v3 (21/21), v4 (38/38), v5 (30/30), v6 (34/34) con 4449 tests, 0 failures.
> 240+ source files, ~270k+ LoC.
>
> **Planes anteriores**: v1 (providers, embeddings, MCP, documents, guardrails),
> v2 (async parity, vector DBs, evaluation, testing),
> v3 (containers, document pipeline, speech/audio, CI/CD maturity),
> v4 (workflows, prompt signatures, A2A, advanced memory, online eval, streaming guardrails),
> v5 (GEPA/MIPROv2, MCP v2, voice agents, media gen, distillation, OTel GenAI,
>     durable execution, constrained decoding, memory evolution).

---

## Contexto

Tras v5, ai_assistant es el framework de IA mas completo en Rust con 4449 tests,
30+ feature flags y cobertura total en: LLM providers, RAG 5 niveles, multi-agente,
distribuido P2P/QUIC, seguridad, streaming, MCP, voice, media generation,
distillation, constrained decoding, y mas.

El analisis competitivo de febrero 2026 contra LangGraph 2.x, LlamaIndex 0.12,
DSPy 2.6, CrewAI 0.9, OpenAI Agents SDK, Google ADK, Vercel AI SDK 6, AutoGen 0.4
y Microsoft Semantic Kernel 1.x revela **brechas emergentes** en 10 areas:

1. **MCP incompleto** — La spec MCP (Nov 2025) anade Elicitation (input
   estructurado del usuario), Audio content type, JSON-RPC batching y Completions.
   Nuestro `mcp_protocol.rs` implementa Streamable HTTP y OAuth 2.1 pero no estos
   4 endpoints nuevos.

2. **Sin cliente MCP remoto** — Tenemos servidor MCP pero no un cliente que conecte
   a servidores MCP externos via Streamable HTTP. LangChain, Cursor y Claude Code
   ya soportan "remote MCP" como consumidores.

3. **Human-in-the-Loop basico** — Solo hay un flag `auto_approve` en wasm_hooks.
   LangGraph 2.x tiene approval gates con politicas, escalado por confianza,
   correcciones interactivas y audit trail completo.

4. **Prompt optimization estancado** — DSPy 2.6 introduce SIMBA (evolutionary
   optimizer que supera a GEPA en benchmarks), reasoning trace capture y
   automated LLM-as-Judge grading. Nuestro GEPA/MIPROv2 es v5 pero no SIMBA.

5. **Memoria sin extraccion automatica** — Mem0, Zep y LangMem extraen hechos
   automaticamente de conversaciones. Nuestra advanced_memory requiere llamadas
   explicitas a `store_episode()`. Falta un "Memory OS" con scheduler de fondo.

6. **RAG sin chunking discursivo** — Chunking actual es por tamano/overlap.
   LlamaIndex 0.12 ofrece discourse-aware chunking y diversity-aware retrieval
   (MMR). Nuestro RAG es potente pero el chunking es ingenuo.

7. **Sin evaluacion de agentes** — Tenemos metricas de outputs (relevance,
   coherence) pero no evaluacion de trayectorias de agente: correccion de
   uso de tools, eficiencia de pasos, tool-call accuracy. CLEAR (Google) es
   el framework emergente.

8. **Sin MCTS ni reward models** — LangGraph 2.x y AutoGen 0.4 soportan
   Monte Carlo Tree Search para planificacion multi-paso y Process Reward
   Models para verificacion paso-a-paso. Nuestro task_planning es estatico.

9. **Voz sin WebRTC** — OpenAI Realtime API y Google ADK usan WebRTC para
   voz sub-200ms. Nuestro voice_agent.rs usa HTTP streaming (mayor latencia).

10. **Sandbox solo Docker** — Solo soportamos Docker (bollard). Daytona y
    E2B ofrecen backends pluggables: Podman, wasmtime, gVisor, Firecracker.

---

## Decisiones de Diseno (propuestas)

| Decision | Eleccion |
|---|---|
| **MCP Elicitation + Audio + Batching + Completions** | Extender `mcp_protocol.rs` existente — misma API MCP |
| **Remote MCP Client** | Nuevo `src/mcp_client.rs` — feature `mcp` existente |
| **Agent discovery (AGENTS.md)** | Extender `a2a_protocol.rs` — formato complementario |
| **HITL** | Nuevo `src/hitl.rs` — feature `hitl` (nuevo, NOT in `full`) |
| **SIMBA optimizer** | Extender `prompt_signature.rs` — misma API de `Signature` + `EvalMetric` |
| **Reasoning traces** | Extender `prompt_signature.rs` — ReasoningTraceCapture struct |
| **LLM-as-Judge** | Extender `prompt_signature.rs` — JudgeMetric struct |
| **Memory OS** | Extender `advanced_memory.rs` — MemoryExtractor + MemoryScheduler |
| **Discourse chunking** | Extender `rag.rs` — DiscourseChunker struct |
| **Diversity retrieval** | Extender `rag_methods.rs` — DiversityRetriever struct |
| **Hierarchical RAG router** | Extender `rag_methods.rs` — HierarchicalRouter struct |
| **Agent eval** | Nuevo `src/agent_eval.rs` — feature `eval` existente |
| **Red teaming** | Nuevo `src/red_team.rs` — feature `eval` existente |
| **NL policy guards** | Extender `guardrail_pipeline.rs` — NaturalLanguageGuard struct |
| **MCTS planning** | Nuevo `src/mcts_planner.rs` — feature `autonomous` existente |
| **Process reward models** | Extender `mcts_planner.rs` — ProcessRewardModel struct |
| **WebRTC voice** | Extender `voice_agent.rs` — WebRtcTransport + feature `webrtc` (nuevo) |
| **Video analysis** | Extender `media_generation.rs` — VideoAnalyzer struct |
| **Multi-backend sandbox** | Extender `container_sandbox.rs` — SandboxBackend trait + impls |
| **Agent DevTools** | Nuevo `src/agent_devtools.rs` — feature `devtools` (nuevo, NOT in `full`) |

---

## Fase 1 — MCP Spec Completeness (Nov 2025)

### 1.1 MCP Elicitation Protocol

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/mcp_protocol.rs`.

Implementar el protocolo de Elicitation (MCP spec Nov 2025): permite a un servidor
MCP solicitar input estructurado al usuario durante la ejecucion de una herramienta.
Soporta text, select, boolean, number y file upload.

**Tipos clave**:
- `ElicitRequest`: Tipo de solicitud (schema JSON del formulario esperado)
- `ElicitResponse`: Respuesta del usuario con valores validados
- `ElicitationHandler`: Trait para manejar solicitudes (sync o async)
- `ElicitAction`: Enum (Accept, Deny, Dismiss)

**Tests**: ~12

**Estado**: ✅ HECHO — ElicitRequest, ElicitResponse, ElicitationHandler en mcp_protocol.rs

---

### 1.2 MCP Audio Content Type

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/mcp_protocol.rs`.

Anadir soporte para contenido `audio/*` en mensajes MCP. Permite enviar y recibir
audio (WAV, OGG, MP3) como content parts junto a texto e imagen.

**Tipos clave**:
- `AudioContent`: Struct con data (base64), mime_type, transcript opcional
- Integracion en `McpContent` enum existente como nueva variante `Audio(AudioContent)`

**Tests**: ~8

**Estado**: ✅ HECHO — AudioContent variant added to McpContent enum en mcp_protocol.rs

---

### 1.3 MCP JSON-RPC Batching

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/mcp_protocol.rs`.

Implementar JSON-RPC 2.0 batch requests: enviar array de requests en una sola
llamada HTTP, recibir array de responses. Reduce latencia de red para operaciones
multi-tool.

**Tipos clave**:
- `BatchRequest`: Vec de `JsonRpcRequest` serializado como JSON array
- `BatchResponse`: Vec de `JsonRpcResponse` con correlacion por ID
- `BatchExecutor`: Ejecuta requests en paralelo o secuencial segun config
- `BatchConfig`: max_batch_size, parallel_execution, timeout_per_request

**Tests**: ~10

**Estado**: ✅ HECHO — BatchExecutor, BatchConfig en mcp_protocol.rs

---

### 1.4 MCP Completions & Suggestions

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/mcp_protocol.rs`.

Endpoint `completion/complete` que sugiere valores para argumentos de resource
templates y prompt arguments. Permite autocompletado en IDEs y CLIs que consumen MCP.

**Tipos clave**:
- `CompletionRequest`: ref (uri/name) + argument name + partial value
- `CompletionResult`: Vec de sugerencias con metadata (hasMore, total)
- `CompletionProvider` trait: implementaciones registrables por recurso

**Tests**: ~8

**Estado**: ✅ HECHO — CompletionProvider trait en mcp_protocol.rs

---

## Fase 2 — Remote MCP Client & Agent Discovery

### 2.1 Remote MCP Client

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/mcp_client.rs` (feature `mcp`).

Cliente MCP que conecta a servidores MCP remotos via Streamable HTTP transport.
Permite que nuestros agentes consuman herramientas expuestas por servidores MCP
externos (VS Code extensions, bases de datos, APIs empresariales).

**Tipos clave**:
- `McpClient`: Cliente con transport configurable (Streamable HTTP, SSE fallback)
- `McpClientConfig`: url, auth (OAuth 2.1 / Bearer token), timeout, retries
- `RemoteToolRegistry`: Descubre y cachea herramientas remotas via `tools/list`
- `RemoteResourceBrowser`: Lista y lee recursos remotos via `resources/list`
- `McpClientSession`: Manejo de sesion con reconnect automatico
- `McpClientPool`: Pool de conexiones a multiples servidores MCP

**Tests**: ~25

**Estado**: ✅ HECHO — RemoteMcpClient, McpClientConfig, RemoteToolRegistry, McpClientPool en NEW mcp_client.rs

---

### 2.2 AGENTS.md Convention

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/a2a_protocol.rs`.

Implementar la convencion AGENTS.md para descubrimiento de agentes en repositorios.
Un archivo AGENTS.md en la raiz de un repo describe las capacidades, endpoints y
protocolos soportados por los agentes disponibles.

**Tipos clave**:
- `AgentsMdParser`: Parser de formato AGENTS.md (YAML frontmatter + markdown)
- `AgentsMdEntry`: Nombre, descripcion, protocolos (A2A, MCP, ACP), endpoint
- `AgentsMdDiscovery`: Descubrimiento via GitHub API / filesystem scan

**Tests**: ~10

**Estado**: ✅ HECHO — AgentsMdParser en a2a_protocol.rs

---

### 2.3 Agent Communication Protocol (ACP) Bridge

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/a2a_protocol.rs`.

Bridge entre nuestro protocolo A2A y el ACP (Agent Communication Protocol) de
Google/BeeAI. Permite interoperabilidad bidireccional: agentes ACP pueden invocar
nuestros agentes y viceversa.

**Tipos clave**:
- `AcpMessage`: Formato de mensaje ACP (content parts, metadata, run_id)
- `AcpBridge`: Traduce A2A ↔ ACP manteniendo semantica
- `AcpAgentAdapter`: Adapta nuestro AgentCard a ACP agent descriptor

**Tests**: ~10

**Estado**: ✅ HECHO — AcpBridge en a2a_protocol.rs

---

## Fase 3 — Human-in-the-Loop (HITL)

### 3.1 Tool Approval Gates

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Muy Alto

Nuevo archivo: `src/hitl.rs` (feature `hitl`).

Sistema de aprobacion que pausa la ejecucion de un agente antes de ejecutar
herramientas sensibles, esperando aprobacion humana. Integra con el ciclo
autonomo existente.

**Tipos clave**:
- `ApprovalGate`: Trait con metodo `request_approval(tool_call) -> ApprovalDecision`
- `ApprovalDecision`: Approve, Deny(reason), Modify(modified_args), Timeout
- `ApprovalRequest`: tool_name, args, agent_context, estimated_impact
- `ChannelApprovalGate`: Implementacion via mpsc channels (para CLI/GUI)
- `CallbackApprovalGate`: Implementacion via callback function
- `ApprovalLog`: Registro auditado de todas las decisiones

**Tests**: ~18

**Estado**: ✅ HECHO — HitlApprovalGate trait, ChannelApprovalGate, CallbackApprovalGate en NEW hitl.rs

---

### 3.2 Confidence-Based Escalation

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/hitl.rs`.

Escalado automatico a supervision humana cuando la confianza del agente cae
por debajo de un umbral configurable. Mide confianza por multiples senales:
temperatura del modelo, ambiguedad de la query, historial de errores.

**Tipos clave**:
- `ConfidenceEstimator`: Trait para estimar confianza (0.0-1.0) de una decision
- `EscalationPolicy`: Umbrales por tipo de accion (tool_call, response, planning)
- `EscalationTrigger`: ConfidenceBelow(f64), ConsecutiveErrors(usize), CostAbove(f64)
- `SignalAggregator`: Combina multiples senales de confianza (weighted average)

**Tests**: ~14

**Estado**: ✅ HECHO — ConfidenceEstimator, EscalationPolicy en hitl.rs

---

### 3.3 Interactive Corrections

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/hitl.rs`.

Permite que un humano corrija outputs del agente durante ejecucion multi-paso.
El agente incorpora la correccion y ajusta pasos posteriores.

**Tipos clave**:
- `CorrectionPoint`: Punto en la ejecucion donde el humano puede intervenir
- `Correction`: Tipo (ReplaceOutput, ModifyPlan, AddContext, SkipStep)
- `CorrectionHandler`: Trait para recibir correcciones (async)
- `CorrectionHistory`: Historial de correcciones para aprendizaje

**Tests**: ~12

**Estado**: ✅ HECHO — CorrectionHistory, CorrectionHandler en hitl.rs

---

### 3.4 Declarative Approval Policies

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/hitl.rs`.

Politicas declarativas (TOML/JSON) que definen cuando se requiere aprobacion
humana sin escribir codigo. Basado en patrones: tool name, estimated cost,
data sensitivity, agent confidence.

**Tipos clave**:
- `ApprovalPolicy`: Struct con rules (Vec de PolicyRule)
- `PolicyRule`: condition (ToolMatch, CostAbove, SensitiveData) + action (Approve/Deny/Ask)
- `PolicyEngine`: Evalua rules contra ApprovalRequest, devuelve decision
- `PolicyLoader`: Carga politicas desde TOML/JSON

**Tests**: ~10

**Estado**: ✅ HECHO — PolicyEngine, PolicyLoader en hitl.rs

---

## Fase 4 — Advanced Prompt Optimization v2

### 4.1 SIMBA Optimizer

**Prioridad**: ALTA | **Esfuerzo**: XL | **Impacto**: Muy Alto

Extender `src/prompt_signature.rs`.

SIMBA (Simulated Annealing + Multi-Armed Bandit Adaptation): optimizador
evolutivo que combina mutaciones guiadas por LLM con seleccion por torneo y
cooling schedule. En benchmarks de DSPy 2.6, supera a GEPA en convergencia
(40% menos evaluaciones para mismo resultado).

**Tipos clave**:
- `SimbaOptimizer`: Optimizador principal con config de temperatura, cooling rate
- `SimbaConfig`: population_size, generations, mutation_rate, cooling_schedule
- `CoolingSchedule`: Linear, Exponential, Adaptive (auto-ajusta segun mejora)
- `MutationStrategy`: LlmGuided (pide al LLM variantes), RandomPerturbation, Crossover
- `TournamentSelector`: Seleccion por torneo con elitismo configurable

**Tests**: ~18

**Estado**: ✅ HECHO — SimbaOptimizer en prompt_signature.rs

---

### 4.2 Reasoning Trace Capture

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/prompt_signature.rs`.

Capturar y almacenar cadenas de razonamiento (chain-of-thought) generadas por
modelos. Las trazas se usan para: (a) optimizar prompts extrayendo patrones
exitosos, (b) debugging, (c) destilacion supervisada.

**Tipos clave**:
- `ReasoningTrace`: Vec de pasos con texto, confianza, token_count
- `ReasoningStep`: thought, conclusion, evidence, confidence
- `TraceExtractor`: Parsea output de modelo para extraer CoT (regex + heuristics)
- `TraceStore`: Almacena trazas indexadas por signature + input hash
- `TraceAnalyzer`: Identifica patrones exitosos vs fallidos en trazas

**Tests**: ~14

**Estado**: ✅ HECHO — ReasoningTrace, TraceExtractor en prompt_signature.rs

---

### 4.3 LLM-as-Judge Automated Grading

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/prompt_signature.rs`.

Usar un LLM como evaluador automatico de calidad de outputs. Configurable con
rubrics en lenguaje natural. Integra con el sistema de metricas existente.

**Tipos clave**:
- `JudgeMetric`: Implementa EvalMetric usando un LLM para evaluar
- `JudgeConfig`: model_id, rubric (String), scoring_scale, few_shot_examples
- `JudgeRubric`: Rubrica estructurada con criterios y pesos
- `JudgeResult`: score, reasoning, per_criterion_scores
- `CalibrationSet`: Conjunto de referencia para calibrar al judge

**Tests**: ~12

**Estado**: ✅ HECHO — JudgeMetric, PromptJudgeResult en prompt_signature.rs

---

## Fase 5 — Memory OS

### 5.1 Automatic Memory Extraction

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/advanced_memory.rs`.

Extraccion automatica de hechos, entidades y procedimientos de conversaciones
sin llamadas explicitas. Un `MemoryExtractor` procesa cada turno de conversacion
y decide que almacenar (hechos nuevos, actualizaciones de entidades, nuevos
procedimientos).

**Tipos clave**:
- `MemoryExtractor`: Procesa ConversationTurn → Vec<MemoryExtraction>
- `MemoryExtraction`: Enum (NewFact, EntityUpdate, NewProcedure, Correction)
- `ExtractionConfig`: min_confidence, categories_to_extract, max_extractions_per_turn
- `ExtractionRule`: Regla heuristica/regex para tipos comunes (fechas, nombres, preferencias)
- `LlmExtractionStrategy`: Usa LLM para extraccion semantica profunda

**Tests**: ~16

**Estado**: ✅ HECHO — MemoryExtractor en advanced_memory.rs

---

### 5.2 Memory Scheduler

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/advanced_memory.rs`.

Scheduler de fondo que ejecuta operaciones de mantenimiento de memoria:
consolidacion periodica, decay de memorias antiguas, compresion de episodios
similares, garbage collection de memorias irrelevantes.

**Tipos clave**:
- `MemoryScheduler`: Scheduler con tareas programables
- `SchedulerTask`: Enum (Consolidate, Decay, Compress, GarbageCollect)
- `SchedulerConfig`: intervals por tarea, batch_size, max_duration
- `ConsolidationJob`: Agrupa episodios similares en procedimientos
- `DecayJob`: Reduce relevancia de memorias no accedidas
- `CompressionJob`: Resumir cadenas de memorias relacionadas

**Tests**: ~14

**Estado**: ✅ HECHO — MemoryScheduler en advanced_memory.rs

---

### 5.3 Cross-Session Memory Sharing

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/advanced_memory.rs`.

Compartir memorias entre instancias de agente. Un `MemoryBus` permite publicar
y suscribirse a memorias con filtros por categoria, entidad o relevancia.

**Tipos clave**:
- `MemoryBus`: Canal pub/sub para memorias entre agentes
- `MemoryFilter`: Filtro por categoria, entidad, confianza minima, recency
- `SharedMemoryPool`: Pool de memorias compartidas con control de acceso
- `MemorySyncPolicy`: Eager (sync inmediato), Lazy (bajo demanda), Periodic

**Tests**: ~12

**Estado**: ✅ HECHO — SharedMemoryPool, MemoryBus en advanced_memory.rs

---

### 5.4 Memory Search Optimization

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/advanced_memory.rs`.

Busqueda hibrida optimizada para memorias: combina keyword matching, embedding
similarity y recency scoring con pesos configurables.

**Tipos clave**:
- `MemorySearchEngine`: Motor de busqueda con ranking hibrido
- `SearchWeights`: keyword_weight, embedding_weight, recency_weight, access_freq_weight
- `MemoryIndex`: Indice invertido para busqueda rapida por keyword
- `SearchResult`: memory, relevance_score, match_reasons

**Tests**: ~10

**Estado**: ✅ HECHO — MemorySearchEngine, MemorySearchResult en advanced_memory.rs

---

## Fase 6 — Advanced RAG v2

### 6.1 Discourse-Aware Chunking

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/rag.rs`.

Chunking que respeta la estructura discursiva del documento: detecta limites
de seccion, parrafo, lista y argumento, evitando cortar a mitad de una idea.
Usa heuristicas de coherencia textual (cosine similarity entre oraciones
consecutivas) para encontrar puntos de corte naturales.

**Tipos clave**:
- `DiscourseChunker`: Chunker con deteccion de limites semanticos
- `DiscourseConfig`: min_chunk_size, max_chunk_size, coherence_threshold
- `BoundaryDetector`: Identifica limites naturales (headings, paragraphs, lists)
- `CoherenceScorer`: Calcula coherencia entre oraciones consecutivas (TF-IDF cosine)
- `ChunkMetadata`: section_hierarchy, discourse_type, coherence_score

**Tests**: ~14

**Estado**: ✅ HECHO — DiscourseChunker, BoundaryDetector, CoherenceScorer en rag.rs

---

### 6.2 Diversity-Aware Retrieval (MMR)

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/rag_methods.rs`.

Maximal Marginal Relevance (MMR): selecciona documentos que son relevantes a la
query pero diversos entre si, evitando redundancia en el contexto.

**Tipos clave**:
- `DiversityRetriever`: Wrapper sobre cualquier retriever que aplica MMR
- `MmrConfig`: lambda (relevance vs diversity tradeoff), top_k, diversity_metric
- `DiversityMetric`: Cosine, Jaccard, SemanticDistance
- `MmrScorer`: Calcula score MMR iterativamente (greedy selection)

**Tests**: ~12

**Estado**: ✅ HECHO — DiversityRetriever, MmrScorer en rag_methods.rs

---

### 6.3 Hierarchical RAG Router

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/rag_methods.rs`.

Router que dirige queries a diferentes niveles de RAG segun complejidad:
simple lookup → BM25, factual → dense retrieval, multi-hop → Graph RAG,
analytical → RAPTOR. Reduce latencia para queries simples.

**Tipos clave**:
- `HierarchicalRouter`: Clasifica queries y ruta a retriever optimo
- `QueryComplexity`: Simple, Factual, MultiHop, Analytical, Conversational
- `QueryClassifier`: Clasifica query por complejidad (heuristic + optional LLM)
- `RouterConfig`: mapping de QueryComplexity → retriever pipeline
- `RoutingDecision`: chosen_retriever, confidence, fallback_retriever

**Tests**: ~12

**Estado**: ✅ HECHO — HierarchicalRouter, QueryClassifier en rag_methods.rs

---

## Fase 7 — Agent Evaluation & Red Teaming

### 7.1 Agent Trajectory Analysis

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/agent_eval.rs` (feature `eval`).

Framework de evaluacion que analiza trayectorias completas de agentes:
secuencias de tool calls, decisiones de planificacion, uso de memoria.
Mide eficiencia, correccion y costo.

**Tipos clave**:
- `TrajectoryRecorder`: Registra cada paso del agente (tool call, response, decision)
- `TrajectoryStep`: action, input, output, duration, tokens, cost
- `TrajectoryAnalyzer`: Analiza trayectoria contra ground truth
- `AgentMetrics`: step_count, tool_accuracy, plan_efficiency, cost_per_task, time_to_complete
- `TrajectoryComparator`: Compara trayectorias de agente vs human baseline
- `EvalReport`: Reporte con metricas, desglose por paso, recomendaciones

**Tests**: ~20

**Estado**: ✅ HECHO — TrajectoryRecorder, TrajectoryAnalyzer en NEW agent_eval.rs

---

### 7.2 Tool-Call Accuracy Metrics

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/agent_eval.rs`.

Metricas especificas para evaluar si el agente usa herramientas correctamente:
tool elegida correcta, argumentos validos, resultados utilizados adecuadamente.

**Tipos clave**:
- `ToolCallEvaluator`: Evalua calls contra expected calls
- `ToolAccuracyMetrics`: precision, recall, f1, argument_accuracy, result_utilization
- `ExpectedToolCall`: tool_name, expected_args (partial match), expected_order
- `ToolCallMatcher`: Alinea actual vs expected calls (order-sensitive y order-insensitive)

**Tests**: ~14

**Estado**: ✅ HECHO — ToolCallEvaluator, ToolAccuracyMetrics en agent_eval.rs

---

### 7.3 Automated Red Teaming

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/red_team.rs` (feature `eval`).

Suite de red teaming automatizado que genera prompts adversariales, los ejecuta
contra el agente, y reporta vulnerabilidades. Cubre: prompt injection, jailbreaks,
data exfiltration, tool misuse, instruction following bypass.

**Tipos clave**:
- `RedTeamSuite`: Suite de ataques configurables
- `AttackCategory`: PromptInjection, Jailbreak, DataExfil, ToolMisuse, InstructionBypass
- `AttackGenerator`: Genera variantes de ataques por categoria
- `AttackTemplate`: Template parametrizable con mutaciones
- `RedTeamResult`: vulnerability_found, category, severity, reproduction_steps
- `RedTeamReport`: Agregado de resultados con risk_score y recomendaciones
- `DefenseEvaluator`: Verifica que guardrails existentes detectan los ataques

**Tests**: ~18

**Estado**: ✅ HECHO — RedTeamSuite, AttackGenerator, DefenseEvaluator en NEW red_team.rs

---

### 7.4 Natural Language Policy Guards

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/guardrail_pipeline.rs`.

Definir politicas de seguridad en lenguaje natural que se compilan a guardrails
ejecutables. Ejemplo: "Never reveal internal API keys" → PatternGuard + PII guard.

**Tipos clave**:
- `NaturalLanguageGuard`: Guard que evalua output contra politicas NL
- `PolicyStatement`: Politica en texto natural con prioridad y scope
- `PolicyCompiler`: Compila statements a checkers ejecutables (regex + semantic)
- `SemanticChecker`: Usa embedding similarity para evaluar cumplimiento
- `PolicyViolation`: statement, severity, evidence, suggested_fix

**Tests**: ~12

**Estado**: ✅ HECHO — NaturalLanguageGuard, PolicyCompiler en guardrail_pipeline.rs

---

## Fase 8 — Reasoning & Planning

### 8.1 MCTS Planning

**Prioridad**: ALTA | **Esfuerzo**: XL | **Impacto**: Muy Alto

Nuevo archivo: `src/mcts_planner.rs` (feature `autonomous`).

Monte Carlo Tree Search para planificacion multi-paso de agentes. Explora
arboles de decisiones posibles, simulando resultados de cada accion antes
de comprometerse. Ideal para tareas complejas donde el orden importa.

**Tipos clave**:
- `MctsPlanner`: Planificador MCTS con UCB1 selection
- `MctsConfig`: max_iterations, exploration_constant (C), max_depth, simulation_budget
- `MctsNode`: state, action, visits, total_reward, children, parent
- `MctsState`: Trait que define el estado del problema (available_actions, is_terminal, reward)
- `AgentMctsState`: Implementacion de MctsState para agentes (tools como acciones)
- `SimulationPolicy`: Random, Heuristic, LlmGuided (usa LLM para simular)
- `MctsResult`: best_action_sequence, confidence, explored_nodes, tree_depth

**Tests**: ~20

**Estado**: ✅ HECHO — MctsPlanner, MctsState trait, AgentMctsState en NEW mcts_planner.rs

---

### 8.2 Process Reward Models

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/mcts_planner.rs`.

Verificacion paso-a-paso de cadenas de razonamiento. Cada paso recibe un score
de correccion, permitiendo detectar errores tempranos antes de que se propaguen.
Complementa MCTS como funcion de reward mas precisa que outcome-based.

**Tipos clave**:
- `ProcessRewardModel`: Trait para scoring de pasos individuales
- `StepScore`: score (0.0-1.0), confidence, feedback
- `RuleBasedPRM`: Verificacion por reglas (type checking, constraint satisfaction)
- `LlmPRM`: Usa LLM para evaluar correccion de cada paso
- `PrmAggregator`: Combina scores de multiples pasos (min, product, weighted_avg)

**Tests**: ~14

**Estado**: ✅ HECHO — ProcessRewardModel, RuleBasedPRM en mcts_planner.rs

---

### 8.3 Plan Refinement Loop

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/mcts_planner.rs`.

Loop iterativo que mejora planes basado en feedback de ejecucion. Cuando un
paso falla o el resultado es suboptimo, el planner re-planifica desde ese punto
con el contexto actualizado.

**Tipos clave**:
- `RefinementLoop`: Ejecuta plan, evalua, re-planifica si necesario
- `RefinementConfig`: max_iterations, improvement_threshold, re_plan_on_failure
- `ExecutionFeedback`: step_index, success, error, actual_vs_expected
- `RefinementStrategy`: ReplanFromFailure, ReplanEntire, PatchStep

**Tests**: ~10

**Estado**: ✅ HECHO — RefinementLoop, RefinementConfig en mcts_planner.rs

---

## Fase 9 — Voice & Multimodal v2

### 9.1 WebRTC Voice Transport

**Prioridad**: ALTA | **Esfuerzo**: XL | **Impacto**: Muy Alto

Extender `src/voice_agent.rs` (nuevo feature `webrtc`).

Transport WebRTC para voz en tiempo real con latencia sub-200ms. Implementa
SDP offer/answer, ICE candidate exchange y RTP audio streaming. Compatible
con OpenAI Realtime API y Google ADK.

**Tipos clave**:
- `WebRtcTransport`: Transport layer con SDP negotiation
- `WebRtcConfig`: stun_servers, turn_servers, audio_codec (Opus/PCM), sample_rate
- `SdpOffer`/`SdpAnswer`: Structs para Session Description Protocol
- `IceCandidate`: Candidato ICE con prioridad y tipo (host/srflx/relay)
- `RtpStream`: Stream de audio RTP con jitter buffer
- `WebRtcSession`: Sesion activa con stats (latency, packet_loss, jitter)

**Tests**: ~16

**Estado**: ✅ HECHO — WebRtcTransport, SdpOffer, WebRtcSession en voice_agent.rs

---

### 9.2 Speech-to-Speech Pipeline

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/voice_agent.rs`.

Pipeline directo audio→audio que evita la transcripcion intermedia a texto
cuando el modelo lo soporta (ej: GPT-4o-audio, Gemini 2.0). Reduce latencia
~300ms al eliminar STT+TTS intermedios.

**Tipos clave**:
- `SpeechToSpeechPipeline`: Pipeline directo sin texto intermedio
- `S2SConfig`: model_id, input_format, output_format, voice_id
- `AudioModelCapability`: Enum indicando si modelo soporta audio nativo
- `S2SFallback`: Fallback automatico a STT→LLM→TTS si modelo no soporta S2S

**Tests**: ~12

**Estado**: ✅ HECHO — SpeechToSpeechPipeline en voice_agent.rs

---

### 9.3 Video Frame Analysis

**Prioridad**: BAJA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/media_generation.rs`.

Analisis de video extrayendo frames clave y procesandolos con modelos de vision.
Soporta: resumen de video, extraccion de objetos/acciones, timestamped descriptions.

**Tipos clave**:
- `VideoAnalyzer`: Analiza video extrayendo y procesando frames
- `FrameExtractionStrategy`: FixedInterval, SceneChange, KeyframeOnly
- `VideoAnalysisConfig`: max_frames, frame_interval_ms, analysis_prompt
- `VideoAnalysisResult`: frame_descriptions, summary, timeline, key_moments

**Tests**: ~10

**Estado**: ✅ HECHO — VideoAnalyzer en media_generation.rs

---

## Fase 10 — Platform & Infrastructure

### 10.1 Multi-Backend Sandbox

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/container_sandbox.rs`.

Abstraer el backend de sandbox con un trait pluggable. Ademas de Docker,
soportar: Podman (rootless containers), wasmtime (WASM sandbox) y procesos
aislados con namespaces (Linux).

**Tipos clave**:
- `SandboxBackend` trait: create, execute, cleanup, is_available
- `PodmanBackend`: Implementacion via Podman CLI (compatible Docker images)
- `WasmSandbox`: Ejecucion en wasmtime con limits de memoria/CPU
- `ProcessSandbox`: Proceso aislado con seccomp/namespaces (Linux)
- `SandboxSelector`: Auto-seleccion del mejor backend disponible
- `SandboxConfig`: backend preference order, resource limits, network policy

**Tests**: ~16

**Estado**: ✅ HECHO — SandboxBackend trait, PodmanBackend, WasmSandbox en container_sandbox.rs

---

### 10.2 Agent Deployment Profiles

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/agent_definition.rs`.

Perfiles de deployment declarativos (TOML) que especifican como desplegar un
agente: sandbox backend, recursos, networking, auto-scaling, health checks.

**Tipos clave**:
- `DeploymentProfile`: Struct con sandbox, resources, networking, health_check
- `ResourceLimits`: max_memory_mb, max_cpu_percent, max_disk_mb, max_runtime_secs
- `NetworkingConfig`: allowed_hosts, dns_policy, proxy
- `HealthCheck`: endpoint, interval, timeout, unhealthy_threshold
- `ProfileLoader`: Carga perfiles desde TOML/JSON

**Tests**: ~10

**Estado**: ✅ HECHO — DeploymentProfile, ProfileLoader en agent_definition.rs

---

### 10.3 Agent DevTools

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Alto

Nuevo archivo: `src/agent_devtools.rs` (feature `devtools`).

Herramientas de desarrollo para debugging y profiling de agentes: step-through
execution, execution replay, performance profiling, state inspection.

**Tipos clave**:
- `AgentDebugger`: Depurador con breakpoints en tool calls y decision points
- `Breakpoint`: Enum (BeforeToolCall(name), AfterStep(n), OnConfidenceBelow(f64))
- `ExecutionRecorder`: Graba ejecucion completa para replay
- `ExecutionReplay`: Reproduce ejecucion grabada paso a paso
- `PerformanceProfiler`: Mide tiempo/tokens/costo por paso
- `StateInspector`: Inspecciona estado interno del agente en cualquier punto
- `DevToolsConfig`: enable_recording, enable_profiling, breakpoints

**Tests**: ~18

**Estado**: ✅ HECHO — AgentDebugger, ExecutionRecorder, PerformanceProfiler en NEW agent_devtools.rs

---

## Resumen

| Fase | Items | Estado | Ficheros | Tests |
|------|-------|--------|----------|-------|
| 1. MCP Spec Completeness | 4 | ✅ 4/4 HECHO | mcp_protocol.rs (ext) | ~38 |
| 2. Remote MCP & Discovery | 3 | ✅ 3/3 HECHO | mcp_client.rs (new), a2a_protocol.rs (ext) | ~45 |
| 3. Human-in-the-Loop | 4 | ✅ 4/4 HECHO | hitl.rs (new) | ~54 |
| 4. Prompt Optimization v2 | 3 | ✅ 3/3 HECHO | prompt_signature.rs (ext) | ~44 |
| 5. Memory OS | 4 | ✅ 4/4 HECHO | advanced_memory.rs (ext) | ~52 |
| 6. Advanced RAG v2 | 3 | ✅ 3/3 HECHO | rag.rs (ext), rag_methods.rs (ext) | ~38 |
| 7. Eval & Red Teaming | 4 | ✅ 4/4 HECHO | agent_eval.rs (new), red_team.rs (new), guardrail_pipeline.rs (ext) | ~64 |
| 8. Reasoning & Planning | 3 | ✅ 3/3 HECHO | mcts_planner.rs (new) | ~44 |
| 9. Voice & Multimodal v2 | 3 | ✅ 3/3 HECHO | voice_agent.rs (ext), media_generation.rs (ext) | ~38 |
| 10. Platform & Infrastructure | 3 | ✅ 3/3 HECHO | container_sandbox.rs (ext), agent_definition.rs (ext), agent_devtools.rs (new) | ~44 |
| **TOTAL** | **34** | **✅ 34/34 HECHO** | **6 nuevos + 12 extendidos** | **~461** |

**Total tests**: 4449 (3907 pre-v6 + 542 nuevos)

---

## Ficheros Nuevos (6)

| Fichero | Items | Feature Gate |
|---------|-------|-------------|
| `src/mcp_client.rs` | 2.1 | `mcp` |
| `src/hitl.rs` | 3.1-3.4 | `hitl` (nuevo) |
| `src/agent_eval.rs` | 7.1-7.2 | `eval` |
| `src/red_team.rs` | 7.3 | `eval` |
| `src/mcts_planner.rs` | 8.1-8.3 | `autonomous` |
| `src/agent_devtools.rs` | 10.3 | `devtools` (nuevo) |

## Ficheros Existentes a Extender (12)

| Fichero | Items |
|---------|-------|
| `src/mcp_protocol.rs` | 1.1-1.4 |
| `src/a2a_protocol.rs` | 2.2-2.3 |
| `src/prompt_signature.rs` | 4.1-4.3 |
| `src/advanced_memory.rs` | 5.1-5.4 |
| `src/rag.rs` | 6.1 |
| `src/rag_methods.rs` | 6.2-6.3 |
| `src/guardrail_pipeline.rs` | 7.4 |
| `src/voice_agent.rs` | 9.1-9.2 |
| `src/media_generation.rs` | 9.3 |
| `src/container_sandbox.rs` | 10.1 |
| `src/agent_definition.rs` | 10.2 |

## Feature Flags Nuevos

| Feature | En `full` | Dependencias |
|---------|-----------|-------------|
| `hitl` | NO | — |
| `webrtc` | NO | `dep:tokio` |
| `devtools` | NO | — |

---

## Prioridades

**CRITICAS** (hacer primero):
- 1.1 MCP Elicitation
- 2.1 Remote MCP Client
- 3.1 Tool Approval Gates
- 5.1 Automatic Memory Extraction

**ALTAS** (hacer segundo):
- 1.2-1.3 MCP Audio + Batching
- 3.2 Confidence Escalation
- 4.1 SIMBA Optimizer
- 4.2 Reasoning Traces
- 5.2 Memory Scheduler
- 6.1-6.2 Discourse Chunking + MMR
- 7.1-7.3 Agent Eval + Red Teaming
- 8.1 MCTS Planning
- 9.1 WebRTC Voice
- 10.1 Multi-Backend Sandbox

**MEDIAS** (hacer tercero):
- 1.4 MCP Completions
- 2.2-2.3 AGENTS.md + ACP Bridge
- 3.3-3.4 Corrections + Policies
- 4.3 LLM-as-Judge
- 5.3-5.4 Cross-Session Memory + Search
- 6.3 Hierarchical Router
- 7.4 NL Policy Guards
- 8.2-8.3 Process Rewards + Refinement
- 9.2 Speech-to-Speech
- 10.2-10.3 Deploy Profiles + DevTools

**BAJAS**:
- 9.3 Video Frame Analysis
