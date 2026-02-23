# Plan de Mejoras para ai_assistant — v5

> **Estado: 30/30 items COMPLETE**

> Documento generado el 2026-02-22.
> Basado en completitud de v1 (39/39), v2 (22/22), v3 (21/21), v4 (38/38) con 3401+ tests, 0 failures.
> 224+ source files, ~232k+ LoC.
>
> **Planes anteriores**: v1 (providers, embeddings, MCP, documents, guardrails),
> v2 (async parity, vector DBs, evaluation, testing),
> v3 (containers, document pipeline, speech/audio, CI/CD maturity),
> v4 (workflows, prompt signatures, A2A, advanced memory, online eval, streaming guardrails).

---

## Contexto

Tras v4, ai_assistant es el framework de IA mas completo en Rust, con paridad
o ventaja sobre LangChain, LlamaIndex, CrewAI y Semantic Kernel en la mayoria
de capacidades. Sin embargo, el analisis competitivo de febrero 2026 revela
**brechas emergentes** en 5 areas clave:

1. **Optimizacion de prompts limitada** — DSPy ha lanzado GEPA (Genetic Pareto
   Optimization) y MIPROv2, que superan a prompts escritos por humanos.
   Nuestro `prompt_signature.rs` tiene BootstrapFewShot y busqueda bayesiana
   basica pero no algoritmos geneticos ni propuestas multi-etapa.

2. **MCP desactualizado** — MCP migra a v2 (marzo 2026) con Streamable HTTP
   Transport, OAuth 2.1, Tool Annotations y elision de SSE. Nuestro
   `mcp_protocol.rs` implementa spec 2025-03-26.

3. **Voz unidireccional** — OpenAI Agents SDK y Google ADK ofrecen agentes
   de voz bidireccionales con deteccion de interrupciones en tiempo real.
   Nuestro `speech.rs` solo hace STT/TTS secuencial.

4. **Sin generacion de imagen/video** — Vercel AI SDK 6 integra generacion
   de video (Sora, Runway, Wan) e imagen (DALL-E 3, Flux). Solo tenemos
   vision (comprension de imagen), no generacion.

5. **Sin pipeline traza-a-destilacion** — OpenAI y NVIDIA impulsan el Data
   Flywheel: observar agentes via trazas, recopilar trayectorias exitosas,
   destilar modelos mas pequenos. Tenemos telemetria y fine-tuning por
   separado, pero no el pipeline cerrado.

Ademas, tendencias emergentes en:
- **OpenTelemetry GenAI Semantic Conventions** (estandarizacion de atributos)
- **Constrained decoding** para modelos locales (XGrammar, llguidance)
- **Patrones de conversacion nombrados** (Swarm, Debate, Round-Robin)
- **Ejecucion durable automatica** (persistencia en cada paso, no opt-in)
- **Memoria auto-evolutiva** (MemRL: refuerzo sobre memoria episodica)

---

## Decisiones de Diseno (propuestas)

| Decision | Eleccion |
|---|---|
| **GEPA/MIPROv2** | Extender `prompt_signature.rs` existente — misma API de `Signature` + `EvalMetric` |
| **MCP v2** | Extender `mcp_protocol.rs` — anadir transport layer, tool annotations, OAuth |
| **Feature flag MCP v2** | Usar `mcp` existente (no nuevo flag, es evolucion) |
| **Voice agents** | Nuevo `src/voice_agent.rs` — feature `voice-agent` (pesado: tokio streams) |
| **Image/video gen** | Nuevo `src/media_generation.rs` — feature `media-generation` |
| **Distillation pipeline** | Nuevo `src/distillation.rs` — feature `distillation` |
| **Constrained decoding** | Nuevo `src/constrained_decoding.rs` — feature `constrained-decoding` |
| **OTel GenAI** | Extender `opentelemetry_integration.rs` — bajo feature `analytics` existente |
| **Conversation patterns** | Extender `multi_agent.rs` — bajo feature `multi-agent` existente |
| **Durable execution** | Extender `event_workflow.rs` + `autonomous_loop.rs` — features existentes |
| **Agent definitions** | Nuevo `src/agent_definition.rs` — siempre disponible (ligero, solo parsing TOML) |
| **Memory evolution** | Extender `advanced_memory.rs` — feature `advanced-memory` existente |

---

## Fase 1 — Optimizacion Avanzada de Prompts (DSPy Parity)

### 1.1 GEPA — Genetic Pareto Optimizer

**Prioridad**: CRITICA | **Esfuerzo**: XL | **Impacto**: Muy Alto

Extender `src/prompt_signature.rs`.

Optimizador genetico multi-objetivo inspirado en DSPy GEPA (julio 2025).
Evoluciona poblaciones de prompts compilados maximizando multiples metricas
simultaneamente (calidad, costo, latencia) via frente de Pareto.

**Tipos clave**:
- `GEPAOptimizer`: Motor genetico con poblacion, seleccion, cruce, mutacion
- `GEPAConfig`: tam_poblacion, generaciones, tasa_mutacion, tasa_cruce, elitismo
- `ParetoFront`: Conjunto de soluciones no-dominadas
- `ParetoSolution`: CompiledPrompt + vector de scores multi-objetivo
- `GeneticOperator` trait: Crossover, Mutation, Selection pluggables

**Algoritmo**:
1. Inicializar poblacion de N prompts compilados (variando demos, instrucciones)
2. Evaluar cada individuo contra K metricas (ExactMatch, F1, costo, latencia)
3. Seleccion por torneo con crowding distance (NSGA-II)
4. Cruce: combinar demos de dos padres, interpolar instrucciones
5. Mutacion: permutar demos, reescribir instrucciones, agregar/quitar campos
6. Repetir G generaciones, retornar frente de Pareto

**Tests**: ~20 (evolucion basica, frente de Pareto, convergencia, diversidad)

---

### 1.2 MIPROv2 — Multi-stage Instruction Proposal Optimizer

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/prompt_signature.rs`.

Pipeline de optimizacion en 3 etapas inspirado en DSPy MIPROv2:
bootstrap de demos → propuesta de instrucciones → busqueda discreta.

**Tipos clave**:
- `MIPROv2Optimizer`: Pipeline de 3 etapas
- `MIPROv2Config`: max_bootstrapped_demos, max_labeled_demos, num_candidates, num_trials
- `InstructionProposer`: Genera candidatas de instrucciones via LLM (grounded en demos)
- `DiscreteSearchStrategy` enum: Exhaustive, Random, Bayesian

**Algoritmo**:
1. **Bootstrap**: Ejecutar signature con ejemplos, recopilar trazas exitosas como demos
2. **Propose**: Usar LLM para generar N instrucciones candidatas basadas en demos
3. **Search**: Evaluar combinaciones (instruccion, demos) con la metrica, seleccionar mejor

**Tests**: ~15 (bootstrap, propuesta, busqueda, pipeline completo)

---

### 1.3 Prompt Assertions & Constraints

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/prompt_signature.rs`.

Restricciones programaticas que guian la optimizacion, inspiradas en DSPy Assertions.
Una assertion es una condicion que DEBE cumplirse en la salida; el optimizador
penaliza prompts que la violan.

**Tipos clave**:
- `PromptAssertion` trait: `fn check(&self, output: &str) -> AssertionResult`
- `AssertionResult`: Pass / Fail(reason) / Warn(reason)
- `LengthAssertion`: min/max caracteres o tokens
- `FormatAssertion`: regex que la salida debe cumplir
- `ContainsAssertion`: keywords que deben aparecer
- `JsonSchemaAssertion`: salida debe ser JSON valido contra schema
- `CustomAssertion`: closure definida por usuario

**Integracion**: Las assertions se registran en `Signature` y se aplican como
penalizacion adicional en todos los optimizadores (GEPA, MIPROv2, Grid, Bayesian).

**Tests**: ~12 (cada tipo de assertion, integracion con optimizadores)

---

### 1.4 LM Adapters — Compilacion Aware del Provider

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/prompt_signature.rs`.

Los prompts compilados se formatean distinto segun el provider: chat API vs
completion API, function calling vs text, multimodal vs text-only. Un LM Adapter
traduce la Signature compilada al formato optimo del provider destino.

**Tipos clave**:
- `LmAdapter` trait: `fn compile_for_provider(&self, compiled: &CompiledPrompt, provider: &AiProvider) -> FormattedPrompt`
- `ChatAdapter`: Formatea como mensajes (system + few-shot as user/assistant turns)
- `CompletionAdapter`: Formatea como prompt unico con delimitadores
- `FunctionCallingAdapter`: Usa tool_use/function_call del provider para structured output
- `FormattedPrompt`: Representacion final lista para enviar al LLM

**Tests**: ~10 (cada adapter, round-trip, provider detection)

---

## Fase 2 — MCP v2 Protocol

### 2.1 Streamable HTTP Transport

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/mcp_protocol.rs`.

MCP v2 (2025-11-05+) reemplaza SSE con Streamable HTTP: el servidor responde
a POST requests con JSON (respuesta inmediata) o SSE stream (operacion larga).
El cliente detecta automaticamente via Content-Type. Esto elimina la necesidad
de mantener conexiones SSE persistentes.

**Tipos clave**:
- `StreamableHttpTransport`: Nuevo transport que envia POST, detecta Content-Type
- `TransportMode` enum: `StdIO | SSE | StreamableHTTP`
- `McpSession`: Session management con session-id header
- `McpSessionStore` trait: Almacena sessions activas (in-memory o SQLite)

**Cambios**:
- Anadir `TransportMode` al `McpClient` existente
- POST a endpoint unico (no /sse + /messages separados)
- Session-id via `Mcp-Session` header
- Auto-detect: si respuesta es `text/event-stream` → SSE, si `application/json` → direct

**Tests**: ~15 (transport detection, session management, backwards compat con v1)

---

### 2.2 OAuth 2.1 + Dynamic Client Registration

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/mcp_protocol.rs`.

MCP v2 requiere OAuth 2.1 para autenticacion (reemplaza API keys simples).
Incluye Authorization Server discovery via RFC 8414, PKCE obligatorio,
y opcionalmente Dynamic Client Registration (RFC 7591).

**Tipos clave**:
- `McpOAuthConfig`: authorization_endpoint, token_endpoint, client_id, scopes
- `McpAuthFlow`: Implementa OAuth 2.1 con PKCE (S256)
- `OAuthTokenManager`: Almacena, refresca y revoca tokens
- `DynamicClientRegistration`: Registro automatico de clientes via RFC 7591
- `AuthorizationServerMetadata`: Discovery via `/.well-known/oauth-authorization-server`

**Flujo**:
1. Cliente descubre metadata del auth server
2. Si no tiene client_id → Dynamic Client Registration
3. Authorization Code + PKCE → token
4. Token refresh automatico antes de expiracion
5. Token incluido en headers de MCP requests

**Tests**: ~12 (discovery, PKCE, token refresh, registration, error handling)

---

### 2.3 Tool Annotations

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/mcp_protocol.rs`.

MCP v2 anade anotaciones a herramientas para indicar comportamiento:
read-only, destructive, idempotent, open-world (puede interactuar con
entidades no especificadas en parametros). El agente usa estas anotaciones
para decidir si necesita confirmacion humana.

**Tipos clave**:
- `ToolAnnotations`: struct con campos booleanos
  - `read_only`: No modifica estado (default: false)
  - `destructive`: Puede eliminar datos (default: true — conservador)
  - `idempotent`: Seguro re-ejecutar (default: false)
  - `open_world`: Afecta entidades externas (default: true)
- Integracion con `ToolDefinition` existente

**Cambios**: Anadir campo `annotations: Option<ToolAnnotations>` a `McpToolDef`.
Los guardrails pueden usar `destructive` para auto-requerir aprobacion.

**Tests**: ~8 (serialization, defaults, integracion con guardrails)

---

## Fase 3 — Agentes de Voz en Tiempo Real

### 3.1 Bidirectional Audio Streaming

**Prioridad**: ALTA | **Esfuerzo**: XL | **Impacto**: Muy Alto

Nuevo archivo: `src/voice_agent.rs` — feature `voice-agent`.

Stream de audio bidireccional: el usuario habla, el agente escucha en tiempo
real (no espera fin de frase), procesa, y responde con voz sintetizada.
Usa WebSocket o gRPC bidireccional. Inspirado en OpenAI Realtime API y Google ADK.

**Tipos clave**:
- `VoiceAgent`: Agente conversacional con audio bidireccional
- `VoiceAgentConfig`: modelo STT, modelo TTS, vad_config, interruption_policy
- `AudioStream`: Stream de chunks de audio (PCM 16-bit, 16kHz)
- `VoiceSession`: Sesion con estado (listening, processing, speaking, interrupted)
- `VoiceTransport` trait: WebSocket, gRPC, o local (pipes)

**Arquitectura**:
1. Thread de captura: recibe audio → buffer circular
2. VAD: detecta inicio/fin de habla → segmentos
3. STT: transcribe segmentos → texto
4. LLM: genera respuesta (con context/tools)
5. TTS: sintetiza respuesta → audio
6. Thread de playback: emite audio al cliente

**Tests**: ~18 (session lifecycle, state transitions, config, mock streams)

---

### 3.2 Voice Activity Detection + Interruption Handling

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

En `src/voice_agent.rs`.

Deteccion de actividad de voz (VAD) para segmentar habla del ruido, y
manejo de interrupciones: si el usuario habla mientras el agente responde,
el agente para de hablar y escucha.

**Tipos clave**:
- `VadConfig`: energy_threshold, silence_duration_ms, min_speech_duration_ms
- `VadDetector`: Analiza frames de audio, emite SpeechStart/SpeechEnd events
- `InterruptionPolicy` enum: `Immediate` (para al instante), `EndSentence` (termina frase), `Never` (no interrumpible)
- `InterruptionEvent`: timestamp, partial_response (lo que alcanzo a decir)

**Algoritmo VAD** (energy-based, sin ML):
1. Calcular RMS energy por frame (20ms)
2. Media movil de energia con ventana de 300ms
3. Si energia > threshold durante min_speech_duration → SpeechStart
4. Si energia < threshold durante silence_duration → SpeechEnd

**Tests**: ~12 (VAD detection, interruption policies, edge cases)

---

### 3.3 Conversation Turn Management

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

En `src/voice_agent.rs`.

Gestion de turnos conversacionales: quien habla cuando, buffering de
contexto, y mantenimiento del historial de conversacion con timestamps
y duracion de cada turno.

**Tipos clave**:
- `ConversationTurn`: speaker (User/Agent), transcript, audio_duration, timestamp
- `TurnManager`: Controla turnos, detecta solapamientos, mantiene historial
- `TurnPolicy` enum: `StrictAlternating` | `NaturalOverlap` | `PushToTalk`
- `VoiceConversationHistory`: Lista de turns con busqueda temporal

**Tests**: ~10 (turn transitions, overlap detection, history)

---

## Fase 4 — Generacion de Imagen y Video

### 4.1 Image Generation Providers

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Nuevo archivo: `src/media_generation.rs` — feature `media-generation`.

Interfaz unificada para generar imagenes via multiples providers.
Sigue el patron de `cloud_providers.rs` con routing por provider.

**Tipos clave**:
- `ImageGenerationProvider` trait: `fn generate_image(&self, prompt: &str, config: &ImageGenConfig) -> Result<GeneratedImage>`
- `ImageGenConfig`: width, height, style, quality, num_images, negative_prompt
- `GeneratedImage`: bytes, format (PNG/JPEG/WebP), revised_prompt, seed
- Providers:
  - `DallEProvider`: OpenAI DALL-E 3 (via /v1/images/generations)
  - `StableDiffusionProvider`: Stability AI (via REST API)
  - `FluxProvider`: Black Forest Labs Flux (via API)
  - `LocalDiffusionProvider`: SD.Next / ComfyUI local (via HTTP API)

**Tests**: ~14 (trait interface, config validation, provider routing, mock responses)

---

### 4.2 Image Editing

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

En `src/media_generation.rs`.

Operaciones de edicion sobre imagenes existentes: inpainting (rellenar
areas enmascaradas), outpainting (extender bordes), y transformacion de estilo.

**Tipos clave**:
- `ImageEditProvider` trait: `fn edit_image(&self, image: &[u8], mask: Option<&[u8]>, prompt: &str, config: &ImageEditConfig) -> Result<GeneratedImage>`
- `ImageEditConfig`: operation (Inpaint/Outpaint/StyleTransfer), strength, guidance_scale
- `ImageEditOperation` enum: Inpaint, Outpaint, StyleTransfer, Upscale, RemoveBackground
- Provider support: DALL-E edit endpoint, Stability AI edit, local ComfyUI workflows

**Tests**: ~10 (cada operacion, mask validation, provider routing)

---

### 4.3 Video Generation Providers

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Alto

En `src/media_generation.rs`.

Generacion de video a partir de texto o imagen. APIs asincronas (submit job →
poll status → download result). Inspirado en Vercel AI SDK 6.

**Tipos clave**:
- `VideoGenerationProvider` trait: `fn generate_video(&self, prompt: &str, config: &VideoGenConfig) -> Result<VideoJob>`
- `VideoGenConfig`: duration_seconds, fps, resolution, aspect_ratio, style, seed
- `VideoJob`: job_id, status (Queued/Processing/Complete/Failed), progress_pct
- `GeneratedVideo`: bytes, format (MP4/WebM), duration, resolution
- Providers:
  - `RunwayProvider`: Runway ML Gen-3 (via REST API)
  - `SoraProvider`: OpenAI Sora (cuando disponible via API)
  - `WanProvider`: Alibaba Wan video (via API)
  - `ReplicateVideoProvider`: Replicate hosted models

**Tests**: ~12 (job lifecycle, polling, config validation, provider routing)

---

## Fase 5 — Pipeline Traza-a-Destilacion

### 5.1 Agent Trajectory Collector

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/distillation.rs` — feature `distillation`.

Captura estructurada de trayectorias de agentes: cada paso (LLM call,
tool use, decision, resultado) se registra como un `TrajectoryStep`.
Las trayectorias se pueden etiquetar (exitosa/fallida) automatica o manualmente.

**Tipos clave**:
- `TrajectoryCollector`: Intercepta ejecuciones de agentes y registra pasos
- `Trajectory`: Vec de TrajectoryStep + metadata (agent_id, task, duration, outcome)
- `TrajectoryStep`: StepType (LlmCall/ToolUse/Decision/Observation), input, output, tokens, latency
- `TrajectoryOutcome` enum: Success(score) | Failure(reason) | Partial(score)
- `TrajectoryStore` trait: Persistencia (in-memory, SQLite, JSONL)

**Integracion**: Hook en `agentic_loop.rs` y `event_workflow.rs` para captura
automatica. Activacion via config flag (no overhead si desactivado).

**Tests**: ~15 (captura, persistencia, etiquetado, filtering)

---

### 5.2 Trajectory Scorer & Filter

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

En `src/distillation.rs`.

Evalua y filtra trayectorias para seleccionar las mejores como datos
de entrenamiento. Usa metricas existentes (OnlineEvaluator hooks) mas
criterios especificos de trayectoria.

**Tipos clave**:
- `TrajectoryScorer` trait: `fn score(&self, trajectory: &Trajectory) -> f64`
- `OutcomeScorer`: Score basado en resultado final (success=1.0, fail=0.0)
- `EfficiencyScorer`: Penaliza trayectorias con muchos pasos o tokens
- `DiversityScorer`: Premia trayectorias que cubren casos distintos
- `TrajectoryFilter`: Filtra por score minimo, max steps, max tokens
- `TrajectoryDataset`: Coleccion filtrada lista para distilacion

**Tests**: ~12 (cada scorer, filtering, dataset construction)

---

### 5.3 Distillation Dataset Builder

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

En `src/distillation.rs`.

Convierte trayectorias filtradas en datasets de fine-tuning en formatos
estandar (JSONL OpenAI, Alpaca, ShareGPT). Cada trayectoria se
"aplana" en pares (input, output) o conversaciones multi-turno.

**Tipos clave**:
- `DatasetBuilder`: Convierte TrajectoryDataset a formato de fine-tuning
- `DatasetFormat` enum: OpenAIJsonl, Alpaca, ShareGPT, Custom
- `DatasetConfig`: max_examples, train_test_split, dedup, shuffle
- `FlatteningStrategy` enum: LastStepOnly, AllSteps, SummaryOnly

**Cada formato**:
- OpenAI JSONL: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- Alpaca: `{"instruction": ..., "input": ..., "output": ...}`
- ShareGPT: `{"conversations": [{"from": "human", "value": ...}, ...]}`

**Tests**: ~10 (cada formato, splitting, dedup)

---

### 5.4 Data Flywheel Orchestrator

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

En `src/distillation.rs`.

Orquestador que cierra el ciclo: colectar trazas → evaluar → filtrar →
construir dataset → (trigger fine-tuning externo) → evaluar modelo nuevo →
repetir. Inspirado en NVIDIA Data Flywheel Blueprint.

**Tipos clave**:
- `DataFlywheel`: Orquesta el pipeline completo
- `FlywheelConfig`: collection_window, min_trajectories, score_threshold, format, output_path
- `FlywheelCycle`: Representa una ejecucion del ciclo (timestamp, stats, dataset_path)
- `FlywheelTrigger` trait: Callback cuando el dataset esta listo (subir a OpenAI, HuggingFace, etc.)

**El ciclo**:
1. Recopilar N trayectorias nuevas desde TrajectoryStore
2. Scorear y filtrar (top-K por calidad y diversidad)
3. Construir dataset en formato elegido
4. Invocar trigger (webhook, upload, log)
5. Registrar stats del ciclo

**Tests**: ~10 (ciclo completo, triggers, stats, config validation)

---

## Fase 6 — OpenTelemetry GenAI Semantic Conventions

### 6.1 GenAI Semantic Attributes

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/opentelemetry_integration.rs`.

Adoptar las convenciones semanticas OTel para GenAI (estandar emergente 2025-2026).
Todos los spans de LLM calls usan atributos normalizados que herramientas de
observabilidad (Grafana, Datadog, Honeycomb) reconocen automaticamente.

**Atributos estandar**:
- `gen_ai.system`: "openai", "anthropic", "ollama", etc.
- `gen_ai.request.model`: nombre del modelo
- `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.request.top_p`
- `gen_ai.response.model`: modelo real usado (puede diferir del solicitado)
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.response.finish_reasons`: ["stop"], ["tool_calls"], etc.

**Tipos clave**:
- `GenAiSpanAttributes`: Builder para atributos normalizados
- `GenAiEventAttributes`: Para eventos dentro del span (prompt, completion, tool_call)
- `set_genai_attributes(span, attrs)`: Helper para instrumentar cualquier LLM call

**Tests**: ~10 (attribute building, span creation, event recording)

---

### 6.2 Hierarchical Agent Tracing

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/opentelemetry_integration.rs`.

Trazas jerarquicas para agentes multi-turno: un span raiz por sesion de agente,
sub-spans por turno, sub-sub-spans por LLM call y tool use. Permite visualizar
el "arbol de ejecucion" completo de un agente.

**Jerarquia de spans**:
```
agent_session (root)
  ├── agent_turn_1
  │   ├── llm_call (gen_ai attributes)
  │   ├── tool_call: search_web
  │   └── llm_call (gen_ai attributes)
  ├── agent_turn_2
  │   ├── llm_call
  │   └── tool_call: code_execute
  └── agent_turn_3
      └── llm_call (final response)
```

**Tipos clave**:
- `AgentTracer`: Crea y gestiona spans jerarquicos para sesiones de agente
- `AgentSpanContext`: Contexto propagado entre turns (trace_id, parent_span_id)
- `TurnSpan`: Span de un turno con metadata (turn_number, role, tokens)
- `ToolSpan`: Span de una llamada a herramienta (tool_name, duration, success)

**Integracion**: Hook automatico en `agentic_loop.rs` cuando OTel esta activo.

**Tests**: ~12 (jerarquia, propagacion de contexto, multi-agent)

---

### 6.3 Cost Attribution & Budget Enforcement

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/opentelemetry_integration.rs` + `src/cost_tracking.rs`.

Atribuir costos a cada span (LLM call, tool call, etc.) y acumular por
sesion/agente/usuario. Enforcement: rechazar requests que excedan presupuesto.

**Tipos clave**:
- `CostAttributor`: Calcula costo por span (input_tokens * price_per_1k + output_tokens * price_per_1k)
- `CostBudget`: Presupuesto por sesion/agente/usuario con limites
- `BudgetEnforcer`: Middleware que verifica presupuesto antes de LLM call
- `CostReport`: Desglose de costos por provider, modelo, agente, sesion

**Precios**: Tabla configurable de precios por modelo (actualizable en runtime).
Default con precios publicos de OpenAI, Anthropic, etc.

**Tests**: ~10 (calculo, enforcement, report generation, budget exceeded)

---

## Fase 7 — Ejecucion Durable y Patrones de Agente

### 7.1 Automatic State Persistence

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/event_workflow.rs` + `src/autonomous_loop.rs`.

Persistencia automatica de estado en cada paso del workflow/agente.
No requiere llamar a checkpoint() manualmente — cada transicion de
estado se persiste automaticamente. Resume exacto tras crash.

**Tipos clave**:
- `DurableExecutor`: Wrapper que auto-persiste estado antes/despues de cada nodo
- `DurableConfig`: backend (InMemory/SQLite/Custom), auto_checkpoint (bool), retention_policy
- `RetentionPolicy`: KeepAll, KeepLast(n), KeepDuration(Duration), KeepCheckpointsOnly
- `RecoveryManager`: Detecta ejecuciones interrumpidas, resume desde ultimo estado

**Cambios**:
- Anadir `DurableConfig` opcional a `WorkflowRunner`
- Anadir `DurableConfig` opcional a `AutonomousAgent`
- Si configurado, cada paso se persiste automaticamente

**Tests**: ~14 (auto-persist, recovery, retention, concurrent access)

---

### 7.2 Named Conversation Patterns

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Medio

Extender `src/multi_agent.rs`.

Patrones de conversacion pre-construidos para multi-agente, inspirados en
AG2/AutoGen. En lugar de orquestar manualmente, el usuario elige un patron
nombrado y configura los participantes.

**Patrones**:
- `Swarm`: N agentes procesan tareas de una cola compartida, el primero disponible toma la siguiente
- `Debate`: 2+ agentes argumentan posiciones opuestas, un juez sintetiza
- `RoundRobin`: Cada agente habla por turno en orden fijo
- `Sequential`: Pipeline lineal donde la salida de uno es entrada del siguiente
- `NestedChat`: Un agente puede invocar un sub-grupo de agentes como "herramienta"
- `Broadcast`: Un mensaje se envia a todos, se recopilan todas las respuestas

**Tipos clave**:
- `ConversationPattern` enum: Swarm, Debate, RoundRobin, Sequential, NestedChat, Broadcast
- `PatternConfig`: max_rounds, termination_condition, judge_agent (para Debate)
- `PatternRunner`: Ejecuta un patron con N agentes y devuelve resultado

**Tests**: ~18 (cada patron x basico + edge cases)

---

### 7.3 Declarative Agent Definitions

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Nuevo archivo: `src/agent_definition.rs` — siempre disponible (core).

Definir agentes en TOML/YAML en lugar de codigo. Carga en runtime, permite
hot-reload de configuracion. Inspirado en Microsoft Agent Framework.

**Formato TOML ejemplo**:
```toml
[agent]
name = "research_assistant"
role = "Analyst"
system_prompt = "You are a research analyst..."
model = "openai/gpt-4o"
temperature = 0.3
max_tokens = 4096

[[tools]]
name = "web_search"
needs_approval = true

[[tools]]
name = "code_execute"
needs_approval = false

[memory]
type = "episodic"
max_episodes = 1000

[guardrails]
max_tokens_per_response = 2000
block_pii = true
```

**Tipos clave**:
- `AgentDefinition`: Struct deserializable desde TOML/YAML/JSON
- `AgentDefinitionLoader`: Carga desde archivo o string
- `AgentBuilder::from_definition(def)`: Construye agente desde definicion
- `ToolRef`: Referencia a herramienta por nombre (resuelta en runtime)

**Tests**: ~12 (parsing TOML, parsing YAML, builder, validation, error handling)

---

### 7.4 Agent Handoffs

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/multi_agent.rs`.

Un agente puede "pasar" la conversacion a otro agente explicitamente,
con transferencia de contexto. Inspirado en OpenAI Agents SDK "handoffs".

**Tipos clave**:
- `HandoffRequest`: from_agent, to_agent, reason, context_transfer_policy
- `ContextTransferPolicy` enum: Full (todo el historial), Summary (resumen), LastN(n), Custom
- `HandoffManager`: Registra agentes disponibles, ejecuta transferencias
- `HandoffResult`: success, transferred_context_tokens, continuation

**Flujo**:
1. Agente A decide que otro agente es mas adecuado
2. Emite HandoffRequest con destino y politica
3. HandoffManager transfiere contexto segun politica
4. Agente B continua la conversacion

**Tests**: ~10 (handoff basico, cada policy, chain handoffs, circular detection)

---

## Fase 8 — Constrained Decoding para Modelos Locales

### 8.1 Grammar-Guided Generation

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Nuevo archivo: `src/constrained_decoding.rs` — feature `constrained-decoding`.

Generacion guiada por gramatica para modelos locales (Ollama, llama.cpp).
En lugar de esperar que el modelo produzca JSON valido y validar despues,
la gramatica restringe la generacion token-a-token para que SOLO pueda
producir output valido.

**Tipos clave**:
- `Grammar`: Representacion de una gramatica formal (GBNF format)
- `GrammarRule`: Regla individual (terminal, alternation, repetition, group)
- `GrammarBuilder`: API fluida para construir gramaticas programaticamente
- `GrammarConstraint`: Constraint que se envia al provider local

**Integracion con providers**:
- Ollama: campo `grammar` en /api/chat (GBNF nativo desde llama.cpp)
- LM Studio: campo `grammar` en API compatible
- vLLM: guided decoding con outlines/xgrammar

**Tests**: ~12 (grammar construction, GBNF serialization, builder API)

---

### 8.2 JSON Schema → Grammar Compiler

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

En `src/constrained_decoding.rs`.

Compila un JSON Schema a gramatica GBNF automaticamente. El usuario
define la estructura esperada como JSON Schema (o como struct Rust
con derive) y el compilador genera la gramatica optima.

**Tipos clave**:
- `SchemaToGrammar`: Compila JSON Schema → Grammar
- `JsonSchemaNode` enum: Object, Array, String, Number, Boolean, Null, Enum, Ref
- `GbnfEmitter`: Emite string GBNF desde Grammar

**Mappings**:
- `{"type": "string"}` → `"\"" [^"]* "\""`
- `{"type": "number"}` → `"-"? [0-9]+ ("." [0-9]+)?`
- `{"type": "object", "properties": {...}}` → `"{" ws prop1 "," ws prop2 ... "}"`
- `{"type": "array", "items": {...}}` → `"[" ws item ("," ws item)* "]"`
- `{"enum": ["a", "b"]}` → `"\"a\"" | "\"b\""`

**Tests**: ~14 (cada tipo JSON Schema, nested objects, arrays, enums, $ref)

---

### 8.3 Streaming Structured Validation

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

En `src/constrained_decoding.rs`.

Validacion incremental de output estructurado durante streaming.
A medida que llegan tokens, valida que el output parcial sigue siendo
compatible con el schema esperado. Puede abortar early si el output
diverge irrecuperablemente.

**Tipos clave**:
- `StreamingValidator`: Valida tokens incrementalmente contra un schema
- `ValidationState` enum: Valid, Partial(expected_next), Invalid(reason)
- `StreamingValidationConfig`: max_recovery_attempts, abort_on_invalid

**Algoritmo**:
1. Mantener stack de estado del parser JSON
2. Cada nuevo token: avanzar parser, verificar consistencia con schema
3. Si invalido: intentar recovery (ignorar whitespace, cerrar strings)
4. Si irrecuperable: emitir abort con contexto

**Tests**: ~10 (streaming valido, recovery, abort, nested structures)

---

## Fase 9 — Evolucion de Memoria

### 9.1 Enhanced Memory Consolidation Pipeline

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/advanced_memory.rs`.

Pipeline de consolidacion mejorado que extrae automaticamente hechos
semanticos (preferencias, patrones, reglas) de memoria episodica.
Va mas alla del MemoryConsolidator actual anadiendo LLM-based extraction
y confidence scoring.

**Tipos clave**:
- `SemanticFact`: Hecho extraido (subject, predicate, object, confidence, source_episodes)
- `FactExtractor` trait: Extrae hechos de episodios
- `PatternFactExtractor`: Basado en patrones/regex (ligero)
- `LlmFactExtractor`: Usa LLM para extraer hechos (calidad alta, costo alto)
- `FactStore`: Almacena hechos con dedup y merge de confidence
- `ConsolidationSchedule`: Configura frecuencia de consolidacion (cron, on-demand, threshold)

**Tests**: ~12 (extraccion, dedup, confidence merge, scheduling)

---

### 9.2 Temporal Memory Graphs

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/advanced_memory.rs`.

Representar relaciones temporales entre episodios como un grafo dirigido.
Permite queries como "que paso despues de X" o "cual fue la causa de Y".

**Tipos clave**:
- `TemporalEdge`: Relacion temporal entre episodios (Before, After, Causes, EnabledBy, CoOccurs)
- `TemporalGraph`: Grafo de episodios con aristas temporales
- `TemporalQuery`: Busqueda por relacion temporal ("que causo X", "que siguio a Y")
- `CausalChain`: Secuencia de episodios conectados causalmente

**Integracion**: Se construye automaticamente al agregar episodios si los
timestamps y contextos sugieren relacion causal.

**Tests**: ~10 (graph construction, temporal queries, causal chains)

---

### 9.3 Self-Evolving Procedures (MemRL-style)

**Prioridad**: BAJA | **Esfuerzo**: L | **Impacto**: Medio

Extender `src/advanced_memory.rs`.

Las proceduras en ProceduralStore evolucionan automaticamente basandose
en feedback: si una procedura falla consistentemente, su confidence baja
y eventualmente se reemplaza. Si una nueva estrategia emerge de episodios
recientes, se crea una nueva procedura. Inspirado en MemRL (ICLR 2026).

**Tipos clave**:
- `ProcedureEvolver`: Analiza feedback de proceduras y las actualiza
- `EvolutionConfig`: success_boost, failure_penalty, min_confidence_to_keep, auto_create_threshold
- `ProcedureFeedback`: procedure_id, outcome (Success/Failure), context
- `EvolutionReport`: procedures_updated, procedures_created, procedures_retired

**Algoritmo**:
1. Acumular feedback por procedura
2. Ajustar confidence: +success_boost por exito, -failure_penalty por fallo
3. Si confidence < min → retirar procedura
4. Analizar episodios recientes sin procedura → si patron emerge → crear nueva
5. Registrar cambios en EvolutionReport

**Tests**: ~10 (feedback, confidence adjustment, retirement, auto-creation)

---

## Resumen

| Fase | Items | Esfuerzo | Ficheros |
|------|-------|----------|----------|
| 1. Prompt Optimization | 4 | XL+L+M+M | prompt_signature.rs (ext) |
| 2. MCP v2 | 3 | L+L+S | mcp_protocol.rs (ext) |
| 3. Voice Agents | 3 | XL+L+M | voice_agent.rs (NUEVO) |
| 4. Media Generation | 3 | L+M+L | media_generation.rs (NUEVO) |
| 5. Distillation | 4 | L+M+M+M | distillation.rs (NUEVO) |
| 6. OTel GenAI | 3 | M+L+M | opentelemetry_integration.rs (ext) |
| 7. Agent Patterns | 4 | L+L+M+M | multi_agent.rs (ext), agent_definition.rs (NUEVO), event_workflow.rs (ext) |
| 8. Constrained Decoding | 3 | L+L+M | constrained_decoding.rs (NUEVO) |
| 9. Memory Evolution | 3 | M+M+L | advanced_memory.rs (ext) |
| **TOTAL** | **30** | | **5 nuevos + 7 extendidos** |

### Ficheros nuevos (5)

| Fichero | Feature Gate | Items |
|---------|-------------|-------|
| `src/voice_agent.rs` | `voice-agent` | 3.1, 3.2, 3.3 |
| `src/media_generation.rs` | `media-generation` | 4.1, 4.2, 4.3 |
| `src/distillation.rs` | `distillation` | 5.1, 5.2, 5.3, 5.4 |
| `src/agent_definition.rs` | siempre disponible | 7.3 |
| `src/constrained_decoding.rs` | `constrained-decoding` | 8.1, 8.2, 8.3 |

### Ficheros extendidos (7)

| Fichero | Items |
|---------|-------|
| `src/prompt_signature.rs` | 1.1, 1.2, 1.3, 1.4 |
| `src/mcp_protocol.rs` | 2.1, 2.2, 2.3 |
| `src/opentelemetry_integration.rs` | 6.1, 6.2, 6.3 |
| `src/multi_agent.rs` | 7.2, 7.4 |
| `src/event_workflow.rs` | 7.1 |
| `src/autonomous_loop.rs` | 7.1 |
| `src/advanced_memory.rs` | 9.1, 9.2, 9.3 |

### Estimacion de tests nuevos: ~270

**Total estimado tras v5**: ~3670 tests

---

## Tabla resumen completa

| # | Item | Prioridad | Esfuerzo | Impacto | Inspirado en | Estado |
|---|------|-----------|----------|---------|-------------|--------|
| 1.1 | GEPA Genetic Pareto Optimizer | CRITICA | XL | Muy Alto | DSPy GEPA | HECHO |
| 1.2 | MIPROv2 Multi-stage Optimizer | ALTA | L | Alto | DSPy MIPROv2 | HECHO |
| 1.3 | Prompt Assertions & Constraints | MEDIA | M | Medio | DSPy Assertions | HECHO |
| 1.4 | LM Adapters | MEDIA | M | Medio | DSPy LM Adapters | HECHO |
| 2.1 | Streamable HTTP Transport | CRITICA | L | Muy Alto | MCP v2 spec | HECHO |
| 2.2 | OAuth 2.1 + Dynamic Registration | ALTA | L | Alto | MCP v2 spec | HECHO |
| 2.3 | Tool Annotations | MEDIA | S | Medio | MCP v2 spec | HECHO |
| 3.1 | Bidirectional Audio Streaming | ALTA | XL | Muy Alto | OpenAI Realtime, Google ADK | HECHO |
| 3.2 | VAD + Interruption Handling | ALTA | L | Alto | OpenAI Agents SDK | HECHO |
| 3.3 | Conversation Turn Management | MEDIA | M | Medio | Google ADK | HECHO |
| 4.1 | Image Generation Providers | ALTA | L | Alto | Vercel AI SDK 6 | HECHO |
| 4.2 | Image Editing | MEDIA | M | Medio | Vercel AI SDK 6, Stability AI | HECHO |
| 4.3 | Video Generation Providers | MEDIA | L | Alto | Vercel AI SDK 6, Runway | HECHO |
| 5.1 | Trajectory Collector | ALTA | L | Muy Alto | OpenAI, NVIDIA Flywheel | HECHO |
| 5.2 | Trajectory Scorer & Filter | ALTA | M | Alto | NVIDIA Flywheel | HECHO |
| 5.3 | Distillation Dataset Builder | MEDIA | M | Alto | OpenAI distillation | HECHO |
| 5.4 | Data Flywheel Orchestrator | MEDIA | M | Alto | NVIDIA Blueprint | HECHO |
| 6.1 | GenAI Semantic Attributes | ALTA | M | Alto | OTel GenAI conventions | HECHO |
| 6.2 | Hierarchical Agent Tracing | ALTA | L | Alto | AG2, CrewAI | HECHO |
| 6.3 | Cost Attribution & Budget | MEDIA | M | Medio | LiteLLM | HECHO |
| 7.1 | Automatic State Persistence | ALTA | L | Alto | LangGraph, Pydantic AI | HECHO |
| 7.2 | Named Conversation Patterns | MEDIA | L | Medio | AG2/AutoGen | HECHO |
| 7.3 | Declarative Agent Definitions | MEDIA | M | Medio | Microsoft Agent Framework | HECHO |
| 7.4 | Agent Handoffs | MEDIA | M | Medio | OpenAI Agents SDK | HECHO |
| 8.1 | Grammar-Guided Generation | ALTA | L | Alto | XGrammar, llguidance | HECHO |
| 8.2 | JSON Schema → Grammar Compiler | ALTA | L | Alto | XGrammar, Outlines | HECHO |
| 8.3 | Streaming Structured Validation | MEDIA | M | Medio | Pydantic AI | HECHO |
| 9.1 | Enhanced Memory Consolidation | MEDIA | M | Medio | Academic research | HECHO |
| 9.2 | Temporal Memory Graphs | MEDIA | M | Medio | MemAgents (ICLR 2026) | HECHO |
| 9.3 | Self-Evolving Procedures | BAJA | L | Medio | MemRL (ICLR 2026) | HECHO |

## Resumen de Implementacion

- **Tests nuevos**: 506 (total: 3907)
- **Ficheros nuevos**: voice_agent.rs, media_generation.rs, distillation.rs, agent_definition.rs, constrained_decoding.rs
- **Ficheros extendidos**: prompt_signature.rs, mcp_protocol.rs, opentelemetry_integration.rs, multi_agent.rs, event_workflow.rs, advanced_memory.rs, guardrail_pipeline.rs
- **Features nuevos**: voice-agent, media-generation, distillation, constrained-decoding
- **0 warnings**, 0 failures
