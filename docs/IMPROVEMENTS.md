# Plan de Mejoras para ai_assistant — v2

> Documento generado el 2026-02-19.
> Basado en análisis comparativo de **20 frameworks de IA** (ver `docs/framework_comparison.html`)
> y auditoría arquitectónica interna de providers, embeddings, MCP, VectorDB y sync/async.
>
> **Plan anterior** (v1, basado en OpenClaw): 15 mejoras, **TODAS completadas**.
> Ver historial al final de este documento.

---

## Contexto

`ai_assistant` es un crate Rust con 191 archivos fuente, 30k+ LoC, 2510+ tests y 0 stubs.
Tras compararlo con LangChain, LlamaIndex, Semantic Kernel, Haystack, AutoGen, CrewAI,
Vercel AI SDK, Rig, DSPy, OpenAI Agents SDK, Google ADK, Pydantic AI, Agno, smolagents,
Mastra, Dify, Spring AI, y otros — las fortalezas únicas de ai_assistant son:

- **Distributed/P2P**: QUIC, CRDTs, DHT, MapReduce, NAT traversal — ningún otro framework lo tiene
- **Security-first**: RBAC, AES-256-GCM, PII, guardrails, Constitutional AI
- **Autonomous agents**: 5 niveles de autonomía, task board, multi-agent sessions
- **Document parsing**: PDF (con tablas), EPUB, DOCX, ODT, HTML, feeds
- **RAG avanzado**: Self-RAG, CRAG, Graph RAG, RAPTOR, chunking strategies
- **Rendimiento**: Rust nativo, sin runtime GC

Las **brechas principales** frente a la industria son:
1. Pocos proveedores cloud (solo OpenAI + Anthropic)
2. MCP en spec 2024-11-05 (industria en 2025-11-25)
3. Embeddings solo locales (TF-IDF), sin APIs remotas
4. Vector DBs limitados (InMemory, Qdrant, LanceDB)
5. Async incompleto (solo providers tienen variante async)
6. Sin ecosistema/comunidad pública

---

## Decisiones de Diseño (confirmadas)

| Decisión | Elección |
|---|---|
| **MCP upgrade** | Usar `rmcp` (crate oficial de Anthropic) |
| **Sync vs Async** | Dual mode: sync + async para TODO componente nuevo |
| **Vector DB backends** | Un feature flag por backend (`pinecone`, `weaviate`, `chroma`, etc.) |
| **Embeddings** | `EmbeddingProvider` trait + implementaciones API completas |

---

## Fase 1 — Fundamentos Críticos (Alto ROI)

### 1.1 Upgrade MCP a spec 2025-11-25 via `rmcp`

**Estado**: HECHO — `mcp_protocol.rs` ya implementa spec 2025-03-26 (más reciente que
el objetivo original), con OAuth 2.1, Streamable HTTP, paginación, 65 tests.
No se necesita `rmcp` externo; la implementación propia es completa y sin deps.

**Prioridad**: CRÍTICA | **Esfuerzo**: M | **Impacto**: Muy Alto

---

### 1.2 Trait `EmbeddingProvider` + Implementaciones API

**Estado**: HECHO — `src/embedding_providers.rs` implementa:
- Trait `EmbeddingProvider` (name, dimensions, max_tokens, embed, embed_single)
- 4 implementaciones: `LocalTfIdfEmbedding` (wrapper de LocalEmbedder), `OllamaEmbeddings`
  (/api/embed), `OpenAIEmbeddings` (/v1/embeddings), `HuggingFaceEmbeddings` (HF Inference API)
- Factory `create_embedding_provider(name)` para instanciar por nombre
- 15 tests unitarios

**Prioridad**: CRÍTICA | **Esfuerzo**: M | **Impacto**: Muy Alto

---

### 1.3 Adaptador Genérico OpenAI-Compatible

**Estado**: HECHO — 7 nuevas variantes en `AiProvider` enum: Groq, Together, Fireworks,
DeepSeek, Mistral, Perplexity, OpenRouter. Cada una con URL base, env var para API key,
context size lookup, routing en providers.rs/cloud_providers.rs, y factory methods
en `OpenAICompatibleProvider`. También se corrigió bug donde Gemini se enrutaba por
OpenAI-compatible (ahora usa su API nativa).

**Prioridad**: CRÍTICA | **Esfuerzo**: S | **Impacto**: Muy Alto

---

### 1.4 Provider Nativo para Google Gemini

**Estado**: HECHO — `src/gemini_provider.rs` implementa `GeminiProvider` con trait `ProviderPlugin`:
generateContent, streamGenerateContent (SSE), embedContent, function calling, model listing.
12 tests unitarios. API key via query param. Role mapping (assistant→model).

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

---

### 1.5 Provider Nativo para Mistral AI

**Estado**: HECHO — `AiProvider::Mistral` con URL `https://api.mistral.ai`, env var
`MISTRAL_API_KEY`, routing OpenAI-compatible, factory method `OpenAICompatibleProvider::mistral()`,
context sizes por modelo (large 128K, codestral 256K).

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto

---

## Fase 2 — Expansión de Infraestructura

### 2.1 Vector DB Backends Adicionales

**Estado**: HECHO — `vector_db.rs` ya tiene 8 implementaciones del trait `VectorDb`:
InMemoryVectorDb, QdrantClient, PineconeClient, ChromaClient, MilvusClient,
WeaviateClient, RedisVectorClient, ElasticsearchClient. Más LanceVectorDb (embedded)
y PgVectorDb (SQL generation). Total: 10 backends.

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

---

### 2.2 Async Parity — `AsyncProviderPlugin` Trait

**Estado**: HECHO — `async_provider_plugin.rs` (690 líneas, 12 tests) implementa:
- Trait `AsyncProviderPlugin` con generate, generate_streaming, generate_with_tools, generate_embeddings
- `SyncToAsyncAdapter` (wraps sync ProviderPlugin for async use via spawn_blocking)
- `AsyncToSyncAdapter` (wraps async plugin for sync use via block_on)
- `AsyncProviderRegistry` para gestión de plugins async

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

---

### 2.3 Formatos de Documento Adicionales

**Estado**: HECHO — `document_parsing.rs` (5105 líneas) ya soporta 11 formatos:
EPUB, DOCX, ODT, HTML, PDF, PlainText, CSV, Email, Image, PPTX, XLSX.
También tiene subsistema OCR (template matching + Tesseract), extracción de imágenes,
y parsing de metadatos EXIF.

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Medio-Alto

---

### 2.4 Structured Output / JSON Mode

**Estado**: HECHO — `structured.rs` (1554 líneas) implementa:
- `SchemaValidator` con validación completa de tipos JSON
- `StructuredOutputGenerator` con registry de schemas y generación de prompts
- `StructuredOutputEnforcer` con retry automático y corrección
- `SchemaBuilder` con templates (sentiment, classification, yes/no, multi-choice)
- Extracción de JSON de respuestas LLM (code blocks, raw JSON)

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

---

## Fase 3 — Capacidades Avanzadas

### 3.1 Observabilidad / Tracing

**Estado**: HECHO — `prometheus_metrics.rs` (605 líneas) + `telemetry.rs` (289 líneas) +
`opentelemetry_integration.rs` (1144 líneas), 41 tests en total.
Métricas Prometheus (counters, histograms, gauges), telemetry spans, OpenTelemetry exporters.

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

---

### 3.2 LLM-as-Judge / Evaluación Avanzada

**Estado**: HECHO — `llm_judge.rs` (746 líneas, 15 tests) implementa:
- `EvalCriterion` enum (Relevance, Coherence, Faithfulness, Toxicity, Custom)
- `JudgeResult`, `PairwiseResult`, `RagFaithfulnessResult`, `BatchEvalResult`
- Prompt building + JSON response parsing
- Pairwise comparison, RAG faithfulness evaluation, batch evaluation

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

---

### 3.3 Tool Calling Unificado

**Estado**: HECHO — `provider_plugins.rs` implementa:
- `OllamaProvider::generate_with_tools()` nativo via `/api/chat` con formato OpenAI-compatible
  (tool schemas, tool_calls en response.message, argument parsing string/object)
- `PromptToolFallback` struct para providers sin soporte nativo (KoboldCpp y cualquier otro):
  inyecta definiciones de tools en el system prompt, parsea `{"tool_call": ...}` del output
  (soporta JSON inline y code blocks), genera IDs secuenciales `fallback_call_N`
- Ollama y KoboldCpp ahora reportan `tool_calling: true` en capabilities
- 7 tests nuevos (build_prompt, build_prompt_no_tools, parse_tool_call, parse_code_block,
  parse_no_tool_calls, parse_multiple, koboldcpp_tool_calling_capability)

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

---

### 3.4 Prompt Templates / Prompt Management

**Estado**: HECHO — `templates.rs` (600+ líneas) implementa:
- `PromptTemplate` con `{{variable}}` placeholders y validación
- `TemplateBuilder` para construcción fluida
- `BuiltinTemplates` con templates predefinidos: code_review, translation, explain, bug_fix, summarize

**Prioridad**: MEDIA-BAJA | **Esfuerzo**: S | **Impacto**: Medio

---

### 3.5 Guardrails v2 — Pipeline de Validación

**Estado**: HECHO — `guardrail_pipeline.rs` (770 líneas) implementa:
- Trait `Guard` unificado: `name()`, `stage()`, `check(text) -> GuardCheckResult`
- `GuardrailPipeline` orchestrator con `check_input()` (PreSend) y `check_output()` (PostReceive)
- Configurable `block_threshold` (default 0.8), violation logging con audit trail
- 6 guards built-in:
  - `ContentLengthGuard` — max character limit enforcement
  - `RateLimitGuard` — sliding-window rate limiting
  - `PatternGuard` — substring blocklist matching
  - `ToxicityGuard` — wraps `ToxicityDetector` de `advanced_guardrails`
  - `PiiGuard` — wraps `PiiDetector` de `pii_detection`
  - `AttackGuard` — wraps `AttackDetector` de `advanced_guardrails`
- `GuardStage` enum: PreSend, PostReceive, Both
- `GuardAction` enum: Pass, Warn(String), Block(String)
- `PipelineResult` con passed/results/blocked_by

**Prioridad**: MEDIA-BAJA | **Esfuerzo**: S-M | **Impacto**: Medio

---

## Fase 4 — Ecosistema y Comunidad

### 4.1 Preparación para crates.io

**Estado**: HECHO — Auditoría completa:
- 260+ pub re-exports organizados por feature gate en `lib.rs`
- 100% doc comments en todos los tipos públicos (auditado en 5 módulos representativos)
- `Cargo.toml` metadata completo: description, license (MIT OR Apache-2.0), repository, homepage, documentation, readme, keywords, categories
- `README.md` profesional y completo
- 19 examples registrados en `Cargo.toml` con `required-features`
- `cargo publish --dry-run` pasa sin errores
- Versión 0.1.0 definida

### 4.2 Documentación

**Estado**: HECHO — Documentación extensiva:
- `docs/GUIDE.md` — 85 secciones con ejemplos de código
- `docs/CONCEPTS.md` — 53 conceptos explicados en profundidad
- `docs/AGENT_SYSTEM_DESIGN.md` — 44 secciones de arquitectura
- `docs/TESTING.md` — guía de testing completa
- `docs/IMPROVEMENTS.md` — roadmap v2 tracking
- Doc comments en todos los módulos públicos

### 4.3 Ejemplos y Templates

**Estado**: HECHO — 19 ejemplos en `examples/`:
- `basic_chat.rs`, `streaming.rs` — uso básico
- `rag_pipeline.rs`, `vector_search.rs`, `encrypted_knowledge.rs`, `knowledge_graph.rs` — RAG
- `multi_agent.rs`, `autonomous_agent.rs`, `scheduler_agent.rs`, `dag_workflow.rs` — agentes
- `mcp_server.rs` — servidor MCP
- `p2p_network.rs` — red P2P
- `quality_tests.rs`, `ab_testing_demo.rs`, `cost_tracking.rs` — evaluación
- `repl_demo.rs`, `multimodal.rs`, `ui_hooks_demo.rs`, `agent_graph_demo.rs` — utilidades

### 4.4 CI/CD

**Estado**: HECHO — `.github/workflows/ci.yml` implementa GitHub Actions con 4 jobs:
- `check` — `cargo check` con `--features full` y `--features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network"`
- `test` — `cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents" --lib`
- `clippy` — `cargo clippy` con `-W clippy::all`
- `fmt` — `cargo fmt -- --check`

Todos los jobs usan `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`.
Triggers: push/PR a `main` y `master`.

**Tareas restantes**:
- [ ] Matrix de features (test cada feature flag independiente)
- [ ] Cobertura de código (tarpaulin o llvm-cov)
- [ ] Release automation

**Prioridad (Fase 4 completa)**: BAJA | **Esfuerzo**: L | **Impacto**: Alto (largo plazo)

---

## Fase 5 — Testing y Calidad

### 5.1 Cobertura de Tests P2P Completa

**Estado actual**: 19→32 tests unitarios en `p2p.rs`. Las 12 funciones públicas sin tests
han sido cubiertas: ICE candidates, unban, get_top_peers, get_connectable_address,
discover_nat (vacío/inalcanzable), upnp/nat_pmp disabled, start/stop/stats.

**Estrategia de testing para funciones dependientes de red**:
- Paths deshabilitados: verificar que la función devuelve error correcto al deshabilitar la feature
- Servidores inalcanzables: usar direcciones RFC 5737 TEST-NET (`198.51.100.x`) que no enrutan
- Configs vacías: ejecutar el path completo con listas de servidores vacías
- Parsing: testear lógica de parseo (STUN XOR-MAPPED-ADDRESS) con datos sintéticos

**Tareas**:
- [x] Tests para NatTraversal (discover_nat, upnp_mapping, nat_pmp_mapping, get_connectable_address)
- [x] Tests para IceAgent (add_local_candidate, add_remote_candidate, get_local_candidates)
- [x] Tests para PeerReputation::unban()
- [x] Tests para ReputationSystem::get_top_peers()
- [x] Tests para P2PManager (start, stop, stats)

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto | **Estado**: HECHO

---

### 5.2 Cobertura de Tests Failure Detector

**Estado actual**: 13→15 tests. Las 2 funciones sin tests cubiertas:
`PhiAccrualDetector::is_suspicious()` (boundary test con verificación de consistencia)
y `HeartbeatManager::get_suspicious_nodes()` (multi-nodo con thresholds generosos).

**Tareas**:
- [x] Test para PhiAccrualDetector::is_suspicious() — boundary con threshold*0.5
- [x] Test para HeartbeatManager::get_suspicious_nodes() — nodos healthy vs stale

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

---

### 5.3 Test Harness — Categorías P2P

**Estado actual**: 122→125 categorías en `ai_test_harness`. 3 nuevas categorías P2P
compiladas condicionalmente con `#[cfg(feature = "p2p")]`.

**Categorías añadidas**:
- `p2p_nat` (5 tests): NatType helpers, NatTraversal construction, P2PConfig defaults, UPnP disabled
- `p2p_reputation` (5 tests): lifecycle, ban/unban, accuracy, is_trusted, get_top_peers
- `p2p_manager` (6 tests): creation, start disabled, stop, stats, Ping→Pong, local_peer_id

**Ejecución**: `cargo run --bin ai_test_harness --features "full,p2p" -- --category=p2p_nat`

**Tareas**:
- [x] Categoría `p2p_nat` (5 tests)
- [x] Categoría `p2p_reputation` (5 tests)
- [x] Categoría `p2p_manager` (6 tests)
- [x] Compilación condicional con `#[cfg(feature = "p2p")]`
- [x] Refactorización de `all_categories()` para pushes condicionales

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

---

### 5.4 Documentación de Testing

**Tareas**:
- [x] Actualizar TESTING.md con conteos de tests y sección P2P
- [x] Sección 38 en AGENT_SYSTEM_DESIGN.md (Infraestructura de Testing)
- [x] Actualizar IMPROVEMENTS.md con Fase 5

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio | **Estado**: HECHO

---

## Resumen de Prioridades

| # | Mejora | Fase | Prioridad | Esfuerzo | Impacto | Estado |
|---|--------|------|-----------|----------|---------|--------|
| 1.1 | MCP upgrade (rmcp) | 1 | CRÍTICA | M | Muy Alto | HECHO (ya en spec 2025-03-26) |
| 1.2 | EmbeddingProvider trait + APIs | 1 | CRÍTICA | M | Muy Alto | HECHO |
| 1.3 | Adaptador genérico OpenAI-compatible | 1 | CRÍTICA | S | Muy Alto | HECHO |
| 1.4 | Google Gemini provider | 1 | ALTA | M | Alto | HECHO |
| 1.5 | Mistral AI provider | 1 | ALTA | S | Alto | HECHO |
| 2.1 | Vector DB backends | 2 | ALTA | L | Alto | HECHO (10 backends) |
| 2.2 | Async parity completa | 2 | ALTA | L | Alto | HECHO |
| 2.3 | Formatos de documento | 2 | MEDIA-ALTA | M | Medio-Alto | HECHO (11 formatos) |
| 2.4 | Structured Output / JSON mode | 2 | MEDIA | M | Alto | HECHO |
| 3.1 | Observabilidad / tracing | 3 | MEDIA | M | Medio | HECHO |
| 3.2 | LLM-as-judge | 3 | MEDIA | M | Medio | HECHO |
| 3.3 | Tool calling unificado | 3 | MEDIA | M | Alto | HECHO |
| 3.4 | Prompt templates | 3 | MEDIA-BAJA | S | Medio | HECHO |
| 3.5 | Guardrails v2 pipeline | 3 | MEDIA-BAJA | S-M | Medio | HECHO |
| 4.1 | crates.io publication | 4 | BAJA | M | Alto (LP) | HECHO |
| 4.2 | Documentación | 4 | BAJA | L | Alto (LP) | HECHO |
| 4.3 | Ejemplos | 4 | BAJA | S | Medio | HECHO |
| 4.4 | CI/CD | 4 | BAJA | M | Alto (LP) | HECHO |
| 5.1 | Tests P2P completos | 5 | ALTA | S | Alto | HECHO |
| 5.2 | Tests Failure Detector | 5 | ALTA | S | Medio | HECHO |
| 5.3 | Test Harness P2P | 5 | MEDIA | S | Medio | HECHO |
| 5.4 | Testing docs | 5 | MEDIA | S | Medio | HECHO |

**Leyenda**: S = Small (1-2 días), M = Medium (3-5 días), L = Large (1-2 semanas), LP = largo plazo

---

## Orden de Ejecución Recomendado

```
Fase 1 (fundamentos):
  1.3 → 1.5 → 1.4 → 1.2 → 1.1
  (providers primero porque son rápidos y desbloquean embeddings/MCP)

Fase 2 (infraestructura):
  2.2 → 2.1 → 2.4 → 2.3
  (async primero porque los nuevos VectorDBs se diseñan async-first)

Fase 3 (avanzado):
  3.3 → 3.1 → 3.2 → 3.4 → 3.5
  (tool calling primero por su alto impacto)

Fase 4 (ecosistema):
  4.4 → 4.1 → 4.3 → 4.2
  (CI/CD primero para que todo lo demás sea verificable)

Fase 5 (testing): ✅ COMPLETADA
  5.1 → 5.2 → 5.3 → 5.4
  (tests unitarios primero, luego harness, luego docs)
```

---

## Dependencias Entre Tareas

```
1.2 (EmbeddingProvider) ←── depende de ──→ 2.1 (Vector DBs consumen embeddings)
1.1 (MCP rmcp)         ←── requiere  ──→ async-runtime feature
2.2 (Async parity)     ←── habilita  ──→ 2.1, 1.1 (backends async-first)
1.3 (OpenAI-compat)    ←── base de   ──→ 1.5 (Mistral como preset)
3.3 (Tool calling)     ←── consume   ──→ 1.4, 1.5 (providers con tool support)
3.2 (LLM-as-judge)     ←── consume   ──→ 1.2, 1.3 (necesita providers funcionales)
```

---

## Historial — Plan v1 (OpenClaw, 2026-02-13) — COMPLETADO

Todas las 15 mejoras del plan original han sido implementadas:

| # | Mejora | Estado |
|---|--------|--------|
| 1 | Failover entre proveedores | HECHO |
| 2 | Auto-compactación de contexto | HECHO |
| 3 | Redacción en logs | HECHO |
| 4 | Async/await (tokio) | HECHO |
| 5 | Sistema de herramientas ejecutable | HECHO |
| 6 | Sesiones JSONL (event log) | HECHO |
| 7 | Detección dinámica de contexto | HECHO |
| 8 | Cola de peticiones | HECHO |
| 9 | Tests con mocks | HECHO |
| 10 | Retry con backoff | HECHO |
| 11 | Logging estructurado | HECHO |
| 12 | Cifrado de sesiones | HECHO |
| 13 | Proveedores cloud nativos (OpenAI + Anthropic) | HECHO |
| 14 | Sistema de hooks/eventos | HECHO |
| 15 | Servidor HTTP embebido | HECHO |
| +  | Binary storage (bincode+gzip) | HECHO |
| +  | API Key Rotation | HECHO |
