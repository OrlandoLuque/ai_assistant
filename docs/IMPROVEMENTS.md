# Plan de Mejoras para ai_assistant — v2

> Documento generado el 2026-02-19.
> Basado en análisis comparativo de **20 frameworks de IA** (ver `docs/framework_comparison.html`)
> y auditoría arquitectónica interna de providers, embeddings, MCP, VectorDB y sync/async.
>
> **Plan anterior** (v1, basado en OpenClaw): 15 mejoras, **TODAS completadas**.
> Ver historial al final de este documento.

---

## Contexto

`ai_assistant` es un crate Rust con 191 archivos fuente, 30k+ LoC, 1786+ tests y 0 stubs.
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

**Estado actual**: `mcp_protocol.rs` implementa JSON-RPC básico, spec 2024-11-05.
McpServer maneja requests in-process. McpClient usa `ureq` sync. Sin streaming,
sin OAuth, sin server identity.

**Objetivo**: Compatibilidad completa con el ecosistema MCP (10,000+ servidores).

**Implementación**:
```toml
# Cargo.toml
rmcp = { version = "0.1", features = ["transport-sse", "transport-stdio"], optional = true }

[features]
mcp = ["dep:rmcp"]
```

**Tareas**:
- [ ] Añadir dependencia `rmcp` bajo feature flag `mcp`
- [ ] Crear `src/mcp_rmcp.rs` con wrapper sobre `rmcp` que exponga nuestra API actual
- [ ] Implementar `McpServerRmcp` compatible con Streamable HTTP + stdio transport
- [ ] Implementar `McpClientRmcp` con descubrimiento de herramientas remoto
- [ ] Migrar registros de tools/resources/prompts existentes a formato rmcp
- [ ] Mantener `mcp_protocol.rs` como fallback ligero (sin dep externa)
- [ ] Tests: conexión a un MCP server estándar (mock), listado de tools, ejecución
- [ ] Dual mode: sync wrapper + async nativo (rmcp es async por naturaleza)

**Riesgo**: rmcp es async-first (tokio). Requiere el feature `async-runtime`.
Mitigación: el fallback sin rmcp sigue siendo nuestro MCP sync actual.

**Prioridad**: CRÍTICA | **Esfuerzo**: M | **Impacto**: Muy Alto

---

### 1.2 Trait `EmbeddingProvider` + Implementaciones API

**Estado actual**: `embeddings.rs` solo tiene `LocalEmbedder` (TF-IDF con hashing trick).
`OllamaProvider` y `OpenAICompatibleProvider` tienen `generate_embeddings()` en el trait
`ProviderPlugin`, pero no hay un trait dedicado ni gestión de modelos de embedding.

**Objetivo**: Trait unificado para embeddings locales y remotos, con implementaciones completas.

**Implementación**:
```rust
// src/embedding_providers.rs (nuevo)

/// Trait para proveedores de embeddings (sync)
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> usize;
    fn max_tokens(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text])?.remove(0))
    }
}

/// Trait async (feature = "async-runtime")
#[cfg(feature = "async-runtime")]
#[async_trait]
pub trait AsyncEmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> usize;
    fn max_tokens(&self) -> usize;
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
```

**Implementaciones previstas**:

| Provider | Feature flag | API endpoint | Modelos |
|---|---|---|---|
| `LocalTfIdf` | (siempre) | N/A | TF-IDF local (wrapper de `LocalEmbedder`) |
| `OllamaEmbeddings` | (siempre) | `POST /api/embed` | nomic-embed-text, mxbai-embed-large, etc. |
| `OpenAIEmbeddings` | `openai-embeddings` | `POST /v1/embeddings` | text-embedding-3-small/large, ada-002 |
| `CohereEmbeddings` | `cohere-embeddings` | `POST /v2/embed` | embed-v4.0, embed-multilingual |
| `HuggingFaceEmbeddings` | `hf-embeddings` | `POST /pipeline/feature-extraction` | sentence-transformers, BGE, E5 |
| `VoyageEmbeddings` | `voyage-embeddings` | `POST /v1/embeddings` | voyage-3, voyage-code-3 |

**Integración con RAG**: `RagPipeline` y `RagAdvanced` recibirán un `Box<dyn EmbeddingProvider>`
en vez de depender solo de `LocalEmbedder`.

**Prioridad**: CRÍTICA | **Esfuerzo**: M | **Impacto**: Muy Alto

---

### 1.3 Adaptador Genérico OpenAI-Compatible

**Estado actual**: `OpenAICompatibleProvider` en `provider_plugins.rs` ya existe y soporta
`/v1/chat/completions` + `/v1/models` + `/v1/embeddings`. Pero requiere configuración manual
de `base_url`.

**Objetivo**: Presets para los proveedores OpenAI-compatible más populares, con auto-detección
de capacidades.

**Implementación**:
```rust
// Extender config.rs AiProvider enum
pub enum AiProvider {
    // ... existentes ...
    Groq,            // https://api.groq.com/openai
    Together,        // https://api.together.xyz
    Fireworks,       // https://api.fireworks.ai/inference
    DeepSeek,        // https://api.deepseek.com
    Perplexity,      // https://api.perplexity.ai
    OpenRouter,      // https://openrouter.ai/api
}
```

**Tareas**:
- [ ] Añadir variantes al enum `AiProvider` con sus URLs base por defecto
- [ ] Mapear cada nueva variante a `OpenAICompatibleProvider` en `provider_plugins.rs`
- [ ] Resolver API keys desde env vars (`GROQ_API_KEY`, `TOGETHER_API_KEY`, etc.)
- [ ] Auto-detección de capacidades por provider (tool calling, vision, embeddings)
- [ ] Context size lookup hardcodeado para modelos populares de cada provider
- [ ] Tests unitarios con mock HTTP para cada preset

**Riesgo**: Bajo. Reutiliza infraestructura existente al 95%.

**Prioridad**: CRÍTICA | **Esfuerzo**: S | **Impacto**: Muy Alto

---

### 1.4 Provider Nativo para Google Gemini

**Estado actual**: No soportado.

**Objetivo**: Soporte completo del tercer mayor proveedor de IA.

**Implementación**:
```rust
// Nuevo en config.rs
AiProvider::Gemini  // https://generativelanguage.googleapis.com

// Nuevo: src/gemini_provider.rs
pub struct GeminiProvider { api_key: String, base_url: String }
```

API de Gemini usa formato propio (`/v1beta/models/{model}:generateContent`), no OpenAI-compatible.

**Tareas**:
- [ ] Añadir `AiProvider::Gemini` al enum
- [ ] Implementar `GeminiProvider` con trait `ProviderPlugin`
- [ ] Soporte de `generateContent` (sync + async)
- [ ] Soporte de streaming via SSE
- [ ] Model listing via `GET /v1beta/models`
- [ ] Embeddings via `embedContent` endpoint
- [ ] API key desde `GEMINI_API_KEY` / `GOOGLE_API_KEY`
- [ ] Tests con mock HTTP

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

---

### 1.5 Provider Nativo para Mistral AI

**Estado actual**: No soportado. Aunque Mistral es OpenAI-compatible en chat completions,
tiene endpoints propios para embeddings y capacidades específicas (tool calling nativo, JSON mode).

**Tareas**:
- [ ] Añadir `AiProvider::Mistral` con URL `https://api.mistral.ai`
- [ ] Implementar como `OpenAICompatibleProvider` preset con capacidades específicas
- [ ] Embeddings via `/v1/embeddings` (mistral-embed)
- [ ] API key desde `MISTRAL_API_KEY`
- [ ] Tests con mock HTTP

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto

---

## Fase 2 — Expansión de Infraestructura

### 2.1 Vector DB Backends Adicionales

**Estado actual**: `VectorDb` trait con 3 implementaciones: `InMemoryVectorDb`,
`QdrantClient` (REST via ureq), `LanceVectorDb` (embedded).

**Objetivo**: Backends para los vector DBs más populares del ecosistema.

**Implementación**: Un feature flag por backend, cada uno trae sus deps.

```toml
[features]
pinecone = ["dep:reqwest", "dep:tokio"]     # REST API, async-only con sync bridge
weaviate = ["dep:reqwest", "dep:tokio"]     # REST + GraphQL
chroma = []                                  # REST via ureq (ligero)
elasticsearch = ["dep:reqwest", "dep:tokio"] # REST API
pgvector = ["dep:tokio-postgres"]           # PostgreSQL wire protocol
milvus = ["dep:reqwest", "dep:tokio"]       # REST API (v2)
```

**Para cada backend**:
- [ ] Implementar `VectorDb` trait (sync)
- [ ] Implementar `AsyncVectorDb` trait (async, donde aplique)
- [ ] Manejo de conexión, retry, health check
- [ ] Soporte de metadata filtering (traducción a query nativa)
- [ ] Tests con mock HTTP / embedded
- [ ] Documentación de configuración

**Orden de implementación** (por popularidad/demanda):
1. Chroma (ligero, ureq, popular en dev local)
2. Pinecone (líder cloud, async)
3. pgvector (popular en producción, SQL estándar)
4. Weaviate (GraphQL, hybrid search nativo)
5. Elasticsearch (enterprise, KNN + BM25)
6. Milvus (escala masiva)

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

---

### 2.2 Async Parity — `AsyncProviderPlugin` Trait

**Estado actual** (auditoría detallada):

| Componente | Sync | Async |
|---|---|---|
| `HttpClient` / `UreqClient` | Si | No |
| `AsyncHttpClient` / `ReqwestClient` | Bridge | Si |
| Funciones provider (`providers.rs`) | Si | No (excepto `async_providers.rs`) |
| `ProviderPlugin` trait + 5 impls | Si | No |
| `McpServer` / `McpClient` | Si | No |
| `VectorDb` trait + 3 impls | Si | No |
| `EmbeddingProvider` (nuevo) | Si | Si (diseñado así) |
| `embeddings.rs` (local TF-IDF) | Si | N/A (sin I/O) |

`async_providers.rs` tiene funciones async (`fetch_models_async`, `generate_response_async`),
pero el **streaming async es fake** (llama a non-streaming y emite un solo callback).
No existe un `AsyncProviderPlugin` trait.

**Objetivo**: Trait async paralelo al sync para todos los componentes con I/O.

**Tareas**:
- [ ] Crear `AsyncProviderPlugin` trait en `tools.rs`:
  ```rust
  #[cfg(feature = "async-runtime")]
  #[async_trait]
  pub trait AsyncProviderPlugin: Send + Sync {
      fn name(&self) -> &str;
      fn capabilities(&self) -> ProviderCapabilities;
      async fn is_available(&self) -> bool;
      async fn list_models(&self) -> Result<Vec<ModelInfo>>;
      async fn generate(&self, config: &AiConfig, messages: &[ChatMessage], system_prompt: Option<&str>) -> Result<String>;
      async fn generate_streaming(&self, config: &AiConfig, messages: &[ChatMessage], system_prompt: Option<&str>, tx: &tokio::sync::mpsc::Sender<AiResponse>) -> Result<()>;
      async fn generate_with_tools(&self, ...) -> Result<(String, Vec<ToolCall>)> { ... }
      async fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> { ... }
  }
  ```
- [ ] Implementar `AsyncProviderPlugin` para cada provider existente
- [ ] Fix del streaming async fake: implementar SSE parsing real con `reqwest::Response::bytes_stream()`
- [ ] Crear `AsyncVectorDb` trait paralelo a `VectorDb`
- [ ] Bridge sync→async: `block_on` wrapper (ya existe en `async_providers.rs`)
- [ ] Bridge async→sync: wrapper que ejecuta sync en `spawn_blocking`
- [ ] Tests async con `#[tokio::test]`

**Riesgo**: Medio. Duplicación de código significativa. Mitigación: macros para generar
ambas variantes desde una sola definición, o bien el trait sync como wrapper del async.

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

---

### 2.3 Formatos de Documento Adicionales

**Estado actual**: `DocumentParser` soporta PDF, EPUB, DOCX, ODT, HTML, TXT, feeds (RSS/Atom).
Monolítico (no basado en traits).

**Objetivo**: Expandir a formatos enterprise comunes.

**Tareas**:
- [ ] PowerPoint (PPTX): parsear con crate `zip` + XML directo (similar a DOCX)
- [ ] Excel (XLSX): usar crate `calamine` para lectura de hojas
- [ ] CSV: usar crate `csv` para parsing tabular
- [ ] Markdown: parser nativo (AST → texto plano)
- [ ] Considerar trait `DocumentLoader` para extensibilidad:
  ```rust
  pub trait DocumentLoader: Send + Sync {
      fn supported_formats(&self) -> &[DocumentFormat];
      fn parse(&self, content: &[u8], format: DocumentFormat) -> Result<ParsedDocument>;
  }
  ```
- [ ] Feature flags: `documents-office` (PPTX, XLSX), `documents-data` (CSV)

**Prioridad**: MEDIA-ALTA | **Esfuerzo**: M | **Impacto**: Medio-Alto

---

### 2.4 Structured Output / JSON Mode

**Estado actual**: Algunos providers (Ollama, LM Studio, OpenAICompatible) reportan
`json_mode: true` en capabilities. Pero no hay enforcement de schemas ni validación
de output contra un schema JSON esperado.

**Objetivo**: Generación con schema enforcement (como Instructor en Python, Vercel AI SDK
`generateObject`, o Pydantic AI `result_type`).

**Tareas**:
- [ ] Crear `src/structured_output.rs`:
  ```rust
  pub struct StructuredGeneration {
      pub schema: serde_json::Value,  // JSON Schema
      pub retry_on_invalid: bool,
      pub max_retries: u32,
  }
  ```
- [ ] Inyectar schema en system prompt cuando el provider no soporta JSON mode nativo
- [ ] Validar respuesta contra schema (usar crate `jsonschema` o validación manual)
- [ ] Retry automático con error feedback si la validación falla
- [ ] Integración con `ProviderPlugin::generate()` y `AiAssistant::send_message()`
- [ ] Tests con schemas simples y complejos (nested objects, enums, arrays)

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

---

## Fase 3 — Capacidades Avanzadas

### 3.1 Observabilidad / Tracing

**Estado actual**: `log` crate para logging estructurado (mejora v1 #11).
`events.rs` con `EventBus` y `AiEvent`. Sin métricas exportables ni traces.

**Objetivo**: Integración con estándares de observabilidad (OpenTelemetry, Prometheus).

**Tareas**:
- [ ] Feature flag `observability` con `tracing` crate (reemplazo progresivo de `log`)
- [ ] Spans para cada operación (provider call, embedding, RAG query, MCP call)
- [ ] Métricas: latencia por provider, tokens consumidos, cache hits, errores
- [ ] Exportador OpenTelemetry (feature `otel`) para dashboards externos
- [ ] Integration con nuestro `EventBus` existente (propagación automática)
- [ ] Callback de uso de tokens para cost tracking

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

---

### 3.2 LLM-as-Judge / Evaluación Avanzada

**Estado actual**: `evaluation.rs` tiene benchmarks, A/B testing con Welch's t-test,
hallucination detection, ensemble evaluation. Robusto estadísticamente.

**Objetivo**: Añadir evaluación basada en LLM (LLM-as-judge), similar a lo que ofrecen
LangSmith, RAGAS, o DSPy evaluators.

**Tareas**:
- [ ] `LlmJudge` struct que use un provider para evaluar respuestas:
  ```rust
  pub struct LlmJudge {
      provider: Box<dyn ProviderPlugin>,
      criteria: Vec<EvalCriterion>,
  }
  pub enum EvalCriterion { Relevance, Coherence, Faithfulness, Toxicity, Custom(String) }
  ```
- [ ] Evaluación de RAG: faithfulness (¿la respuesta se basa en los documentos?),
  relevance (¿los documentos son relevantes a la query?)
- [ ] Pairwise comparison (¿qué respuesta es mejor?)
- [ ] Batch evaluation con report
- [ ] Integración con `EvaluationRunner` existente

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

---

### 3.3 Tool Calling Unificado

**Estado actual**: `OpenAICompatibleProvider` es el único con `generate_with_tools()`.
`OllamaProvider` no lo implementa (aunque Ollama lo soporta desde v0.3).
El resto de providers ignoran tools.

**Objetivo**: Tool calling en todos los providers que lo soporten.

**Tareas**:
- [ ] Implementar `generate_with_tools()` en `OllamaProvider` (Ollama tool calling API)
- [ ] Implementar para `GeminiProvider` (function calling nativo de Gemini)
- [ ] Implementar para `Anthropic` (tool use nativo de Claude)
- [ ] Fallback para providers sin soporte nativo: inyectar tools como texto en el prompt,
  parsear llamadas a tools del output
- [ ] Estandarizar `ToolCall` / `ToolResult` format across providers
- [ ] Agentic loop: permitir encadenamiento tool call → result → continue

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

---

### 3.4 Prompt Templates / Prompt Management

**Estado actual**: System prompts como strings literales. `build_system_prompt()` y
`build_system_prompt_with_notes()` en `providers.rs` concatenan partes.

**Objetivo**: Sistema de templates reutilizables con variables, similar a LangChain
PromptTemplate o Semantic Kernel prompts.

**Tareas**:
- [ ] Crear `src/prompt_template.rs`:
  ```rust
  pub struct PromptTemplate {
      template: String,          // "You are {role}. Context: {context}"
      variables: HashSet<String>,
  }
  impl PromptTemplate {
      pub fn render(&self, vars: &HashMap<String, String>) -> Result<String>;
  }
  ```
- [ ] Templates composables (system + user + few-shot examples)
- [ ] Serialización/carga desde archivos (YAML, JSON, texto plano)
- [ ] Biblioteca de templates built-in (RAG, summarization, code generation, etc.)
- [ ] Integración con `AiAssistant::send_message()`

**Prioridad**: MEDIA-BAJA | **Esfuerzo**: S | **Impacto**: Medio

---

### 3.5 Guardrails v2 — Pipeline de Validación

**Estado actual**: `guardrails.rs` implementa ContentGuard, TopicGuard, PII detection,
Constitutional AI. Funcional pero invocado manualmente.

**Objetivo**: Pipeline automático de guardrails que se aplique before/after cada
llamada al modelo, configurable por sesión.

**Tareas**:
- [ ] `GuardrailPipeline` con stages pre-send y post-receive:
  ```rust
  pub struct GuardrailPipeline {
      pre_guards: Vec<Box<dyn Guard>>,
      post_guards: Vec<Box<dyn Guard>>,
  }
  pub trait Guard: Send + Sync {
      fn check(&self, content: &str) -> GuardResult;
  }
  pub enum GuardResult { Pass, Warn(String), Block(String) }
  ```
- [ ] Auto-aplicación en `AiAssistant::send_message()` (configurable on/off)
- [ ] Rate limiting guard (tokens/min, requests/min)
- [ ] Content length guard (max tokens en input/output)
- [ ] Custom regex guard para políticas específicas
- [ ] Logging de violaciones

**Prioridad**: MEDIA-BAJA | **Esfuerzo**: S-M | **Impacto**: Medio

---

## Fase 4 — Ecosistema y Comunidad

### 4.1 Preparación para crates.io

**Tareas**:
- [ ] Auditar `pub` exports en `lib.rs` (API pública limpia)
- [ ] Añadir `#[doc]` a todos los tipos públicos
- [ ] Semver: definir versión 0.1.0 inicial
- [ ] `Cargo.toml`: license, repository, description, keywords, categories
- [ ] `README.md` con ejemplos de uso, feature matrix, badges
- [ ] `CHANGELOG.md`
- [ ] Verificar que `cargo publish --dry-run` funciona

### 4.2 Documentación

**Tareas**:
- [ ] Rustdoc completo con ejemplos en cada módulo público
- [ ] Libro/guía (mdbook) con tutoriales paso a paso
- [ ] Diagrama de arquitectura
- [ ] Guía de migración entre versiones

### 4.3 Ejemplos y Templates

**Tareas**:
- [ ] `examples/` directorio con:
  - `basic_chat.rs` — conversación mínima
  - `rag_pipeline.rs` — RAG completo con embeddings
  - `multi_agent.rs` — sesión multi-agente
  - `mcp_server.rs` — servidor MCP con herramientas custom
  - `distributed_agents.rs` — agentes distribuidos

### 4.4 CI/CD

**Tareas**:
- [ ] GitHub Actions: `cargo check`, `cargo test`, `cargo clippy`, `cargo fmt`
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
| 1.1 | MCP upgrade (rmcp) | 1 | CRÍTICA | M | Muy Alto | PENDIENTE |
| 1.2 | EmbeddingProvider trait + APIs | 1 | CRÍTICA | M | Muy Alto | PENDIENTE |
| 1.3 | Adaptador genérico OpenAI-compatible | 1 | CRÍTICA | S | Muy Alto | PENDIENTE |
| 1.4 | Google Gemini provider | 1 | ALTA | M | Alto | PENDIENTE |
| 1.5 | Mistral AI provider | 1 | ALTA | S | Alto | PENDIENTE |
| 2.1 | Vector DB backends | 2 | ALTA | L | Alto | PENDIENTE |
| 2.2 | Async parity completa | 2 | ALTA | L | Alto | PENDIENTE |
| 2.3 | Formatos de documento | 2 | MEDIA-ALTA | M | Medio-Alto | PENDIENTE |
| 2.4 | Structured Output / JSON mode | 2 | MEDIA | M | Alto | PENDIENTE |
| 3.1 | Observabilidad / tracing | 3 | MEDIA | M | Medio | PENDIENTE |
| 3.2 | LLM-as-judge | 3 | MEDIA | M | Medio | PENDIENTE |
| 3.3 | Tool calling unificado | 3 | MEDIA | M | Alto | PENDIENTE |
| 3.4 | Prompt templates | 3 | MEDIA-BAJA | S | Medio | PENDIENTE |
| 3.5 | Guardrails v2 pipeline | 3 | MEDIA-BAJA | S-M | Medio | PENDIENTE |
| 4.1 | crates.io publication | 4 | BAJA | M | Alto (LP) | PENDIENTE |
| 4.2 | Documentación | 4 | BAJA | L | Alto (LP) | PENDIENTE |
| 4.3 | Ejemplos | 4 | BAJA | S | Medio | PENDIENTE |
| 4.4 | CI/CD | 4 | BAJA | M | Alto (LP) | PENDIENTE |
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
