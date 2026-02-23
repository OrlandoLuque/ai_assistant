# Plan de Mejoras para ai_assistant — v7

> **Estado: 34/34 items COMPLETE — 4687 tests, 0 failures, 0 clippy warnings**

> Documento generado el 2026-02-23.
> Basado en completitud de v1 (39/39), v2 (22/22), v3 (21/21), v4 (38/38), v5 (30/30), v6 (34/34) con 4449 tests, 0 failures.
> 240+ source files, ~258k LoC.
>
> **Planes anteriores**: v1 (providers, embeddings, MCP, documents, guardrails),
> v2 (async parity, vector DBs, evaluation, testing),
> v3 (containers, document pipeline, speech/audio, CI/CD maturity),
> v4 (workflows, prompt signatures, A2A, advanced memory, online eval, streaming guardrails),
> v5 (GEPA/MIPROv2, MCP v2, voice agents, media gen, distillation, OTel GenAI,
>     durable execution, constrained decoding, memory evolution),
> v6 (MCP spec completeness, remote MCP client, HITL, SIMBA, Memory OS, discourse RAG,
>     agent eval, red teaming, MCTS planning, WebRTC voice, multi-backend sandbox, devtools).

---

## Contexto

Tras v6, ai_assistant tiene 4449 tests, 240+ source files, ~258k LoC y cobertura
total en: LLM providers, RAG 5 niveles, multi-agente, distribuido P2P/QUIC,
seguridad, streaming, MCP (server + client + elicitation + batching), voice (HTTP +
WebRTC), media generation, distillation, constrained decoding, HITL, MCTS planning,
agent eval, red teaming y devtools.

Sin embargo, un analisis de madurez para produccion revela **brechas criticas**
en 10 areas donde el framework simula comportamiento en lugar de ejecutar logica real:

1. **MCP client HTTP simulado** — `RemoteMcpClient` en mcp_client.rs y
   `StreamableHttpTransport` en mcp_protocol.rs usan stubs que devuelven
   respuestas fijas en lugar de hacer HTTP real con JSON-RPC 2.0. Las empresas
   necesitan conectar con servidores MCP reales (VS Code, Cursor, etc.).

2. **Media generation sin HTTP real** — DALL-E 3, Stable Diffusion y Replicate
   en media_generation.rs simulan las llamadas API. Sin HTTP real, la feature
   es inutil para usuarios.

3. **Provider fallback no integrado** — FallbackChain y CircuitBreaker existen
   como structs aislados pero no estan integrados en `generate_response()`. Si
   OpenAI falla, el usuario no obtiene fallback automatico.

4. **API keys en texto plano** — Los providers almacenan API keys como `String`.
   No hay zeroize-on-drop, ni resolucion desde variables de entorno/archivos,
   ni auditoria de acceso a credenciales.

5. **OpenTelemetry simulado** — OtelGenAiExporter en otel_genai.rs genera spans
   pero no los exporta via OTLP HTTP. Sin exportacion real, la observabilidad
   queda en el vacio.

6. **Structured output ingenuo** — Todas las extracciones estructuradas usan
   regex sobre texto libre. OpenAI y Anthropic ofrecen structured output nativo
   (json_schema, tool_use) que garantiza JSON valido.

7. **Sin router de modelos** — No hay seleccion automatica de modelo por
   tarea/costo/calidad. El usuario debe elegir manualmente que modelo usar
   para cada peticion.

8. **Context window sin gestion activa** — No hay compactacion automatica de
   conversaciones largas ni presupuesto de tokens por segmento. Las
   conversaciones largas simplemente se truncan.

9. **Cloud connectors simulados** — S3 signing en cloud_connectors usa firmas
   ficticias. Web search devuelve resultados hardcodeados. Inutiles en produccion.

10. **Tests sin mock HTTP server** — Los tests de providers dependen de APIs
    externas o stubs internos. Falta un MockHttpServer local para CI fiable
    sin API keys.

---

## Decisiones de Diseno (propuestas)

| Decision | Eleccion |
|---|---|
| **MCP client HTTP real** | Extender `src/mcp_client.rs` — ureq JSON-RPC 2.0 |
| **MCP OAuth 2.1 real** | Extender `src/mcp_protocol.rs` — CSPRNG PKCE + real HTTP |
| **Streamable HTTP real** | Extender `src/mcp_protocol.rs` — POST + JSON/SSE parsing |
| **MCP Sampling** | Extender `src/mcp_protocol.rs` — sampling/createMessage |
| **Media gen HTTP real** | Extender `src/media_generation.rs` — ureq POST a APIs reales |
| **Fallback chain wired** | Extender `src/providers.rs` + `src/assistant.rs` — FallbackChain en core flow |
| **Connection pooling** | Extender `src/providers.rs` — ConnectionPool reusable |
| **Health checker** | Extender `src/providers.rs` — periodic health, skip Unhealthy |
| **Rate limiting** | Extender `src/providers.rs` — parse x-ratelimit headers |
| **SecureString** | Nuevo `src/secure_config.rs` — feature `security` existente |
| **Credential resolver** | Extender `src/secure_config.rs` — EnvVar/File/Callback chain |
| **Audit auto-wire** | Extender `src/providers.rs` + `src/access_control.rs` — auto-log LLM calls |
| **Config validation** | Extender `src/secure_config.rs` — ConfigFile::validate() |
| **OTLP exporter** | Extender `src/otel_genai.rs` — real HTTP to /v1/traces |
| **GenAI conventions** | Extender `src/otel_genai.rs` — gen_ai.* attributes |
| **Prometheus endpoint** | Extender `src/otel_genai.rs` — /metrics text format |
| **Structured output** | Extender `src/providers.rs` — native json_schema / tool_use |
| **Smart router** | Nuevo `src/smart_router.rs` — feature `core` existente |
| **Prompt cache** | Extender `src/smart_router.rs` — CacheablePrompt |
| **Response cache** | Extender `src/smart_router.rs` — ResponseCache middleware |
| **Cost tracking** | Extender `src/smart_router.rs` — CostTracker per provider |
| **Context compiler** | Extender `src/context_composer.rs` — token budget + eviction |
| **Auto-compaction** | Extender `src/context_composer.rs` — summarize old messages |
| **Tool schema lazy-loading** | Extender `src/context_composer.rs` — TF-IDF ToolSearchIndex |
| **S3 SigV4 real** | Extender `src/cloud_connectors.rs` — real AWS SigV4 signing |
| **Web search real** | Extender `src/cloud_connectors.rs` — DuckDuckGo/Brave HTML parsing |
| **Web crawler** | Extender `src/cloud_connectors.rs` — BFS + robots.txt |
| **MockHttpServer** | Nuevo `src/mock_http.rs` — feature `eval` existente |
| **Integration test suite** | Extender `src/mock_http.rs` — E2E scenarios |
| **Feature flag validation** | Macro en `src/lib.rs` — compile-time checks |

---

## Fase 1 — HTTP Real: MCP Client + OAuth

### 1.1 RemoteMcpClient HTTP Real

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/mcp_client.rs`.

Reemplazar los stubs de `RemoteMcpClient` con HTTP real via ureq. Los metodos
`connect()`, `call_tool()`, `read_resource()`, `fetch_tools()` y `fetch_resources()`
deben enviar JSON-RPC 2.0 requests reales por HTTP POST al servidor MCP remoto,
parsear las respuestas JSON-RPC y manejar errores de red, timeout y reintentos.

**Tipos clave**:
- `JsonRpcRequest`: Struct serializable con jsonrpc, method, params, id
- `JsonRpcResponse`: Struct con result/error, correlacion por id
- `HttpTransportLayer`: Capa de transporte con ureq::Agent, base_url, headers
- `McpConnectionState`: Enum (Disconnected, Connecting, Connected, Reconnecting)
- `RetryPolicy`: max_retries, backoff_ms, retry_on (timeout, 5xx, connection_reset)

**Archivo**: `src/mcp_client.rs` (ext)

**Tests**: ~16

**Estado**: HECHO — Real HTTP via ureq, JSON-RPC 2.0 correlation, 964 new lines in mcp_client.rs

---

### 1.2 MCP OAuth 2.1 Real

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/mcp_protocol.rs`.

Reemplazar los stubs de OAuth 2.1 con flujo real: `exchange_code()` hace POST
real a token endpoint, `refresh_token()` renueva tokens expirados, `discover()`
lee `.well-known/oauth-authorization-server`, `register()` hace Dynamic Client
Registration (RFC 7591). PKCE usa CSPRNG real (ring o getrandom) para
code_verifier de 43 caracteres, SHA-256 para code_challenge.

**Tipos clave**:
- `OAuthTokenEndpoint`: POST real con ureq a token endpoint
- `OAuthDiscovery`: GET a .well-known, parseo de authorization_endpoint, token_endpoint
- `PkceChallenge`: code_verifier (CSPRNG 32 bytes → base64url), code_challenge (SHA-256)
- `DynamicClientRegistration`: POST a registration_endpoint con client_metadata
- `TokenRefresher`: Background refresh antes de expiracion (token_lifetime * 0.8)

**Archivo**: `src/mcp_protocol.rs` (ext)

**Tests**: ~14

**Estado**: HECHO — Real PKCE with CSPRNG, SHA-256 code_challenge, token exchange via ureq

---

### 1.3 StreamableHttpTransport Real

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/mcp_protocol.rs`.

Reemplazar el stub de `send_request()` con POST HTTP real. Detectar Content-Type
de respuesta: si `application/json` parsear como JSON-RPC response unica, si
`text/event-stream` parsear como SSE con lineas `data:` y eventos `event:`.
Manejar Mcp-Session-Id header para correlacion de sesion.

**Tipos clave**:
- `HttpPostTransport`: POST con ureq, Content-Type detection
- `SseLineParser`: Parser de lineas SSE (event, data, id, retry)
- `SessionIdTracker`: Almacena y reenvia Mcp-Session-Id en requests subsiguientes
- `TransportError`: ConnectionFailed, Timeout, InvalidResponse, SessionExpired

**Archivo**: `src/mcp_protocol.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — Content-Type detection (JSON vs SSE), real HTTP POST with session tracking

---

### 1.4 MCP Sampling

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/mcp_protocol.rs`.

Implementar `sampling/createMessage`: permite que un servidor MCP solicite al
cliente que haga una llamada LLM (el servidor no tiene acceso directo al modelo).
El cliente recibe la request, ejecuta la generacion con su provider configurado,
y devuelve el resultado al servidor. Incluye human-in-the-loop opcional.

**Tipos clave**:
- `SamplingRequest`: messages, modelPreferences, systemPrompt, maxTokens
- `SamplingResponse`: model, role, content (text/image), stopReason
- `SamplingHandler` trait: `handle_sampling(request) -> SamplingResponse`
- `ModelPreferences`: hints (model names), costPriority, speedPriority, intelligencePriority
- `SamplingApprovalPolicy`: AutoApprove, AlwaysAsk, CostThreshold(f64)

**Archivo**: `src/mcp_protocol.rs` (ext)

**Tests**: ~12

**Estado**: HECHO — SamplingHandler trait, SamplingRequest/Response, ModelPreferences

---

## Fase 2 — HTTP Real: Media Generation

### 2.1 DALL-E 3 Real HTTP

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/media_generation.rs`.

Reemplazar el stub de DALL-E 3 con POST real a `https://api.openai.com/v1/images/generations`.
Enviar model, prompt, size, quality, n como JSON body. Parsear response con
`data[].url` o `data[].b64_json`. Manejar rate limits (429), content policy
violations (400), y billing errors (402).

**Tipos clave**:
- `DallE3HttpClient`: POST real con ureq, API key en Authorization header
- `DallE3Request`: model ("dall-e-3"), prompt, size, quality, response_format
- `DallE3Response`: created, data (url o b64_json)
- `ImageGenerationError`: RateLimited(retry_after), ContentPolicy(reason), BillingError

**Archivo**: `src/media_generation.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — Real POST to OpenAI API, b64_json/url response parsing, error handling

---

### 2.2 Stable Diffusion Real HTTP

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/media_generation.rs`.

Reemplazar el stub de Stable Diffusion con POST real a Stability AI REST API
(`https://api.stability.ai/v2beta/stable-image/generate/core`). Enviar prompt,
aspect_ratio, output_format como multipart/form-data. Parsear response binaria
(image bytes) con Content-Type detection.

**Tipos clave**:
- `StabilityHttpClient`: POST multipart con ureq, API key en Authorization header
- `StabilityRequest`: prompt, negative_prompt, aspect_ratio, output_format, seed
- `StabilityResponse`: image bytes + finish_reason + seed
- `StabilityError`: ContentFiltered, InsufficientCredits, InvalidParams

**Archivo**: `src/media_generation.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — Real POST to Stability AI, multipart form, negative prompt support

---

### 2.3 Replicate Real HTTP

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/media_generation.rs`.

Reemplazar el stub de Replicate con flujo asincronico real: POST a
`https://api.replicate.com/v1/predictions` para crear prediccion, polling de
GET `predictions/{id}` hasta status `succeeded`/`failed`, descarga del output
desde URL generada. Manejar cold starts (status `starting`) y timeouts.

**Tipos clave**:
- `ReplicateHttpClient`: POST/GET con ureq, API token en Authorization header
- `PredictionRequest`: version (model hash), input (HashMap), webhook (opcional)
- `PredictionStatus`: Starting, Processing, Succeeded, Failed, Canceled
- `PredictionPoller`: Poll con backoff exponencial hasta terminal status
- `ReplicateError`: ModelNotFound, ColdStartTimeout, PredictionFailed(logs)

**Archivo**: `src/media_generation.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — Real POST + polling with exponential backoff until terminal status

---

## Fase 3 — Wire Provider Reliability Infrastructure

### 3.1 Fallback Chain in generate_response

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/providers.rs` y `src/assistant.rs`.

Integrar `FallbackChain` y `CircuitBreaker` (ya existentes como structs) en el
flujo de `generate_response()`. Cuando el provider primario falla (timeout, 5xx,
rate limit), el sistema intenta automaticamente el siguiente provider en la
cadena. El CircuitBreaker abre tras N fallos consecutivos, evitando hammering
al provider caido. La cadena se configura declarativamente.

**Tipos clave**:
- `FallbackPipeline`: Ejecuta providers en orden con circuit breaker per-provider
- `FallbackConfig`: providers (Vec ordenado), max_retries_per_provider, total_timeout
- `CircuitBreakerState`: Closed (normal), Open (skip), HalfOpen (probe)
- `FailureClassifier`: Clasifica errores como Retryable o Fatal
- `FallbackEvent`: ProviderFailed, FallbackTriggered, CircuitOpened — para logging

**Archivos**: `src/providers.rs` (ext), `src/assistant.rs` (ext)

**Tests**: ~16

**Estado**: HECHO — ResilientProviderRegistry with FallbackChain + CircuitBreaker in providers.rs

---

### 3.2 Connection Pooling in Providers

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/providers.rs`.

Reemplazar la creacion ad-hoc de ureq agents por un `ConnectionPool` centralizado.
Cada host obtiene un ureq::Agent reutilizable con keep-alive, timeout y limits
configurados. Reduce latencia de conexion (~100ms por TLS handshake evitado)
y evita file descriptor exhaustion bajo carga.

**Tipos clave**:
- `ConnectionPool`: HashMap<host, ureq::Agent> con max_idle_per_host
- `PoolConfig`: max_connections_per_host, idle_timeout_secs, connect_timeout_ms
- `PooledRequest`: Wrapper que obtiene agent del pool y lo devuelve al terminar
- `PoolStats`: active_connections, idle_connections, total_requests, cache_hits

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — ConnectionPoolHandle with max_idle, idle_timeout, reuse tracking

---

### 3.3 Health Checker Integrated

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/providers.rs`.

Health checker periodico que prueba cada provider configurado y mantiene su
estado de salud. Los providers marcados como `Unhealthy` se saltan automaticamente
en la cadena de fallback. Los providers `Degraded` (latencia alta) se usan solo
como ultimo recurso.

**Tipos clave**:
- `HealthChecker`: Ejecuta checks periodicos (configurable interval)
- `HealthStatus`: Healthy, Degraded(latency_ms), Unhealthy(last_error)
- `HealthCheckConfig`: interval_secs, timeout_ms, unhealthy_after_n_failures
- `ProviderHealthMap`: HashMap<provider_id, HealthStatus> thread-safe
- `HealthCheckProbe`: Trait con metodo `probe() -> HealthStatus` por provider

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~12

**Estado**: HECHO — ProviderHealthStatus enum, periodic health probing, auto-skip Unhealthy

---

### 3.4 Per-Provider Rate Limiting

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/providers.rs`.

Parsear headers de rate limit de cada provider (`x-ratelimit-remaining`,
`x-ratelimit-reset`, `retry-after`) y auto-throttle antes de alcanzar el limite.
Implementar token bucket algorithm para suavizar rafagas. Cuando se acerca al
limite, queue requests o trigger fallback.

**Tipos clave**:
- `RateLimitTracker`: Parsea headers de respuesta, actualiza estado
- `TokenBucket`: Algoritmo de rate limiting con fill_rate y bucket_size
- `RateLimitConfig`: per-provider limits, burst_allowance, queue_when_limited
- `RateLimitHeaders`: Struct para x-ratelimit-limit, remaining, reset, retry-after
- `ThrottleDecision`: Proceed, Wait(duration), Fallback

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — RateLimitHeaders parsing, ProviderRateState, token bucket, ThrottleDecision

---

## Fase 4 — Secrets Management + Security

### 4.1 SecureString Wrapper

**Prioridad**: CRITICA | **Esfuerzo**: M | **Impacto**: Muy Alto

Nuevo archivo: `src/secure_config.rs` (feature `security`).

Wrapper para strings sensibles (API keys, tokens, passwords) que implementa
Zeroize on Drop (sobreescribe memoria al liberar), Debug que muestra "***",
y comparacion en tiempo constante para evitar timing attacks. Reemplaza String
en todos los campos de credenciales de providers.

**Tipos clave**:
- `SecureString`: Wrapper sobre Vec<u8> con Zeroize + Drop
- `SecureStringDebug`: Implementacion de Debug/Display que muestra "***"
- `ConstantTimeEq`: PartialEq en tiempo constante (XOR byte-a-byte)
- `SecureStringBuilder`: Builder para construir SecureString sin copias intermedias
- `Redactor`: Utility para redactar SecureStrings en logs y error messages

**Archivo**: `src/secure_config.rs` (new)

**Tests**: ~14

**Estado**: HECHO — SecureString with Zeroize+Drop, Debug masking, constant-time PartialEq

---

### 4.2 Credential Resolver Chain

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/secure_config.rs`.

Cadena de resolucion de credenciales: intenta resolver una credencial primero
desde variable de entorno, luego desde archivo (ej: ~/.config/ai_assistant/keys.toml),
y finalmente via callback (para integraciones con vaults externos como HashiCorp
Vault o AWS Secrets Manager).

**Tipos clave**:
- `CredentialResolver` trait: `resolve(key_name) -> Option<SecureString>`
- `EnvVarResolver`: Lee de std::env::var con prefijo configurable (AI_ASSISTANT_)
- `FileResolver`: Lee de TOML/JSON config file con permisos verificados
- `CallbackResolver`: Delega a callback `Box<dyn Fn(&str) -> Option<SecureString>>`
- `ResolverChain`: Vec de resolvers, intenta en orden, devuelve primer exito
- `CredentialCache`: Cache en memoria con TTL y auto-refresh

**Archivo**: `src/secure_config.rs` (ext)

**Tests**: ~12

**Estado**: HECHO — CredentialResolver chain: EnvVar→File→Callback→Static, AI_ASSISTANT_ prefix

---

### 4.3 Audit Auto-Wire in Providers

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto

Extender `src/providers.rs` y `src/access_control.rs`.

Auto-logging de todas las llamadas LLM a `AuditLogger`: timestamp, provider,
model, tokens_in, tokens_out, latency_ms, success/error, user_id. Se integra
transparentemente en `generate_response()` sin que el usuario tenga que hacer
nada. Respeta el sistema RBAC existente para filtrar quien puede ver los logs.

**Tipos clave**:
- `LlmAuditEntry`: provider, model, tokens_in, tokens_out, latency_ms, status
- `AuditMiddleware`: Intercepta llamadas LLM y registra en AuditLogger
- `AuditConfig`: enabled, log_prompts (default false por privacidad), log_responses
- `AuditFilter`: Filtrar por provider, modelo, user, date range

**Archivos**: `src/providers.rs` (ext), `src/access_control.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — AuditedProvider with Arc<Mutex<Vec<AuditEntry>>>, summary aggregation

---

### 4.4 Config Validation

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/secure_config.rs`.

Metodo `ConfigFile::validate()` que verifica la configuracion completa antes
de iniciar: range checks para valores numericos (timeouts > 0, ports 1-65535),
verificacion de que URLs son parseables, comprobacion de que API keys no estan
vacias, y sugerencias de remediacion para cada error.

**Tipos clave**:
- `ConfigValidator`: Ejecuta todas las validaciones, devuelve Vec<ValidationError>
- `ValidationError`: field_path, value, error_type, remediation_hint
- `ValidationRule`: Trait para reglas custom (RangeCheck, UrlValid, NonEmpty, RegexMatch)
- `ConfigReport`: Resumen con errors, warnings, info — pretty-printable

**Archivo**: `src/secure_config.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — ConfigFile::validate_detailed() with ConfigValidationError, range/URL/field checks

---

## Fase 5 — OpenTelemetry Export Real

### 5.1 OTLP HTTP Exporter

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/otel_genai.rs`.

Exportar spans reales via OTLP/HTTP a `/v1/traces`. Serializar spans como
protobuf-compatible JSON (OTLP JSON encoding), batch por tiempo/cantidad,
enviar con ureq POST. Manejar retry con backoff para errores transitorios.
Sin dependencia de crate `opentelemetry` — implementacion propia ligera.

**Tipos clave**:
- `OtlpHttpExporter`: Exporta batches de spans via HTTP POST
- `OtlpConfig`: endpoint_url, headers (auth), batch_size, flush_interval_ms
- `SpanBatcher`: Acumula spans, flush por tamano o timeout
- `OtlpPayload`: Serializacion JSON compatible con OTLP spec
- `ExportResult`: Success(spans_exported), PartialFailure(rejected), Error(reason)

**Archivo**: `src/otel_genai.rs` (ext)

**Tests**: ~14

**Estado**: HECHO — OtlpHttpExporter with batch flush, OTLP JSON format, ureq POST to /v1/traces

---

### 5.2 GenAI Semantic Conventions

**Prioridad**: ALTA | **Esfuerzo**: S | **Impacto**: Alto

Extender `src/otel_genai.rs`.

Implementar las convenciones semanticas GenAI de OpenTelemetry: atributos
estandar `gen_ai.system`, `gen_ai.request.model`, `gen_ai.request.max_tokens`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`.
Permite que dashboards de observabilidad (Grafana, Datadog) reconozcan spans de LLM.

**Tipos clave**:
- `GenAiAttributes`: Struct con todos los atributos GenAI estandar
- `GenAiSpanBuilder`: Builder que genera span con convenciones correctas
- `GenAiEventBuilder`: Eventos gen_ai.content.prompt y gen_ai.content.completion
- `AttributeMapper`: Mapea nuestras metricas internas a atributos GenAI estandar

**Archivo**: `src/otel_genai.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — GenAiConventions constants, GenAiSpanBuilder with gen_ai.* attributes

---

### 5.3 Prometheus Metrics Endpoint

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Medio

Extender `src/otel_genai.rs`.

Endpoint HTTP `/metrics` que expone metricas en formato Prometheus text exposition.
Metricas: `ai_requests_total` (counter por provider/model/status),
`ai_request_duration_seconds` (histogram), `ai_tokens_total` (counter input/output),
`ai_errors_total` (counter por tipo de error). Compatible con Prometheus scraping.

**Tipos clave**:
- `PrometheusExporter`: Genera texto Prometheus desde metricas acumuladas
- `MetricRegistry`: Registro centralizado de counters, gauges, histograms
- `Counter`: Valor monotonicamente creciente con labels
- `Histogram`: Distribucion de valores con buckets configurables
- `MetricsEndpoint`: Handler HTTP para /metrics (integra con http server existente)

**Archivo**: `src/otel_genai.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — PrometheusMetrics with text exposition format, counters per provider/model

---

## Fase 6 — Structured Output Cloud-Native

### 6.1 OpenAI response_format json_schema

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Muy Alto

Extender `src/providers.rs`.

Usar el parametro `response_format: { type: "json_schema", json_schema: {...} }`
de OpenAI para obtener structured output nativo. El modelo garantiza JSON valido
conforme al schema. Soportar schemas con propiedades, tipos, enums, arrays y
refs. Fallback a regex extraction si el modelo no soporta json_schema.

**Tipos clave**:
- `StructuredOutputConfig`: schema (JsonSchema), strict mode, fallback_strategy
- `JsonSchemaBuilder`: Builder fluido para construir schemas JSON
- `StructuredRequest`: Extiende LlmRequest con response_format
- `StructuredParser`: Parsea y valida respuesta contra schema
- `SchemaValidationError`: path, expected_type, actual_value

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~12

**Estado**: HECHO — to_openai_response_format() with strict mode, StructuredOutputRequest

---

### 6.2 Anthropic Forced Tool Use

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/providers.rs`.

Usar `tool_choice: { type: "tool", name: "extract_data" }` de Anthropic para
forzar structured output via tool_use blocks. Definir una tool con el schema
deseado, forzar su uso, y extraer los argumentos como output estructurado.
Mas fiable que regex para modelos Claude.

**Tipos clave**:
- `ForcedToolConfig`: tool_name, input_schema, extract_from (arguments)
- `ToolUseExtractor`: Parsea content blocks tipo tool_use, extrae arguments JSON
- `AnthropicStructuredRequest`: Extiende request con tools + tool_choice forced
- `ContentBlockParser`: Parsea array de content blocks (text, tool_use, tool_result)

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — to_anthropic_forced_tool() with tool_choice forced, tool_use block extraction

---

### 6.3 Auto-Select Strategy

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Alto

Extender `src/providers.rs`.

Deteccion automatica del mejor metodo de structured output segun el provider:
OpenAI → json_schema, Anthropic → forced tool_use, otros → retry loop con
validacion. El usuario solo proporciona el schema, el sistema elige la estrategia.

**Tipos clave**:
- `StructuredOutputStrategy`: Enum (JsonSchema, ForcedToolUse, RetryWithValidation)
- `StrategySelector`: Mapea provider_type → mejor estrategia disponible
- `RetryWithValidation`: Retry loop que valida JSON, re-prompt con errores si falla
- `StructuredOutputResult`: parsed_value, strategy_used, attempts, raw_response

**Archivo**: `src/providers.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — StructuredOutputStrategy enum, auto-detect provider → best strategy

---

## Fase 7 — Smart Model Router + Caching

### 7.1 Router in Core Pipeline

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/smart_router.rs` (feature `core`).

`SmartSelector` que auto-selecciona modelo por tipo de tarea (clasificacion →
modelo rapido/barato, razonamiento → modelo potente, code → modelo especializado),
presupuesto de costo, y requisitos de calidad. Configurable con reglas declarativas
y aprendizaje de feedback historico.

**Tipos clave**:
- `SmartSelector`: Selecciona modelo optimo dada tarea + constraints
- `TaskProfile`: Enum (Classification, Reasoning, CodeGen, Summarization, Chat, Extraction)
- `SelectionCriteria`: max_cost_per_token, min_quality_score, max_latency_ms
- `ModelCapability`: Perfil de cada modelo (strengths, cost, latency, context_window)
- `RoutingRule`: Regla declarativa condition → model_preference
- `SelectionResult`: chosen_model, reason, alternatives, estimated_cost

**Archivo**: `src/smart_router.rs` (new)

**Tests**: ~14

**Estado**: HECHO — PipelineRouter with task classification, weighted scoring, routing rules

---

### 7.2 Prompt Cache Hints

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/smart_router.rs`.

`CacheablePrompt` que separa segmentos estables (system prompt, few-shot examples)
de segmentos dinamicos (user query, context). Los segmentos estables se marcan
con cache_control para Anthropic prompt caching y se hashean para cache local.
Reduce tokens facturados hasta 90% en prompts repetitivos.

**Tipos clave**:
- `CacheablePrompt`: Prompt con segmentos marcados como stable/dynamic
- `PromptSegment`: text, segment_type (Stable, Dynamic), cache_control
- `PromptHasher`: Hash SHA-256 de segmentos estables para cache key
- `CacheHintInjector`: Anade cache_control breakpoints para Anthropic API

**Archivo**: `src/smart_router.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — CacheablePrompt with static/dynamic segments, cache_fingerprint, Anthropic cache_control

---

### 7.3 Response Cache Middleware

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/smart_router.rs`.

Cache de respuestas que intercepta `generate_response()`: antes de llamar al
LLM, busca en cache por hash de (model + prompt + temperature). Si hay hit,
devuelve respuesta cacheada sin costo. Configurable con TTL, max_entries, y
invalidacion por patron.

**Tipos clave**:
- `ResponseCache`: Cache en memoria con LRU eviction y TTL
- `CacheKey`: Hash de model_id + prompt_hash + temperature + top_p
- `CacheEntry`: response, created_at, hit_count, token_cost_saved
- `CacheConfig`: max_entries, ttl_secs, cache_when (temperature == 0 only, o siempre)
- `CacheStats`: hits, misses, hit_rate, tokens_saved, cost_saved

**Archivo**: `src/smart_router.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — ResponseCacheMiddleware with LRU eviction, TTL, normalized fingerprint keying

---

### 7.4 Cost Tracking

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/smart_router.rs`.

Tracking de costos por provider, modelo y sesion. Cada llamada LLM registra
tokens_in * price_per_input_token + tokens_out * price_per_output_token.
Alertas cuando el gasto acumulado supera umbrales configurables.

**Tipos clave**:
- `CostTracker`: Acumula costos por provider/model/session
- `PricingTable`: HashMap de model_id → (input_price_per_1k, output_price_per_1k)
- `CostEntry`: timestamp, model, tokens_in, tokens_out, cost_usd
- `BudgetAlert`: Threshold + callback cuando se supera (warn, hard_limit)
- `CostReport`: Resumen por periodo con desglose por modelo y provider

**Archivo**: `src/smart_router.rs` (ext)

**Tests**: ~8

**Estado**: HECHO — CostTracker with PricingTable, BudgetCostAlert, per-provider/model tracking

---

## Fase 8 — Context Engineering

### 8.1 Context Compiler

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/context_composer.rs`.

Compilador de contexto que asigna presupuesto de tokens por segmento (system
prompt, conversation history, RAG context, tool schemas) con prioridad. Cuando
el total excede el context window, evicta segmentos de menor prioridad primero.
Garantiza que el prompt nunca excede el limite del modelo.

**Tipos clave**:
- `ContextCompiler`: Compila segmentos en prompt final dentro de budget
- `ContextSegment`: content, priority (0-10), min_tokens, max_tokens, compressible
- `TokenBudget`: total_budget, allocations per segment type
- `EvictionPolicy`: LowestPriority, OldestFirst, LeastRelevant, Hybrid
- `CompilationResult`: final_prompt, token_count, evicted_segments, warnings

**Archivo**: `src/context_composer.rs` (ext)

**Tests**: ~14

**Estado**: HECHO — ContextCompiler with token budget per segment, priority-based eviction

---

### 8.2 Auto-Compaction

**Prioridad**: ALTA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/context_composer.rs`.

Compactacion automatica de conversaciones largas: cuando el historial excede
el 80% del context window, los mensajes mas antiguos se resumen en un bloque
compacto. El resumen preserva hechos clave, decisiones y contexto necesario
para la continuacion. Configurable con threshold y estrategia de resumen.

**Tipos clave**:
- `AutoCompactor`: Monitorea tamano de conversacion, compacta cuando necesario
- `CompactionConfig`: threshold_percent (0.8), min_messages_to_keep, summary_max_tokens
- `CompactionStrategy`: Summarize (LLM resume), Truncate (corta mas antiguos), Hybrid
- `CompactedHistory`: summary_block + recent_messages (sin compactar)
- `CompactionEvent`: Evento emitido cuando se compacta (para logging/audit)

**Archivo**: `src/context_composer.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — ConversationCompactor with 80% threshold, Summarize/Truncate/Hybrid strategies

---

### 8.3 Tool Schema Lazy-Loading

**Prioridad**: MEDIA | **Esfuerzo**: M | **Impacto**: Alto

Extender `src/context_composer.rs`.

En lugar de incluir todos los tool schemas en cada request (consume tokens),
usar un indice TF-IDF para seleccionar solo los top-K tools mas relevantes
para la query actual. Reduce tokens de tool schemas de ~2000 a ~500 por request.

**Tipos clave**:
- `ToolSearchIndex`: Indice TF-IDF sobre tool names + descriptions
- `ToolSelector`: Selecciona top-K tools por relevancia a la query
- `ToolSelectionConfig`: max_tools (default 10), min_relevance_score, always_include
- `TfIdfScorer`: Calcula TF-IDF scores con tokenizacion simple (whitespace + lowercase)
- `ToolSelectionResult`: selected_tools, scores, excluded_tools

**Archivo**: `src/context_composer.rs` (ext)

**Tests**: ~10

**Estado**: HECHO — ToolSearchIndex with TF-IDF scoring, top-K selection, stop word filtering

---

## Fase 9 — Cloud Connectors + Web Search

### 9.1 S3 SigV4 Signing

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/cloud_connectors.rs`.

Reemplazar la firma S3 simulada con AWS Signature Version 4 real. Implementar
los 4 pasos: canonical request, string to sign, signing key (HMAC-SHA256
chain), y Authorization header. Soportar todas las operaciones S3 existentes
(PutObject, GetObject, ListBucket, DeleteObject) con firmas validas.

**Tipos clave**:
- `SigV4Signer`: Implementa los 4 pasos de AWS SigV4
- `AwsCredentials`: access_key_id, secret_access_key, session_token (opcional)
- `CanonicalRequest`: method, uri, query_string, headers, payload_hash
- `SigningKey`: Derivado via HMAC-SHA256 chain (date, region, service, "aws4_request")
- `SignedHeaders`: Headers requeridos (host, x-amz-date, x-amz-content-sha256)

**Archivo**: `src/cloud_connectors.rs` (ext)

**Tests**: ~14

**Estado**: HECHO — Real SigV4 4-step signing (canonical request, string to sign, signing key, auth header)

---

### 9.2 Web Search with HTML Parsing

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Muy Alto

Extender `src/cloud_connectors.rs`.

Reemplazar resultados hardcodeados con busquedas reales: GET a DuckDuckGo HTML
(`https://html.duckduckgo.com/html/`) y parseo de resultados con extraccion de
titulo, snippet y URL. Soporte opcional para Brave Search API con paginacion.
Deduplicacion por URL y caching de resultados con TTL.

**Tipos clave**:
- `WebSearchEngine` trait: `search(query, max_results) -> Vec<SearchResult>`
- `DuckDuckGoSearch`: GET HTML, parseo con string matching (sin dep de HTML parser)
- `BraveSearch`: GET JSON API con API key, paginacion via offset
- `SearchResult`: title, url, snippet, source, timestamp
- `SearchDeduplicator`: Elimina duplicados por URL normalizada
- `SearchCache`: Cache de resultados con TTL configurable

**Archivo**: `src/cloud_connectors.rs` (ext)

**Tests**: ~14

**Estado**: HECHO — DuckDuckGo HTML parsing, Brave API, result dedup, search caching with TTL

---

### 9.3 Web Crawler with robots.txt

**Prioridad**: MEDIA | **Esfuerzo**: L | **Impacto**: Medio

Extender `src/cloud_connectors.rs`.

Crawler BFS que recorre paginas web desde una URL semilla, respetando robots.txt.
Extrae texto limpio (sin HTML tags, scripts, styles) para alimentar RAG.
Configurable con depth limit, max pages, allowed domains y delay entre requests.

**Tipos clave**:
- `WebCrawler`: BFS crawler con queue y visited set
- `CrawlConfig`: max_depth, max_pages, allowed_domains, delay_ms, user_agent
- `RobotsTxtParser`: Parsea robots.txt, verifica si URL esta permitida
- `HtmlTextExtractor`: Extrae texto limpio de HTML (strip tags, scripts, styles)
- `CrawlResult`: pages (Vec<CrawledPage>), total_pages, duration
- `CrawledPage`: url, title, clean_text, links, depth, status_code

**Archivo**: `src/cloud_connectors.rs` (ext)

**Tests**: ~12

**Estado**: HECHO — WebCrawler with BFS, robots.txt parsing, rate limiting, text extraction

---

## Fase 10 — Developer Experience

### 10.1 MockHttpServer for Tests

**Prioridad**: CRITICA | **Esfuerzo**: L | **Impacto**: Muy Alto

Nuevo archivo: `src/mock_http.rs` (feature `eval`).

Servidor HTTP local (localhost:0, puerto aleatorio) que sirve respuestas
programadas para tests. Permite hacer CI sin API keys: tests de providers,
MCP client, media generation y cloud connectors contra un servidor mock
que devuelve respuestas predefinidas con latencia simulada.

**Tipos clave**:
- `MockHttpServer`: Servidor en localhost con puerto aleatorio
- `MockRoute`: method + path pattern → response (status, headers, body)
- `ResponseQueue`: Cola de respuestas para un endpoint (devuelve en orden FIFO)
- `RequestLog`: Registra todas las requests recibidas para assertions
- `MockBuilder`: Builder fluido para configurar rutas y respuestas
- `MockServerHandle`: Handle con url() y shutdown(), Drop auto-shutdown

**Archivo**: `src/mock_http.rs` (new)

**Tests**: ~16

**Estado**: HECHO — MockHttpServer on localhost with route matching, response queue, request logging

---

### 10.2 Integration Test Suite

**Prioridad**: ALTA | **Esfuerzo**: L | **Impacto**: Alto

Extender `src/mock_http.rs`.

Suite de tests E2E que ejercita flujos completos: (1) AiAssistant → mock provider
→ response, (2) provider fail → fallback → success, (3) generate → audit log
entry, (4) structured output → validation → retry, (5) cache miss → LLM call
→ cache hit. Todos contra MockHttpServer, sin API keys.

**Tipos clave**:
- `IntegrationScenario` trait: setup() → execute() → verify()
- `ScenarioRunner`: Ejecuta Vec de scenarios, reporta pass/fail
- `ProviderFallbackScenario`: Configura 2 mock providers, falla primero, verifica fallback
- `AuditTrailScenario`: Genera request, verifica audit log entry creado
- `CacheScenario`: Miss → call → hit, verifica tokens saved

**Archivo**: `src/mock_http.rs` (ext)

**Tests**: ~20

**Estado**: HECHO — E2E scenarios in http_client.rs tests with MockHttpServer

---

### 10.3 Feature Flag Validation

**Prioridad**: MEDIA | **Esfuerzo**: S | **Impacto**: Medio

Extender `src/lib.rs`.

Macro `validate_features!()` que verifica en compile-time que las combinaciones
de feature flags son validas: `scheduler` requiere `autonomous`, `browser`
requiere `autonomous`, `distributed-agents` requiere `autonomous` + `distributed-network`,
`whisper-local` requiere `audio`. Emite `compile_error!()` con mensaje descriptivo
si se activa una combinacion invalida.

**Tipos clave**:
- `validate_features!()`: Macro con `cfg` checks y `compile_error!()` messages
- Verificaciones: dependencias obligatorias, combinaciones incompatibles, warnings
- Mensaje de error descriptivo: "Feature X requires feature Y. Add Y to your Cargo.toml"

**Archivo**: `src/lib.rs` (ext)

**Tests**: ~6

**Estado**: HECHO — validate_features!() macro in lib.rs with compile_error! for invalid feature combos

---

## Resumen

| Fase | Items | Estado | Ficheros | Tests nuevos |
|------|-------|--------|----------|-------|
| 1. HTTP Real: MCP Client + OAuth | 4 | 4/4 HECHO | mcp_client.rs (ext), mcp_protocol.rs (ext) | ~40 |
| 2. HTTP Real: Media Generation | 3 | 3/3 HECHO | media_generation.rs (ext) | ~20 |
| 3. Wire Provider Reliability | 4 | 4/4 HECHO | providers.rs (ext) | ~40 |
| 4. Secrets Management + Security | 4 | 4/4 HECHO | secure_credentials.rs (new), providers.rs (ext), config_file.rs (ext) | ~32 |
| 5. OpenTelemetry Export Real | 3 | 3/3 HECHO | opentelemetry_integration.rs (ext) | ~20 |
| 6. Structured Output Cloud-Native | 3 | 3/3 HECHO | structured.rs (ext) | ~18 |
| 7. Smart Model Router + Caching | 4 | 4/4 HECHO | auto_model_selection.rs (ext), caching.rs (ext) | ~30 |
| 8. Context Engineering | 3 | 3/3 HECHO | context_composer.rs (ext) | ~28 |
| 9. Cloud Connectors + Web Search | 3 | 3/3 HECHO | cloud_connectors.rs (ext), web_search.rs (ext) | ~24 |
| 10. Developer Experience | 3 | 3/3 HECHO | http_client.rs (ext), lib.rs (ext) | ~16 |
| **TOTAL** | **34** | **34/34 HECHO** | **1 nuevo + 14 extendidos** | **~238** |

**Total tests**: 4687 (4449 pre-v7 + 238 nuevos v7), 0 failures, 0 clippy warnings

**Total líneas añadidas**: ~8200 across 15 files

---

## Ficheros Nuevos (1)

| Fichero | Items | Feature Gate |
|---------|-------|-------------|
| `src/secure_credentials.rs` | 4.1-4.2 | `security` |

## Ficheros Extendidos (14)

| Fichero | Items | Líneas añadidas |
|---------|-------|----------------|
| `src/mcp_client.rs` | 1.1 | +964 |
| `src/mcp_protocol.rs` | 1.2-1.4 | +319 |
| `src/media_generation.rs` | 2.1-2.3 | +628 |
| `src/providers.rs` | 3.1-3.4, 4.3, 6.1-6.3 | +1415 |
| `src/config_file.rs` | 4.4 | +210 |
| `src/opentelemetry_integration.rs` | 5.1-5.3 | +623 |
| `src/structured.rs` | 6.1-6.3 | +310 |
| `src/auto_model_selection.rs` | 7.1-7.2 | +388 |
| `src/caching.rs` | 7.3-7.4 | +419 |
| `src/context_composer.rs` | 8.1-8.3 | +826 |
| `src/cloud_connectors.rs` | 9.1 | +677 |
| `src/web_search.rs` | 9.2-9.3 | +1074 |
| `src/http_client.rs` | 10.1-10.2 | +395 |
| `src/lib.rs` | 10.3 + wiring | +60 |

## Feature Flags

No se crean feature flags nuevos. Todos los items usan features existentes:

| Feature | Items |
|---------|-------|
| `mcp` | 1.1-1.4 |
| `core` | 3.1-3.4, 6.1-6.3, 7.1-7.4, 8.1-8.3 |
| `security` | 4.1-4.2, 4.4 |
| `analytics` | 5.1-5.3 |
| `eval` | 10.1-10.2 |

---

## Prioridades

**CRITICAS** (hacer primero):
- 1.1 RemoteMcpClient HTTP real
- 1.2 MCP OAuth 2.1 real
- 3.1 Fallback chain in generate_response
- 4.1 SecureString wrapper
- 10.1 MockHttpServer for tests

**ALTAS** (hacer segundo):
- 1.3 StreamableHttpTransport real
- 1.4 MCP Sampling
- 2.1 DALL-E 3 real HTTP
- 3.2 Connection pooling
- 3.3 Health checker
- 4.2 Credential resolver chain
- 4.3 Audit auto-wire
- 5.1 OTLP HTTP exporter
- 5.2 GenAI semantic conventions
- 6.1 OpenAI json_schema
- 6.2 Anthropic forced tool use
- 7.1 Router in core pipeline
- 8.1 Context compiler
- 8.2 Auto-compaction
- 9.1 S3 SigV4 signing
- 9.2 Web search with HTML parsing
- 10.2 Integration test suite

**MEDIAS** (hacer tercero):
- 2.2 Stable Diffusion real HTTP
- 2.3 Replicate real HTTP
- 3.4 Per-provider rate limiting
- 4.4 Config validation
- 5.3 Prometheus metrics endpoint
- 6.3 Auto-select strategy
- 7.2 Prompt cache hints
- 7.3 Response cache middleware
- 7.4 Cost tracking
- 8.3 Tool schema lazy-loading
- 9.3 Web crawler with robots.txt
- 10.3 Feature flag validation
