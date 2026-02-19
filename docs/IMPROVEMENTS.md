# Mejoras para ai_assistant — Inspiradas en el Análisis de OpenClaw

> Documento generado el 2026-02-13.
> Basado en análisis comparativo de OpenClaw (gateway IA multi-canal, TypeScript, ~100K LOC)
> vs ai_assistant (librería IA local-first, Rust, ~10K+ LOC).
>
> Ver también: `docs/OPENCLAW_ANALYSIS.md` y `docs/OPENCLAW_SECURITY.md`

---

## Resumen

`ai_assistant` ya tiene una base sólida: buen sistema de features, RAG con SQLite/FTS5,
gestión de sesiones, múltiples proveedores locales, streaming, y módulos avanzados
(knowledge graphs, document parsing, decision trees, etc.).

Tras analizar OpenClaw (un proyecto maduro con 15+ canales, 20+ proveedores IA, herramientas
de agente, y uso en producción), identifico las siguientes áreas de mejora organizadas
por **prioridad** e **impacto**.

---

## PRIORIDAD ALTA — Mejoras Arquitectónicas

### 1. Sistema de Failover entre Proveedores

**Estado actual**: Si Ollama no responde, se devuelve un error y punto.

**Lo que hace OpenClaw**: Tiene un sistema de failover con múltiples perfiles de auth,
cooldown por perfil, y cadena de fallback configurable:
```
Perfil primario → cooldown → siguiente perfil → fallback → error
```

**Propuesta para ai_assistant**:

```rust
// En config.rs
pub struct FailoverConfig {
    /// Proveedores en orden de preferencia
    pub providers: Vec<AiProvider>,
    /// Cooldown después de error (en segundos)
    pub cooldown_secs: u64,
    /// Máximo de reintentos antes de pasar al siguiente
    pub max_retries: u32,
}

// En providers.rs
pub struct ProviderPool {
    providers: Vec<ProviderState>,
}

struct ProviderState {
    provider: AiProvider,
    url: String,
    cooldown_until: Option<Instant>,
    consecutive_failures: u32,
}

impl ProviderPool {
    /// Obtiene el siguiente proveedor disponible (no en cooldown)
    pub fn next_available(&mut self) -> Option<&ProviderState> { ... }

    /// Marca un proveedor como fallido
    pub fn mark_failed(&mut self, provider: &AiProvider) { ... }

    /// Marca un proveedor como exitoso (resetea cooldown)
    pub fn mark_success(&mut self, provider: &AiProvider) { ... }
}
```

**Impacto**: Alto. La mayoría de usuarios tendrán Ollama + al menos un backup.
Un fallo de Ollama no debería ser terminal.

---

### 2. Gestión de Contexto con Auto-Compactación

**Estado actual**: `ContextUsage` calcula uso y tiene umbrales de warning/critical,
pero no actúa sobre ellos. Si el contexto se llena, simplemente falla.

**Lo que hace OpenClaw**: Auto-compactación cuando el contexto se desborda, con
truncado de resultados de herramientas como fallback.

**Propuesta**:

```rust
// En context.rs
pub enum ContextAction {
    /// El contexto está bien, proceder normalmente
    Ok,
    /// Comprimir: resumir mensajes antiguos
    Compact { messages_to_summarize: usize },
    /// Truncar: los últimos mensajes no caben, recortar
    Truncate { max_chars: usize },
    /// Crítico: no queda espacio ni para el prompt del sistema
    Abort { reason: String },
}

impl ContextUsage {
    /// Determina qué acción tomar dado el uso actual
    pub fn recommended_action(&self) -> ContextAction {
        if self.usage_percent < 70.0 {
            ContextAction::Ok
        } else if self.usage_percent < 90.0 {
            // Compactar mensajes antiguos para liberar espacio
            let to_summarize = self.conversation_tokens / 3;
            ContextAction::Compact {
                messages_to_summarize: to_summarize / 100, // aprox msgs
            }
        } else if self.usage_percent < 99.0 {
            ContextAction::Truncate {
                max_chars: self.remaining_tokens() * 3, // aprox chars
            }
        } else {
            ContextAction::Abort {
                reason: "Context window full".into(),
            }
        }
    }
}
```

En `assistant.rs`, antes de cada llamada al modelo:
```rust
let usage = self.calculate_context_usage();
match usage.recommended_action() {
    ContextAction::Compact { messages_to_summarize } => {
        self.compact_history(messages_to_summarize);
    }
    ContextAction::Truncate { max_chars } => {
        self.truncate_oldest_messages(max_chars);
    }
    ContextAction::Abort { reason } => {
        return Err(AiError::ResourceLimit(
            ResourceLimitError::ContextOverflow(reason)
        ));
    }
    ContextAction::Ok => {}
}
```
dw
**Impacto**: Alto. Evita errores de "context too long" que frustran al usuario.

---

### 3. Redacción de Datos Sensibles en Logs

**Estado actual**: El crate usa `println!` para debug. Sin redacción.

**Lo que hace OpenClaw**: Sistema de redacción robusto (`src/logging/redact.ts`) que
oculta API keys, tokens, passwords, claves PEM, etc. en todos los logs.

**Propuesta**:

```rust
// Nuevo módulo: src/logging.rs
use regex::Regex;
use std::sync::LazyLock;

static REDACTION_PATTERNS: LazyLock<Vec<(Regex, &str)>> = LazyLock::new(|| vec![
    // API keys genéricas
    (Regex::new(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+").unwrap(),
     "$1=***REDACTED***"),
    // Bearer tokens
    (Regex::new(r"(?i)bearer\s+\S+").unwrap(), "Bearer ***REDACTED***"),
    // OpenAI-style keys
    (Regex::new(r"sk-[a-zA-Z0-9]{20,}").unwrap(), "sk-***REDACTED***"),
]);

pub fn redact(text: &str) -> String {
    let mut result = text.to_string();
    for (pattern, replacement) in REDACTION_PATTERNS.iter() {
        result = pattern.replace_all(&result, *replacement).to_string();
    }
    result
}

/// Macro para logging seguro
macro_rules! safe_log {
    ($($arg:tt)*) => {
        eprintln!("{}", $crate::logging::redact(&format!($($arg)*)));
    };
}
```

**Impacto**: Alto para seguridad. Especialmente si el usuario redirige logs a archivo.

---

### 4. Migración a Async/Await (tokio)

**Estado actual**: HTTP síncrono con `ureq`, threads manuales para streaming.

**Lo que hace OpenClaw**: Node.js es async por naturaleza. Todo el I/O es non-blocking.

**Propuesta**: Migrar gradualmente a `reqwest` (async) + `tokio`:

```toml
# Cargo.toml - nueva dependencia
[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"], optional = true }
reqwest = { version = "0.12", features = ["json", "stream"], optional = true }

[features]
async-runtime = ["tokio", "reqwest"]
```

**Fase 1**: Añadir variantes async de las funciones de providers sin romper la API actual:
```rust
#[cfg(feature = "async-runtime")]
pub async fn fetch_ollama_models_async(base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    let response = reqwest::get(&url).await?;
    let body: serde_json::Value = response.json().await?;
    // ... parseo igual que la versión sync
}
```

**Fase 2**: Streaming via `tokio::sync::mpsc` (unbounded o bounded con backpressure).

**Fase 3**: Paralelizar operaciones independientes (fetch models de múltiples providers).

**Impacto**: Alto a largo plazo. Permite streaming real, múltiples requests concurrentes,
y mejor integración con frameworks web (axum, actix).

---

## PRIORIDAD MEDIA — Funcionalidades Nuevas

### 5. Sistema de Herramientas (Tool Use / Function Calling)

**Estado actual**: El feature flag `tools` existe en Cargo.toml pero la implementación
es básica (definiciones de herramientas, no ejecución real).

**Lo que hace OpenClaw**: Sistema completo de herramientas con:
- Definiciones con schema JSON
- Ejecución con streaming de resultados
- Gating de permisos
- Loop agentic (el modelo puede invocar herramientas iterativamente)

**Propuesta**: Implementar un sistema de herramientas ejecutable:

```rust
// src/tools.rs
pub trait Tool: Send + Sync {
    /// Nombre único de la herramienta
    fn name(&self) -> &str;

    /// Descripción para el modelo
    fn description(&self) -> &str;

    /// Schema de parámetros (JSON Schema)
    fn parameters_schema(&self) -> serde_json::Value;

    /// Ejecutar la herramienta
    fn execute(&self, params: serde_json::Value) -> Result<ToolResult>;
}

pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Box<dyn Tool>) { ... }
    pub fn execute(&self, name: &str, params: Value) -> Result<ToolResult> { ... }
    pub fn schemas_for_model(&self) -> Vec<Value> { ... } // Para enviar al modelo
}
```

**Herramientas built-in sugeridas**:
- `file_read` — Leer archivos locales
- `web_fetch` — Hacer peticiones HTTP
- `rag_search` — Buscar en la base de conocimiento
- `calculator` — Operaciones matemáticas
- `datetime` — Fecha/hora actual

**Impacto**: Medio-Alto. Convierte al asistente de un chatbot en un agente capaz.

---

### 6. Sesiones como Event Log (JSONL)

**Estado actual**: Las sesiones se guardan como JSON completo (todo el array de mensajes
serializado de una vez).

**Lo que hace OpenClaw**: Sesiones como archivos `.jsonl` (append-only):
- Solo añade líneas, nunca reescribe el archivo completo
- Más eficiente para conversaciones largas
- Más resistente a corrupción (un crash no pierde todo el archivo)
- Permite compactación selectiva

**Propuesta**:

```rust
// En session.rs
pub struct JournalSession {
    path: PathBuf,
    /// Cache en memoria de los mensajes actuales
    messages: Vec<ChatMessage>,
}

impl JournalSession {
    /// Añade un mensaje al journal (append, no rewrite)
    pub fn append_message(&mut self, msg: &ChatMessage) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(msg)?;
        writeln!(file, "{}", line)?;
        self.messages.push(msg.clone());
        Ok(())
    }

    /// Carga una sesión desde un archivo JSONL
    pub fn load(path: &Path) -> Result<Self> {
        let mut messages = Vec::new();
        if path.exists() {
            let file = File::open(path)?;
            for line in BufReader::new(file).lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    let msg: ChatMessage = serde_json::from_str(&line)?;
                    messages.push(msg);
                }
            }
        }
        Ok(Self { path: path.to_path_buf(), messages })
    }

    /// Compacta: resume mensajes antiguos y reescribe el archivo
    pub fn compact(&mut self, keep_last: usize) -> Result<()> { ... }
}
```

**Impacto**: Medio. Mejor rendimiento y resiliencia para conversaciones largas.

---

### 7. Detección Dinámica de Contexto del Modelo

**Estado actual**: Tabla hardcodeada de contextos por modelo (`get_model_context_size`).
Si el modelo no está en la tabla, devuelve 8K por defecto.

**Lo que hace OpenClaw**: Consulta al proveedor el tamaño real del contexto,
con caché por modelo.

**Propuesta**: Ya existe `fetch_model_context_size` en providers.rs.
Integrar mejor con caché:

```rust
// En context.rs
use std::collections::HashMap;
use std::sync::Mutex;

static CONTEXT_CACHE: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn get_model_context_size_cached(
    model_name: &str,
    provider: &AiProvider,
    base_url: &str,
) -> usize {
    // 1. Comprobar caché
    if let Some(&size) = CONTEXT_CACHE.lock().unwrap().get(model_name) {
        return size;
    }

    // 2. Intentar consultar al proveedor
    if let Ok(size) = fetch_model_context_size(base_url, model_name) {
        if size > 0 {
            CONTEXT_CACHE.lock().unwrap().insert(model_name.to_string(), size);
            return size;
        }
    }

    // 3. Fallback a tabla estática
    let size = get_model_context_size(model_name);
    CONTEXT_CACHE.lock().unwrap().insert(model_name.to_string(), size);
    size
}
```

**Impacto**: Medio. Evita subestimar/sobreestimar el contexto disponible.

---

### 8. Cola de Peticiones con Prioridad

**Estado actual**: Las peticiones se procesan una a una de forma síncrona.

**Lo que hace OpenClaw**: Cola de comandos con lanes (global + por sesión), que previene
condiciones de carrera y permite priorización.

**Propuesta** (útil cuando ai_assistant se use en un servidor multi-usuario):

```rust
pub struct RequestQueue {
    queue: VecDeque<QueuedRequest>,
    active: Option<QueuedRequest>,
    max_concurrent: usize,
}

struct QueuedRequest {
    id: String,
    session_id: Option<String>,
    priority: RequestPriority,
    message: ChatMessage,
    response_tx: mpsc::Sender<AiResponse>,
}

pub enum RequestPriority {
    High,    // Cancelaciones, comandos del sistema
    Normal,  // Mensajes de usuario
    Low,     // Indexación, tareas de fondo
}
```

**Impacto**: Medio. Necesario si se usa como backend de un servicio.

---

## PRIORIDAD MEDIA-BAJA — Calidad y Operaciones

### 9. Tests con Mocks de Proveedores

**Estado actual**: El test harness es un binario CLI que necesita proveedores reales
para testear. No hay mocks.

**Lo que hace OpenClaw**: Tests extensivos con Vitest, mocking de APIs,
tests unitarios sin dependencias externas.

**Propuesta**: Crear un trait para abstraer las llamadas HTTP:

```rust
// En providers.rs
pub trait HttpClient: Send + Sync {
    fn get(&self, url: &str) -> Result<serde_json::Value>;
    fn post(&self, url: &str, body: &serde_json::Value) -> Result<serde_json::Value>;
}

/// Cliente real que usa ureq
pub struct UreqClient;
impl HttpClient for UreqClient { ... }

/// Cliente mock para tests
#[cfg(test)]
pub struct MockClient {
    responses: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
impl MockClient {
    pub fn with_response(mut self, url: &str, response: Value) -> Self {
        self.responses.insert(url.to_string(), response);
        self
    }
}

#[cfg(test)]
impl HttpClient for MockClient { ... }
```

Esto permite:
```rust
#[test]
fn test_ollama_models_parsing() {
    let client = MockClient::new()
        .with_response("http://localhost:11434/api/tags", json!({
            "models": [{"name": "llama3:latest", "size": 4_000_000_000}]
        }));

    let models = fetch_ollama_models_with_client("http://localhost:11434", &client).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "llama3:latest");
}
```

**Impacto**: Medio. Tests más rápidos y fiables, CI sin Ollama.

---

### 10. Retry con Backoff Exponencial

**Estado actual**: Sin reintentos. Si una petición falla, error inmediato.

**Lo que hace OpenClaw**: Reintentos con cooldown, tracking de errores por perfil.

**Propuesta**:

```rust
// En providers.rs (o un nuevo módulo retry.rs)
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 500,
            max_delay_ms: 10_000,
            backoff_multiplier: 2.0,
        }
    }
}

pub fn with_retry<F, T>(config: &RetryConfig, mut operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let mut delay = config.initial_delay_ms;
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                if attempt < config.max_retries {
                    std::thread::sleep(Duration::from_millis(delay));
                    delay = (delay as f64 * config.backoff_multiplier)
                        .min(config.max_delay_ms as f64) as u64;
                }
            }
        }
    }

    Err(last_error.unwrap())
}
```

**Impacto**: Medio. Resilencia ante errores transitorios de red.

---

### 11. Logging Estructurado

**Estado actual**: `println!` y `eprintln!` para debug.

**Propuesta**: Usar `log` crate + `env_logger` (o `tracing` para async futuro):

```toml
# Cargo.toml
log = "0.4"
env_logger = { version = "0.11", optional = true }

[features]
logging = ["env_logger"]
```

```rust
// En vez de println!, usar:
log::info!("Fetching models from {}", provider);
log::debug!("Response: {:?}", redact(&response));
log::warn!("Context at {}%, approaching limit", usage.usage_percent);
log::error!("Provider {} unavailable: {}", provider, err);
```

**Impacto**: Medio-Bajo. Mejor debugging sin contaminar stdout.

---

### 12. Cifrado de Sesiones en Reposo

**Estado actual**: Las sesiones se guardan en JSON sin cifrar.

**Lo que hace OpenClaw**: Tampoco cifra (depende del OS). Pero ya tenemos `aes-gcm`
como dependencia para KPKG.

**Propuesta**: Reutilizar la infraestructura de cifrado de `encrypted_knowledge`:

```rust
// En session.rs (feature = "rag" ya trae aes-gcm)
#[cfg(feature = "rag")]
impl ChatSessionStore {
    /// Guarda la sesión cifrada con AES-256-GCM
    pub fn save_encrypted(&self, path: &Path, key: &[u8; 32]) -> Result<()> {
        let json = serde_json::to_vec(&self.sessions)?;
        let encrypted = encrypt_aes256gcm(&json, key)?;
        std::fs::write(path, encrypted)?;
        Ok(())
    }

    /// Carga una sesión cifrada
    pub fn load_encrypted(path: &Path, key: &[u8; 32]) -> Result<Self> {
        let encrypted = std::fs::read(path)?;
        let decrypted = decrypt_aes256gcm(&encrypted, key)?;
        let sessions: Vec<ChatSession> = serde_json::from_slice(&decrypted)?;
        Ok(Self { sessions, current_session_id: None })
    }
}
```

**Impacto**: Medio para usuarios con datos sensibles (como búsqueda de empleo).

---

## PRIORIDAD BAJA — Futuro

### 13. Soporte para Proveedores Cloud (Anthropic, OpenAI)

**Estado actual**: Solo soporta proveedores locales (Ollama, LM Studio, KoboldCpp, etc.)
y OpenAI-compatible genérico.

**Lo que hace OpenClaw**: 20+ proveedores incluyendo Anthropic Claude, OpenAI, Google
Gemini, AWS Bedrock, etc.

**Propuesta**: Añadir `AiProvider::Anthropic` y `AiProvider::OpenAI` con sus APIs
nativas (no solo OpenAI-compatible). Esto daría acceso a:
- Extended thinking de Claude
- Function calling nativo de cada proveedor
- Prompt caching de Anthropic
- Modelos más potentes para tareas complejas

**Impacto**: Bajo por ahora (el crate es local-first), pero Alto si se quiere
ampliar el público objetivo.

---

### 14. Sistema de Hooks/Eventos

**Estado actual**: Sin sistema de eventos.

**Lo que hace OpenClaw**: Sistema de hooks event-driven con lifecycle completo.

**Propuesta**:

```rust
pub enum AiEvent {
    SessionStarted { session_id: String },
    MessageSent { role: String, content: String },
    ResponseReceived { content: String, tokens: usize },
    ContextWarning { usage_percent: f32 },
    ProviderFailed { provider: String, error: String },
    ToolExecuted { name: String, result: String },
}

pub trait EventHandler: Send + Sync {
    fn on_event(&self, event: &AiEvent);
}

// En AiAssistant:
pub fn add_event_handler(&mut self, handler: Box<dyn EventHandler>) { ... }
```

**Impacto**: Bajo. Útil para integración con UIs y telemetría custom.

---

### 15. WebSocket / HTTP Server Embebido

**Estado actual**: Es una librería pura, sin servidor.

**Lo que hace OpenClaw**: Gateway WebSocket completo con broadcasting.

**Propuesta futura**: Un feature `server` que exponga un mini-gateway:

```toml
[features]
server = ["axum", "tokio", "async-runtime"]
```

Permitiría usar `ai_assistant` como backend de una UI web, app móvil, etc.

**Impacto**: Bajo por ahora. Sería un proyecto separado que use el crate como dependencia.

---

## Resumen de Prioridades

| # | Mejora | Prioridad | Dificultad | Impacto | Estado |
|---|--------|-----------|------------|---------|--------|
| 1 | Failover entre proveedores | ALTA | Media | Alto | HECHO |
| 2 | Auto-compactación de contexto | ALTA | Media | Alto | HECHO |
| 3 | Redacción en logs | ALTA | Baja | Alto (seguridad) | HECHO |
| 4 | Async/await (tokio) | ALTA | Alta | Alto (largo plazo) | HECHO |
| 5 | Sistema de herramientas ejecutable | MEDIA | Alta | Alto | HECHO |
| 6 | Sesiones JSONL (event log) | MEDIA | Media | Medio | HECHO |
| 7 | Detección dinámica de contexto | MEDIA | Baja | Medio | HECHO |
| 8 | Cola de peticiones | MEDIA | Media | Medio | HECHO |
| 9 | Tests con mocks | MEDIA-BAJA | Media | Medio | HECHO |
| 10 | Retry con backoff | MEDIA-BAJA | Baja | Medio | HECHO |
| 11 | Logging estructurado | MEDIA-BAJA | Baja | Medio-Bajo | HECHO |
| 12 | Cifrado de sesiones | MEDIA-BAJA | Baja | Medio | HECHO |
| 13 | Proveedores cloud nativos | BAJA | Alta | Bajo (ahora) | HECHO |
| 14 | Sistema de hooks/eventos | BAJA | Media | Bajo | HECHO |
| 15 | Servidor embebido | BAJA | Alta | Bajo | HECHO |

---

## Lo que ai_assistant ya hace MEJOR que OpenClaw

No todo son mejoras necesarias. Hay cosas donde `ai_assistant` destaca:

1. **Knowledge Graph con Graph RAG**: OpenClaw tiene memoria vectorial básica;
   nosotros tenemos un grafo de conocimiento completo con SQLite + FTS5 + traversal
   multi-hop + extracción de entidades.

2. **Document Parsing robusto**: EPUB, DOCX, ODT, HTML, PDF, tablas...
   OpenClaw depende de herramientas externas para esto.

3. **Encrypted Knowledge Packages (KPKG)**: Sistema propio de paquetes cifrados
   con AES-256-GCM. OpenClaw no tiene equivalente.

4. **Decision Trees + Task Planning**: Módulos de lógica y planificación que
   OpenClaw no tiene (su agente depende del modelo para planificar).

5. **Crawl Policy + Feed Monitor**: Infraestructura de crawling con respeto a
   robots.txt que OpenClaw no implementa.

6. **Content Versioning con diffs**: Seguimiento de cambios en contenido con
   algoritmo LCS. OpenClaw solo tiene snapshots.

7. **Entity Enrichment con dedup fuzzy**: Jaccard index sobre bigrams para
   deduplicación de entidades. Más sofisticado que lo que OpenClaw ofrece.

8. **Rendimiento**: Rust vs Node.js. Para operaciones CPU-bound (parsing,
   chunking, búsqueda vectorial), ai_assistant será significativamente más rápido.

---

## Plan de Implementación Sugerido

**Fase 1 (inmediata, 1-2 semanas)**:
- [x] #3 Redacción en logs — `log_redaction.rs` con macro `safe_log!`, 8 tests
- [x] #10 Retry con backoff — `RetryExecutor` integrado en `providers.rs`, clasificación de errores
- [x] #11 Logging estructurado — Migración a `log` crate (0.4), reemplazo de `println!`/`eprintln!` con `log::info!`/`log::debug!`/etc.

**Fase 2 (corto plazo, 2-4 semanas)**:
- [x] #1 Failover entre proveedores — `FallbackChain` integrada en `assistant.rs`, fallback automático en `send_message`
- [x] #2 Auto-compactación de contexto — `ConversationCompactor` integrada en `assistant.rs`, compactación pre-envío
- [x] #7 Detección dinámica de contexto — Cache global en `context.rs` con `get_model_context_size_cached()`, 5 tests

**Fase 3 (medio plazo, 1-2 meses)**:
- [x] #5 Sistema de herramientas unificado — `unified_tools.rs` fusiona 4 módulos (tools, tool_use, tool_calling, function_calling), builder pattern, multi-format parsing, 39 tests
- [x] #6 Sesiones JSONL — `JournalSession` con append-only, compactación atómica, migración desde `ChatSession`, 7 tests
- [x] #9 Tests con mocks — `HttpClient` trait + `MockHttpClient` en `http_client.rs`, funciones `_with_client()` en providers, 4 tests
- [x] #12 Cifrado de sesiones — `save_encrypted()`/`load_encrypted()` con AES-256-GCM en `session.rs`, 3 tests

**Fase 4 (largo plazo)**:
- [x] #4 Async/await — `async_providers.rs` con `reqwest` + `tokio` detrás de feature `async-runtime`, `AsyncHttpClient` trait, `ReqwestClient`, modelo/generación async, bridge blocking, 11 tests
- [x] #8 Cola de peticiones — `request_queue.rs` con cola thread-safe con prioridad (Low/Normal/High), Condvar para blocking, session removal, 13 tests
- [x] #13 Proveedores cloud nativos — `cloud_providers.rs` con soporte OpenAI y Anthropic, resolución de API keys (config + env vars), 9 tests
- [x] #14 Sistema de hooks/eventos — `events.rs` con `EventBus`, `AiEvent` enum (20+ variantes), `EventHandler` trait, handlers filtered/logging/collecting, 12 tests
- [x] #15 Servidor HTTP embebido — `server.rs` con `TcpListener`, endpoints REST (/health, /models, /chat, /config), CORS, background server, 9 tests

**Adicional (no en plan original)**:
- [x] Binary storage — `internal_storage.rs` con bincode+gzip, auto-detección, herramientas de debug, 8 tests
- [x] Migración de session.rs a internal_storage — `save_to_file()`/`load_to_file()` usan bincode, backward-compatible con JSON
- [x] Migración de módulos restantes — `conversation_snapshot`, `multi_layer_graph`, `content_versioning`, `feed_monitor`, `memory`, `metrics`, `entities` con alternativas bincode
- [x] API Key Rotation — `ApiKeyManager` integrado en `assistant.rs`, rotación en HTTP 429
