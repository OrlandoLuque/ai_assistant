# Prompt para sesión de implementación de mejoras en ai_assistant

> Copia y pega todo este texto como mensaje en una nueva sesión de Claude Code
> abierta en el directorio del proyecto ai_assistant_standalone.

---

Eres un ingeniero Rust experto. Tu tarea es implementar una serie de mejoras en el crate `ai_assistant`, una librería Rust para integración con LLMs locales (Ollama, LM Studio, KoboldCpp, etc.). El crate ya es extenso (~1250 líneas solo en lib.rs de re-exports) con muchos módulos y feature flags.

## CONTEXTO CRÍTICO

Tras analizar OpenClaw (un gateway IA multi-canal maduro, open-source, TypeScript, ~100K LOC), hemos identificado mejoras aplicables a nuestro crate. PERO el crate ya tiene muchos módulos implementados que podrían cubrir parte de las necesidades. Tu trabajo es:

1. **Primero leer** los archivos existentes relevantes para entender qué ya hay
2. **Luego integrar** funcionalidad existente que no está conectada al flujo principal
3. **Finalmente implementar** lo que falta

Lee el documento `docs/IMPROVEMENTS.md` para el análisis completo de las 15 mejoras propuestas. Ese documento fue escrito antes de descubrir que muchos módulos ya existen, así que debes adaptar las propuestas a la realidad del código.

## MÓDULOS YA EXISTENTES QUE DEBES REVISAR ANTES DE IMPLEMENTAR

Antes de crear código nuevo, LEE estos módulos porque probablemente ya cubren parte de lo que necesitamos:

- `src/fallback.rs` — FallbackChain, FallbackProvider, ProviderState, ProviderStatus (¿está conectado a assistant.rs?)
- `src/retry.rs` — RetryConfig, RetryExecutor, CircuitBreaker, ResilientExecutor (¿se usa en providers.rs?)
- `src/context_window.rs` — ContextWindow, ContextWindowConfig, EvictionStrategy (¿se usa en assistant.rs?)
- `src/conversation_compaction.rs` — ConversationCompactor, CompactionConfig (¿se integra con el flujo de mensajes?)
- `src/streaming.rs` — StreamBuffer, BackpressureStream, RateLimitedStream
- `src/debug.rs` — DebugLogger, DebugConfig, RequestInspector
- `src/async_support.rs` — AsyncResult, spawn_blocking, channels async
- `src/connection_pool.rs` — ConnectionPool, PoolConfig
- `src/api_key_rotation.rs` — ApiKeyManager, RotationConfig
- `src/plugins.rs` — Plugin trait, PluginManager
- `src/tools.rs` + `src/tool_use.rs` + `src/tool_calling.rs` + `src/agentic_loop.rs` (feature "tools")
- `src/health_check.rs` — HealthChecker, ProviderHealth
- `src/keepalive.rs` — KeepaliveManager
- `src/memory.rs` — MemoryManager, WorkingMemory
- `src/content_encryption.rs` (feature "security") — ContentEncryptor, EncryptedMessageStore
- `src/pii_detection.rs` (feature "security") — PiiDetector, RedactionStrategy

## TAREAS POR ORDEN DE PRIORIDAD

Implementa cada tarea en orden. Antes de cada tarea, lee los archivos relevantes. Después de cada tarea, verifica que compila con `cargo check --features full`. Si algo no compila, arréglalo antes de continuar.

---

### FASE 1: INTEGRACIÓN DE MÓDULOS EXISTENTES (lo más valioso)

#### Tarea 1.1: Integrar FallbackChain en AiAssistant

**Lee primero**: `src/fallback.rs`, `src/assistant.rs`, `src/providers.rs`

El módulo `fallback.rs` tiene `FallbackChain` con `FallbackProvider`, cooldown, y estado. Pero probablemente NO está conectado al flujo principal de `AiAssistant`. Tu trabajo:

1. Revisar si `AiAssistant` usa `FallbackChain` cuando llama a `generate_response_streaming`
2. Si no, integrarlo: cuando el proveedor primario falla, que intente automáticamente con el siguiente
3. Añadir un campo `fallback_chain: Option<FallbackChain>` a `AiAssistant` (o similar)
4. Añadir un método `configure_fallback(&mut self, providers: Vec<AiProvider>)` a `AiAssistant`
5. Modificar el flujo de envío de mensajes para usar el fallback chain

**No crear tipos nuevos si ya existen en fallback.rs. Reutilizar.**

#### Tarea 1.2: Integrar RetryExecutor en llamadas a proveedores

**Lee primero**: `src/retry.rs`, `src/providers.rs`

El módulo `retry.rs` tiene `RetryExecutor` y `CircuitBreaker`. Verificar:

1. ¿Se usa en `providers.rs` cuando se hace fetch de modelos o generate response?
2. Si no, envolver las llamadas HTTP en `retry_with_config()` o `RetryExecutor`
3. Especialmente para `fetch_ollama_models`, `fetch_openai_compatible_models`, `generate_response_streaming`
4. Usar backoff exponencial para errores de red (timeouts, connection refused)
5. NO reintentar en errores de validación o auth (solo errores transitorios)

#### Tarea 1.3: Integrar ContextWindow + ConversationCompactor en el flujo de mensajes

**Lee primero**: `src/context_window.rs`, `src/conversation_compaction.rs`, `src/context.rs`, `src/assistant.rs`

1. Verificar si `AiAssistant` usa `ContextWindow` o `ConversationCompactor` antes de enviar mensajes
2. Si no, integrar: antes de cada `generate_response`, calcular `ContextUsage` y si está en warning/critical, compactar automáticamente
3. La compactación debe: resumir mensajes antiguos (o eliminarlos según EvictionStrategy) para liberar espacio
4. Añadir un campo de configuración a `AiAssistant` para activar/desactivar auto-compactación
5. El flujo debe ser: usuario envía mensaje → calcular contexto → si >70% compactar → enviar al modelo

#### Tarea 1.4: Integrar ApiKeyRotation con el sistema de proveedores

**Lee primero**: `src/api_key_rotation.rs`, `src/providers.rs`, `src/config.rs`

1. Verificar si `ApiKeyManager` se usa para rotar entre múltiples API keys
2. Si no, permitir configurar múltiples keys por proveedor
3. Cuando una key falla (rate limited), rotar automáticamente a la siguiente

---

### FASE 2: IMPLEMENTAR LO QUE FALTA

#### Tarea 2.1: Redacción de datos sensibles en logs

**Lee primero**: `src/debug.rs`, `src/pii_detection.rs` (si existe con feature security)

El crate probablemente usa `println!` o `eprintln!` para debug en algunos sitios, y `DebugLogger` en otros. Necesitamos:

1. Crear un módulo `src/log_redaction.rs` (siempre disponible, sin feature flag) con:
   - Función `redact(text: &str) -> String` que oculta patrones sensibles
   - Patrones: API keys (`sk-*`, `key-*`), Bearer tokens, passwords en URLs, PEM keys
   - Macro `safe_log!` que aplica redacción antes de imprimir
2. Si `pii_detection.rs` ya tiene patrones similares, reutilizar los regexes de ahí
3. Buscar todos los `println!` y `eprintln!` en el crate que puedan imprimir datos sensibles y reemplazar por `safe_log!` o al menos pasar por `redact()`
4. NO reemplazar TODOS los println — solo los que podrían contener datos sensibles (respuestas HTTP, headers, URLs con tokens, etc.)

#### Tarea 2.2: Sesiones en formato JSONL (Event Log)

**Lee primero**: `src/session.rs`, `src/persistence.rs`, `src/conversation_snapshot.rs`

Las sesiones actualmente se guardan como JSON completo. Necesitamos un formato JSONL alternativo:

1. Añadir a `session.rs` una struct `JournalSession` que:
   - Almacene mensajes en un archivo `.jsonl` (una línea JSON por mensaje)
   - `append_message()`: añade al final del archivo (no reescribe)
   - `load()`: lee el archivo línea por línea
   - `compact()`: resume mensajes antiguos y reescribe
2. Mantener compatibilidad: `ChatSessionStore` sigue funcionando igual
3. `JournalSession` es una alternativa para conversaciones largas
4. Añadir un método `to_journal(&self) -> JournalSession` en `ChatSession` para migrar

#### Tarea 2.3: Detección dinámica de contexto del modelo con caché

**Lee primero**: `src/context.rs`, `src/providers.rs` (buscar `fetch_model_context_size`)

1. La función `fetch_model_context_size` ya existe en providers.rs. Verificar si se usa
2. Crear `get_model_context_size_cached()` en context.rs que:
   - Primero busque en caché en memoria (HashMap estático con LazyLock + Mutex)
   - Luego intente consultar al proveedor via `fetch_model_context_size`
   - Finalmente caiga a la tabla estática `get_model_context_size`
3. Usar `get_model_context_size_cached` en todos los sitios donde se llame a `get_model_context_size`

#### Tarea 2.4: Cifrado opcional de sesiones en reposo

**Lee primero**: `src/content_encryption.rs` (feature security), `src/encrypted_knowledge.rs` (feature rag), `src/session.rs`

1. Si `content_encryption.rs` ya tiene `ContentEncryptor` con AES, reutilizarlo
2. Si no, usar `aes-gcm` (ya es dependencia del feature rag)
3. Añadir a `ChatSessionStore`:
   - `save_encrypted(&self, path, key)` — cifra con AES-256-GCM antes de escribir
   - `load_encrypted(path, key)` — descifra al cargar
4. Gated bajo `#[cfg(feature = "rag")]` ya que `aes-gcm` es dependencia de rag
5. Reutilizar las funciones de cifrado de `encrypted_knowledge` si es posible (para no duplicar código)

---

### FASE 3: MIGRACIÓN A BINCODE PARA ALMACENAMIENTO INTERNO

Tras analizar el crate, hemos identificado que **~40 sitios en 14+ archivos** usan `serde_json` para almacenamiento interno que el usuario NUNCA lee directamente. JSON es obligatorio para APIs externas (Ollama, OpenAI, etc.) y archivos de usuario (config, exports), pero para estado interno es puro desperdicio.

**Contexto de rendimiento**:
- bincode es **10-50x más rápido** que serde_json en serialización/deserialización
- bincode ocupa **60-70% menos** espacio que JSON
- bincode + gzip (ya tenemos `flate2`) ocupa **85-90% menos**
- Todos los tipos ya tienen `#[derive(Serialize, Deserialize)]` — bincode funciona sin cambios en structs

**Regla de oro**: JSON se queda para APIs externas y archivos que el usuario lee. bincode para todo lo demás.

#### Tarea 3.1: Añadir dependencia bincode y crear módulo de serialización interna

**Lee primero**: `Cargo.toml`, `src/cache_compression.rs` (ya usa flate2), `src/export.rs` (ya tiene formatos)

1. Añadir `bincode` a Cargo.toml como dependencia opcional:
   ```toml
   bincode = { version = "1", optional = true }
   ```
2. Añadir un feature flag:
   ```toml
   binary-storage = ["bincode"]
   ```
3. Añadir `"binary-storage"` a la lista del feature `full`
4. Crear módulo `src/internal_storage.rs` con:
   ```rust
   //! Internal storage format abstraction.
   //! Uses bincode+gzip when feature "binary-storage" is enabled, JSON otherwise.

   use std::path::Path;
   use anyhow::Result;
   use serde::{Serialize, de::DeserializeOwned};

   /// Save data to a file using the optimal internal format.
   /// With "binary-storage": bincode + gzip compression.
   /// Without: JSON pretty-printed (for debugging).
   pub fn save_internal<T: Serialize>(data: &T, path: &Path) -> Result<()> { ... }

   /// Load data from a file saved with save_internal.
   pub fn load_internal<T: DeserializeOwned>(path: &Path) -> Result<T> { ... }

   /// Serialize data to bytes (in-memory, no file I/O).
   pub fn serialize_internal<T: Serialize>(data: &T) -> Result<Vec<u8>> { ... }

   /// Deserialize data from bytes.
   pub fn deserialize_internal<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> { ... }
   ```
5. La implementación debe:
   - Con feature `binary-storage`: usar `bincode::serialize()` + `flate2::write::GzEncoder` para guardar, y `flate2::read::GzDecoder` + `bincode::deserialize()` para cargar
   - Sin feature `binary-storage`: usar `serde_json::to_string_pretty()` / `from_str()` como fallback (útil para debugging)
   - Detectar automáticamente el formato al cargar (si empieza con `{` o `[` es JSON, si no es bincode+gzip) para compatibilidad con archivos existentes
6. Registrar el módulo en `lib.rs` y exportar las funciones públicas

#### Tarea 3.2: Herramientas de depuración para bincode

**Lee primero**: `src/internal_storage.rs` (recién creado), `src/debug.rs`

bincode no es legible por humanos, así que necesitamos herramientas para inspeccionarlo cuando algo falle. Añadir a `internal_storage.rs`:

1. **Función `dump_as_json`**: Lee un archivo bincode+gzip y lo imprime como JSON legible:
   ```rust
   /// Deserializa un archivo interno y lo devuelve como JSON pretty-printed.
   /// Útil para depuración: `dump_as_json::<ChatSessionStore>(path)`
   pub fn dump_as_json<T: Serialize + DeserializeOwned>(path: &Path) -> Result<String> {
       let data: T = load_internal(path)?;
       Ok(serde_json::to_string_pretty(&data)?)
   }
   ```

2. **Función `convert_to_json`**: Convierte un archivo bincode a JSON en disco (para inspección manual):
   ```rust
   /// Convierte un archivo interno (bincode+gzip) a un archivo JSON legible.
   /// El archivo JSON se crea junto al original con extensión .debug.json
   pub fn convert_to_json<T: Serialize + DeserializeOwned>(path: &Path) -> Result<PathBuf> {
       let json = dump_as_json::<T>(path)?;
       let debug_path = path.with_extension("debug.json");
       std::fs::write(&debug_path, &json)?;
       Ok(debug_path)
   }
   ```

3. **Función `convert_json_to_binary`**: Para migrar archivos JSON existentes a bincode:
   ```rust
   /// Convierte un archivo JSON existente al formato binario interno.
   pub fn convert_json_to_binary<T: Serialize + DeserializeOwned>(
       json_path: &Path,
       binary_path: &Path,
   ) -> Result<()> {
       let json_str = std::fs::read_to_string(json_path)?;
       let data: T = serde_json::from_str(&json_str)?;
       save_internal(&data, binary_path)
   }
   ```

4. **Función `file_info`**: Muestra metadatos de un archivo interno:
   ```rust
   pub struct InternalFileInfo {
       pub path: PathBuf,
       pub format: StorageFormat,          // Binary o Json
       pub size_bytes: u64,                // Tamaño en disco
       pub uncompressed_bytes: Option<u64>, // Tamaño sin comprimir (si bincode+gzip)
       pub compression_ratio: Option<f64>,  // Ratio de compresión
   }

   pub fn file_info(path: &Path) -> Result<InternalFileInfo> { ... }
   ```

5. **Macro `debug_dump!`** (solo en debug builds):
   ```rust
   /// En debug builds, guarda una copia JSON junto al archivo binario.
   /// En release, no hace nada (zero-cost).
   #[cfg(debug_assertions)]
   macro_rules! debug_dump {
       ($data:expr, $path:expr) => {
           if let Ok(json) = serde_json::to_string_pretty($data) {
               let _ = std::fs::write($path.with_extension("debug.json"), &json);
           }
       };
   }

   #[cfg(not(debug_assertions))]
   macro_rules! debug_dump {
       ($data:expr, $path:expr) => {};
   }
   ```

6. Opcionalmente, en `save_internal()`, llamar a `debug_dump!` para que en builds de desarrollo siempre se genere un .debug.json al lado del archivo binario. En release, el macro se elimina por el compilador (zero-cost).

7. Tests:
   - Guardar como bincode, dump_as_json, verificar que el JSON resultante es válido y contiene los datos correctos
   - convert_to_json + convert_json_to_binary: round-trip completo
   - file_info muestra formato y tamaños correctos
   - Auto-detección: load_internal lee tanto archivos JSON legacy como bincode nuevos

#### Tarea 3.3: Migrar almacenamiento de sesiones a internal_storage

**Lee primero**: `src/session.rs`

1. En `ChatSessionStore::save_to_file()`: reemplazar `serde_json::to_string_pretty()` por `internal_storage::save_internal()`
2. En `ChatSessionStore::load_from_file()`: reemplazar `serde_json::from_str()` por `internal_storage::load_internal()`
3. La función `load_internal` debe auto-detectar formato (JSON legacy vs bincode nuevo) para no romper sesiones existentes
4. Test: crear sesión, guardar, cargar, verificar que es idéntica

#### Tarea 3.4: Migrar almacenamiento de snapshots y estado interno

**Lee primero**: Cada archivo antes de modificarlo

Migrar estos módulos para usar `internal_storage::save_internal` / `load_internal` en vez de `serde_json` directo. **SOLO para funciones de guardado/carga a disco de estado interno** (NO tocar nada que vaya a APIs ni archivos de usuario):

1. `src/conversation_snapshot.rs` — snapshots a disco
2. `src/content_versioning.rs` — historial de versiones
3. `src/feed_monitor.rs` — estado del monitor de feeds
4. `src/memory.rs` — almacenamiento de memorias
5. `src/metrics.rs` — exportación de métricas a disco
6. `src/entities.rs` — almacenamiento de facts
7. `src/decision_tree.rs` — guardado de árboles
8. `src/auto_indexing.rs` — estado del indexador (si feature rag)
9. `src/incremental_sync.rs` — estado de sincronización
10. `src/multi_layer_graph.rs` — estado del grafo
11. `src/benchmark.rs` — resultados de benchmarks
12. `src/persistence.rs` — backups (ya comprime con flate2, integrar con el nuevo sistema)

**Para cada archivo**:
- Leer primero
- Identificar las funciones save/load que usan `serde_json` para disco
- Reemplazar por `internal_storage::save_internal()` / `load_internal()`
- Mantener auto-detección de formato para compatibilidad con archivos existentes
- Verificar que compila después de cada cambio

**NO tocar**:
- `src/providers.rs` — APIs externas, JSON obligatorio
- `src/config_file.rs` — archivos de usuario, JSON/TOML legible
- `src/export.rs` — exports de usuario, multi-formato ya soportado
- `src/openapi_export.rs` — estándar OpenAPI requiere JSON
- Cualquier `serde_json::json!()` usado para construir request bodies de API
- Cualquier `response.into_json()` usado para parsear respuestas de API

---

### FASE 4: MEJORAS DE TESTING

#### Tarea 4.1: Trait HttpClient para mocking de proveedores

**Lee primero**: `src/providers.rs`, `src/connection_pool.rs`

1. Crear un trait `HttpClient` en providers.rs (o un módulo nuevo `src/http_client.rs`):
   ```rust
   pub trait HttpClient: Send + Sync {
       fn get_json(&self, url: &str, timeout_secs: u64) -> Result<serde_json::Value>;
       fn post_json(&self, url: &str, body: &serde_json::Value, timeout_secs: u64) -> Result<serde_json::Value>;
   }
   ```
2. Implementar `UreqClient` que envuelve las llamadas actuales de `ureq`
3. Modificar las funciones de providers.rs para aceptar `&dyn HttpClient` en vez de hacer `ureq::get()` directamente
4. Mantener las funciones públicas actuales como wrappers que usan `UreqClient` por defecto (backward compatible)
5. Crear un `MockHttpClient` bajo `#[cfg(test)]` con respuestas configurables
6. Escribir al menos 3 tests:
   - Test de parseo de respuesta de Ollama models
   - Test de parseo de respuesta OpenAI-compatible models
   - Test de error de conexión (el mock devuelve error)

---

## REGLAS GENERALES

1. **Lee antes de escribir**: Siempre lee los archivos que vas a modificar ANTES de editarlos
2. **Compila después de cada tarea**: `cargo check --features full` (y `cargo check --features core` para verificar que no rompes builds mínimos)
3. **No duplicar código**: Si algo ya existe en un módulo, importarlo y reutilizarlo
4. **Backward compatible**: No romper la API pública existente. Añadir, no reemplazar
5. **Feature gates**: Respetar los feature flags existentes. Código que necesita `rusqlite` va bajo `#[cfg(feature = "rag")]`, etc.
6. **Tests**: Añadir al menos un test por cada funcionalidad nueva. Pueden ir en `#[cfg(test)] mod tests` al final del módulo
7. **Dependencias**: La única dependencia nueva permitida es `bincode = { version = "1", optional = true }` para la Fase 3. No añadir ninguna otra. Ya tenemos `regex`, `serde`, `serde_json`, `ureq`, `aes-gcm`, `flate2`, etc.
8. **Documentar**: Añadir `///` doc comments a funciones y structs públicas nuevas
9. **Errores**: Usar `anyhow::Result` para funciones internas y `AiError`/`AiResult` para la API pública
10. **Naming**: Seguir las convenciones existentes del crate (snake_case, structs con PascalCase, etc.)

## VERIFICACIÓN FINAL

Después de completar todas las tareas:
1. `cargo check --features full` debe compilar sin errores
2. `cargo check --features core` debe compilar sin errores
3. `cargo check` (default features = full) debe compilar sin errores
4. `cargo test --features full` debe pasar todos los tests existentes + los nuevos
5. Actualizar `PENDING.md` con un resumen de lo implementado
6. Actualizar `docs/IMPROVEMENTS.md` marcando las tareas completadas con ✅
