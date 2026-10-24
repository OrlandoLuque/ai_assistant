//! HTTP client abstraction for provider communication.
//!
//! Defines a trait `HttpClient` that abstracts HTTP operations, allowing
//! the provider layer to be tested with mock clients. The default
//! implementation `UreqClient` wraps the `ureq` HTTP library.
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::http_client::{HttpClient, UreqClient};
//!
//! let client = UreqClient;
//! let result = client.get_json("http://localhost:11434/api/tags", 5);
//! ```

use anyhow::{Context, Result};
use std::time::Duration;

/// Tope por defecto del cuerpo que `send` se trae a memoria: 10 MiB, el mismo
/// que `ureq` aplica en `read_to_vec`.
///
/// Se **falla** al pasarse, no se trunca. Un cuerpo truncado es JSON invalido o,
/// peor, JSON valido con menos datos.
pub const DEFAULT_MAX_BODY_BYTES: u64 = 10 * 1024 * 1024;

/// Metodo HTTP.
///
/// Los seis que el barrido de 2026-10-23 encontro en uso: `GET` y `POST` en 35 y
/// 32 ficheros, y `PUT`/`DELETE`/`HEAD`/`PATCH` en seis entre todos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST`.
    Post,
    /// `PUT`.
    Put,
    /// `DELETE`.
    Delete,
    /// `HEAD`.
    Head,
    /// `PATCH`.
    Patch,
    /// `OPTIONS`.
    ///
    /// Se anadio en V429, y el hueco es instructivo: el barrido de V427 midio
    /// los metodos que YO habia puesto en la lista de patrones, y `OPTIONS` no
    /// estaba. Lo destapo el primer consumidor que lo necesitaba
    /// (`vllm_capability`, que sondea si existe la ruta de LoRA), no la
    /// medicion. Un barrido mide lo que le preguntas.
    Options,
}

impl HttpMethod {
    /// El nombre del metodo tal como va en la peticion.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Patch => "PATCH",
            Self::Options => "OPTIONS",
        }
    }

    /// Si el metodo lleva cuerpo.
    pub fn takes_body(self) -> bool {
        matches!(self, Self::Post | Self::Put | Self::Patch)
    }
}

/// Una peticion HTTP, con lo que los 60 modulos que hablan `ureq` a pelo
/// resultaron necesitar de verdad.
///
/// ```rust,no_run
/// use ai_assistant::{HttpClient, HttpRequest, UreqClient};
/// use std::time::Duration;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let request = HttpRequest::post_json(
///     "http://localhost:11434/api/chat",
///     &serde_json::json!({"model": "llama3.1:8b"}),
/// )?
/// .header("Authorization", "Bearer un-token")
/// .timeout(Duration::from_secs(60));
///
/// // Un no-2xx NO es un error de transporte: vuelve con su cuerpo.
/// let response = UreqClient.send(&request)?;
/// if response.is_success() {
///     let body = response.json()?;
///     println!("{}", body);
/// } else {
///     // Y aqui sigue estando el mensaje del proveedor, y el Retry-After.
///     eprintln!("HTTP {}: {}", response.status, response.text()?);
///     if let Some(wait) = response.header("retry-after") {
///         eprintln!("reintentar en {}s", wait);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// La lista no es de diseno, es **medida** (2026-10-23, N77): un plazo global en
/// 45 ficheros, cabeceras propias en 35, cuerpo JSON en 21, y la cola de
/// `PUT`/`DELETE`/`HEAD`, redirecciones y plazo de conexion en menos de diez
/// cada una.
///
/// Campos publicos y constructores a proposito: un tipo que no se puede
/// construir desde fuera de la crate no es un puerto, es un detalle interno con
/// `pub` delante.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// El metodo.
    pub method: HttpMethod,
    /// La URL completa.
    pub url: String,
    /// Cabeceras a enviar, en orden.
    pub headers: Vec<(String, String)>,
    /// El cuerpo, en bytes.
    ///
    /// Bytes y no `serde_json::Value` porque de los 24 ficheros que leen JSON
    /// hay 23 que leen texto plano, y forzar JSON los habria dejado fuera.
    pub body: Option<Vec<u8>>,
    /// Plazo total de la peticion.
    pub timeout: Duration,
    /// Plazo SOLO para establecer la conexion, mas corto que el total.
    ///
    /// Sin esto, `ureq` 3 ante un puerto cerrado gasta todo el presupuesto
    /// reintentando y devuelve un timeout global, con lo que "no hay nadie
    /// escuchando" y "el servidor tarda" se vuelven el mismo error. Medido en
    /// V422.
    pub connect_timeout: Option<Duration>,
    /// Cuantas redirecciones seguir. `Some(0)` las prohibe (proteccion SSRF).
    pub max_redirects: Option<u32>,
    /// Tope del cuerpo a traer a memoria. Ver [`DEFAULT_MAX_BODY_BYTES`].
    pub max_body_bytes: u64,
}

impl HttpRequest {
    /// Una peticion con el metodo y la URL dados y todo lo demas por defecto.
    pub fn new(method: HttpMethod, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            timeout: Duration::from_secs(30),
            connect_timeout: None,
            max_redirects: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }

    /// Un `GET`.
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(HttpMethod::Get, url)
    }

    /// Un `POST` con cuerpo JSON y su `Content-Type` ya puesto.
    ///
    /// La cabecera va aqui porque `ureq::send_json` la ponia sola y quitarla sin
    /// darse cuenta fue una de las trampas de la subida a `ureq` 3: algunos
    /// servidores rechazan JSON sin declarar.
    pub fn post_json(url: impl Into<String>, body: &serde_json::Value) -> Result<Self> {
        let bytes = serde_json::to_vec(body).context("failed to serialize request body")?;
        Ok(Self::new(HttpMethod::Post, url)
            .header("Content-Type", "application/json")
            .body(bytes))
    }

    /// Anade una cabecera.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Pone el cuerpo.
    pub fn body(mut self, bytes: Vec<u8>) -> Self {
        self.body = Some(bytes);
        self
    }

    /// Pone el plazo total.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Pone el plazo de conexion.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Limita las redirecciones. `0` las prohibe.
    pub fn max_redirects(mut self, max: u32) -> Self {
        self.max_redirects = Some(max);
        self
    }

    /// Cambia el tope del cuerpo.
    pub fn max_body_bytes(mut self, max: u64) -> Self {
        self.max_body_bytes = max;
        self
    }
}

/// Una respuesta HTTP, **venga o no con un 2xx**.
///
/// Esa es la decision de fondo del puerto. En `ureq` 3,
/// `Error::StatusCode(u16)` no lleva la respuesta, asi que los diez sitios que
/// necesitaban el cuerpo del error tuvieron que pedir
/// `http_status_as_error(false)` uno por uno (V422), y los que se olvidaran
/// perderian en silencio el mensaje del proveedor y el `Retry-After` del 429.
/// Aqui no hay interruptor: el no-2xx **siempre** llega entero, y decidir que
/// hacer con el es del llamante.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Codigo de estado.
    pub status: u16,
    /// Cabeceras de la respuesta, en orden.
    pub headers: Vec<(String, String)>,
    /// Cuerpo en bytes.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Si el estado esta en 200-299.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Busca una cabecera **ignorando mayusculas**.
    ///
    /// Los nombres de cabecera son insensibles a mayusculas por RFC 9110, y un
    /// servidor puede responder `retry-after` donde otro manda `Retry-After`.
    /// Comparar literalmente es un fallo que solo aparece contra el servidor que
    /// no te esperabas.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }

    /// El cuerpo como texto.
    pub fn text(&self) -> Result<String> {
        String::from_utf8(self.body.clone()).context("response body is not valid UTF-8")
    }

    /// El cuerpo parseado como JSON.
    pub fn json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.body).context("failed to parse JSON response")
    }

    /// Error si el estado no es 2xx, **con el cuerpo dentro del mensaje**.
    ///
    /// Lo que convierte "el servidor dijo que no" en un diagnostico: el cuerpo
    /// es donde el proveedor explica por que.
    pub fn error_for_status(&self) -> Result<()> {
        if self.is_success() {
            return Ok(());
        }
        let body = String::from_utf8_lossy(&self.body);
        let shown: String = body.chars().take(500).collect();
        anyhow::bail!(
            "HTTP {}{}",
            self.status,
            if shown.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", shown)
            }
        )
    }
}

/// Busca una cabecera por nombre **ignorando mayusculas**, que es lo que dice
/// el RFC 9110: los nombres de cabecera son insensibles a mayusculas, y un
/// servidor manda `retry-after` donde otro manda `Retry-After`.
///
/// Una sola implementacion para [`HttpResponse`] y [`HttpStream`]. Nacio
/// duplicada --- copie el cuerpo al anadir `HttpStream` en V429 --- y lo cazo la
/// puerta de mutacion, que dijo "la spec dice 1 ocurrencia, el fichero tiene 2".
/// Que es justo la razon de ser de esa cuenta: dos copias de una regla son dos
/// sitios donde arreglarla, y uno donde olvidarse.
fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Una respuesta cuyo cuerpo se lee como FLUJO, con su estado delante.
///
/// El estado va aqui y no se descarta por una razon concreta, encontrada
/// migrando el descargador de corpus en V429: `send_streaming` devolvia solo el
/// lector, asi que un 404 con una pagina de error HTML se escribia en el
/// `.part` y se renombraba como si fuera el corpus. **Un flujo sin su estado es
/// un OK mas estrecho que la pregunta.**
///
/// Las cabeceras vienen tambien porque son lo que dice el tamano esperado
/// (`Content-Length`) y el tipo, y decidir si seguir leyendo sin ellas es
/// decidir a ciegas.
pub struct HttpStream {
    /// Codigo de estado.
    pub status: u16,
    /// Cabeceras de la respuesta.
    pub headers: Vec<(String, String)>,
    /// El cuerpo, sin traer a memoria.
    pub body: Box<dyn std::io::Read + Send>,
}

impl HttpStream {
    /// Si el estado esta en 200-299.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Busca una cabecera ignorando mayusculas, igual que [`HttpResponse::header`].
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }
}

/// Abstraction over HTTP operations for provider communication.
///
/// Implementing this trait allows swapping the real HTTP client with a mock
/// for testing provider parsing logic without network access.
pub trait HttpClient: Send + Sync {
    /// Envia la peticion y devuelve la respuesta, **sea o no un 2xx**.
    ///
    /// Es el unico metodo que un implementador tiene que escribir; todo lo demas
    /// del trait se deriva de aqui.
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse>;

    /// Envia la peticion y devuelve el cuerpo como flujo, sin traerlo a memoria.
    ///
    /// Separado de [`HttpClient::send`] **a proposito**, no por simetria: 18 de
    /// los 60 modulos medidos usan `into_reader()`, y lo que leen son descargas
    /// de modelos y flujos SSE. Meter eso en un `Vec<u8>` no seria un refactor,
    /// seria cargarse el caso de uso --- un GGUF de 5 GB en RAM.
    fn send_streaming(&self, request: &HttpRequest) -> Result<HttpStream>;

    /// `GET` que devuelve JSON, o error si el estado no es 2xx.
    ///
    /// Se conserva porque `providers` y `async_providers` la llaman. Antes
    /// perdia el cuerpo del no-2xx (en `ureq` 3 `Error::StatusCode` no lo
    /// lleva); ahora lo mete en el mensaje, que es estrictamente mejor para
    /// diagnosticar.
    fn get_json(&self, url: &str, timeout_secs: u64) -> Result<serde_json::Value> {
        let request = HttpRequest::get(url).timeout(Duration::from_secs(timeout_secs));
        let response = self
            .send(&request)
            .with_context(|| format!("GET request failed: {}", url))?;
        response
            .error_for_status()
            .with_context(|| format!("GET {}", url))?;
        response.json()
    }

    /// `POST` con cuerpo JSON que devuelve JSON, o error si el estado no es 2xx.
    fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        let request = HttpRequest::post_json(url, body)?.timeout(Duration::from_secs(timeout_secs));
        let response = self
            .send(&request)
            .with_context(|| format!("POST request failed: {}", url))?;
        response
            .error_for_status()
            .with_context(|| format!("POST {}", url))?;
        response.json()
    }

    /// `POST` con cuerpo JSON cuya respuesta se lee como flujo.
    fn post_streaming(
        &self,
        url: &str,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<Box<dyn std::io::Read + Send>> {
        let request = HttpRequest::post_json(url, body)?.timeout(Duration::from_secs(timeout_secs));
        let stream = self
            .send_streaming(&request)
            .with_context(|| format!("POST streaming request failed: {}", url))?;
        // Conserva el comportamiento de antes --- `.call()` sin
        // `http_status_as_error(false)` erraba ante un no-2xx --- y ahora puede
        // hacerlo mirando el estado en vez de dependiendo de un interruptor.
        if !stream.is_success() {
            anyhow::bail!("POST {} returned HTTP {}", url, stream.status);
        }
        Ok(stream.body)
    }
}

/// Default HTTP client using the `ureq` library.
///
/// This is the production client used by all provider functions.
/// Wraps `ureq::get()` and `ureq::post()` with configurable timeouts.
pub struct UreqClient;

impl UreqClient {
    /// Construye la peticion de `ureq` comun a `send` y `send_streaming`.
    ///
    /// `http_status_as_error(false)` va aqui y **no es opcional**: es lo que
    /// hace que el no-2xx vuelva con cuerpo y cabeceras, que es el contrato de
    /// [`HttpResponse`]. Ponerlo en un solo sitio es justamente el punto del
    /// puerto --- antes eran diez decisiones sueltas y ocho las tomaban bien.
    fn build(request: &HttpRequest) -> Result<ureq::http::Request<Vec<u8>>> {
        let mut builder = ureq::http::Request::builder()
            .method(request.method.as_str())
            .uri(&request.url);
        for (name, value) in &request.headers {
            builder = builder.header(name, value);
        }
        builder
            .body(request.body.clone().unwrap_or_default())
            .map_err(|e| {
                anyhow::anyhow!(
                    "bad {} request for {}: {}",
                    request.method.as_str(),
                    request.url,
                    e
                )
            })
    }

    /// El agente con la configuracion de ESTA peticion.
    fn agent(request: &HttpRequest) -> ureq::Agent {
        let mut config = ureq::Agent::config_builder()
            .timeout_global(Some(request.timeout))
            .http_status_as_error(false);
        if let Some(connect) = request.connect_timeout {
            config = config.timeout_connect(Some(connect));
        }
        if let Some(max) = request.max_redirects {
            config = config.max_redirects(max);
        }
        ureq::Agent::new_with_config(config.build())
    }
}

impl HttpClient for UreqClient {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse> {
        let http_request = Self::build(request)?;
        let mut response = Self::agent(request)
            .run(http_request)
            .with_context(|| format!("{} {}", request.method.as_str(), request.url))?;

        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    v.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let body = response
            .body_mut()
            .with_config()
            .limit(request.max_body_bytes)
            .read_to_vec()
            .with_context(|| {
                format!(
                    "failed to read response body from {} (limit {} bytes)",
                    request.url, request.max_body_bytes
                )
            })?;

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    fn send_streaming(&self, request: &HttpRequest) -> Result<HttpStream> {
        let http_request = Self::build(request)?;
        let response = Self::agent(request)
            .run(http_request)
            .with_context(|| format!("{} {} (streaming)", request.method.as_str(), request.url))?;
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    v.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        Ok(HttpStream {
            status,
            headers,
            body: Box::new(response.into_body().into_reader()),
        })
    }
}

// ============================================================================
// Parsing functions (shared between real and mock clients)
// ============================================================================

use crate::config::AiProvider;
use crate::models::{format_size, ModelInfo};

/// Parse an Ollama `/api/tags` response into a list of ModelInfo.
///
/// This is the core parsing logic extracted from `fetch_ollama_models` so it
/// can be tested with mock data without network access.
pub fn parse_ollama_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(model_list) = body.get("models").and_then(|m| m.as_array()) {
        for model in model_list {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                models.push(ModelInfo {
                    name: name.to_string(),
                    provider: AiProvider::Ollama,
                    size: model.get("size").and_then(|s| s.as_u64()).map(format_size),
                    modified_at: model
                        .get("modified_at")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string()),
                    capabilities: None,
                });
            }
        }
    }
    models
}

/// Parse an OpenAI-compatible `/v1/models` response into a list of ModelInfo.
pub fn parse_openai_models(body: &serde_json::Value, provider: AiProvider) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        for model in data {
            if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                models.push(ModelInfo {
                    name: id.to_string(),
                    provider: provider.clone(),
                    size: None,
                    modified_at: None,
                    capabilities: None,
                });
            }
        }
    }
    models
}

/// Parse a Kobold.cpp `/api/v1/model` response into a list of ModelInfo.
pub fn parse_kobold_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    if let Some(result) = body.get("result").and_then(|r| r.as_str()) {
        if !result.is_empty() && result != "Read Only" {
            models.push(ModelInfo {
                name: result.to_string(),
                provider: AiProvider::KoboldCpp,
                size: None,
                modified_at: None,
                capabilities: None,
            });
        }
    }
    models
}

/// Fetch Ollama models using a provided HttpClient.
///
/// This is the testable version that accepts any HttpClient implementation.
/// The public `fetch_ollama_models` in providers.rs delegates to this.
pub fn fetch_ollama_models_with(client: &dyn HttpClient, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    let body = client
        .get_json(&url, 5)
        .context("Failed to connect to Ollama")?;
    Ok(parse_ollama_models(&body))
}

/// Fetch OpenAI-compatible models using a provided HttpClient.
pub fn fetch_openai_models_with(
    client: &dyn HttpClient,
    base_url: &str,
    provider: AiProvider,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url);
    let body = client
        .get_json(&url, 5)
        .context("Failed to connect to OpenAI-compatible API")?;
    Ok(parse_openai_models(&body, provider))
}

/// Fetch Kobold.cpp models using a provided HttpClient.
pub fn fetch_kobold_models_with(client: &dyn HttpClient, base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);
    let body = client
        .get_json(&url, 5)
        .context("Failed to connect to Kobold.cpp")?;
    Ok(parse_kobold_models(&body))
}

// ============================================================================
// Feature flag validation macro
// ============================================================================

/// Macro that validates feature flag combinations at compile time.
/// Produces a clear compilation error if invalid combinations are used.
///
/// Usage in lib.rs (added during wiring):
/// ```rust,ignore
/// validate_features!();
/// ```
#[macro_export]
macro_rules! validate_features {
    () => {
        // webrtc requires voice-agent
        #[cfg(all(feature = "webrtc", not(feature = "voice-agent")))]
        compile_error!(
            "Feature 'webrtc' requires 'voice-agent'. Add voice-agent to your Cargo.toml features."
        );

        // browser requires autonomous
        #[cfg(all(feature = "browser", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'browser' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // scheduler requires autonomous
        #[cfg(all(feature = "scheduler", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'scheduler' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // butler requires autonomous
        #[cfg(all(feature = "butler", not(feature = "autonomous")))]
        compile_error!(
            "Feature 'butler' requires 'autonomous'. Add autonomous to your Cargo.toml features."
        );

        // distributed-agents requires autonomous + distributed-network
        #[cfg(all(feature = "distributed-agents", not(feature = "autonomous")))]
        compile_error!("Feature 'distributed-agents' requires 'autonomous'.");

        #[cfg(all(feature = "distributed-agents", not(feature = "distributed-network")))]
        compile_error!("Feature 'distributed-agents' requires 'distributed-network'.");

        // whisper-local requires audio
        #[cfg(all(feature = "whisper-local", not(feature = "audio")))]
        compile_error!("Feature 'whisper-local' requires 'audio'.");
    };
}

// ============================================================================
// Mock HTTP Server (test-only)
// ============================================================================

/// A mock HTTP server that listens on localhost and serves queued responses.
/// Used for testing modules that make real HTTP calls (mcp_client, media_generation, etc.)
///
/// # Example
/// ```rust,no_run
/// let mut server = MockHttpServer::start();
/// server.enqueue_json(200, serde_json::json!({"result": "ok"}));
/// // Use server.url() as the base URL for the module under test
/// let url = server.url();
/// // ... make HTTP call to url ...
/// let received = server.last_request();
/// ```
#[cfg(test)]
pub struct MockHttpServer {
    addr: std::net::SocketAddr,
    /// Queued responses: (status_code, content_type, body)
    /// Cola de respuestas: (estado, content-type, cabeceras extra, cuerpo).
    #[allow(clippy::type_complexity)]
    responses: std::sync::Arc<std::sync::Mutex<Vec<(u16, String, Vec<(String, String)>, String)>>>,
    /// Received requests: (method, path, body)
    requests: std::sync::Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    heads: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// Background thread handle
    _handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(test)]
impl MockHttpServer {
    /// Start a new mock server on a random port.
    pub fn start() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let responses = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        // La cabecera cruda, para poder comprobar que una cabecera que el
        // llamante puso llego de verdad al cable. Hasta V427 el mock la parseaba
        // y la tiraba, asi que ningun test podia probarlo --- y "el puerto
        // manda cabeceras" era justo la capacidad que le faltaba al trait.
        let heads: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let resp_clone = responses.clone();
        let req_clone = requests.clone();
        let head_clone = heads.clone();
        let ready_clone = ready.clone();

        let handle = std::thread::spawn(move || {
            // Set non-blocking with short timeout so the thread can exit
            listener.set_nonblocking(false).ok();

            // Signal that the server thread is ready to accept connections
            ready_clone.store(true, std::sync::atomic::Ordering::Release);

            // Accept connections in a loop
            for stream in listener.incoming() {
                match stream {
                    Ok(mut stream) => {
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                            .ok();

                        // Read the WHOLE request: headers, then exactly
                        // Content-Length bytes of body.
                        //
                        // This used to be a single `read()`. TCP is a stream, so
                        // under load a request arrives split across segments —
                        // the server would read the first chunk, reply, and close
                        // while the client was still writing, which the client
                        // sees as a connection reset. That made
                        // `test_otlp_exporter_flush_with_mock` and
                        // `test_mock_server_post_streaming` fail intermittently
                        // (measured: 2 of 3 full runs) while passing 5/5 in
                        // isolation — a broken helper, not fragile tests.
                        let mut raw: Vec<u8> = Vec::with_capacity(8192);
                        let mut chunk = [0u8; 4096];
                        let request_str = loop {
                            let n = match std::io::Read::read(&mut stream, &mut chunk) {
                                Ok(0) => break String::from_utf8_lossy(&raw).to_string(),
                                Ok(n) => n,
                                Err(_) => break String::from_utf8_lossy(&raw).to_string(),
                            };
                            raw.extend_from_slice(&chunk[..n]);

                            let text = String::from_utf8_lossy(&raw);
                            let Some(head_end) = text.find("\r\n\r\n") else {
                                continue; // headers still incomplete
                            };
                            // Content-Length decides how much body to expect; a
                            // request without one carries no body.
                            let want: usize = text[..head_end]
                                .lines()
                                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                                .and_then(|l| l.split(':').nth(1))
                                .and_then(|v| v.trim().parse().ok())
                                .unwrap_or(0);
                            if raw.len() >= head_end + 4 + want {
                                break text.to_string();
                            }
                        };

                        // Parse method, path, body
                        let lines: Vec<&str> = request_str.split("\r\n").collect();
                        let first_line = lines.first().unwrap_or(&"");
                        let parts: Vec<&str> = first_line.split(' ').collect();
                        let method = parts.first().unwrap_or(&"GET").to_string();
                        let path = parts.get(1).unwrap_or(&"/").to_string();

                        // Body is after the empty line
                        let body = if let Some(pos) = request_str.find("\r\n\r\n") {
                            request_str[pos + 4..].to_string()
                        } else {
                            String::new()
                        };

                        // La cabecera cruda es todo lo anterior a la linea vacia.
                        let head = request_str
                            .find("\r\n\r\n")
                            .map(|pos| request_str[..pos].to_string())
                            .unwrap_or_else(|| request_str.clone());
                        head_clone.lock().unwrap().push(head);

                        req_clone.lock().unwrap().push((method, path, body));

                        // Get next response from queue
                        let (status, content_type, extra_headers, resp_body) = {
                            let mut q = resp_clone.lock().unwrap();
                            if q.is_empty() {
                                (
                                    404,
                                    "text/plain".to_string(),
                                    Vec::new(),
                                    "No responses queued".to_string(),
                                )
                            } else {
                                q.remove(0)
                            }
                        };

                        // Map status code to reason phrase
                        let reason = match status {
                            200 => "OK",
                            201 => "Created",
                            204 => "No Content",
                            400 => "Bad Request",
                            404 => "Not Found",
                            500 => "Internal Server Error",
                            _ => "Unknown",
                        };

                        // Write HTTP response
                        let extra: String = extra_headers
                            .iter()
                            .map(|(k, v): &(String, String)| format!("{}: {}\r\n", k, v))
                            .collect();
                        let response = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
                            status, reason, content_type, resp_body.len(), extra, resp_body
                        );
                        std::io::Write::write_all(&mut stream, response.as_bytes()).ok();
                    }
                    Err(_) => break,
                }
            }
        });

        // Wait for server thread to be ready before returning
        while !ready.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::yield_now();
        }

        Self {
            addr,
            responses,
            requests,
            heads,
            _handle: Some(handle),
        }
    }

    /// Get the server's base URL (e.g., "http://127.0.0.1:12345")
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Enqueue a JSON response
    pub fn enqueue_json(&self, status: u16, body: serde_json::Value) {
        self.responses.lock().unwrap().push((
            status,
            "application/json".to_string(),
            Vec::new(),
            body.to_string(),
        ));
    }

    /// Encola una respuesta con cabeceras propias.
    ///
    /// Existe para poder devolver un `Retry-After` --- el 429 de N46 --- y
    /// comprobar que el puerto lo entrega al llamante.
    pub fn enqueue_with_headers(&self, status: u16, headers: Vec<(String, String)>, body: &str) {
        self.responses.lock().unwrap().push((
            status,
            "text/plain".to_string(),
            headers,
            body.to_string(),
        ));
    }

    /// La cabecera CRUDA de la ultima peticion recibida, tal como llego.
    ///
    /// Sondea un momento, igual que `last_request`, porque el hilo del servidor
    /// puede ir por detras del cliente.
    pub fn last_request_head(&self) -> Option<String> {
        for _ in 0..50 {
            if let Some(head) = self.heads.lock().unwrap().last().cloned() {
                return Some(head);
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        None
    }

    /// Enqueue a plain text response
    pub fn enqueue_text(&self, status: u16, body: &str) {
        self.responses.lock().unwrap().push((
            status,
            "text/plain".to_string(),
            Vec::new(),
            body.to_string(),
        ));
    }

    /// Get the last received request (method, path, body).
    /// Polls briefly to allow the server thread to process in-flight requests.
    pub fn last_request(&self) -> Option<(String, String, String)> {
        for _ in 0..50 {
            let result = self.requests.lock().unwrap().last().cloned();
            if result.is_some() {
                return result;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.requests.lock().unwrap().last().cloned()
    }

    /// Get all received requests
    pub fn all_requests(&self) -> Vec<(String, String, String)> {
        self.requests.lock().unwrap().clone()
    }

    /// Get the number of requests received
    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock HTTP client for testing provider parsing without network access.
    struct MockHttpClient {
        responses: Mutex<Vec<Result<serde_json::Value>>>,
    }

    impl MockHttpClient {
        fn with_response(value: serde_json::Value) -> Self {
            Self {
                responses: Mutex::new(vec![Ok(value)]),
            }
        }

        fn with_error(msg: &str) -> Self {
            Self {
                responses: Mutex::new(vec![Err(anyhow::anyhow!("{}", msg))]),
            }
        }
    }

    impl HttpClient for MockHttpClient {
        fn send(&self, _request: &HttpRequest) -> Result<HttpResponse> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No mock responses configured");
            }
            let value = responses.remove(0)?;
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: serde_json::to_vec(&value)?,
            })
        }

        fn send_streaming(&self, _request: &HttpRequest) -> Result<HttpStream> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No mock responses configured");
            }
            let value = responses.remove(0)?;
            let bytes = serde_json::to_vec(&value)?;
            Ok(HttpStream {
                status: 200,
                headers: Vec::new(),
                body: Box::new(std::io::Cursor::new(bytes)),
            })
        }
    }

    // ======================================================================
    // El puerto (N77). Cada test fija una capacidad que el trait de tres
    // metodos NO podia expresar, y por la que 60 modulos lo esquivaban.
    // ======================================================================

    /// LA razon por la que el puerto no se usaba. `get_json(url, timeout_secs)`
    /// no tiene donde poner un `Authorization`, asi que los 35 ficheros medidos
    /// que mandan cabeceras --- o sea todos los proveedores cloud --- no tenian
    /// forma de usarlo. No era indisciplina: era un parametro que faltaba.
    #[test]
    fn test_port_sends_custom_headers() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"ok": true}));

        let request = HttpRequest::get(format!("{}/v1/models", server.url()))
            .header("Authorization", "Bearer sk-not-a-real-key")
            .header("X-Custom-Thing", "42")
            .timeout(Duration::from_secs(5));
        let response = UreqClient.send(&request).expect("the mock server answered");
        assert!(response.is_success());

        let head = server
            .last_request_head()
            .expect("the server recorded the request head");
        // En MINUSCULAS a proposito: `ureq` 3 normaliza los nombres de cabecera
        // al mandarlos (son insensibles a mayusculas por RFC 9110), asi que
        // buscar "Authorization:" literal falla aunque la cabecera SI viaje. Lo
        // descubri haciendo fallar este test y volcando la peticion cruda.
        let lower = head.to_ascii_lowercase();
        assert!(
            lower.contains("authorization: bearer sk-not-a-real-key"),
            "head = {}",
            head
        );
        assert!(lower.contains("x-custom-thing: 42"), "head = {}", head);
    }

    /// El contrato de fondo: el no-2xx vuelve ENTERO, con cuerpo y estado, y no
    /// como un error sin cuerpo. En `ureq` 3, `Error::StatusCode(u16)` no lleva
    /// la respuesta, asi que diez sitios tuvieron que pedir
    /// `http_status_as_error(false)` uno a uno (V422) --- y el que se olvidara
    /// perdia en silencio el mensaje del proveedor.
    #[test]
    fn test_port_returns_non_2xx_with_its_body() {
        let server = MockHttpServer::start();
        server.enqueue_json(
            404,
            serde_json::json!({"error": {"message": "model not found"}}),
        );

        let request = HttpRequest::get(format!("{}/v1/models/nope", server.url()))
            .timeout(Duration::from_secs(5));
        let response = UreqClient
            .send(&request)
            .expect("a 404 is an answer, not a transport failure");

        assert_eq!(response.status, 404);
        assert!(!response.is_success());
        let json = response.json().expect("the error body is still JSON");
        assert_eq!(json["error"]["message"], "model not found");

        // Y el mensaje de `error_for_status` LLEVA el cuerpo, que es lo que
        // convierte "el servidor dijo que no" en un diagnostico.
        let err = response
            .error_for_status()
            .expect_err("404 is not a success");
        let text = format!("{}", err);
        assert!(text.contains("404"), "err = {}", text);
        assert!(text.contains("model not found"), "err = {}", text);
    }

    /// El caso de N46: el backoff honesto necesita el `Retry-After` de un 429.
    /// Y los nombres de cabecera son insensibles a mayusculas por RFC 9110, asi
    /// que compararlos literalmente es un fallo que solo aparece contra el
    /// servidor que no te esperabas.
    #[test]
    fn test_port_reads_response_headers_ignoring_case() {
        let server = MockHttpServer::start();
        server.enqueue_with_headers(
            500,
            vec![("Retry-After".to_string(), "42".to_string())],
            "slow down",
        );

        let request =
            HttpRequest::get(format!("{}/search", server.url())).timeout(Duration::from_secs(5));
        let response = UreqClient.send(&request).expect("the mock server answered");

        assert_eq!(response.header("Retry-After"), Some("42"));
        assert_eq!(response.header("retry-after"), Some("42"));
        assert_eq!(response.header("RETRY-AFTER"), Some("42"));
        // Y una que no esta es None, no una cadena vacia.
        assert_eq!(response.header("X-Not-Sent"), None);
        assert_eq!(response.text().expect("utf-8"), "slow down");
    }

    /// Los seis metodos medidos, no solo los dos del trait viejo. `PUT`,
    /// `DELETE` y `HEAD` aparecian en seis ficheros entre todos, y ninguno tenia
    /// por donde entrar.
    #[test]
    fn test_port_covers_methods_beyond_get_and_post() {
        let server = MockHttpServer::start();
        server.enqueue_text(200, "deleted");
        let request = HttpRequest::new(HttpMethod::Delete, format!("{}/thing/1", server.url()))
            .timeout(Duration::from_secs(5));
        UreqClient.send(&request).expect("DELETE went through");
        let (method, path, _) = server.last_request().expect("recorded");
        assert_eq!(method, "DELETE");
        assert_eq!(path, "/thing/1");

        let server2 = MockHttpServer::start();
        server2.enqueue_text(200, "put");
        let request = HttpRequest::new(HttpMethod::Put, format!("{}/thing/2", server2.url()))
            .body(b"payload".to_vec())
            .timeout(Duration::from_secs(5));
        UreqClient.send(&request).expect("PUT went through");
        let (method, _, body) = server2.last_request().expect("recorded");
        assert_eq!(method, "PUT");
        assert_eq!(body, "payload");
    }

    /// Un cuerpo por encima del tope FALLA, no se trunca a medias. Un cuerpo
    /// truncado es JSON invalido o --- mucho peor --- JSON valido con menos
    /// datos, que es un resultado silenciosamente equivocado.
    #[test]
    fn test_port_body_limit_fails_instead_of_truncating() {
        let server = MockHttpServer::start();
        let big = "x".repeat(4096);
        server.enqueue_text(200, &big);

        let request = HttpRequest::get(format!("{}/big", server.url()))
            .timeout(Duration::from_secs(5))
            .max_body_bytes(64);
        let result = UreqClient.send(&request);
        assert!(
            result.is_err(),
            "4 KiB with a 64-byte limit must fail, not come back short"
        );
    }

    /// `send_json` de `ureq` ponia el `Content-Type` por su cuenta, y perderlo
    /// sin darse cuenta fue una de las trampas de la subida a `ureq` 3: algunos
    /// servidores rechazan JSON sin declarar. Aqui se comprueba EN EL CABLE.
    #[test]
    fn test_port_post_json_declares_its_content_type() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"ok": true}));

        let request = HttpRequest::post_json(
            format!("{}/v1/chat", server.url()),
            &serde_json::json!({"model": "m"}),
        )
        .expect("serialises")
        .timeout(Duration::from_secs(5));
        UreqClient.send(&request).expect("the mock server answered");

        let head = server.last_request_head().expect("recorded");
        assert!(
            head.to_ascii_lowercase()
                .contains("content-type: application/json"),
            "head = {}",
            head
        );
        let (_, _, body) = server.last_request().expect("recorded");
        assert_eq!(body, r#"{"model":"m"}"#);
    }

    /// Los tres metodos de conveniencia siguen funcionando sobre `send`, que es
    /// lo que permite que `providers` y `async_providers` no cambien una linea.
    /// Y `get_json` ahora SI da error con un no-2xx, como antes --- comprobado,
    /// porque un metodo derivado es donde se pierde un comportamiento sin que
    /// nadie lo note.
    #[test]
    fn test_convenience_methods_still_delegate_and_still_reject_non_2xx() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"models": []}));
        let json = UreqClient
            .get_json(&format!("{}/api/tags", server.url()), 5)
            .expect("200 parses");
        assert!(json["models"].is_array());

        let server2 = MockHttpServer::start();
        server2.enqueue_json(500, serde_json::json!({"error": "boom"}));
        let err = UreqClient
            .get_json(&format!("{}/api/tags", server2.url()), 5)
            .expect_err("a 500 must still be an error for get_json");
        let text = format!("{:#}", err);
        assert!(text.contains("500"), "err = {}", text);
        assert!(
            text.contains("boom"),
            "el cuerpo del error va dentro: {}",
            text
        );
    }

    /// Fija las dos diferencias observables entre el puerto y el
    /// `ureq::get(..).call()` que sustituye, para que no se redescubran:
    ///
    /// 1. **Los nombres de cabecera viajan en minusculas.** `ureq` 3 los
    ///    normaliza. Es legal --- RFC 9110 los declara insensibles a
    ///    mayusculas --- pero rompe cualquier comparacion literal, y me costo
    ///    un test en rojo averiguarlo.
    /// 2. **Un `GET` sin cuerpo manda `content-length: 0`**, que
    ///    `ureq::get(..).call()` no mandaba. Tambien legal (RFC 9110 8.6), y
    ///    apuntado aqui porque un servidor quisquilloso es justo el tipo de
    ///    cosa que no se ve hasta produccion.
    #[test]
    fn test_port_wire_format_differences_are_pinned() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({}));
        let request = HttpRequest::get(format!("{}/x", server.url()))
            .header("X-Mixed-Case", "v")
            .timeout(Duration::from_secs(5));
        UreqClient.send(&request).expect("answered");

        let head = server.last_request_head().expect("recorded");
        assert!(
            head.contains("x-mixed-case: v"),
            "los nombres van en minusculas: {}",
            head
        );
        assert!(
            !head.contains("X-Mixed-Case"),
            "y NO en el caso original: {}",
            head
        );
        assert!(
            head.to_ascii_lowercase().contains("content-length: 0"),
            "un GET sin cuerpo manda content-length: 0: {}",
            head
        );
    }

    /// `takes_body` existe para que un llamante pueda preguntarlo; probado para
    /// que no sea una funcion publica que nadie ha ejecutado nunca.
    #[test]
    fn test_http_method_names_and_body_expectations() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
        for m in [HttpMethod::Post, HttpMethod::Put, HttpMethod::Patch] {
            assert!(m.takes_body(), "{:?} lleva cuerpo", m);
        }
        for m in [
            HttpMethod::Get,
            HttpMethod::Delete,
            HttpMethod::Head,
            HttpMethod::Options,
        ] {
            assert!(!m.takes_body(), "{:?} no lleva cuerpo", m);
        }
        assert_eq!(HttpMethod::Options.as_str(), "OPTIONS");
    }

    /// El hueco que destapo el descargador de corpus en V429: si
    /// `send_streaming` no trae el estado, un 404 con una pagina de error HTML
    /// se escribe en el fichero de destino y se renombra como si fuera el
    /// corpus. El flujo lleva su estado, y eso se comprueba.
    #[test]
    fn test_streaming_carries_its_status_and_headers() {
        let server = MockHttpServer::start();
        server.enqueue_with_headers(
            404,
            vec![("X-Why".to_string(), "no-such-corpus".to_string())],
            "<html>not found</html>",
        );

        let request = HttpRequest::get(format!("{}/corpus.tar.gz", server.url()))
            .timeout(Duration::from_secs(5));
        let mut stream = UreqClient
            .send_streaming(&request)
            .expect("un 404 es una respuesta, no un fallo de transporte");

        assert_eq!(stream.status, 404);
        assert!(!stream.is_success(), "y el llamante puede PREGUNTARLO");
        assert_eq!(stream.header("x-why"), Some("no-such-corpus"));

        // El cuerpo tambien esta --- es la pagina de error, no el corpus.
        let mut body = String::new();
        std::io::Read::read_to_string(&mut stream.body, &mut body).expect("se lee");
        assert!(body.contains("not found"), "body = {}", body);
    }

    #[test]
    fn test_parse_ollama_models_response() {
        let body = serde_json::json!({
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 4_000_000_000u64,
                    "modified_at": "2024-01-15T10:30:00Z"
                },
                {
                    "name": "qwen2.5:7b",
                    "size": 7_000_000_000u64,
                    "modified_at": "2024-02-01T08:00:00Z"
                }
            ]
        });

        let models = parse_ollama_models(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "llama3.2:latest");
        assert_eq!(models[0].provider, AiProvider::Ollama);
        assert!(models[0].size.is_some());
        assert_eq!(models[1].name, "qwen2.5:7b");
    }

    #[test]
    fn test_parse_ollama_empty_response() {
        let body = serde_json::json!({ "models": [] });
        let models = parse_ollama_models(&body);
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_openai_models_response() {
        let body = serde_json::json!({
            "data": [
                { "id": "gpt-4", "object": "model" },
                { "id": "gpt-3.5-turbo", "object": "model" }
            ]
        });

        let models = parse_openai_models(&body, AiProvider::LMStudio);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "gpt-4");
        assert_eq!(models[0].provider, AiProvider::LMStudio);
        assert_eq!(models[1].name, "gpt-3.5-turbo");
    }

    #[test]
    fn test_parse_kobold_models_response() {
        let body = serde_json::json!({ "result": "mistral-7b-instruct" });
        let models = parse_kobold_models(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "mistral-7b-instruct");
        assert_eq!(models[0].provider, AiProvider::KoboldCpp);
    }

    #[test]
    fn test_parse_kobold_read_only_ignored() {
        let body = serde_json::json!({ "result": "Read Only" });
        let models = parse_kobold_models(&body);
        assert!(models.is_empty());
    }

    #[test]
    fn test_fetch_ollama_with_mock() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "models": [
                { "name": "test-model:latest", "size": 1000000 }
            ]
        }));

        let models = fetch_ollama_models_with(&mock, "http://localhost:11434").unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test-model:latest");
    }

    #[test]
    fn test_fetch_openai_with_mock() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "data": [
                { "id": "local-model-1" },
                { "id": "local-model-2" }
            ]
        }));

        let models =
            fetch_openai_models_with(&mock, "http://localhost:1234", AiProvider::LMStudio).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_fetch_with_connection_error() {
        let mock = MockHttpClient::with_error("Connection refused");
        let result = fetch_ollama_models_with(&mock, "http://localhost:11434");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Ollama") || err.contains("Connection refused"));
    }

    #[test]
    fn test_mock_streaming_response() {
        let mock = MockHttpClient::with_response(serde_json::json!({
            "message": { "content": "Hello" },
            "done": false
        }));

        let reader = mock
            .post_streaming(
                "http://localhost:11434/api/chat",
                &serde_json::json!({}),
                300,
            )
            .unwrap();

        let mut content = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut content).unwrap();
        assert!(content.contains("Hello"));
    }

    #[test]
    fn test_parse_malformed_json_returns_empty() {
        // Missing "models" key
        let body = serde_json::json!({ "error": "bad request" });
        assert!(parse_ollama_models(&body).is_empty());

        // Missing "data" key
        assert!(parse_openai_models(&body, AiProvider::LMStudio).is_empty());

        // Missing "result" key
        assert!(parse_kobold_models(&body).is_empty());
    }

    // ========================================================================
    // MockHttpServer tests (Item 10.1 + 10.2)
    // ========================================================================

    #[test]
    fn test_mock_server_start() {
        let server = MockHttpServer::start();
        let url = server.url();
        assert!(url.starts_with("http://127.0.0.1:"));
        // Port should be > 0
        let port_str = url.rsplit(':').next().unwrap();
        let port: u16 = port_str.parse().unwrap();
        assert!(port > 0);
    }

    #[test]
    fn test_mock_server_basic_json_response() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"status": "ok"}));

        let client = UreqClient;
        let result = client.get_json(&format!("{}/health", server.url()), 5);
        assert!(result.is_ok());
        let body = result.unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[test]
    fn test_mock_server_post_json() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"id": 1}));

        let client = UreqClient;
        let result = client.post_json(
            &format!("{}/api/chat", server.url()),
            &serde_json::json!({"message": "hello"}),
            5,
        );
        assert!(result.is_ok());

        let (method, path, _body) = server.last_request().unwrap();
        assert_eq!(method, "POST");
        assert!(path.contains("/api/chat"));
    }

    #[test]
    fn test_mock_server_error_response() {
        let server = MockHttpServer::start();
        server.enqueue_json(500, serde_json::json!({"error": "internal"}));

        let client = UreqClient;
        let result = client.get_json(&format!("{}/fail", server.url()), 5);
        // ureq treats 500 as an error
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_server_multiple_requests() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({"n": 1}));
        server.enqueue_json(200, serde_json::json!({"n": 2}));

        let client = UreqClient;
        let r1 = client.get_json(&format!("{}/a", server.url()), 5).unwrap();
        let r2 = client.get_json(&format!("{}/b", server.url()), 5).unwrap();

        assert_eq!(r1["n"], 1);
        assert_eq!(r2["n"], 2);
        assert_eq!(server.request_count(), 2);
    }

    #[test]
    fn test_mock_server_no_response_queued() {
        let server = MockHttpServer::start();
        // Don't enqueue anything — server returns 404
        let client = UreqClient;
        let result = client.get_json(&format!("{}/empty", server.url()), 5);
        // Should get 404 or error
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_server_text_response() {
        let server = MockHttpServer::start();
        server.enqueue_text(200, "Hello, world!");

        // Use a raw TCP connection to verify the text response
        let mut stream = std::net::TcpStream::connect(server.addr).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .ok();
        let request = "GET /text HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        std::io::Write::write_all(&mut stream, request.as_bytes()).unwrap();

        let mut response = String::new();
        std::io::Read::read_to_string(&mut stream, &mut response).unwrap();
        assert!(response.contains("Hello, world!"));
        assert!(response.contains("text/plain"));
    }

    #[test]
    fn test_mock_server_request_count() {
        let server = MockHttpServer::start();
        assert_eq!(server.request_count(), 0);

        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));

        let client = UreqClient;
        let _ = client.get_json(&format!("{}/1", server.url()), 5);
        assert_eq!(server.request_count(), 1);

        let _ = client.get_json(&format!("{}/2", server.url()), 5);
        assert_eq!(server.request_count(), 2);

        let _ = client.get_json(&format!("{}/3", server.url()), 5);
        assert_eq!(server.request_count(), 3);
    }

    #[test]
    fn test_mock_server_all_requests() {
        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({}));
        server.enqueue_json(200, serde_json::json!({}));

        let client = UreqClient;
        let _ = client.get_json(&format!("{}/first", server.url()), 5);
        let _ = client.post_json(
            &format!("{}/second", server.url()),
            &serde_json::json!({"key": "value"}),
            5,
        );

        let requests = server.all_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].0, "GET");
        assert!(requests[0].1.contains("/first"));
        assert_eq!(requests[1].0, "POST");
        assert!(requests[1].1.contains("/second"));
    }

    #[test]
    fn test_mock_server_post_streaming() {
        let server = MockHttpServer::start();
        server.enqueue_json(
            200,
            serde_json::json!({"message": {"content": "streamed"}, "done": false}),
        );

        let client = UreqClient;
        let reader = client
            .post_streaming(
                &format!("{}/api/chat", server.url()),
                &serde_json::json!({"model": "test", "stream": true}),
                5,
            )
            .unwrap();

        let mut content = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut content).unwrap();
        assert!(content.contains("streamed"));

        let (method, path, _) = server.last_request().unwrap();
        assert_eq!(method, "POST");
        assert!(path.contains("/api/chat"));
    }

    // ========================================================================
    // Feature flag validation macro test (Item 10.3)
    // ========================================================================

    #[test]
    fn test_validate_features_macro_exists() {
        // This test verifies the macro compiles.
        // Invalid feature combinations would cause compile errors.
        validate_features!();
    }
}
