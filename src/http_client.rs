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
use std::io::Read as _;
use std::time::Duration;

/// Tope por defecto del cuerpo que `send` se trae a memoria: 10 MiB, el mismo
/// que `ureq` aplica en `read_to_vec`.
///
/// Se **falla** al pasarse, no se trunca. Un cuerpo truncado es JSON invalido o,
/// peor, JSON valido con menos datos.
pub const DEFAULT_MAX_BODY_BYTES: u64 = 10 * 1024 * 1024;

/// Plazo de ATASCO por defecto para una transferencia: 60 s de silencio.
///
/// Sesenta y no treinta porque un servidor de LLM que carga el modelo en frio
/// tarda en contestar la primera cabecera, y eso cuenta como lectura: en este
/// mismo repositorio se midio una recarga de 166 s cuando Ollama tuvo que
/// repartir capas con la CPU. Para una **descarga** 60 s de silencio son
/// muchisimos --- los bytes fluyen desde el primero --- y para un **stream de
/// tokens** son el margen justo de la primera ficha.
///
/// Ver [`HttpRequest::stall_timeout`].
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Plazo TOTAL por defecto de una conexion de vida larga: una hora.
///
/// Es un **tope**, no una politica: lo que corta un servidor muerto es el plazo
/// de atasco. Esto solo esta para que una conexion que gotea un byte cada
/// cincuenta segundos no se quede indefinidamente.
///
/// Una hora y no `None` porque una biblioteca que retiene un hilo y un socket
/// para siempre no es educada; donde el tamano es de verdad arbitrario --- un
/// GGUF de 70 GB --- el sitio lo dice y pone `None` a mano.
pub const DEFAULT_LONG_LIVED_TIMEOUT: Duration = Duration::from_secs(3600);

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

    /// El metodo a partir de su nombre, **sin defecto silencioso**.
    ///
    /// Devuelve `None` para un nombre que no conoce, y eso es deliberado: lo
    /// natural seria caer en `Get`, y entonces un `request("PATCH", ..)` mal
    /// escrito mandaria un `GET` --- la peticion saldria, el servidor
    /// contestaria algo, y nadie veria que se pidio otra cosa.
    ///
    /// Es el inverso exacto de [`HttpMethod::as_str`], y hay un test que
    /// recorre las siete variantes para que siga siendolo cuando se anada una
    /// octava. Acepta el nombre en cualquier caja: los sitios que lo necesitan
    /// lo traen de una cadena de configuracion.
    pub fn from_name(name: &str) -> Option<Self> {
        let upper = name.to_ascii_uppercase();
        [
            Self::Get,
            Self::Post,
            Self::Put,
            Self::Delete,
            Self::Head,
            Self::Patch,
            Self::Options,
        ]
        .into_iter()
        .find(|method| method.as_str() == upper)
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
    /// Plazo de ATASCO: cuanto se espera **sin que llegue un byte**, no cuanto
    /// dura la peticion entera.
    ///
    /// Es el plazo que sirve para una conexion de vida larga que da senales de
    /// vida --- un stream de tokens, la descarga de un corpus de 2 GB. Para
    /// esas, un plazo total es la herramienta equivocada en las **dos**
    /// direcciones: corto mata una generacion legitima de seis minutos, y largo
    /// tarda ese mismo rato en darse cuenta de que el servidor se murio. Un
    /// plazo de atasco corta a los N segundos de silencio y no le pone techo a
    /// lo que va bien.
    ///
    /// `None` --- el defecto --- es lo que `ureq` hace hoy: ningun plazo por
    /// lectura. Es deliberado: el campo llego despues de los 55 sitios, y
    /// encenderlo de golpe para todos habria cambiado el comportamiento de
    /// peticiones que nadie habia medido.
    ///
    /// No confundir con el modulo `stall_detection`, que es **otra cosa**: alli
    /// el atasco es semantico --- el modelo repite la misma llamada a
    /// herramienta en circulos --- y se mide en turnos, no en bytes.
    pub stall_timeout: Option<Duration>,
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
            stall_timeout: None,
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
        Self::new(HttpMethod::Post, url).json(body)
    }

    /// Pone un cuerpo JSON con su `Content-Type`, **con el metodo que sea**.
    ///
    /// [`HttpRequest::post_json`] cubre los 21 sitios medidos que mandan JSON
    /// por `POST`, pero no todo JSON va por `POST`: Elasticsearch indexa con
    /// `PUT` y hay APIs que actualizan con `PATCH`. Antes de esto, el unico
    /// camino era `.body(serde_json::to_vec(..)?)` mas la cabecera a mano ---
    /// o sea, el sitio volvia a decidir lo que el puerto existe para decidir
    /// una vez.
    pub fn json(self, body: &serde_json::Value) -> Result<Self> {
        let bytes = serde_json::to_vec(body).context("failed to serialize request body")?;
        Ok(self.header("Content-Type", "application/json").body(bytes))
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

    /// Pone el plazo de atasco. Ver [`HttpRequest::stall_timeout`].
    ///
    /// Ojo con usarlo suelto: **sobre el plazo total por defecto no hace nada**.
    /// El total corta a los 30 s y un atasco de 30 s nunca llega a dispararse.
    /// Para una conexion de vida larga, lo que se quiere es
    /// [`HttpRequest::long_lived`], que pone los dos a la vez.
    pub fn stall_timeout(mut self, timeout: Duration) -> Self {
        self.stall_timeout = Some(timeout);
        self
    }

    /// Configura la peticion para una conexion de **vida larga que da senales
    /// de vida**: plazo total generoso y plazo de atasco corto.
    ///
    /// Los dos valores van juntos porque uno sin el otro no sirve, y esa es la
    /// razon de que esto sea un metodo y no dos:
    ///
    /// * Solo el de atasco, sobre el total por defecto de 30 s --- el total
    ///   corta primero y el de atasco no se entera. Es el error facil.
    /// * Solo el total, subiendolo --- es exactamente lo que esto viene a
    ///   sustituir: 300 s de total son 300 s de espera ante un servidor muerto.
    ///
    /// `stall` tiene que ser **menor** que `total` para servir de algo; si no,
    /// queda inerte (lo prueba
    /// `test_a_stall_longer_than_the_total_is_inert`). No se valida en tiempo
    /// de ejecucion a proposito: el puerto no devuelve errores de
    /// configuracion, y un `panic!` por esto seria peor que el plazo.
    pub fn long_lived(mut self, total: Duration, stall: Duration) -> Self {
        self.timeout = total;
        self.stall_timeout = Some(stall);
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

    /// Error si el estado no es 2xx, leyendo hasta 4 KiB del cuerpo para el
    /// mensaje.
    ///
    /// Existe porque en un flujo el cuerpo del error es tan util como en una
    /// respuesta bufferizada --- es donde el proveedor explica el rechazo --- y
    /// sin esto el llamante tendria que elegir entre perderlo o leerse el flujo
    /// entero. El tope de 4 KiB es para que una pagina de error de 5 MB no se
    /// cuele en un mensaje de error.
    ///
    /// Y es lo que impide el fallo que de verdad importa en una ruta de
    /// streaming: **transmitir el cuerpo de un error como si fuera la respuesta
    /// del modelo**.
    pub fn error_for_status(&mut self) -> Result<()> {
        if self.is_success() {
            return Ok(());
        }
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut (&mut self.body).take(4096), &mut buf);
        let shown = String::from_utf8_lossy(&buf);
        let shown = shown.trim();
        anyhow::bail!(
            "HTTP {}{}",
            self.status,
            if shown.is_empty() {
                String::new()
            } else {
                format!(": {}", shown)
            }
        )
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

/// El agente compartido de todo el proceso, y **el pool de conexiones vive
/// aqui**.
///
/// Esto no es una optimizacion especulativa: esta medido. En `ureq` 3,
/// `ureq::get(url)` construye `Agent::new_with_defaults()` **en cada llamada**
/// (`lib.rs:662` de ureq 3.5.0), asi que los ~50 sitios de la crate escritos de
/// esa forma **nunca han reutilizado una conexion**: TCP y TLS completos cada
/// vez. Y la primera version de este puerto hacia lo mismo, un `Agent` por
/// `send`.
///
/// Con un agente compartido, los 277 sitios de produccion que vayan quedando
/// detras del puerto comparten pool. Lo que lo hace posible es que `ureq` deja
/// configurar **por peticion** --- con [`ureq::Agent::configure_request`] --- lo
/// que antes obligaba a un agente por peticion: plazos, redirecciones y el
/// `http_status_as_error`. Comprobado en `config.rs`: los cuatro ajustes estan
/// en el `impl` generico de `ConfigBuilder`, no en el de ambito de agente.
///
/// `Agent` es `Clone` y su pool lleva un `Mutex` dentro, asi que compartirlo
/// entre hilos es su uso previsto.
static SHARED_AGENT: std::sync::LazyLock<ureq::Agent> =
    std::sync::LazyLock::new(ureq::Agent::new_with_defaults);

/// Default HTTP client using the `ureq` library.
///
/// This is the production client used by all provider functions.
/// Wraps `ureq::get()` and `ureq::post()` with configurable timeouts.
pub struct UreqClient;

impl UreqClient {
    /// Construye la peticion de `ureq` **ya configurada**, comun a `send` y a
    /// `send_streaming`.
    ///
    /// `http_status_as_error(false)` va aqui y **no es opcional**: es lo que
    /// hace que el no-2xx vuelva con cuerpo y cabeceras, que es el contrato de
    /// [`HttpResponse`]. Ponerlo en un solo sitio es el punto del puerto ---
    /// antes eran diez decisiones sueltas y dos de ellas estaban mal.
    ///
    /// La configuracion se aplica **sobre el agente compartido**, no creando uno
    /// nuevo: ver [`SHARED_AGENT`].
    fn prepare(request: &HttpRequest) -> Result<ureq::http::Request<Vec<u8>>> {
        let mut builder = ureq::http::Request::builder()
            .method(request.method.as_str())
            .uri(&request.url);
        for (name, value) in &request.headers {
            builder = builder.header(name, value);
        }
        let http_request = builder
            .body(request.body.clone().unwrap_or_default())
            .map_err(|e| {
                anyhow::anyhow!(
                    "bad {} request for {}: {}",
                    request.method.as_str(),
                    request.url,
                    e
                )
            })?;

        let mut config = SHARED_AGENT
            .configure_request(http_request)
            .timeout_global(Some(request.timeout))
            .http_status_as_error(false);
        if let Some(connect) = request.connect_timeout {
            config = config.timeout_connect(Some(connect));
        }
        // `timeout_per_read` y no `timeout_recv_body`: el que reinicia la cuenta
        // en cada lectura es este. `timeout_recv_body` es un total para el
        // cuerpo, o sea otro plazo total --- justo lo que no queremos.
        if let Some(stall) = request.stall_timeout {
            config = config.timeout_per_read(Some(stall));
        }
        if let Some(max) = request.max_redirects {
            config = config.max_redirects(max);
        }
        Ok(config.build())
    }
}

impl HttpClient for UreqClient {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse> {
        let mut response = SHARED_AGENT
            .run(Self::prepare(request)?)
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
        let response = SHARED_AGENT
            .run(Self::prepare(request)?)
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

// ============================================================================
// Stalling / dribbling server (test-only)
// ============================================================================

/// Un servidor que manda las cabeceras, un trozo del cuerpo, y luego **se
/// calla sin cerrar la conexion**.
///
/// No sirve `MockHttpServer` para esto: escribe la respuesta entera de una
/// vez. Lo que hace falta aqui es lo contrario, y es el unico caso de
/// atasco realista --- el servidor no se murio: el TCP sigue abierto, y un
/// cierre lo veria el cliente al instante. Simplemente dejo de hablar. Eso
/// es lo que un plazo total no sabe distinguir de "va lento".
#[cfg(any(test, feature = "test-utils"))]
pub struct StallingServer {
    addr: std::net::SocketAddr,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(any(test, feature = "test-utils"))]
impl StallingServer {
    /// Promete 1000 bytes, entrega cinco, y se calla para siempre.
    pub fn start() -> Self {
        Self::spawn(1000, 5, Duration::ZERO)
    }

    /// El caso CONTRARIO: una transferencia **lenta pero viva**. Entrega el
    /// cuerpo entero, byte a byte, esperando `gap` entre cada uno.
    ///
    /// Es el caso que separa un plazo de atasco de un plazo total: cualquier
    /// plazo total lo bastante corto para detectar un servidor muerto mata
    /// tambien esto, y un plazo por lectura no, porque ningun hueco
    /// individual llega al limite. Sin este servidor, cambiar
    /// `timeout_per_read` por `timeout_recv_body` pasaba todos los tests ---
    /// los dos matan un cuerpo que no llega nunca, y solo uno de los dos
    /// deja pasar uno que llega despacio. Lo dijo la mutacion M23, no yo.
    pub fn dribbling(bytes: usize, gap: Duration) -> Self {
        Self::spawn(bytes, bytes, gap)
    }

    fn spawn(promised: usize, sent: usize, gap: Duration) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut chunk = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut chunk);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\
                     Content-Length: {}\r\nConnection: keep-alive\r\n\r\n",
                    promised
                );
                if std::io::Write::write_all(&mut stream, head.as_bytes()).is_err() {
                    continue;
                }
                let mut broken = false;
                for _ in 0..sent {
                    if std::io::Write::write_all(&mut stream, b"x").is_err() {
                        broken = true;
                        break;
                    }
                    std::io::Write::flush(&mut stream).ok();
                    if !gap.is_zero() {
                        std::thread::sleep(gap);
                    }
                }
                if broken {
                    continue;
                }
                // Con el socket en la mano hasta que el test acabe: si se
                // prometio mas de lo enviado, el cliente sigue esperando lo
                // que falta. Un CIERRE lo veria al instante, y eso es otro
                // error --- no un atasco.
                while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                return;
            }
        });
        Self { addr, stop }
    }

    pub fn url(&self) -> String {
        format!("http://{}/atascado", self.addr)
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Drop for StallingServer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Un servidor HTTP de mentira que escucha en localhost y sirve respuestas
/// encoladas.
///
/// Disponible para los tests de esta crate y, tras la feature **`test-utils`**,
/// para quien la consuma. Antes de V443 vivia bajo `#[cfg(test)]` a secas, asi
/// que escribir `MockHttpServer::start()` en otra crate **no compilaba** y la
/// unica salida era montarse un servidor TCP a mano --- que es exactamente el
/// trabajo que esto ya hacia aqui dentro.
///
/// ```rust
/// # #[cfg(feature = "test-utils")]
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use ai_assistant::{HttpClient, HttpRequest, MockHttpServer, UreqClient};
///
/// let server = MockHttpServer::start();
/// server.enqueue_json(200, serde_json::json!({"result": "ok"}));
///
/// let response = UreqClient.send(&HttpRequest::get(server.url()))?;
/// assert_eq!(response.status, 200);
/// assert_eq!(response.json()?["result"], "ok");
///
/// // Y se puede comprobar lo que LLEGO al cable, no solo lo que volvio.
/// let (method, path, _body) = server.last_request().expect("recorded");
/// assert_eq!(method, "GET");
/// assert_eq!(path, "/");
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "test-utils"))]
/// # fn main() {}
/// ```
///
/// Para probar un **atasco** --- un servidor que manda las cabeceras y se calla
/// --- esta [`StallingServer`], que es otra cosa: este escribe la respuesta
/// entera de una vez.
#[cfg(any(test, feature = "test-utils"))]
pub struct MockHttpServer {
    addr: std::net::SocketAddr,
    /// Queued responses: (status_code, content_type, body)
    /// Cola de respuestas: (estado, content-type, cabeceras extra, cuerpo).
    #[allow(clippy::type_complexity)]
    responses: std::sync::Arc<std::sync::Mutex<Vec<(u16, String, Vec<(String, String)>, String)>>>,
    /// Received requests: (method, path, body)
    requests: std::sync::Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    heads: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    connections: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Background thread handle
    _handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(any(test, feature = "test-utils"))]
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
        // Cuantas CONEXIONES se aceptaron, no cuantas peticiones llegaron. Es
        // la unica forma de ver el pool desde fuera: N peticiones seguidas sobre
        // un agente con pool usan 1 conexion, y sin pool usan N.
        let connections = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let resp_clone = responses.clone();
        let req_clone = requests.clone();
        let head_clone = heads.clone();
        let conn_clone = connections.clone();
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
                        conn_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                            .ok();

                        // Una conexion puede llevar VARIAS peticiones. Hasta
                        // V431 el mock contestaba una y cerraba
                        // (`Connection: close`), asi que ningun test podia
                        // distinguir un cliente con pool de uno sin el --- el
                        // servidor forzaba una conexion por peticion. Se sale
                        // del bucle cuando el cliente cierra (lectura vacia).
                        loop {
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

                            // Una lectura vacia significa que el cliente cerro la
                            // conexion: no hay mas peticiones por aqui.
                            if request_str.trim().is_empty() {
                                break;
                            }

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
                            // HTTP/1.1: si el cliente pide `Connection: close`, el servidor
                            // DEBE cerrar tras responder. Ignorarlo es un fallo de
                            // protocolo, y se noto enseguida al poner keep-alive: un test
                            // que lee hasta EOF se quedaba colgado hasta el plazo de
                            // lectura. Se mira solo la CABECERA --- en el cuerpo,
                            // "connection: close" es texto cualquiera.
                            let client_wants_close =
                                head.to_ascii_lowercase().contains("connection: close");

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
                            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: {}\r\n\r\n{}",
                            status, reason, content_type, resp_body.len(), extra, if client_wants_close { "close" } else { "keep-alive" }, resp_body
                        );
                            std::io::Write::write_all(&mut stream, response.as_bytes()).ok();

                            if client_wants_close {
                                break;
                            }
                        } // fin del bucle de peticiones de ESTA conexion
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
            connections,
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

    /// Cuantas CONEXIONES TCP se aceptaron.
    ///
    /// Distinto de `request_count()` a proposito: es la diferencia entre un
    /// cliente que reutiliza la conexion y uno que abre una por peticion, y es
    /// lo que hace observable el pool sin medir tiempos.
    pub fn connection_count(&self) -> usize {
        self.connections.load(std::sync::atomic::Ordering::Relaxed)
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

    /// `from_name` recorre las variantes para que sea el inverso EXACTO de
    /// `as_str`, y el test las recorre tambien --- no una lista escrita a mano,
    /// que es como se queda fuera la octava variante que alguien anada.
    ///
    /// `OPTIONS` ya se quedo fuera una vez de un barrido que miraba solo los
    /// patrones que yo habia listado (ver el comentario de la variante).
    #[test]
    fn test_method_from_name_is_the_exact_inverse_of_as_str() {
        let all = [
            HttpMethod::Get,
            HttpMethod::Post,
            HttpMethod::Put,
            HttpMethod::Delete,
            HttpMethod::Head,
            HttpMethod::Patch,
            HttpMethod::Options,
        ];
        for method in all {
            assert_eq!(
                HttpMethod::from_name(method.as_str()),
                Some(method),
                "{} no vuelve de su propio nombre",
                method.as_str()
            );
            assert_eq!(
                HttpMethod::from_name(&method.as_str().to_ascii_lowercase()),
                Some(method),
                "{} en minusculas tiene que valer: el nombre llega de configuracion",
                method.as_str()
            );
        }

        // Y lo que de verdad justifica el `Option`: un nombre desconocido NO cae
        // en `Get`. Si cayera, un metodo mal escrito mandaria un `GET`, el
        // servidor contestaria algo y nadie veria que se pidio otra cosa.
        assert_eq!(HttpMethod::from_name("TRACE"), None);
        assert_eq!(HttpMethod::from_name("GETT"), None);
        assert_eq!(HttpMethod::from_name(""), None);
    }

    /// El cuerpo JSON con un metodo que no es `POST`: Elasticsearch indexa con
    /// `PUT`, y hasta V433 el puerto solo sabia poner JSON en un `POST`.
    ///
    /// Se comprueba **en el cable**, no en los campos del struct: que el metodo
    /// viaje como `PUT` y que el `Content-Type` vaya puesto. Mirar
    /// `request.headers` solo probaria que el constructor hace lo que dice su
    /// propio codigo.
    #[test]
    fn test_json_body_travels_with_any_method() {
        let server = MockHttpServer::start();
        server.enqueue_json(201, serde_json::json!({"result": "created"}));

        let request = HttpRequest::new(HttpMethod::Put, format!("{}/index/_doc/1", server.url()))
            .json(&serde_json::json!({"vector": [0.5]}))
            .expect("un Value siempre serializa");
        let response = UreqClient.send(&request).expect("el mock contesto");
        assert_eq!(response.status, 201);

        let head = server
            .last_request_head()
            .expect("el servidor guardo la cabecera")
            .to_ascii_lowercase();
        assert!(head.starts_with("put /index/_doc/1"), "head = {}", head);
        assert!(
            head.contains("content-type: application/json"),
            "head = {}",
            head
        );

        let (_, _, body) = server.last_request().expect("y el cuerpo");
        assert!(body.contains("\"vector\""), "body = {}", body);
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

    /// La prueba de que el pool existe, y es **determinista**: tres peticiones
    /// seguidas al mismo servidor usan UNA conexion.
    ///
    /// Se cuentan conexiones y no milisegundos a proposito. Una medida de
    /// latencia contra un servidor en localhost es ruido --- no hay ni RTT ni
    /// handshake TLS que ahorrar --- asi que diria "no se nota" sobre algo que
    /// si esta pasando. El numero de conexiones TCP lo dice sin ambiguedad.
    ///
    /// Sin agente compartido serian tres, porque `ureq::get(url)` construye un
    /// `Agent` por llamada y con el un pool nuevo. Y hasta V431 este test **no
    /// se podia escribir**: el mock contestaba y cerraba la conexion, asi que
    /// forzaba una por peticion y tapaba la diferencia que se quiere medir.
    #[test]
    fn test_shared_agent_reuses_one_connection() {
        let server = MockHttpServer::start();
        for _ in 0..3 {
            server.enqueue_json(200, serde_json::json!({"ok": true}));
        }

        for i in 0..3 {
            let request =
                HttpRequest::get(format!("{}/ping", server.url())).timeout(Duration::from_secs(5));
            let response = UreqClient
                .send(&request)
                .unwrap_or_else(|e| panic!("peticion {} fallo: {:#}", i, e));
            assert!(response.is_success());
        }

        assert_eq!(server.request_count(), 3, "llegaron las tres peticiones");
        assert_eq!(
            server.connection_count(),
            1,
            "tres peticiones con pool son UNA conexion; {} significa que el              agente compartido no esta reutilizando",
            server.connection_count()
        );
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
    // Plazo de atasco (N163)
    // ========================================================================

    #[test]
    fn test_the_stall_timeout_cuts_a_silent_server_without_burning_the_total() {
        let server = StallingServer::start();
        let started = std::time::Instant::now();
        let err = UreqClient
            .send(
                &HttpRequest::get(server.url())
                    .long_lived(Duration::from_secs(10), Duration::from_millis(300)),
            )
            .expect_err("un servidor que se calla a mitad del cuerpo tiene que fallar");
        let elapsed = started.elapsed();
        // Lo que se prueba NO es que falle --- un plazo total tambien falla ---
        // sino que falla PRONTO: a los 300 ms de silencio, no a los 10 s.
        assert!(
            elapsed < Duration::from_secs(3),
            "corto a los {:?}, o sea que actuo el plazo total y no el de atasco: {}",
            elapsed,
            err
        );
    }

    /// El caso separador del de arriba.
    ///
    /// Sin esto, "fallo en 300 ms" podria deberse a cualquier otra cosa --- un
    /// `Content-Length` que `ureq` no se cree, un cierre que yo no vi --- y el
    /// test anterior pasaria sin probar nada. Aqui el mismo servidor, el mismo
    /// camino y **sin** plazo de atasco tarda todo el total.
    #[test]
    fn test_without_a_stall_timeout_the_same_server_burns_the_whole_total() {
        let server = StallingServer::start();
        let started = std::time::Instant::now();
        UreqClient
            .send(&HttpRequest::get(server.url()).timeout(Duration::from_secs(2)))
            .expect_err("sin plazo de atasco corta el total, pero corta");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(1500),
            "fallo a los {:?} SIN plazo de atasco puesto, asi que el test de arriba \
             no demuestra nada: algo mas esta cortando",
            elapsed
        );
    }

    /// Lo que afirma la documentacion de [`HttpRequest::long_lived`].
    #[test]
    fn test_a_stall_longer_than_the_total_is_inert() {
        let server = StallingServer::start();
        let started = std::time::Instant::now();
        UreqClient
            .send(
                &HttpRequest::get(server.url())
                    .long_lived(Duration::from_secs(2), Duration::from_secs(30)),
            )
            .expect_err("corta el total");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(1500),
            "corto a los {:?} con un atasco de 30 s sobre un total de 2 s, \
             o sea que el atasco NO es lo que la documentacion dice que es",
            elapsed
        );
    }

    /// El caso de verdad: el plazo tiene que seguir vivo en el lector que
    /// `send_streaming` devuelve, que es donde se consumen los tokens.
    ///
    /// Que la configuracion se aplique a la peticion no implica que alcance a
    /// las lecturas de despues --- son dos cosas, y la segunda es la que
    /// importa para un stream.
    #[test]
    fn test_the_stall_timeout_survives_into_the_streaming_reader() {
        let server = StallingServer::start();
        let mut stream = UreqClient
            .send_streaming(
                &HttpRequest::get(server.url())
                    .long_lived(Duration::from_secs(10), Duration::from_millis(300)),
            )
            .expect("las cabeceras si llegan");
        assert!(stream.is_success());

        let started = std::time::Instant::now();
        let mut body = Vec::new();
        let err = std::io::Read::read_to_end(&mut stream.body, &mut body)
            .expect_err("el cuerpo se queda a medias");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "el lector tardo {:?}, o sea que el plazo de atasco NO llega al flujo: {}",
            elapsed,
            err
        );
        // Y lo que si llego, llego: el atasco no se come los bytes buenos.
        assert_eq!(&body[..], b"xxxxx");
    }

    /// La otra mitad, y la que de verdad justifica el campo: una transferencia
    /// **lenta pero viva** llega entera.
    ///
    /// Doce bytes con 120 ms entre cada uno son casi un segundo y medio de
    /// transferencia, muy por encima del plazo de atasco de 400 ms --- y aun asi
    /// pasa, porque ningun hueco individual llega a 400 ms. Un plazo total de
    /// 400 ms la habria matado, y un plazo total generoso no habria detectado un
    /// servidor muerto.
    ///
    /// Este test existe porque la mutacion M23 ---
    /// `timeout_per_read` -> `timeout_recv_body` --- SOBREVIVIO a los tres
    /// anteriores: los dos plazos matan un cuerpo que no llega nunca, y solo uno
    /// de los dos deja pasar uno que llega despacio.
    #[test]
    fn test_a_slow_but_alive_transfer_is_not_a_stall() {
        let server = StallingServer::dribbling(12, Duration::from_millis(120));
        let started = std::time::Instant::now();
        let response = UreqClient
            .send(
                &HttpRequest::get(server.url())
                    .long_lived(Duration::from_secs(10), Duration::from_millis(400)),
            )
            .expect("una transferencia lenta NO es un atasco");
        let elapsed = started.elapsed();
        assert!(response.is_success());
        assert_eq!(response.body.len(), 12, "llego el cuerpo entero");
        // Y tardo lo que tenia que tardar: si hubiera acabado en 50 ms, el
        // servidor no estaria goteando y el test no probaria nada.
        assert!(
            elapsed >= Duration::from_millis(800),
            "acabo en {:?}, asi que el servidor no goteo y este test no separa nada",
            elapsed
        );
    }

    #[test]
    fn test_long_lived_sets_both_halves_of_the_decision() {
        let req = HttpRequest::get("http://x/")
            .long_lived(Duration::from_secs(600), Duration::from_secs(30));
        assert_eq!(req.timeout, Duration::from_secs(600));
        assert_eq!(req.stall_timeout, Some(Duration::from_secs(30)));
        // Y el defecto sigue siendo `None`: encenderlo para los 55 sitios de
        // golpe habria cambiado el comportamiento de peticiones sin medir.
        assert_eq!(HttpRequest::get("http://x/").stall_timeout, None);
        assert_eq!(
            HttpRequest::get("http://x/").timeout,
            Duration::from_secs(30)
        );
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
