//! Web search integration
//!
//! Search the web and integrate results into AI responses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::http_client::{HttpClient, HttpRequest, HttpResponse, UreqClient};

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub source: String,
    pub date: Option<String>,
    pub relevance_score: f64,
}

impl SearchResult {
    pub fn new(title: &str, url: &str, snippet: &str) -> Self {
        Self {
            title: title.to_string(),
            url: url.to_string(),
            snippet: snippet.to_string(),
            source: extract_domain(url),
            date: None,
            relevance_score: 0.0,
        }
    }

    pub fn with_date(mut self, date: &str) -> Self {
        self.date = Some(date.to_string());
        self
    }

    pub fn with_relevance(mut self, score: f64) -> Self {
        self.relevance_score = score;
        self
    }
}

fn extract_domain(url: &str) -> String {
    url.split("//")
        .nth(1)
        .and_then(|s| s.split('/').next())
        .unwrap_or("unknown")
        .to_string()
}

/// Search configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SearchConfig {
    /// Maximum results to return
    pub max_results: usize,
    /// Timeout for requests
    pub timeout: Duration,
    /// Safe search level
    pub safe_search: SafeSearch,
    /// Language filter
    pub language: Option<String>,
    /// Region filter
    pub region: Option<String>,
    /// Time filter
    pub time_range: Option<TimeRange>,
}

with_setters!(SearchConfig {
    with_max_results -> max_results: usize,
    with_timeout -> timeout: Duration,
    with_safe_search -> safe_search: SafeSearch,
    with_language -> language: Option<String>,
    with_region -> region: Option<String>,
    with_time_range -> time_range: Option<TimeRange>,
});

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            timeout: Duration::from_secs(10),
            safe_search: SafeSearch::Moderate,
            language: None,
            region: None,
            time_range: None,
        }
    }
}

/// Safe search level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SafeSearch {
    Off,
    Moderate,
    Strict,
}

/// Time range filter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimeRange {
    Day,
    Week,
    Month,
    Year,
    All,
}

/// Search provider trait
///
/// # Stability
///
/// New methods may be added to this trait in minor versions with default
/// implementations. Required methods will only change in major versions.
pub trait SearchProvider: Send + Sync {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError>;
    fn name(&self) -> &str;
}

/// Manda la peticion por el puerto y traduce el ESTADO antes de que el llamante
/// vea el cuerpo.
///
/// Tres decisiones que estaban repetidas en los siete proveedores, y una que no
/// estaba en ninguno:
///
/// - un fallo de transporte es [`SearchError::Network`], como antes;
/// - un 429 o un 503 es [`SearchError::RateLimit`] ---la variante que existia
///   desde el principio y que **no construia nadie**---. Los siete colapsaban
///   todo no-2xx en `Network`, asi que «frena» y «no hay ruta al host» llegaban
///   al llamante como el mismo error. Importa porque la cuota gratuita de
///   Brave, Google y SerpAPI se agota a diario: lo primero que encuentra una
///   busqueda ancha es un 429, y el mensaje mandaba a mirar la red en vez de el
///   plan. El mismo defecto estaba ya encontrado y arreglado en
///   `academic_search` ---su comentario lo cuenta con estas palabras--- y el
///   arreglo se quedo en el modulo donde se aprendio;
/// - cualquier otro no-2xx sigue siendo `Network`, pero ahora **con el cuerpo
///   dentro del mensaje**, que es donde el proveedor explica el motivo. Con
///   `ureq` a pelo eso se perdia: su `Error::StatusCode(u16)` no lleva la
///   respuesta, asi que un 401 por clave caducada y un 403 por dominio no
///   autorizado decian lo mismo.
///
/// Aqui **no** se reintenta, a diferencia de `academic_search`: el
/// [`WebSearchEngine`] ya recorre los proveedores en orden y pasa al siguiente
/// cuando uno falla, asi que dormir dentro de `search()` retrasaria justo al
/// sustituto. La clasificacion del estado si es la misma funcion, no una copia.
fn send_search(
    client: &dyn HttpClient,
    request: &HttpRequest,
) -> Result<HttpResponse, SearchError> {
    let response = client
        .send(request)
        .map_err(|e| SearchError::Network(e.to_string()))?;
    if crate::http_client::status_says_slow_down(response.status) {
        return Err(SearchError::RateLimit);
    }
    response
        .error_for_status()
        .map_err(|e| SearchError::Network(e.to_string()))?;
    Ok(response)
}

/// Lo mismo, y el cuerpo parseado como JSON. Lo usan los seis proveedores de
/// API; el de DuckDuckGo raspa HTML y pide el texto.
fn search_json(
    client: &dyn HttpClient,
    request: &HttpRequest,
) -> Result<serde_json::Value, SearchError> {
    send_search(client, request)?
        .json()
        .map_err(|e| SearchError::Parse(e.to_string()))
}

/// DuckDuckGo search provider (HTML scraping - for educational purposes)
pub struct DuckDuckGoProvider {
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl DuckDuckGoProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://html.duckduckgo.com/html".to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP.
    ///
    /// Para un test ---un doble contesta sin red y sin depender de que haya un
    /// puerto libre--- o para un llamante que quiera poner su propio cliente
    /// (proxy, metricas, reintentos) sin que este modulo sepa nada de eso.
    ///
    /// Los otros seis proveedores llevan el mismo constructor. No esta en el
    /// trait a proposito: `SearchProvider::search` promete en su doc que los
    /// metodos requeridos solo cambian en una version mayor, y pasar el cliente
    /// por el metodo habria roto a cualquiera que lo implemente fuera.
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl Default for DuckDuckGoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchProvider for DuckDuckGoProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let encoded_query = urlencoding::encode(query);
        let url = format!("{}/?q={}", self.base_url, encoded_query);

        let request = HttpRequest::get(url)
            .timeout(config.timeout)
            .header("User-Agent", "Mozilla/5.0 (compatible; AIAssistant/1.0)");

        // El fallo de lectura ya no es de red: detras del puerto el cuerpo
        // llega entero o no llega ---un corte lo cuenta `send`---, asi que lo
        // unico que queda aqui es «esto no es UTF-8», que es un `Parse`.
        let html = send_search(&*self.client, &request)?
            .text()
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        // El mismo lector que el proveedor con cache, y por eso dos expresiones
        // regulares se fueron de aqui: ver [`parse_ddg_results`], que explica en
        // que cuatro cosas perdian.
        Ok(parse_ddg_results(&html, config.max_results))
    }

    fn name(&self) -> &str {
        "DuckDuckGo"
    }
}

/// Simple HTML entity decoder
fn html_decode(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Brave Search provider (requires API key)
pub struct BraveSearchProvider {
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl BraveSearchProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.search.brave.com/res/v1/web/search".to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for BraveSearchProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "{}?q={}&count={}",
            self.base_url,
            urlencoding::encode(query),
            config.max_results
        );

        let request = HttpRequest::get(url)
            .timeout(config.timeout)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json");

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();

        if let Some(web) = json
            .get("web")
            .and_then(|w| w.get("results"))
            .and_then(|r| r.as_array())
        {
            for item in web {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let snippet = item
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");

                results.push(SearchResult::new(title, url, snippet));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "Brave Search"
    }
}

/// SearXNG provider (self-hosted search)
pub struct SearXNGProvider {
    instance_url: String,
    client: Arc<dyn HttpClient>,
}

impl SearXNGProvider {
    pub fn new(instance_url: &str) -> Self {
        Self {
            instance_url: instance_url.trim_end_matches('/').to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for SearXNGProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "{}/search?q={}&format=json",
            self.instance_url,
            urlencoding::encode(query)
        );

        let request = HttpRequest::get(url).timeout(config.timeout);

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();

        if let Some(items) = json.get("results").and_then(|r| r.as_array()) {
            for item in items.iter().take(config.max_results) {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let snippet = item.get("content").and_then(|c| c.as_str()).unwrap_or("");

                results.push(SearchResult::new(title, url, snippet));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "SearXNG"
    }
}

// =============================================================================
// Additional Search Providers (API-based)
// =============================================================================

/// Google Custom Search API provider.
///
/// Requires a Google API key and a Custom Search Engine ID (cx).
/// See: https://developers.google.com/custom-search/v1/overview
pub struct GoogleSearchProvider {
    api_key: String,
    cx: String,
    client: Arc<dyn HttpClient>,
}

impl GoogleSearchProvider {
    pub fn new(api_key: &str, cx: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            cx: cx.to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    ///
    /// En este y en los tres siguientes es la **unica** forma de probarlos: la
    /// URL del servicio esta incrustada, asi que no hay donde apuntar un
    /// servidor de pruebas.
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for GoogleSearchProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}&num={}",
            self.api_key,
            self.cx,
            urlencoding::encode(query),
            config.max_results.min(10), // Google max is 10 per page
        );

        let request = HttpRequest::get(url).timeout(config.timeout);

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();
        if let Some(items) = json.get("items").and_then(|i| i.as_array()) {
            for item in items {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let link = item.get("link").and_then(|l| l.as_str()).unwrap_or("");
                let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
                results.push(SearchResult::new(title, link, snippet));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "Google Custom Search"
    }
}

/// Bing Web Search API provider.
///
/// Requires a Microsoft Azure Cognitive Services API key.
/// See: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
pub struct BingSearchProvider {
    api_key: String,
    client: Arc<dyn HttpClient>,
}

impl BingSearchProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for BingSearchProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://api.bing.microsoft.com/v7.0/search?q={}&count={}",
            urlencoding::encode(query),
            config.max_results,
        );

        let request = HttpRequest::get(url)
            .timeout(config.timeout)
            .header("Ocp-Apim-Subscription-Key", self.api_key.as_str());

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();
        if let Some(pages) = json
            .get("webPages")
            .and_then(|w| w.get("value"))
            .and_then(|v| v.as_array())
        {
            for item in pages {
                let title = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
                results.push(SearchResult::new(title, url, snippet));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "Bing Web Search"
    }
}

/// SerpAPI provider — unified search API supporting Google, Bing, Yahoo, and more.
///
/// See: https://serpapi.com/
pub struct SerpApiProvider {
    api_key: String,
    engine: String,
    client: Arc<dyn HttpClient>,
}

impl SerpApiProvider {
    /// Create a SerpAPI provider using Google as the search engine.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            engine: "google".to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Create with a specific search engine (e.g., "google", "bing", "yahoo").
    pub fn with_engine(api_key: &str, engine: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            engine: engine.to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for SerpApiProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://serpapi.com/search.json?api_key={}&engine={}&q={}&num={}",
            self.api_key,
            self.engine,
            urlencoding::encode(query),
            config.max_results,
        );

        let request = HttpRequest::get(url).timeout(config.timeout);

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();
        if let Some(organic) = json.get("organic_results").and_then(|o| o.as_array()) {
            for item in organic {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let link = item.get("link").and_then(|l| l.as_str()).unwrap_or("");
                let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
                results.push(SearchResult::new(title, link, snippet));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "SerpAPI"
    }
}

/// Tavily Search API provider — AI-optimized search with content extraction.
///
/// See: https://tavily.com/
pub struct TavilyProvider {
    api_key: String,
    client: Arc<dyn HttpClient>,
}

impl TavilyProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }
}

impl SearchProvider for TavilyProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let body = serde_json::json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": config.max_results,
            "search_depth": "basic",
        });

        // `post_json` pone el `Content-Type` por contrato del puerto, que es
        // donde debe estar: ponerlo a mano era la trampa de la subida a ureq 3.
        let request = HttpRequest::post_json("https://api.tavily.com/search", &body)
            .map_err(|e| SearchError::Parse(e.to_string()))?
            .timeout(config.timeout);

        let json = search_json(&*self.client, &request)?;

        let mut results = Vec::new();
        if let Some(items) = json.get("results").and_then(|r| r.as_array()) {
            for item in items {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let content = item.get("content").and_then(|c| c.as_str()).unwrap_or("");
                results.push(SearchResult::new(title, url, content));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "Tavily"
    }
}

/// Web search manager
pub struct WebSearchEngine {
    providers: Vec<Box<dyn SearchProvider>>,
    config: SearchConfig,
    cache: HashMap<String, (Vec<SearchResult>, std::time::Instant)>,
    cache_ttl: Duration,
}

impl WebSearchEngine {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            providers: Vec::new(),
            config,
            cache: HashMap::new(),
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    pub fn add_provider(&mut self, provider: Box<dyn SearchProvider>) {
        self.providers.push(provider);
    }

    /// Search using all providers
    pub fn search(&mut self, query: &str) -> Result<Vec<SearchResult>, SearchError> {
        // Check cache
        let cache_key = query.to_lowercase();
        if let Some((results, time)) = self.cache.get(&cache_key) {
            if time.elapsed() < self.cache_ttl {
                return Ok(results.clone());
            }
        }

        // Try each provider
        for provider in &self.providers {
            match provider.search(query, &self.config) {
                Ok(results) if !results.is_empty() => {
                    self.cache
                        .insert(cache_key, (results.clone(), std::time::Instant::now()));
                    return Ok(results);
                }
                _ => continue,
            }
        }

        Err(SearchError::NoResults)
    }

    /// Search and format results for AI context
    pub fn search_for_context(
        &mut self,
        query: &str,
        max_chars: usize,
    ) -> Result<String, SearchError> {
        let results = self.search(query)?;

        let mut context = format!("Web search results for '{}':\n\n", query);
        let mut total_chars = context.len();

        for (i, result) in results.iter().enumerate() {
            let entry = format!(
                "{}. {} ({})\n{}\n\n",
                i + 1,
                result.title,
                result.source,
                result.snippet
            );

            if total_chars + entry.len() > max_chars {
                break;
            }

            context.push_str(&entry);
            total_chars += entry.len();
        }

        Ok(context)
    }

    /// Extract facts from search results
    pub fn extract_facts(&mut self, query: &str) -> Result<Vec<String>, SearchError> {
        let results = self.search(query)?;

        let mut facts = Vec::new();
        for result in results {
            // Split snippet into sentences
            for sentence in result.snippet.split(|c| c == '.' || c == '!' || c == '?') {
                let trimmed = sentence.trim();
                if trimmed.len() > 20 && trimmed.len() < 200 {
                    facts.push(trimmed.to_string());
                }
            }
        }

        Ok(facts)
    }
}

impl Default for WebSearchEngine {
    fn default() -> Self {
        let mut manager = Self::new(SearchConfig::default());
        manager.add_provider(Box::new(DuckDuckGoProvider::new()));
        manager
    }
}

// ============================================================================
// Enhanced DuckDuckGo Provider with Caching and Rate Limiting
// ============================================================================

/// Enhanced DuckDuckGo search with HTML result parsing, caching, and rate limiting.
///
/// Hasta V471 **no lo podia usar nadie**. No implementaba [`SearchProvider`],
/// asi que `WebSearchEngine::add_provider` no lo aceptaba, y el desencaje era
/// doble: el trait pide `&self` y su busqueda tomaba `&mut self`, y pide
/// `(query, &SearchConfig)` donde esta tomaba `(query, max_results)`. O sea que
/// el proveedor **con** cache y limitacion de ritmo ---las dos cosas que quieres
/// al raspar el HTML de un buscador--- se quedaba fuera del motor, y el que si
/// estaba dentro no tiene ninguna de las dos. Lo usaba solo su propio test.
///
/// Ahora encaja: la cache y el reloj viven detras de un `Mutex` cada uno, que es
/// lo que convierte `&mut self` en `&self` sin pedirle nada al llamante.
pub struct EnhancedDuckDuckGoProvider {
    /// Rate limit: minimum seconds between requests.
    min_interval_secs: f64,
    /// Instante de la ultima peticion **mandada**, no de la ultima que salio
    /// bien. Ver [`EnhancedDuckDuckGoProvider::pace`].
    last_request: Mutex<Option<std::time::Instant>>,
    /// Result cache (query -> (results, timestamp)).
    cache: Mutex<HashMap<String, (Vec<SearchResult>, std::time::Instant)>>,
    /// Cache TTL in seconds.
    cache_ttl_secs: u64,
    /// Quien habla HTTP.
    client: Arc<dyn HttpClient>,
}

impl EnhancedDuckDuckGoProvider {
    pub fn new() -> Self {
        Self {
            min_interval_secs: 1.0,
            last_request: Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
            cache_ttl_secs: 300,
            client: Arc::new(UreqClient),
        }
    }

    /// Set the minimum interval between requests (rate limiting).
    pub fn with_min_interval(mut self, secs: f64) -> Self {
        self.min_interval_secs = secs;
        self
    }

    /// Set the cache TTL in seconds.
    pub fn with_cache_ttl(mut self, secs: u64) -> Self {
        self.cache_ttl_secs = secs;
        self
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// Search with caching and rate limiting.
    ///
    /// Sigue aqui por compatibilidad y delega en la del trait. El plazo de 10 s
    /// que llevaba incrustado es exactamente el de `SearchConfig::default()`,
    /// asi que delegar no cambia lo que sale al cable.
    pub fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        SearchProvider::search(
            self,
            query,
            &SearchConfig {
                max_results,
                ..SearchConfig::default()
            },
        )
    }

    /// La respuesta cacheada, si existe y no ha caducado, recortada al tope que
    /// pide quien pregunta **ahora**.
    ///
    /// Y lo que hay guardado es lo que pidio el PRIMERO: la pagina se trocea ya
    /// con su `max_results`, asi que un llamante posterior que quiera mas
    /// recibe menos y nada se lo dice. Es el comportamiento de antes de V471
    /// ---el codigo viejo tambien cacheaba lo ya recortado--- y no lo cambio
    /// aqui, pero conviene que este escrito: esta cache no guarda «la respuesta
    /// de DuckDuckGo», guarda «la respuesta recortada que pidio alguien».
    fn cached(&self, key: &str, max_results: usize) -> Option<Vec<SearchResult>> {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let (results, ts) = cache.get(key)?;
        if ts.elapsed().as_secs() >= self.cache_ttl_secs {
            return None;
        }
        Some(results[..results.len().min(max_results)].to_vec())
    }

    fn remember(&self, key: String, results: &[SearchResult]) {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, (results.to_vec(), std::time::Instant::now()));
    }

    /// Espera lo que falte del intervalo minimo y apunta **esta** peticion.
    ///
    /// Apuntarla aqui, antes de mandarla, y no despues de que salga bien, es un
    /// arreglo: `last_request` solo se escribia tras un exito, asi que un bucle
    /// de fallos ---justo el caso en que el servidor te quiere lejos--- no
    /// esperaba nada entre intento e intento.
    ///
    /// El candado se mantiene **durante** la espera a proposito. Soltarlo antes
    /// dejaria pasar a dos hilos a la vez: los dos leerian «ya toca» y el ritmo
    /// real seria el doble del pedido.
    fn pace(&self) {
        let mut last = self.last_request.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(previous) = *last {
            let elapsed = previous.elapsed().as_secs_f64();
            if elapsed < self.min_interval_secs {
                std::thread::sleep(Duration::from_secs_f64(self.min_interval_secs - elapsed));
            }
        }
        *last = Some(std::time::Instant::now());
    }
}

impl SearchProvider for EnhancedDuckDuckGoProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let cache_key = query.to_lowercase();
        if let Some(hit) = self.cached(&cache_key, config.max_results) {
            return Ok(hit);
        }

        self.pace();

        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            search_urlencode(query)
        );
        let request = HttpRequest::get(url)
            .timeout(config.timeout)
            .header("User-Agent", "Mozilla/5.0 (compatible; AIAssistant/1.0)");

        let html = send_search(&*self.client, &request)?
            .text()
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        let results = parse_ddg_results(&html, config.max_results);
        self.remember(cache_key, &results);
        Ok(results)
    }

    fn name(&self) -> &str {
        "DuckDuckGo (cached)"
    }
}

impl Default for EnhancedDuckDuckGoProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Lee la pagina de resultados HTML de DuckDuckGo. **El unico lector.**
///
/// Habia dos. `DuckDuckGoProvider` traia el suyo con dos expresiones
/// regulares y este barria cadenas, y no era una cuestion de gusto: el de
/// regex perdia en las cuatro.
///
/// | | regex | este |
/// |---|---|---|
/// | orden de atributos | exigia `href` DESPUES de `class` | `extract_attr`, indiferente |
/// | titulo con `<b>` | `([^<]*)` lo cortaba en el primer resaltado | `strip_html_tags` |
/// | resumen | el i-esimo del documento entero, suponiendo 1:1 | el que sigue a ESE enlace |
/// | `uddg=` | no cortaba en `&`: arrastraba `&rut=...` | corta y decodifica |
///
/// Las dos del medio no son casos raros: DuckDuckGo envuelve en `<b>` cada
/// termino de la consulta que aparece en el titulo o en el resumen, o sea que
/// el lector de regex truncaba **casi siempre**.
fn parse_ddg_results(html: &str, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // DuckDuckGo HTML results are in <a class="result__a" href="...">Title</a>
    // and <a class="result__snippet" ...>Description...</a>
    // We use simple string scanning without regex.
    let result_marker = "class=\"result__a\"";
    let snippet_marker = "class=\"result__snippet\"";

    // Collect all result blocks
    let mut pos = 0;
    while pos < html.len() && results.len() < max_results {
        // Find next result link
        let link_start = match html[pos..].find(result_marker) {
            Some(i) => pos + i,
            None => break,
        };

        // Find href in this tag
        let tag_start = html[..link_start].rfind('<').unwrap_or(link_start);
        let tag_end = match html[link_start..].find('>') {
            Some(i) => link_start + i,
            None => break,
        };

        let tag_content = &html[tag_start..tag_end + 1];
        let href = extract_attr(tag_content, "href").unwrap_or_default();

        // Extract title (text between > and </a>)
        let title_start = tag_end + 1;
        let title_end = match html[title_start..].find("</a>") {
            Some(i) => title_start + i,
            None => break,
        };
        let raw_title = strip_html_tags(&html[title_start..title_end]);

        // Find snippet after the link
        let snippet_search_start = title_end;
        let snippet_text = if let Some(snip_pos) = html[snippet_search_start..].find(snippet_marker)
        {
            let snip_abs = snippet_search_start + snip_pos;
            // Find the closing > of this tag
            let snip_tag_end = match html[snip_abs..].find('>') {
                Some(i) => snip_abs + i + 1,
                None => break,
            };
            // Find closing </a> or </span>
            let snip_content_end = html[snip_tag_end..]
                .find("</a>")
                .or_else(|| html[snip_tag_end..].find("</span>"))
                .map(|i| snip_tag_end + i)
                .unwrap_or(snip_tag_end);
            strip_html_tags(&html[snip_tag_end..snip_content_end])
        } else {
            String::new()
        };

        // Decode DuckDuckGo redirect URL
        let actual_url = if href.contains("uddg=") {
            href.split("uddg=")
                .nth(1)
                .and_then(|s| s.split('&').next())
                .map(|s| simple_urldecode(s))
                .unwrap_or(href.clone())
        } else {
            href.clone()
        };

        if !actual_url.is_empty() && !raw_title.is_empty() {
            results.push(SearchResult::new(
                &html_decode(&raw_title),
                &actual_url,
                &html_decode(&snippet_text),
            ));
        }

        pos = title_end + 4; // Skip past </a>
    }

    results
}

/// URL-encode a query string for search.
fn search_urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                String::from(b as char)
            }
            b' ' => "+".to_string(),
            _ => format!("%{:02X}", b),
        })
        .collect()
}

/// Simple URL decoder for redirect URLs.
fn simple_urldecode(s: &str) -> String {
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00"),
                16,
            ) {
                result.push(byte as char);
                i += 3;
                continue;
            }
        } else if bytes[i] == b'+' {
            result.push(' ');
            i += 1;
            continue;
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Extract an attribute value from an HTML tag string.
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

/// Strip HTML tags from a string, returning only text content.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

/// Deduplicate search results by URL.
pub fn dedup_results(results: &mut Vec<SearchResult>) {
    let mut seen = std::collections::HashSet::new();
    results.retain(|r| seen.insert(r.url.clone()));
}

// ============================================================================
// Web Crawler with robots.txt Compliance
// ============================================================================

/// Parsed robots.txt rules for a specific user agent.
#[derive(Debug, Clone)]
pub struct RobotsRules {
    /// Disallowed paths.
    pub disallowed: Vec<String>,
    /// Allowed paths (override disallow).
    pub allowed: Vec<String>,
    /// Crawl delay in seconds.
    pub crawl_delay: Option<f64>,
}

impl RobotsRules {
    /// Create empty (allow-all) rules.
    pub fn allow_all() -> Self {
        Self {
            disallowed: Vec::new(),
            allowed: Vec::new(),
            crawl_delay: None,
        }
    }

    /// Reglas que prohiben el sitio entero.
    ///
    /// Es lo que pide el RFC 9309 ante un 5xx (§2.3.1.4): el fichero queda
    /// «unreachable», o sea indefinido, y entonces el rastreador *«MUST assume
    /// complete disallow»*. No es lo mismo que un 4xx (§2.3.1.3), donde el
    /// fichero simplemente no existe y el rastreador **puede** acceder a todo.
    ///
    /// Esa distincion es la que faltaba, y la explica entera
    /// `WebCrawler::fetch_robots_rules`. Va en texto y no como enlace a
    /// proposito: esa funcion es privada, y un enlace de rustdoc desde un
    /// elemento publico a uno privado **es un enlace roto** --- lo canto la
    /// puerta `check_doc_links` con «<private-link>:0», que es exactamente para
    /// lo que esta.
    pub fn disallow_all() -> Self {
        Self {
            disallowed: vec!["/".to_string()],
            allowed: Vec::new(),
            crawl_delay: None,
        }
    }
}

/// A crawled page with extracted content.
#[derive(Debug, Clone)]
pub struct CrawledPage {
    /// The URL that was crawled.
    pub url: String,
    /// Page title extracted from <title> tag.
    pub title: Option<String>,
    /// Clean text content (HTML tags stripped).
    pub text: String,
    /// Links found on the page.
    pub links: Vec<String>,
    /// Crawl depth from the start URL.
    pub depth: usize,
}

/// Web crawler for document ingestion with robots.txt compliance.
pub struct WebCrawler {
    /// Maximum crawl depth.
    max_depth: usize,
    /// Maximum pages to crawl.
    max_pages: usize,
    /// Delay between requests in milliseconds.
    delay_ms: u64,
    /// User agent string.
    user_agent: String,
    /// robots.txt cache per domain.
    robots_cache: HashMap<String, RobotsRules>,
    /// Visited URLs.
    visited: std::collections::HashSet<String>,
    /// Quien habla HTTP.
    client: Arc<dyn HttpClient>,
}

impl WebCrawler {
    pub fn new() -> Self {
        Self {
            max_depth: 2,
            max_pages: 50,
            delay_ms: 500,
            user_agent: "AIAssistantCrawler/1.0".to_string(),
            robots_cache: HashMap::new(),
            visited: std::collections::HashSet::new(),
            client: Arc::new(UreqClient),
        }
    }

    /// Cambia quien habla HTTP. Ver [`DuckDuckGoProvider::with_http_client`].
    ///
    /// Aqui importa el doble: sin esto, probar la politica de `robots.txt`
    /// exigiria un servidor que contestara 404, 503 y basura a voluntad.
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_max_pages(mut self, pages: usize) -> Self {
        self.max_pages = pages;
        self
    }

    pub fn with_delay_ms(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self
    }

    pub fn with_user_agent(mut self, ua: &str) -> Self {
        self.user_agent = ua.to_string();
        self
    }

    /// Crawl starting from a URL, following links up to max_depth.
    pub fn crawl(&mut self, start_url: &str) -> Vec<CrawledPage> {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_url.to_string(), 0usize));
        let mut results = Vec::new();

        while let Some((url, depth)) = queue.pop_front() {
            if depth > self.max_depth || results.len() >= self.max_pages {
                break;
            }
            if self.visited.contains(&url) {
                continue;
            }
            if !self.is_allowed(&url) {
                self.visited.insert(url.clone());
                continue;
            }

            self.visited.insert(url.clone());

            // Respect crawl delay
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }

            if let Ok(page) = self.fetch_page(&url, depth) {
                // Only enqueue links if we haven't hit max depth
                if depth < self.max_depth {
                    for link in &page.links {
                        if !self.visited.contains(link) {
                            queue.push_back((link.clone(), depth + 1));
                        }
                    }
                }
                results.push(page);
            }
        }

        results
    }

    /// Trae el `robots.txt` de un dominio y lo interpreta **con la politica de
    /// estados del RFC 9309**.
    ///
    /// Antes esto estaba **dos veces**, en `fetch_robots` y dentro de
    /// `is_allowed`, y la segunda copia lo admitia en un comentario: *«For now,
    /// inline the fetch to avoid borrow issues»*. El prestamo se resuelve
    /// pidiendo las piezas por argumento en vez de `&self`, que es lo que hace
    /// esta funcion asociada. Dos copias de una regla son dos sitios donde
    /// arreglarla y uno donde olvidarse --- y aqui la regla estaba mal en las
    /// dos.
    ///
    /// **Lo que estaba mal**: los *tres* caminos de fallo ---error de
    /// transporte, no-2xx y cuerpo ilegible--- acababan en `allow_all()`. Para
    /// un **4xx** eso es correcto y la norma lo dice (§2.3.1.3: *«then the
    /// crawler MAY access any resources on the server»*). Para un **5xx** es lo
    /// contrario de lo que dice (§2.3.1.4: el fichero queda indefinido y el
    /// rastreador *«MUST assume complete disallow»*), y un servidor que se esta
    /// cayendo es justo al que no hay que entrarle a rastrear.
    ///
    /// Dos decisiones que **no** son literales del RFC y por eso van escritas:
    ///
    /// - un fallo de transporte no es un 5xx, pero es la misma condicion ---no
    ///   hay fichero que interpretar--- asi que se trata igual;
    /// - un 3xx que llegue hasta aqui significa que se agoto el limite de
    ///   redirecciones, y el RFC permite entonces tomar el fichero por
    ///   inexistente (§2.3.1.2 con §2.3.1.3), o sea `allow_all`.
    ///
    /// Y el cuerpo se decodifica **con perdida** en vez de descartarse: el
    /// `read_to_string().unwrap_or_default()` de antes convertia un
    /// `robots.txt` con un solo byte invalido en uno vacio, o sea en permiso
    /// total, cuando lo que pide la norma es seguir las reglas que **si** se
    /// pueden leer.
    fn fetch_robots_rules(client: &dyn HttpClient, user_agent: &str, domain: &str) -> RobotsRules {
        let request = HttpRequest::get(format!("https://{domain}/robots.txt"))
            .timeout(Duration::from_secs(5))
            .header("User-Agent", user_agent);

        match client.send(&request) {
            Ok(response) if response.is_success() => {
                let content = String::from_utf8_lossy(&response.body);
                Self::parse_robots_txt(&content, user_agent)
            }
            Ok(response) if (300..500).contains(&response.status) => RobotsRules::allow_all(),
            Ok(_) | Err(_) => RobotsRules::disallow_all(),
        }
    }

    /// Fetch robots.txt for a domain and cache it.
    pub fn fetch_robots(&mut self, domain: &str) -> Option<&RobotsRules> {
        if !self.robots_cache.contains_key(domain) {
            let rules = Self::fetch_robots_rules(&*self.client, &self.user_agent, domain);
            self.robots_cache.insert(domain.to_string(), rules);
        }
        self.robots_cache.get(domain)
    }

    /// Check if a URL is allowed by robots.txt.
    pub fn is_allowed(&mut self, url: &str) -> bool {
        let domain = match Self::extract_domain(url) {
            Some(d) => d,
            None => return false,
        };
        let path = Self::extract_path(url);

        // Un solo camino hacia las reglas: `fetch_robots` las cachea y las
        // devuelve. La copia inline que habia aqui ya no existe, y con ella se
        // va la posibilidad de que las dos respondan distinto al mismo servidor.
        match self.fetch_robots(&domain) {
            Some(rules) => Self::path_allowed(rules, &path),
            None => true,
        }
    }

    /// Check if a path is allowed by the given rules.
    fn path_allowed(rules: &RobotsRules, path: &str) -> bool {
        // Check allowed first (overrides disallow)
        for allowed in &rules.allowed {
            if path.starts_with(allowed) {
                return true;
            }
        }
        // Check disallowed
        for disallowed in &rules.disallowed {
            if disallowed == "/" {
                return false;
            }
            if path.starts_with(disallowed) {
                return false;
            }
        }
        true
    }

    /// Parse robots.txt content for a given user agent.
    pub fn parse_robots_txt(content: &str, user_agent: &str) -> RobotsRules {
        let mut rules = RobotsRules::allow_all();
        let ua_lower = user_agent.to_lowercase();

        // Extract the first token of the user agent (before '/')
        let ua_token = ua_lower.split('/').next().unwrap_or(&ua_lower);

        let mut in_matching_section = false;
        let mut found_specific = false;

        // First pass: look for user-agent-specific rules
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(ua_val) = line
                .strip_prefix("User-agent:")
                .or_else(|| line.strip_prefix("user-agent:"))
            {
                let val = ua_val.trim().to_lowercase();
                if val == "*" || val == ua_token || ua_token.starts_with(&val) {
                    in_matching_section = true;
                    if val != "*" {
                        found_specific = true;
                    }
                } else {
                    in_matching_section = false;
                }
                continue;
            }

            if !in_matching_section {
                continue;
            }

            if let Some(path) = line
                .strip_prefix("Disallow:")
                .or_else(|| line.strip_prefix("disallow:"))
            {
                let path = path.trim();
                if !path.is_empty() {
                    rules.disallowed.push(path.to_string());
                }
            } else if let Some(path) = line
                .strip_prefix("Allow:")
                .or_else(|| line.strip_prefix("allow:"))
            {
                let path = path.trim();
                if !path.is_empty() {
                    rules.allowed.push(path.to_string());
                }
            } else if let Some(delay) = line
                .strip_prefix("Crawl-delay:")
                .or_else(|| line.strip_prefix("crawl-delay:"))
            {
                if let Ok(d) = delay.trim().parse::<f64>() {
                    rules.crawl_delay = Some(d);
                }
            }
        }

        // If we found specific rules, filter out wildcard rules collected during the pass
        // Actually our simple parser collects all matching sections, which is fine
        let _ = found_specific;

        rules
    }

    /// Fetch a page and extract text + links.
    fn fetch_page(&self, url: &str, depth: usize) -> Result<CrawledPage, String> {
        let request = HttpRequest::get(url)
            .timeout(Duration::from_secs(15))
            .header("User-Agent", self.user_agent.as_str());

        let response = self
            .client
            .send(&request)
            .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

        // Detras del puerto un no-2xx vuelve como `Ok`, asi que el error hay
        // que pedirlo: sin esto, la pagina de error de un 404 se rastrearia
        // como si fuera contenido, con sus enlaces y todo.
        response
            .error_for_status()
            .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

        // Aqui SI se exige UTF-8 valido, al contrario que con el `robots.txt`:
        // de un `robots.txt` roto interesan las reglas que se puedan leer, pero
        // una pagina que no decodifica no es texto que valga la pena trocear, y
        // decodificar con perdida convertiria basura binaria en «contenido».
        let html = response
            .text()
            .map_err(|e| format!("Failed to read {}: {}", url, e))?;

        let title = Self::extract_title(&html);
        let text = Self::extract_text(&html);
        let base_url = Self::base_url_from(url);
        let links = Self::extract_links(&html, &base_url);

        Ok(CrawledPage {
            url: url.to_string(),
            title,
            text,
            links,
            depth,
        })
    }

    /// Extract clean text from HTML (strip tags, collapse whitespace).
    pub fn extract_text(html: &str) -> String {
        // Remove script and style blocks first
        let mut cleaned = html.to_string();

        // Remove <script>...</script> blocks
        while let Some(start) = cleaned.find("<script") {
            if let Some(end) = cleaned[start..].find("</script>") {
                cleaned = format!("{}{}", &cleaned[..start], &cleaned[start + end + 9..]);
            } else {
                break;
            }
        }

        // Remove <style>...</style> blocks
        while let Some(start) = cleaned.find("<style") {
            if let Some(end) = cleaned[start..].find("</style>") {
                cleaned = format!("{}{}", &cleaned[..start], &cleaned[start + end + 8..]);
            } else {
                break;
            }
        }

        // Strip remaining HTML tags
        let text = strip_html_tags(&cleaned);

        // Collapse whitespace
        let mut result = String::new();
        let mut last_was_space = false;
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }
        result.trim().to_string()
    }

    /// Extract links from HTML, resolving relative URLs against a base URL.
    pub fn extract_links(html: &str, base_url: &str) -> Vec<String> {
        let mut links = Vec::new();
        let mut pos = 0;

        while pos < html.len() {
            // Find <a ... href="...">
            let a_start = match html[pos..].find("<a ").or_else(|| html[pos..].find("<A ")) {
                Some(i) => pos + i,
                None => break,
            };

            let tag_end = match html[a_start..].find('>') {
                Some(i) => a_start + i,
                None => break,
            };

            let tag = &html[a_start..tag_end + 1];
            if let Some(href) = extract_attr(tag, "href") {
                let resolved = Self::resolve_url(&href, base_url);
                // Only include http(s) links, skip anchors, javascript, mailto
                if (resolved.starts_with("http://") || resolved.starts_with("https://"))
                    && !resolved.contains("javascript:")
                    && !resolved.contains("mailto:")
                {
                    // Remove fragment
                    let clean = resolved.split('#').next().unwrap_or(&resolved).to_string();
                    if !clean.is_empty() {
                        links.push(clean);
                    }
                }
            }

            pos = tag_end + 1;
        }

        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        links.retain(|l| seen.insert(l.clone()));

        links
    }

    /// Extract the <title> tag content.
    pub fn extract_title(html: &str) -> Option<String> {
        // Locate the tags in the ORIGINAL html (case-insensitive), so the byte
        // offsets are valid there. A `to_lowercase()` copy drifts on
        // length-changing chars before the title → slicing could panic.
        let (_, open_end) = crate::text_util::find_ci_range(html, "<title>")?;
        let (rel_close, _) = crate::text_util::find_ci_range(&html[open_end..], "</title>")?;
        let title = html[open_end..open_end + rel_close].trim().to_string();
        if title.is_empty() {
            None
        } else {
            Some(html_decode(&title))
        }
    }

    /// Resolve a potentially relative URL against a base URL.
    fn resolve_url(href: &str, base_url: &str) -> String {
        if href.starts_with("http://") || href.starts_with("https://") {
            return href.to_string();
        }
        if href.starts_with("//") {
            return format!("https:{}", href);
        }
        if href.starts_with('/') {
            // Absolute path — combine with base origin
            let origin = Self::extract_origin(base_url);
            return format!("{}{}", origin, href);
        }
        // Relative path
        let base = if base_url.ends_with('/') {
            base_url.to_string()
        } else {
            // Remove last path segment
            match base_url.rfind('/') {
                Some(i) if i > 8 => format!("{}/", &base_url[..i]),
                _ => format!("{}/", base_url),
            }
        };
        format!("{}{}", base, href)
    }

    /// Extract the origin (scheme + host) from a URL.
    fn extract_origin(url: &str) -> String {
        if let Some(idx) = url.find("://") {
            let after_scheme = &url[idx + 3..];
            if let Some(slash) = after_scheme.find('/') {
                return url[..idx + 3 + slash].to_string();
            }
        }
        url.to_string()
    }

    /// Extract domain from a URL.
    fn extract_domain(url: &str) -> Option<String> {
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))?;
        Some(
            without_scheme
                .split('/')
                .next()
                .unwrap_or(without_scheme)
                .to_string(),
        )
    }

    /// Extract path from a URL.
    fn extract_path(url: &str) -> String {
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url);
        match without_scheme.find('/') {
            Some(i) => without_scheme[i..].to_string(),
            None => "/".to_string(),
        }
    }

    /// Compute base URL for resolving relative links.
    fn base_url_from(url: &str) -> String {
        // Remove query string and fragment
        let clean = url.split('?').next().unwrap_or(url);
        let clean = clean.split('#').next().unwrap_or(clean);
        clean.to_string()
    }
}

impl Default for WebCrawler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Claim-oriented search helper (V83)
// =============================================================================

/// Search for evidence related to a specific factual claim.
///
/// Extracts keywords from the claim and searches using the provided engine.
/// Returns results sorted by relevance to the claim.
pub fn search_for_claim(
    engine: &mut WebSearchEngine,
    claim: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>, SearchError> {
    // Extract keywords from the claim (filter stopwords, keep significant words)
    let stopwords = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "and", "but", "or", "nor", "not", "no", "so", "yet", "for", "at", "by", "to", "from", "in",
        "on", "of", "with", "that", "this", "it", "its", "very", "also",
    ];

    let keywords: Vec<&str> = claim
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .filter(|w| !stopwords.contains(&w.to_lowercase().as_str()))
        .collect();

    if keywords.is_empty() {
        return Err(SearchError::InvalidQuery);
    }

    // Build query from keywords (max 8 keywords for focused search)
    let query: String = keywords
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>()
        .join(" ");

    let mut results = engine.search(&query)?;
    results.truncate(max_results);

    // Score results by relevance to the original claim
    let claim_lower = claim.to_lowercase();
    let claim_words: std::collections::HashSet<&str> = claim_lower
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    for result in &mut results {
        let snippet_lower = result.snippet.to_lowercase();
        let snippet_words: std::collections::HashSet<&str> = snippet_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        let intersection = claim_words.intersection(&snippet_words).count();
        let union = claim_words.union(&snippet_words).count();

        result.relevance_score = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };
    }

    // Sort by relevance descending
    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

// =============================================================================
// Academic Search Adapter
// =============================================================================

/// Adapter that wraps an academic search provider to implement `SearchProvider`.
///
/// This allows academic search results to flow through the standard web search
/// pipeline (e.g., for fact verification, claim checking).
#[cfg(feature = "research")]
pub struct AcademicSearchAdapter {
    /// Which academic source to search
    pub source: crate::academic_search::AcademicSource,
}

#[cfg(feature = "research")]
impl AcademicSearchAdapter {
    /// Create a new adapter for the given academic source.
    pub fn new(source: crate::academic_search::AcademicSource) -> Self {
        Self { source }
    }

    /// Create adapters for all default providers (arXiv, Semantic Scholar, PubMed).
    pub fn default_providers() -> Vec<Self> {
        use crate::academic_search::AcademicSource;
        vec![
            Self::new(AcademicSource::ArXiv),
            Self::new(AcademicSource::SemanticScholar),
            Self::new(AcademicSource::PubMed),
        ]
    }

    /// Convert an AcademicPaper to a SearchResult.
    fn paper_to_result(paper: &crate::academic_search::AcademicPaper) -> SearchResult {
        let snippet = paper
            .abstract_text
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(300)
            .collect::<String>();

        let url = paper.url.as_deref().unwrap_or("").to_string();

        let mut result = SearchResult::new(&paper.title, &url, &snippet);
        result.source = paper.source.display_name().to_string();
        if let Some(year) = paper.year {
            result.date = Some(year.to_string());
        }
        result
    }
}

#[cfg(feature = "research")]
impl SearchProvider for AcademicSearchAdapter {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        use crate::academic_search::*;

        let academic_config = AcademicSearchConfig {
            max_results: config.max_results,
            timeout: config.timeout,
            ..Default::default()
        };

        let provider: Box<dyn AcademicSearchProvider> = match self.source {
            AcademicSource::ArXiv => Box::new(ArxivProvider::new()),
            AcademicSource::SemanticScholar => Box::new(SemanticScholarProvider::new()),
            AcademicSource::PubMed => Box::new(PubMedProvider::new()),
            AcademicSource::OpenAlex => Box::new(OpenAlexProvider::new()),
            AcademicSource::CrossRef => Box::new(CrossrefProvider::new()),
            // `Supplied` marks a record the caller handed us, not a database we
            // can query. Refusing beats substituting a provider: the caller
            // would otherwise get results from a corpus they never asked for.
            _ => return Err(SearchError::InvalidQuery),
        };

        let papers = provider
            .search_papers(query, &academic_config)
            .map_err(|e| SearchError::Network(e.to_string()))?;

        Ok(papers.iter().map(Self::paper_to_result).collect())
    }

    fn name(&self) -> &str {
        match self.source {
            crate::academic_search::AcademicSource::ArXiv => "arXiv",
            crate::academic_search::AcademicSource::SemanticScholar => "Semantic Scholar",
            crate::academic_search::AcademicSource::PubMed => "PubMed",
            _ => "Academic",
        }
    }
}

/// Search error
#[derive(Debug)]
#[non_exhaustive]
pub enum SearchError {
    Network(String),
    Parse(String),
    NoResults,
    RateLimit,
    InvalidQuery,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "Network error: {}", e),
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::NoResults => write!(f, "No results found"),
            Self::RateLimit => write!(f, "Rate limited"),
            Self::InvalidQuery => write!(f, "Invalid query"),
        }
    }
}

impl std::error::Error for SearchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result() {
        let result = SearchResult::new("Test", "https://example.com/page", "Snippet");
        assert_eq!(result.source, "example.com");
    }

    #[test]
    fn test_html_decode() {
        assert_eq!(html_decode("&amp;"), "&");
        assert_eq!(html_decode("&lt;tag&gt;"), "<tag>");
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("http://sub.example.org/"), "sub.example.org");
    }

    #[test]
    fn test_config() {
        let config = SearchConfig::default();
        assert_eq!(config.max_results, 10);
        assert_eq!(config.safe_search, SafeSearch::Moderate);
    }

    // ========================================================================
    // Enhanced DuckDuckGo Provider Tests
    // ========================================================================

    #[test]
    fn test_urlencoded() {
        assert_eq!(search_urlencode("hello world"), "hello+world");
        assert_eq!(search_urlencode("a/b"), "a%2Fb");
        assert_eq!(search_urlencode("test"), "test");
        assert_eq!(search_urlencode("foo@bar"), "foo%40bar");
        assert_eq!(search_urlencode("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn test_ddg_html_parsing() {
        let html = r##"
        <div class="result">
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">Example Page One</a>
            <a class="result__snippet" href="#">This is the first result snippet</a>
        </div>
        <div class="result">
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fpage2">Example Page Two</a>
            <a class="result__snippet" href="#">This is the second result snippet</a>
        </div>
        "##;

        let results = parse_ddg_results(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example Page One");
        assert_eq!(results[0].url, "https://example.com/page1");
        assert_eq!(results[0].snippet, "This is the first result snippet");
        assert_eq!(results[1].title, "Example Page Two");
        assert_eq!(results[1].url, "https://example.org/page2");
    }

    // `test_search_cache_hit` y `test_search_cache_expired` estaban aqui y
    // ahora viven en `transport_tests`, con los dobles. El segundo aceptaba
    // `Ok` **o** `Err` y por tanto no podia fallar; ademas llamaba a
    // duckduckgo.com de verdad desde la suite. Las dos versiones nuevas
    // comprueban lo que estos decian comprobar: que la cache **evita** la
    // peticion y que una caducada la **provoca**.

    #[test]
    fn test_dedup_results() {
        let mut results = vec![
            SearchResult::new("A", "https://example.com/a", "snippet a"),
            SearchResult::new("B", "https://example.com/b", "snippet b"),
            SearchResult::new("A dup", "https://example.com/a", "snippet a dup"),
            SearchResult::new("C", "https://example.com/c", "snippet c"),
            SearchResult::new("B dup", "https://example.com/b", "snippet b dup"),
        ];
        dedup_results(&mut results);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title, "A");
        assert_eq!(results[1].title, "B");
        assert_eq!(results[2].title, "C");
    }

    #[test]
    fn test_dedup_empty() {
        let mut results: Vec<SearchResult> = Vec::new();
        dedup_results(&mut results);
        assert!(results.is_empty());
    }

    // ========================================================================
    // Web Crawler Tests
    // ========================================================================

    #[test]
    fn test_robots_txt_parsing() {
        let content = r#"
User-agent: *
Disallow: /private/
Disallow: /admin/
Allow: /admin/public/
Crawl-delay: 2

User-agent: Googlebot
Disallow: /nogoogle/
"#;
        let rules = WebCrawler::parse_robots_txt(content, "AIAssistantCrawler/1.0");
        assert!(rules.disallowed.contains(&"/private/".to_string()));
        assert!(rules.disallowed.contains(&"/admin/".to_string()));
        assert!(rules.allowed.contains(&"/admin/public/".to_string()));
        assert_eq!(rules.crawl_delay, Some(2.0));
    }

    #[test]
    fn test_robots_disallowed_path() {
        let rules = RobotsRules {
            disallowed: vec!["/private/".to_string(), "/secret/".to_string()],
            allowed: vec![],
            crawl_delay: None,
        };
        assert!(!WebCrawler::path_allowed(&rules, "/private/page.html"));
        assert!(!WebCrawler::path_allowed(&rules, "/secret/data"));
        assert!(WebCrawler::path_allowed(&rules, "/public/page.html"));
    }

    #[test]
    fn test_robots_allowed_path() {
        let rules = RobotsRules {
            disallowed: vec!["/admin/".to_string()],
            allowed: vec!["/admin/public/".to_string()],
            crawl_delay: None,
        };
        // Allowed overrides disallow
        assert!(WebCrawler::path_allowed(&rules, "/admin/public/page.html"));
        // Still disallowed for other admin paths
        assert!(!WebCrawler::path_allowed(&rules, "/admin/secret/page.html"));
    }

    #[test]
    fn test_robots_crawl_delay() {
        let content = "User-agent: *\nCrawl-delay: 5.5\nDisallow: /tmp/\n";
        let rules = WebCrawler::parse_robots_txt(content, "TestBot/1.0");
        assert_eq!(rules.crawl_delay, Some(5.5));
    }

    #[test]
    fn test_crawler_max_depth() {
        let mut crawler = WebCrawler::new().with_max_depth(0).with_delay_ms(0);
        // With max_depth=0, only the start URL should be crawled (depth 0)
        // Since we can't make real HTTP calls, verify the setting
        assert_eq!(crawler.max_depth, 0);
        assert_eq!(crawler.max_pages, 50);

        // Crawl returns empty because the network call will fail
        let results = crawler.crawl("https://nonexistent.example.com/");
        // Should have tried to crawl depth=0 (the start URL), may be empty due to network error
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_crawler_max_pages() {
        let crawler = WebCrawler::new().with_max_pages(3).with_delay_ms(0);
        assert_eq!(crawler.max_pages, 3);
    }

    #[test]
    fn test_extract_text_strips_tags() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        let text = WebCrawler::extract_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<h1>"));
        assert!(!text.contains("<p>"));
        assert!(!text.contains("<b>"));
    }

    #[test]
    fn test_extract_text_strips_script_style() {
        let html = r#"<html>
            <head><style>body { color: red; }</style></head>
            <body>
                <p>Visible text</p>
                <script>var x = 1;</script>
                <p>More text</p>
            </body>
        </html>"#;
        let text = WebCrawler::extract_text(html);
        assert!(text.contains("Visible text"));
        assert!(text.contains("More text"));
        assert!(!text.contains("color: red"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn test_extract_links_absolute() {
        let html = r#"<html><body>
            <a href="https://example.com/page1">Link 1</a>
            <a href="https://example.org/page2">Link 2</a>
        </body></html>"#;
        let links = WebCrawler::extract_links(html, "https://base.com/");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://example.com/page1");
        assert_eq!(links[1], "https://example.org/page2");
    }

    #[test]
    fn test_extract_links_relative() {
        let html = r#"<html><body>
            <a href="/about">About</a>
            <a href="contact.html">Contact</a>
            <a href="//cdn.example.com/resource">CDN</a>
        </body></html>"#;
        let links = WebCrawler::extract_links(html, "https://example.com/pages/index.html");
        assert!(links.contains(&"https://example.com/about".to_string()));
        assert!(links.contains(&"https://example.com/pages/contact.html".to_string()));
        assert!(links.contains(&"https://cdn.example.com/resource".to_string()));
    }

    #[test]
    fn test_extract_title() {
        let html = "<html><head><title>My Page Title</title></head><body></body></html>";
        assert_eq!(
            WebCrawler::extract_title(html),
            Some("My Page Title".to_string())
        );
    }

    #[test]
    fn test_extract_title_missing() {
        let html = "<html><head></head><body>No title</body></html>";
        assert_eq!(WebCrawler::extract_title(html), None);
    }

    #[test]
    fn test_extract_title_empty() {
        let html = "<html><head><title></title></head><body></body></html>";
        assert_eq!(WebCrawler::extract_title(html), None);
    }

    #[test]
    fn test_crawler_respects_visited() {
        let mut crawler = WebCrawler::new().with_delay_ms(0);
        // Mark a URL as visited
        crawler
            .visited
            .insert("https://example.com/visited".to_string());

        // The visited set should prevent re-crawling
        assert!(crawler.visited.contains("https://example.com/visited"));

        // Crawling with a visited start URL should produce no results
        let results = crawler.crawl("https://example.com/visited");
        assert!(results.is_empty());
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>bold</b>"), "bold");
        assert_eq!(strip_html_tags("no tags"), "no tags");
        assert_eq!(
            strip_html_tags("<a href=\"url\">link</a> text"),
            "link text"
        );
    }

    #[test]
    fn test_simple_urldecode() {
        assert_eq!(simple_urldecode("hello%20world"), "hello world");
        assert_eq!(simple_urldecode("a%2Fb"), "a/b");
        assert_eq!(simple_urldecode("test+query"), "test query");
        assert_eq!(simple_urldecode("plain"), "plain");
    }

    #[test]
    fn test_resolve_url_absolute() {
        let url = WebCrawler::resolve_url("https://other.com/page", "https://base.com/");
        assert_eq!(url, "https://other.com/page");
    }

    #[test]
    fn test_resolve_url_relative_root() {
        let url = WebCrawler::resolve_url("/about", "https://example.com/pages/index.html");
        assert_eq!(url, "https://example.com/about");
    }

    #[test]
    fn test_resolve_url_relative_path() {
        let url = WebCrawler::resolve_url("contact.html", "https://example.com/pages/index.html");
        assert_eq!(url, "https://example.com/pages/contact.html");
    }

    #[test]
    fn test_resolve_url_protocol_relative() {
        let url = WebCrawler::resolve_url("//cdn.example.com/res", "https://example.com/");
        assert_eq!(url, "https://cdn.example.com/res");
    }

    #[test]
    fn test_extract_links_dedup() {
        let html = r#"<html><body>
            <a href="https://example.com/page1">Link 1</a>
            <a href="https://example.com/page1">Link 1 again</a>
        </body></html>"#;
        let links = WebCrawler::extract_links(html, "https://base.com/");
        assert_eq!(links.len(), 1);
    }

    #[test]
    fn test_extract_links_filters_javascript() {
        let html = r#"<html><body>
            <a href="javascript:void(0)">JS Link</a>
            <a href="mailto:test@example.com">Email</a>
            <a href="https://example.com/real">Real Link</a>
        </body></html>"#;
        let links = WebCrawler::extract_links(html, "https://base.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0], "https://example.com/real");
    }

    #[test]
    fn test_robots_disallow_all() {
        let content = "User-agent: *\nDisallow: /\n";
        let rules = WebCrawler::parse_robots_txt(content, "TestBot/1.0");
        assert!(!WebCrawler::path_allowed(&rules, "/anything"));
        assert!(!WebCrawler::path_allowed(&rules, "/"));
    }

    #[test]
    fn test_robots_empty_allows_all() {
        let content = "";
        let rules = WebCrawler::parse_robots_txt(content, "TestBot/1.0");
        assert!(WebCrawler::path_allowed(&rules, "/anything"));
        assert!(rules.disallowed.is_empty());
    }

    // === V83: search_for_claim ===

    #[test]
    fn test_search_for_claim_with_cached_engine() {
        let mut engine = WebSearchEngine::new(SearchConfig::default());
        // Manually insert a cached result
        let cached = vec![
            SearchResult::new(
                "Python info",
                "https://python.org",
                "Python is a programming language",
            ),
            SearchResult::new(
                "Unrelated",
                "https://other.com",
                "Cooking recipes for pasta",
            ),
        ];
        engine.cache.insert(
            "python programming language".to_string(),
            (cached, std::time::Instant::now()),
        );

        let results = search_for_claim(&mut engine, "Python is a programming language", 10);
        assert!(results.is_ok());
        let results = results.unwrap();
        // First result should be the more relevant one
        assert!(!results.is_empty());
        assert!(results[0].relevance_score >= results.last().unwrap().relevance_score);
    }

    #[test]
    fn test_search_for_claim_empty_query() {
        let mut engine = WebSearchEngine::new(SearchConfig::default());
        // All stopwords → empty keywords → InvalidQuery
        let result = search_for_claim(&mut engine, "the is a an", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_for_claim_truncates_results() {
        let mut engine = WebSearchEngine::new(SearchConfig::default());
        let cached = vec![
            SearchResult::new("R1", "https://a.com", "result one content"),
            SearchResult::new("R2", "https://b.com", "result two content"),
            SearchResult::new("R3", "https://c.com", "result three content"),
        ];
        engine.cache.insert(
            "result content".to_string(),
            (cached, std::time::Instant::now()),
        );

        let results = search_for_claim(&mut engine, "result content", 2).unwrap();
        assert!(results.len() <= 2);
    }

    #[cfg(feature = "research")]
    #[test]
    fn test_academic_adapter_creation() {
        use crate::academic_search::AcademicSource;
        let adapter = AcademicSearchAdapter::new(AcademicSource::ArXiv);
        assert_eq!(adapter.name(), "arXiv");

        let adapter_s2 = AcademicSearchAdapter::new(AcademicSource::SemanticScholar);
        assert_eq!(adapter_s2.name(), "Semantic Scholar");

        let adapter_pm = AcademicSearchAdapter::new(AcademicSource::PubMed);
        assert_eq!(adapter_pm.name(), "PubMed");
    }

    #[cfg(feature = "research")]
    #[test]
    fn test_academic_adapter_default_providers() {
        let adapters = AcademicSearchAdapter::default_providers();
        assert_eq!(adapters.len(), 3);
        assert_eq!(adapters[0].name(), "arXiv");
        assert_eq!(adapters[1].name(), "Semantic Scholar");
        assert_eq!(adapters[2].name(), "PubMed");
    }

    #[cfg(feature = "research")]
    #[test]
    fn test_academic_adapter_paper_to_result() {
        use crate::academic_search::{AcademicPaper, AcademicSource, Author};
        let mut paper = AcademicPaper::new("123", "Test Paper Title", AcademicSource::ArXiv);
        paper.authors = vec![Author::new("John Doe")];
        paper.abstract_text = Some("This is the abstract of the paper.".to_string());
        paper.url = Some("https://arxiv.org/abs/123".to_string());
        paper.year = Some(2024);

        let result = AcademicSearchAdapter::paper_to_result(&paper);
        assert_eq!(result.title, "Test Paper Title");
        assert!(result.snippet.contains("abstract"));
        assert_eq!(result.source, "arXiv");
        assert_eq!(result.date.as_deref(), Some("2024"));
    }
}

/// Los siete proveedores, probados sin red.
///
/// Antes de V470 no habia ni un test de ninguno de los siete: lo unico que se
/// probaba de este modulo era el troceado de HTML y las funciones puras. La
/// razon no era descuido ---la URL del servicio esta incrustada en cuatro de
/// ellos, asi que no habia donde apuntar un servidor de pruebas---, y es
/// exactamente el hueco que cierra inyectar el cliente.
#[cfg(test)]
mod transport_tests {
    use super::*;
    use crate::http_client::{HttpMethod, HttpStream};

    /// Contesta una respuesta preparada y guarda la peticion que vio.
    ///
    /// Un doble y no un `MockHttpServer`: ademas de servir para los cuatro de
    /// URL fija, el test no depende de que haya un puerto libre ni de lo que
    /// escuche en localhost.
    struct Canned {
        status: u16,
        body: Vec<u8>,
        seen: Mutex<Vec<HttpRequest>>,
    }

    impl Canned {
        fn new(status: u16, body: &str) -> Self {
            Self {
                status,
                body: body.as_bytes().to_vec(),
                seen: Mutex::new(Vec::new()),
            }
        }

        /// Un cuerpo que **no** es UTF-8 valido, que `String` no puede llevar.
        fn raw(status: u16, body: Vec<u8>) -> Self {
            Self {
                status,
                body,
                seen: Mutex::new(Vec::new()),
            }
        }

        /// La ultima peticion que el proveedor mando de verdad al cable.
        fn seen(&self) -> HttpRequest {
            self.seen
                .lock()
                .expect("el candado del doble")
                .last()
                .cloned()
                .expect("el proveedor no mando ninguna peticion")
        }

        /// Cuantas veces se ha salido al cable. Es lo que permite comprobar que
        /// una cache **evita** una peticion, en vez de suponerlo.
        fn calls(&self) -> usize {
            self.seen.lock().expect("el candado del doble").len()
        }
    }

    impl HttpClient for Canned {
        fn send(&self, request: &HttpRequest) -> anyhow::Result<HttpResponse> {
            self.seen
                .lock()
                .expect("el candado del doble")
                .push(request.clone());
            Ok(HttpResponse {
                status: self.status,
                headers: Vec::new(),
                body: self.body.clone(),
            })
        }

        fn send_streaming(&self, _request: &HttpRequest) -> anyhow::Result<HttpStream> {
            anyhow::bail!("el doble no hace streaming")
        }
    }

    /// No hay respuesta ninguna: el transporte falla.
    struct NoTransport;

    impl HttpClient for NoTransport {
        fn send(&self, _request: &HttpRequest) -> anyhow::Result<HttpResponse> {
            anyhow::bail!("conexion rehusada")
        }

        fn send_streaming(&self, _request: &HttpRequest) -> anyhow::Result<HttpStream> {
            anyhow::bail!("conexion rehusada")
        }
    }

    fn config_with(max_results: usize) -> SearchConfig {
        SearchConfig {
            max_results,
            ..SearchConfig::default()
        }
    }

    #[test]
    fn test_a_429_is_rate_limit_and_not_a_network_error() {
        let spy = Arc::new(Canned::new(429, r#"{"error":"quota exceeded"}"#));
        let provider = BraveSearchProvider::new("k").with_http_client(spy);
        let err = provider
            .search("rust", &config_with(5))
            .expect_err("un 429 no es una lista de resultados");
        assert!(
            matches!(err, SearchError::RateLimit),
            "«frena» tiene que distinguirse de «no hay ruta al host»: {err:?}"
        );
    }

    #[test]
    fn test_a_503_is_rate_limit_too_because_that_is_what_the_server_means() {
        let spy = Arc::new(Canned::new(503, "overloaded"));
        let provider = SearXNGProvider::new("http://searx.invalid").with_http_client(spy);
        let err = provider
            .search("rust", &config_with(5))
            .expect_err("un 503 no es una lista de resultados");
        assert!(matches!(err, SearchError::RateLimit), "{err:?}");
    }

    #[test]
    fn test_a_401_carries_the_body_that_says_why() {
        // Lo que `ureq` tiraba: su `Error::StatusCode(u16)` no lleva la
        // respuesta, asi que «clave caducada» y «dominio no autorizado»
        // llegaban al llamante como el mismo "status code 401".
        let spy = Arc::new(Canned::new(
            401,
            r#"{"message":"subscription token expired"}"#,
        ));
        let provider = BraveSearchProvider::new("k").with_http_client(spy);
        let err = provider
            .search("rust", &config_with(5))
            .expect_err("un 401 no es una lista de resultados");
        let shown = err.to_string();
        assert!(
            shown.contains("subscription token expired"),
            "el motivo que da el proveedor tiene que llegar: {shown}"
        );
        assert!(shown.contains("401"), "y el codigo tambien: {shown}");
    }

    #[test]
    fn test_a_transport_failure_is_still_a_network_error() {
        let provider = BraveSearchProvider::new("k").with_http_client(Arc::new(NoTransport));
        let err = provider
            .search("rust", &config_with(5))
            .expect_err("sin transporte no hay resultados");
        assert!(matches!(err, SearchError::Network(_)), "{err:?}");
    }

    #[test]
    fn test_google_caps_num_at_ten_even_when_asked_for_fifty() {
        let spy = Arc::new(Canned::new(200, r#"{"items":[]}"#));
        let provider = GoogleSearchProvider::new("key", "cx").with_http_client(spy.clone());
        provider
            .search("rust", &config_with(50))
            .expect("un 200 vacio es una lista vacia");
        let url = spy.seen().url;
        assert!(
            url.contains("num=10"),
            "Google sirve 10 por pagina y pedirle 50 es pedir un error: {url}"
        );
    }

    #[test]
    fn test_bing_sends_its_key_in_a_header_and_not_in_the_url() {
        let spy = Arc::new(Canned::new(200, r#"{"webPages":{"value":[]}}"#));
        let provider = BingSearchProvider::new("secreto").with_http_client(spy.clone());
        provider.search("rust", &config_with(3)).expect("200");
        let request = spy.seen();
        assert!(
            request
                .headers
                .iter()
                .any(|(n, v)| n == "Ocp-Apim-Subscription-Key" && v == "secreto"),
            "la clave va en la cabecera: {:?}",
            request.headers
        );
        assert!(
            !request.url.contains("secreto"),
            "y no en la URL, que es lo que acaba en los registros: {}",
            request.url
        );
    }

    #[test]
    fn test_tavily_posts_the_query_as_json_and_reads_content_as_the_snippet() {
        let spy = Arc::new(Canned::new(
            200,
            r#"{"results":[{"title":"T","url":"u","content":"el contenido"}]}"#,
        ));
        let provider = TavilyProvider::new("k").with_http_client(spy.clone());
        let results = provider
            .search("rust async", &config_with(3))
            .expect("200 con un resultado");
        let request = spy.seen();
        assert_eq!(request.method, HttpMethod::Post);
        assert!(
            request
                .headers
                .iter()
                .any(|(n, v)| n == "Content-Type" && v == "application/json"),
            "el Content-Type lo pone el puerto: {:?}",
            request.headers
        );
        let sent = String::from_utf8(request.body.clone().expect("un POST lleva cuerpo"))
            .expect("el cuerpo es UTF-8");
        assert!(
            sent.contains("rust async"),
            "la consulta va en el cuerpo, no en la URL: {sent}"
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "el contenido");
    }

    #[test]
    fn test_serpapi_parses_the_organic_results_and_nothing_else() {
        let spy = Arc::new(Canned::new(
            200,
            r#"{"organic_results":[{"title":"A","link":"la","snippet":"sa"}],
                "ads":[{"title":"anuncio","link":"lb","snippet":"sb"}]}"#,
        ));
        let provider = SerpApiProvider::new("k").with_http_client(spy);
        let results = provider.search("rust", &config_with(5)).expect("200");
        assert_eq!(results.len(), 1, "un anuncio no es un resultado organico");
        assert_eq!(results[0].title, "A");
    }

    #[test]
    fn test_duckduckgo_scrapes_the_html_it_is_served() {
        let html = r#"<a class="result__a" href="https://example.com/p">Un titulo</a>
                      <a class="result__snippet">Un resumen</a>"#;
        let spy = Arc::new(Canned::new(200, html));
        let provider = DuckDuckGoProvider::new().with_http_client(spy);
        let results = provider.search("rust", &config_with(5)).expect("200");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.com/p");
        assert_eq!(results[0].title, "Un titulo");
        assert_eq!(results[0].snippet, "Un resumen");
    }

    #[test]
    fn test_searxng_returns_no_more_than_the_configured_maximum() {
        let spy = Arc::new(Canned::new(
            200,
            r#"{"results":[{"title":"1","url":"u1","content":"c1"},
                           {"title":"2","url":"u2","content":"c2"},
                           {"title":"3","url":"u3","content":"c3"}]}"#,
        ));
        let provider = SearXNGProvider::new("http://searx.invalid/").with_http_client(spy);
        let results = provider.search("rust", &config_with(2)).expect("200");
        assert_eq!(results.len(), 2, "el tope es del llamante, no del servidor");
    }

    // ======================================================================
    // El proveedor con cache, que hasta V471 no podia usar nadie
    // ======================================================================

    const DDG_UN_RESULTADO: &str = r#"<a class="result__a" href="https://example.com/p">Un titulo</a>
                                      <a class="result__snippet">Un resumen</a>"#;

    fn cached_provider(spy: Arc<Canned>) -> EnhancedDuckDuckGoProvider {
        EnhancedDuckDuckGoProvider::new()
            .with_min_interval(0.0)
            .with_http_client(spy)
    }

    #[test]
    fn test_the_engine_can_hold_the_cached_provider_now() {
        // Lo que no se podia escribir antes de V471: `add_provider` pide
        // `Box<dyn SearchProvider>` y este tipo no implementaba el trait, asi
        // que el proveedor con cache y ritmo se quedaba fuera del motor.
        let spy = Arc::new(Canned::new(200, DDG_UN_RESULTADO));
        let mut engine = WebSearchEngine::new(config_with(10));
        engine.add_provider(Box::new(cached_provider(spy.clone())));

        let results = engine
            .search("rust")
            .expect("el motor pregunta al proveedor");
        assert_eq!(results.len(), 1);
        assert_eq!(spy.calls(), 1);
    }

    #[test]
    fn test_the_cache_answers_the_second_identical_query_without_asking_again() {
        let spy = Arc::new(Canned::new(200, DDG_UN_RESULTADO));
        let provider = cached_provider(spy.clone());

        let first = SearchProvider::search(&provider, "rust", &config_with(10)).expect("200");
        let second =
            SearchProvider::search(&provider, "rust", &config_with(10)).expect("de la cache");

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), first.len());
        assert_eq!(second[0].url, first[0].url);
        assert_eq!(
            spy.calls(),
            1,
            "la segunda consulta no tenia que salir al cable"
        );
    }

    #[test]
    fn test_an_expired_entry_does_cause_a_second_request() {
        // El test que esto sustituye aceptaba `Ok` **o** `Err`, o sea que no
        // podia fallar, y ademas llamaba a duckduckgo.com de verdad.
        let spy = Arc::new(Canned::new(200, DDG_UN_RESULTADO));
        let provider = cached_provider(spy.clone()).with_cache_ttl(0);

        SearchProvider::search(&provider, "rust", &config_with(10)).expect("200");
        SearchProvider::search(&provider, "rust", &config_with(10)).expect("200");

        assert_eq!(spy.calls(), 2, "con TTL 0 no hay nada que reutilizar");
    }

    #[test]
    fn test_the_rate_limiter_counts_a_request_that_failed_too() {
        // El defecto: `last_request` solo se apuntaba tras un exito, asi que un
        // bucle de fallos no esperaba nada entre intentos --- justo el caso en
        // que el servidor te quiere lejos.
        let provider = EnhancedDuckDuckGoProvider::new()
            .with_min_interval(0.3)
            .with_http_client(Arc::new(NoTransport));

        let started = std::time::Instant::now();
        for _ in 0..2 {
            assert!(SearchProvider::search(&provider, "rust", &config_with(3)).is_err());
        }

        assert!(
            started.elapsed() >= Duration::from_millis(250),
            "la segunda tenia que esperar su turno, y todo tardo {:?}",
            started.elapsed()
        );
    }

    // ======================================================================
    // Las cuatro cosas en que perdia el lector de regex
    // ======================================================================

    #[test]
    fn test_a_highlighted_term_no_longer_truncates_the_title() {
        // DuckDuckGo envuelve en `<b>` cada termino de la consulta que aparece
        // en el titulo o en el resumen. El lector de regex cortaba en
        // `([^<]*)`, o sea en el primer resaltado: truncaba casi siempre.
        let html = r#"<a class="result__a" href="https://example.com/p">rust <b>async</b> hoy</a>
                      <a class="result__snippet">el <b>async</b> de Rust</a>"#;
        let results = parse_ddg_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "rust async hoy");
        assert_eq!(results[0].snippet, "el async de Rust");
    }

    #[test]
    fn test_the_redirect_url_drops_the_trailing_parameters() {
        // `uddg=` lleva la URL real y detras puede ir `&rut=...`. El lector de
        // regex no cortaba en `&`, asi que se lo llevaba pegado.
        let html =
            r#"<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fp&rut=abc">T</a>"#;
        let results = parse_ddg_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.org/p");
    }

    #[test]
    fn test_the_attributes_may_come_in_the_other_order() {
        // El lector de regex exigia `href` DESPUES de `class`.
        let html = r#"<a href="https://example.com/p" class="result__a">T</a>"#;
        let results = parse_ddg_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.com/p");
    }

    // ======================================================================
    // El rastreador: la politica de `robots.txt` del RFC 9309
    //
    // Los tres caminos de fallo acababan en «todo permitido». Para un 4xx eso
    // es la norma; para un 5xx es lo contrario de la norma.
    // ======================================================================

    fn crawler_with(client: Arc<dyn HttpClient>) -> WebCrawler {
        WebCrawler::new().with_http_client(client)
    }

    #[test]
    fn test_a_404_robots_allows_the_whole_site() {
        // RFC 9309 §2.3.1.3: no hay fichero, «the crawler MAY access any
        // resources on the server». Este camino ya era correcto.
        let mut crawler = crawler_with(Arc::new(Canned::new(404, "not found")));
        assert!(crawler.is_allowed("https://example.invalid/cualquiera"));
    }

    #[test]
    fn test_a_503_robots_disallows_everything_as_the_rfc_requires() {
        // §2.3.1.4: el fichero queda indefinido y el rastreador «MUST assume
        // complete disallow». Antes de V472 esto devolvia `true`.
        let mut crawler = crawler_with(Arc::new(Canned::new(503, "overloaded")));
        assert!(
            !crawler.is_allowed("https://example.invalid/cualquiera"),
            "a un servidor que se esta cayendo no se le entra a rastrear"
        );
    }

    #[test]
    fn test_a_transport_failure_disallows_everything_too() {
        // No es un 5xx, pero es la misma condicion: no hay fichero que
        // interpretar. Antes tambien devolvia `true`.
        let mut crawler = crawler_with(Arc::new(NoTransport));
        assert!(!crawler.is_allowed("https://example.invalid/cualquiera"));
    }

    #[test]
    fn test_an_undecodable_robots_still_applies_the_rules_it_can_read() {
        // El `unwrap_or_default()` de antes convertia esto en un fichero vacio,
        // o sea en permiso total, por un solo byte invalido.
        let mut body = b"User-agent: *\nDisallow: /privado\n".to_vec();
        body.extend_from_slice(&[0xff, 0xfe]); // basura que no es UTF-8
        let mut crawler = crawler_with(Arc::new(Canned::raw(200, body)));

        assert!(
            !crawler.is_allowed("https://example.invalid/privado/x"),
            "la regla legible tiene que seguir aplicandose"
        );
        assert!(crawler.is_allowed("https://example.invalid/publico"));
    }

    #[test]
    fn test_both_entry_points_agree_on_the_same_server() {
        // Lo que la duplicacion no podia garantizar: `fetch_robots` y
        // `is_allowed` leian el mismo servidor con dos copias del codigo.
        let mut crawler = crawler_with(Arc::new(Canned::new(503, "")));
        let desde_fetch = crawler
            .fetch_robots("example.invalid")
            .map(|r| r.disallowed.clone())
            .unwrap_or_default();
        let desde_is_allowed = crawler.is_allowed("https://example.invalid/x");

        assert_eq!(desde_fetch, vec!["/".to_string()]);
        assert!(!desde_is_allowed, "y la otra puerta dice lo mismo");
    }

    #[test]
    fn test_fetch_page_refuses_a_non_2xx_instead_of_crawling_the_error_page() {
        // Detras del puerto un no-2xx vuelve como `Ok`: sin `error_for_status`
        // la pagina de un 404 se rastrearia como contenido, con sus enlaces.
        let crawler = crawler_with(Arc::new(Canned::new(
            404,
            "<html><title>No existe</title><a href=\"/otra\">otra</a></html>",
        )));
        let fallo = crawler
            .fetch_page("https://example.invalid/x", 0)
            .expect_err("un 404 no es una pagina rastreada");
        assert!(
            fallo.contains("404"),
            "y el codigo va en el mensaje: {fallo}"
        );
    }

    #[test]
    fn test_fetch_page_reads_the_page_it_is_served() {
        let crawler = crawler_with(Arc::new(Canned::new(
            200,
            "<html><title>Hola</title><body>texto<a href=\"https://example.invalid/b\">b</a></body></html>",
        )));
        let page = crawler
            .fetch_page("https://example.invalid/a", 1)
            .expect("200");
        assert_eq!(page.title.as_deref(), Some("Hola"));
        assert_eq!(page.depth, 1);
        assert!(page.text.contains("texto"));
        assert!(page.links.iter().any(|l| l.ends_with("/b")));
    }
}
