//! Crawl policy management
//!
//! Parses robots.txt files, checks URL permissions, implements per-domain
//! rate limiting, and discovers sitemaps. Provides a complete crawl policy
//! engine suitable for respectful web crawling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::http_client::{HttpClient, HttpRequest, UreqClient};

// ─────────────────────────────────────────────────────────────────────────────
// Robots.txt types
// ─────────────────────────────────────────────────────────────────────────────

/// A single rule from a robots.txt file (Allow or Disallow directive).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotsRule {
    /// The path pattern this rule applies to.
    pub path: String,
    /// Whether this path is allowed (`true`) or disallowed (`false`).
    pub allowed: bool,
}

/// Directives for a specific user-agent section in robots.txt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotsDirectives {
    /// The user-agent string this section applies to.
    pub user_agent: String,
    /// Ordered list of rules (Allow/Disallow) for this user-agent.
    pub rules: Vec<RobotsRule>,
    /// Optional crawl delay in seconds.
    pub crawl_delay: Option<f64>,
    /// Sitemaps declared within this user-agent section.
    pub sitemaps: Vec<String>,
}

/// A fully parsed robots.txt file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRobotsTxt {
    /// Directives keyed by lowercase user-agent string.
    pub directives: HashMap<String, RobotsDirectives>,
    /// Global sitemaps declared outside any user-agent section.
    pub sitemaps: Vec<String>,
    /// The raw text content of the robots.txt file.
    pub raw: String,
}

impl ParsedRobotsTxt {
    /// Un robots.txt que no prohibe nada, como el de un dominio que contesta
    /// que no tiene fichero.
    ///
    /// Definido como el parseo del fichero **vacio**, y no a mano, para que «sin
    /// reglas» signifique aqui exactamente lo que significa para el parser.
    pub fn allow_all() -> Self {
        CrawlPolicy::parse_robots_txt("")
    }

    /// Un robots.txt que lo prohibe todo, para cuando NO se ha podido leer el de
    /// verdad: el 5xx y la falta de respuesta del RFC 9309 seccion 2.3.1.4.
    ///
    /// Tambien definido pasando el texto por el parser ---`User-agent: *` mas
    /// `Disallow: /`---, para que no haya una segunda idea de lo que ese par de
    /// lineas significa. El campo `raw` queda con ese texto, que es sintetico:
    /// ningun dominio lo mando.
    pub fn disallow_all() -> Self {
        CrawlPolicy::parse_robots_txt("User-agent: *\nDisallow: /\n")
    }

    /// Check whether the given path is allowed for the specified user-agent.
    ///
    /// Uses the exact user-agent match first, then falls back to the wildcard `*` section.
    /// If no matching directive exists, access is allowed by default.
    pub fn is_allowed(&self, user_agent: &str, path: &str) -> bool {
        if let Some(directives) = self.get_directives(user_agent) {
            // Find the most specific matching rule (longest path pattern wins)
            let mut best_match: Option<&RobotsRule> = None;
            let mut best_len = 0;

            for rule in &directives.rules {
                if Self::path_matches_pattern(path, &rule.path) {
                    let pattern_len = rule.path.len();
                    if pattern_len >= best_len {
                        best_len = pattern_len;
                        best_match = Some(rule);
                    }
                }
            }

            match best_match {
                Some(rule) => rule.allowed,
                None => true, // No matching rule means allowed
            }
        } else {
            true // No directives for this user-agent means allowed
        }
    }

    /// Get the crawl delay for the specified user-agent.
    pub fn crawl_delay(&self, user_agent: &str) -> Option<f64> {
        self.get_directives(user_agent).and_then(|d| d.crawl_delay)
    }

    /// Collect all sitemaps from global declarations and all user-agent sections.
    pub fn all_sitemaps(&self) -> Vec<&str> {
        let mut result: Vec<&str> = self.sitemaps.iter().map(|s| s.as_str()).collect();
        for directives in self.directives.values() {
            for sitemap in &directives.sitemaps {
                let s = sitemap.as_str();
                if !result.contains(&s) {
                    result.push(s);
                }
            }
        }
        result
    }

    /// Get the directives for a user-agent, falling back to `*` wildcard.
    fn get_directives(&self, user_agent: &str) -> Option<&RobotsDirectives> {
        let ua_lower = user_agent.to_lowercase();
        self.directives
            .get(&ua_lower)
            .or_else(|| self.directives.get("*"))
    }

    /// Check if a path matches a robots.txt pattern.
    ///
    /// Supports `*` as a wildcard matching any sequence of characters,
    /// and `$` at the end as an end-of-string anchor.
    fn path_matches_pattern(path: &str, pattern: &str) -> bool {
        if pattern.is_empty() {
            return true;
        }

        let anchored_end = pattern.ends_with('$');
        let effective_pattern = if anchored_end {
            &pattern[..pattern.len() - 1]
        } else {
            pattern
        };

        // Build a regex from the pattern
        let mut regex_str = String::from("^");
        for ch in effective_pattern.chars() {
            match ch {
                '*' => regex_str.push_str(".*"),
                '.' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '|' | '\\' => {
                    regex_str.push('\\');
                    regex_str.push(ch);
                }
                _ => regex_str.push(ch),
            }
        }
        if anchored_end {
            regex_str.push('$');
        }

        match Regex::new(&regex_str) {
            Ok(re) => re.is_match(path),
            Err(_) => {
                // Fallback: simple prefix match
                path.starts_with(effective_pattern)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sitemap types
// ─────────────────────────────────────────────────────────────────────────────

/// Change frequency hint for a sitemap entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum ChangeFrequency {
    Always,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
    Never,
}

impl ChangeFrequency {
    /// Parse a changefreq string from a sitemap XML.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "always" => Some(Self::Always),
            "hourly" => Some(Self::Hourly),
            "daily" => Some(Self::Daily),
            "weekly" => Some(Self::Weekly),
            "monthly" => Some(Self::Monthly),
            "yearly" => Some(Self::Yearly),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

/// A single entry in a sitemap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitemapEntry {
    /// The URL of the page.
    pub url: String,
    /// Last modification date (ISO 8601 format).
    pub last_modified: Option<String>,
    /// How frequently the page is likely to change.
    pub change_frequency: Option<ChangeFrequency>,
    /// Priority of this URL relative to other URLs on the site (0.0 to 1.0).
    pub priority: Option<f64>,
}

/// A parsed sitemap file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSitemap {
    /// URL entries found in this sitemap.
    pub entries: Vec<SitemapEntry>,
    /// References to sub-sitemaps (for sitemap index files).
    pub sub_sitemaps: Vec<String>,
    /// Whether this sitemap is an index file pointing to other sitemaps.
    pub is_index: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for crawl policy behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CrawlPolicyConfig {
    /// User-agent string to identify this crawler.
    pub user_agent: String,
    /// Default delay between requests to the same domain (milliseconds).
    pub default_delay_ms: u64,
    /// Maximum delay to respect from robots.txt Crawl-delay (milliseconds).
    pub max_delay_ms: u64,
    /// Whether to respect robots.txt directives.
    pub respect_robots_txt: bool,
    /// Whether to cache parsed robots.txt files.
    pub cache_robots_txt: bool,
    /// Time-to-live for cached robots.txt entries (seconds).
    pub robots_cache_ttl_secs: u64,
    /// HTTP request timeout (milliseconds).
    pub timeout_ms: u64,
}

with_setters!(CrawlPolicyConfig {
    with_user_agent -> user_agent: String,
    with_default_delay_ms -> default_delay_ms: u64,
    with_max_delay_ms -> max_delay_ms: u64,
    with_respect_robots_txt -> respect_robots_txt: bool,
    with_cache_robots_txt -> cache_robots_txt: bool,
    with_robots_cache_ttl_secs -> robots_cache_ttl_secs: u64,
    with_timeout_ms -> timeout_ms: u64,
});

impl Default for CrawlPolicyConfig {
    fn default() -> Self {
        Self {
            user_agent: "ai-assistant-crawler/0.1".to_string(),
            default_delay_ms: 1000,
            max_delay_ms: 60_000,
            respect_robots_txt: true,
            cache_robots_txt: true,
            robots_cache_ttl_secs: 86_400, // 24 hours
            timeout_ms: 10_000,
        }
    }
}

impl CrawlPolicyConfig {
    /// Get the default delay as a `Duration`.
    pub fn default_delay(&self) -> Duration {
        Duration::from_millis(self.default_delay_ms)
    }

    /// Get the maximum delay as a `Duration`.
    pub fn max_delay(&self) -> Duration {
        Duration::from_millis(self.max_delay_ms)
    }

    /// Get the robots.txt cache TTL as a `Duration`.
    pub fn robots_cache_ttl(&self) -> Duration {
        Duration::from_secs(self.robots_cache_ttl_secs)
    }

    /// Get the HTTP timeout as a `Duration`.
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rate limiting state
// ─────────────────────────────────────────────────────────────────────────────

/// Internal rate-limiting state for a single domain.
#[derive(Debug, Clone)]
struct DomainRateState {
    /// When the last request was made to this domain.
    last_request: Instant,
    /// The delay to enforce between requests.
    delay: Duration,
    /// Total number of requests made to this domain.
    request_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// CrawlPolicy engine
// ─────────────────────────────────────────────────────────────────────────────

/// The main crawl policy engine.
///
/// Manages robots.txt parsing and caching, per-domain rate limiting,
/// and sitemap discovery for respectful web crawling.
pub struct CrawlPolicy {
    /// Configuration for crawl behavior.
    pub config: CrawlPolicyConfig,
    /// Cache of parsed robots.txt files keyed by domain, with fetch timestamp.
    robots_cache: HashMap<String, (ParsedRobotsTxt, Instant)>,
    /// Per-domain rate limiting state.
    rate_state: HashMap<String, DomainRateState>,
    /// El transporte.
    ///
    /// Inyectable porque las dos URL de este modulo se **construyen** aqui
    /// dentro ---`https://{dominio}/robots.txt`---, asi que no hay donde apuntar
    /// un servidor de pruebas y lo unico sustituible es esto.
    client: Arc<dyn HttpClient>,
}

impl CrawlPolicy {
    /// Create a new crawl policy engine with the given configuration.
    pub fn new(config: CrawlPolicyConfig) -> Self {
        Self {
            config,
            robots_cache: HashMap::new(),
            rate_state: HashMap::new(),
            client: Arc::new(UreqClient),
        }
    }

    /// El mismo motor con otro transporte.
    ///
    /// Es lo que hace probables los dos caminos de red de este modulo. Antes de
    /// V473 no habia **ni un** test de ninguno de los dos, y por eso nada cazaba
    /// que un dominio caido se leyera como permiso para rastrearlo entero.
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Static parsing methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Parse a robots.txt file content into a structured representation.
    pub fn parse_robots_txt(content: &str) -> ParsedRobotsTxt {
        let mut directives: HashMap<String, RobotsDirectives> = HashMap::new();
        let mut global_sitemaps: Vec<String> = Vec::new();
        let mut current_agents: Vec<String> = Vec::new();
        let mut in_agent_section = false;

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                if line.is_empty() && in_agent_section {
                    // Empty line can reset the current section context for
                    // subsequent User-agent declarations
                    current_agents.clear();
                    in_agent_section = false;
                }
                continue;
            }

            // Parse directive: value
            let (directive, value) = match line.split_once(':') {
                Some((d, v)) => (d.trim().to_lowercase(), v.trim().to_string()),
                None => continue,
            };

            // Strip inline comments from value
            let value = value.split('#').next().unwrap_or("").trim().to_string();

            match directive.as_str() {
                "user-agent" => {
                    let agent = value.to_lowercase();
                    if !in_agent_section {
                        current_agents.clear();
                    }
                    current_agents.push(agent.clone());
                    in_agent_section = true;

                    // Ensure entry exists
                    directives
                        .entry(agent.clone())
                        .or_insert_with(|| RobotsDirectives {
                            user_agent: agent,
                            rules: Vec::new(),
                            crawl_delay: None,
                            sitemaps: Vec::new(),
                        });
                }
                "allow" => {
                    in_agent_section = true;
                    if current_agents.is_empty() {
                        current_agents.push("*".to_string());
                        directives
                            .entry("*".to_string())
                            .or_insert_with(|| RobotsDirectives {
                                user_agent: "*".to_string(),
                                rules: Vec::new(),
                                crawl_delay: None,
                                sitemaps: Vec::new(),
                            });
                    }
                    let rule = RobotsRule {
                        path: value,
                        allowed: true,
                    };
                    for agent in &current_agents {
                        if let Some(d) = directives.get_mut(agent) {
                            d.rules.push(rule.clone());
                        }
                    }
                }
                "disallow" => {
                    in_agent_section = true;
                    if current_agents.is_empty() {
                        current_agents.push("*".to_string());
                        directives
                            .entry("*".to_string())
                            .or_insert_with(|| RobotsDirectives {
                                user_agent: "*".to_string(),
                                rules: Vec::new(),
                                crawl_delay: None,
                                sitemaps: Vec::new(),
                            });
                    }
                    let rule = RobotsRule {
                        path: value,
                        allowed: false,
                    };
                    for agent in &current_agents {
                        if let Some(d) = directives.get_mut(agent) {
                            d.rules.push(rule.clone());
                        }
                    }
                }
                "crawl-delay" => {
                    if let Ok(delay) = value.parse::<f64>() {
                        for agent in &current_agents {
                            if let Some(d) = directives.get_mut(agent) {
                                d.crawl_delay = Some(delay);
                            }
                        }
                    }
                }
                "sitemap" => {
                    if current_agents.is_empty() || !in_agent_section {
                        global_sitemaps.push(value);
                    } else {
                        for agent in &current_agents {
                            if let Some(d) = directives.get_mut(agent) {
                                d.sitemaps.push(value.clone());
                            }
                        }
                    }
                }
                _ => {} // Ignore unknown directives
            }
        }

        ParsedRobotsTxt {
            directives,
            sitemaps: global_sitemaps,
            raw: content.to_string(),
        }
    }

    /// Parse a sitemap XML document into a structured representation.
    ///
    /// Handles both regular sitemaps (`<urlset>`) and sitemap index files (`<sitemapindex>`).
    pub fn parse_sitemap(content: &str) -> Result<ParsedSitemap> {
        let is_index = content.contains("<sitemapindex");
        let mut entries: Vec<SitemapEntry> = Vec::new();
        let mut sub_sitemaps: Vec<String> = Vec::new();

        if is_index {
            // Parse sitemap index: extract <sitemap><loc>...</loc></sitemap>
            let sitemap_re = Regex::new(r"(?s)<sitemap>(.*?)</sitemap>")
                .map_err(|e| anyhow!("Failed to compile sitemap regex: {}", e))?;
            let loc_re = Regex::new(r"<loc>\s*(.*?)\s*</loc>")
                .map_err(|e| anyhow!("Failed to compile loc regex: {}", e))?;

            for cap in sitemap_re.captures_iter(content) {
                let block = &cap[1];
                if let Some(loc_cap) = loc_re.captures(block) {
                    sub_sitemaps.push(loc_cap[1].to_string());
                }
            }
        } else {
            // Parse regular sitemap: extract <url> entries
            let url_re = Regex::new(r"(?s)<url>(.*?)</url>")
                .map_err(|e| anyhow!("Failed to compile url regex: {}", e))?;
            let loc_re = Regex::new(r"<loc>\s*(.*?)\s*</loc>")
                .map_err(|e| anyhow!("Failed to compile loc regex: {}", e))?;
            let lastmod_re = Regex::new(r"<lastmod>\s*(.*?)\s*</lastmod>")
                .map_err(|e| anyhow!("Failed to compile lastmod regex: {}", e))?;
            let changefreq_re = Regex::new(r"<changefreq>\s*(.*?)\s*</changefreq>")
                .map_err(|e| anyhow!("Failed to compile changefreq regex: {}", e))?;
            let priority_re = Regex::new(r"<priority>\s*(.*?)\s*</priority>")
                .map_err(|e| anyhow!("Failed to compile priority regex: {}", e))?;

            for cap in url_re.captures_iter(content) {
                let block = &cap[1];

                let url = match loc_re.captures(block) {
                    Some(loc_cap) => loc_cap[1].to_string(),
                    None => continue, // Skip entries without a <loc>
                };

                let last_modified = lastmod_re.captures(block).map(|c| c[1].to_string());

                let change_frequency = changefreq_re
                    .captures(block)
                    .and_then(|c| ChangeFrequency::from_str_opt(&c[1]));

                let priority = priority_re
                    .captures(block)
                    .and_then(|c| c[1].parse::<f64>().ok());

                entries.push(SitemapEntry {
                    url,
                    last_modified,
                    change_frequency,
                    priority,
                });
            }
        }

        Ok(ParsedSitemap {
            entries,
            sub_sitemaps,
            is_index,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // URL permission checking
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if a URL is allowed to be crawled according to robots.txt.
    ///
    /// Fetches and caches the robots.txt for the domain if not already cached.
    /// Returns `true` if `respect_robots_txt` is disabled. Lo que se decide
    /// cuando el fichero **no** se puede leer esta en
    /// [`CrawlPolicy::fetch_robots_txt`], con el RFC al lado.
    pub fn is_url_allowed(&mut self, url: &str) -> Result<bool> {
        if !self.config.respect_robots_txt {
            return Ok(true);
        }

        let domain = Self::extract_domain(url)
            .ok_or_else(|| anyhow!("Could not extract domain from URL: {}", url))?;
        let path = Self::extract_path(url);

        // La copia del agente es para poder preguntar por el mientras
        // `fetch_robots_txt` tiene prestado el motor. Aqui habia una rama `else`
        // que concedia permiso «si no se pudo traer el robots.txt» ---imposible,
        // porque la funcion acaba de insertar la entrada--- y lo que de verdad
        // hacia era dejar escrita una politica que ya no es la de abajo.
        let user_agent = self.config.user_agent.clone();
        let parsed = self.fetch_robots_txt(&domain)?;
        Ok(parsed.is_allowed(&user_agent, &path))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Rate limiting
    // ─────────────────────────────────────────────────────────────────────────

    /// Block the current thread until the rate limit for the given domain allows
    /// the next request.
    ///
    /// Updates the rate state after sleeping.
    pub fn wait_for_rate_limit(&mut self, domain: &str) {
        let delay = self.get_delay(domain);

        if let Some(state) = self.rate_state.get(domain) {
            let elapsed = state.last_request.elapsed();
            if elapsed < delay {
                let sleep_time = delay - elapsed;
                std::thread::sleep(sleep_time);
            }
        }

        // Update state after waiting
        let state = self
            .rate_state
            .entry(domain.to_string())
            .or_insert_with(|| DomainRateState {
                last_request: Instant::now(),
                delay,
                request_count: 0,
            });
        state.last_request = Instant::now();
        state.delay = delay;
        state.request_count += 1;
    }

    /// Get the appropriate delay for a domain.
    ///
    /// Checks robots.txt crawl-delay first, then falls back to the configured default.
    /// Clamps to the configured maximum delay.
    pub fn get_delay(&self, domain: &str) -> Duration {
        let robots_delay = self
            .robots_cache
            .get(domain)
            .and_then(|(parsed, _)| parsed.crawl_delay(&self.config.user_agent))
            .map(|secs| Duration::from_secs_f64(secs));

        let delay = robots_delay.unwrap_or_else(|| self.config.default_delay());
        std::cmp::min(delay, self.config.max_delay())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Robots.txt fetching
    // ─────────────────────────────────────────────────────────────────────────

    /// Fetch and cache the robots.txt for a domain.
    ///
    /// Returns a reference to the cached parsed robots.txt. If the cache is still
    /// valid (within TTL), returns the cached version without making a new request.
    ///
    /// # Lo que se decide cuando el fichero no se puede leer
    ///
    /// Los tres caminos de fallo acababan en «todo permitido», y para uno de
    /// ellos eso es lo contrario de lo que manda el RFC 9309:
    ///
    /// | lo que contesto el dominio | RFC 9309 | aqui |
    /// |---|---|---|
    /// | 2xx | 2.3.1.1: parsearlo | se parsea |
    /// | 4xx, o un 3xx que el transporte no resolvio | 2.3.1.3: «the crawler MAY access any resources» | todo permitido |
    /// | 5xx | 2.3.1.4: «MUST assume complete disallow» | todo prohibido |
    /// | nada: conexion, DNS, plazo agotado | 2.3.1.4 cubre lo «unreachable» | todo prohibido |
    /// | los dos ultimos **habiendo copia** | 2.4: la copia vale incluso pasado el dia | se mantiene la copia |
    ///
    /// El 5xx es el que importa: un dominio caido no es un dominio que invite a
    /// rastrearlo entero, y el camino anterior ---`Err(_)` a un `robots.txt`
    /// vacio--- convertia su caida en permiso.
    ///
    /// La copia que se mantiene se vuelve a fechar hoy, o sea que no se pregunta
    /// otra vez hasta que pase el TTL. Es lo que la seccion 2.4 permite, y evita
    /// insistirle a un dominio que ya esta caido.
    pub fn fetch_robots_txt(&mut self, domain: &str) -> Result<&ParsedRobotsTxt> {
        let now = Instant::now();

        // Check if we have a valid cached version
        if self.config.cache_robots_txt {
            if let Some((_, fetched_at)) = self.robots_cache.get(domain) {
                if now.duration_since(*fetched_at) < self.config.robots_cache_ttl() {
                    // Cache is still valid - return reference
                    return Ok(&self
                        .robots_cache
                        .get(domain)
                        .expect("domain just verified")
                        .0);
                }
            }
        }

        // Fetch robots.txt from the domain
        let request = HttpRequest::get(format!("https://{}/robots.txt", domain))
            .timeout(self.config.timeout())
            .header("User-Agent", self.config.user_agent.as_str());

        let fetched = match self.client.send(&request) {
            // 2.3.1.1. El cuerpo se lee **con perdida** porque el RFC permite
            // saltarse los octetos que no sean UTF-8 valido: un fichero con un
            // byte suelto no es un dominio sin reglas, que es en lo que lo
            // convertia el `read_to_string().unwrap_or_default()` de antes.
            Ok(response) if response.is_success() => Some(Self::parse_robots_txt(
                &String::from_utf8_lossy(&response.body),
            )),
            // 2.3.1.3. Detras de este puerto un no-2xx llega como `Ok`, asi que
            // el estado hay que mirarlo: con el `.call()` de `ureq` un 404 venia
            // como `Err` y caia en el mismo saco que no haber servidor.
            Ok(response) if (300..500).contains(&response.status) => {
                Some(ParsedRobotsTxt::allow_all())
            }
            // 2.3.1.4: 5xx, o no hubo respuesta.
            Ok(_) | Err(_) => None,
        };

        let parsed = match (fetched, self.robots_cache.remove(domain)) {
            (Some(parsed), _) => parsed,
            // 2.4: una copia vieja es mejor dato que una prohibicion general.
            (None, Some((cached, _))) => cached,
            (None, None) => ParsedRobotsTxt::disallow_all(),
        };

        self.robots_cache
            .insert(domain.to_string(), (parsed, Instant::now()));

        Ok(&self
            .robots_cache
            .get(domain)
            .expect("domain just inserted")
            .0)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Sitemap discovery
    // ─────────────────────────────────────────────────────────────────────────

    /// Discover sitemaps for a domain by checking its robots.txt.
    ///
    /// If no sitemaps are declared in robots.txt, tries the standard
    /// `/sitemap.xml` location.
    pub fn discover_sitemaps(&mut self, domain: &str) -> Result<Vec<String>> {
        self.fetch_robots_txt(domain)?;

        let mut sitemaps: Vec<String> = Vec::new();

        if let Some((parsed, _)) = self.robots_cache.get(domain) {
            let declared = parsed.all_sitemaps();
            for s in declared {
                if !sitemaps.contains(&s.to_string()) {
                    sitemaps.push(s.to_string());
                }
            }
        }

        // If no sitemaps found, try the standard location
        if sitemaps.is_empty() {
            let default_sitemap = format!("https://{}/sitemap.xml", domain);
            sitemaps.push(default_sitemap);
        }

        Ok(sitemaps)
    }

    /// Fetch and parse a sitemap from a URL.
    ///
    /// Un no-2xx es un error, y el mensaje lleva el **cuerpo** dentro: es donde
    /// el servidor explica por que. Con `.call()` el no-2xx ya era un error, pero
    /// de `ureq` 3 solo se podia sacar el numero.
    pub fn fetch_sitemap(&self, url: &str) -> Result<ParsedSitemap> {
        let request = HttpRequest::get(url)
            .timeout(self.config.timeout())
            .header("User-Agent", self.config.user_agent.as_str());

        let response = self
            .client
            .send(&request)
            .map_err(|e| anyhow!("Failed to fetch sitemap {}: {}", url, e))?;
        response
            .error_for_status()
            .map_err(|e| anyhow!("Failed to fetch sitemap {}: {}", url, e))?;

        let content = response
            .text()
            .map_err(|e| anyhow!("Failed to read sitemap response: {}", e))?;

        Self::parse_sitemap(&content)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Extract the domain (host) from a URL.
    fn extract_domain(url: &str) -> Option<String> {
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url);

        let domain = without_scheme.split('/').next()?;
        let domain = domain.split(':').next()?; // Strip port

        if domain.is_empty() {
            None
        } else {
            Some(domain.to_lowercase())
        }
    }

    /// Extract the path component from a URL.
    fn extract_path(url: &str) -> String {
        let without_scheme = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url);

        match without_scheme.find('/') {
            Some(idx) => without_scheme[idx..].to_string(),
            None => "/".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_robots_txt_basic() {
        let content = r#"
User-agent: *
Disallow: /private/
Disallow: /admin/
Allow: /admin/public/
Crawl-delay: 2

User-agent: googlebot
Disallow: /no-google/
Allow: /

Sitemap: https://example.com/sitemap.xml
"#;
        let parsed = CrawlPolicy::parse_robots_txt(content);

        // Wildcard agent should exist
        assert!(parsed.directives.contains_key("*"));
        let wildcard = parsed.directives.get("*").unwrap();
        assert_eq!(wildcard.rules.len(), 3);
        assert_eq!(wildcard.crawl_delay, Some(2.0));

        // Googlebot directives
        assert!(parsed.directives.contains_key("googlebot"));
        let googlebot = parsed.directives.get("googlebot").unwrap();
        assert_eq!(googlebot.rules.len(), 2);

        // Global sitemap
        assert_eq!(parsed.sitemaps.len(), 1);
        assert_eq!(parsed.sitemaps[0], "https://example.com/sitemap.xml");
    }

    #[test]
    fn test_is_allowed_rules() {
        let content = r#"
User-agent: *
Disallow: /private/
Disallow: /secret
Allow: /private/public/

User-agent: mybot
Disallow: /blocked/
"#;
        let parsed = CrawlPolicy::parse_robots_txt(content);

        // Wildcard user-agent checks
        assert!(!parsed.is_allowed("randombot", "/private/page.html"));
        assert!(parsed.is_allowed("randombot", "/public/page.html"));
        assert!(parsed.is_allowed("randombot", "/private/public/file.txt"));
        assert!(!parsed.is_allowed("randombot", "/secret"));

        // Specific user-agent checks
        assert!(!parsed.is_allowed("mybot", "/blocked/page.html"));
        assert!(parsed.is_allowed("mybot", "/public/page.html"));
    }

    #[test]
    fn test_path_matching_wildcards() {
        // Wildcard pattern matching
        assert!(ParsedRobotsTxt::path_matches_pattern(
            "/api/v1/users",
            "/api/*/users"
        ));
        assert!(ParsedRobotsTxt::path_matches_pattern(
            "/images/photo.jpg",
            "/*.jpg"
        ));
        assert!(!ParsedRobotsTxt::path_matches_pattern(
            "/images/photo.png",
            "/*.jpg"
        ));

        // End-of-string anchor
        assert!(ParsedRobotsTxt::path_matches_pattern(
            "/page.html",
            "*.html$"
        ));
        assert!(!ParsedRobotsTxt::path_matches_pattern(
            "/page.html?query=1",
            "*.html$"
        ));

        // Simple prefix matching
        assert!(ParsedRobotsTxt::path_matches_pattern(
            "/admin/page",
            "/admin/"
        ));
        assert!(!ParsedRobotsTxt::path_matches_pattern(
            "/public/page",
            "/admin/"
        ));
    }

    #[test]
    fn test_parse_sitemap_regular() {
        let content = r#"<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/page1</loc>
    <lastmod>2024-01-15</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://example.com/page2</loc>
    <changefreq>daily</changefreq>
    <priority>0.5</priority>
  </url>
  <url>
    <loc>https://example.com/page3</loc>
  </url>
</urlset>"#;

        let parsed = CrawlPolicy::parse_sitemap(content).unwrap();
        assert!(!parsed.is_index);
        assert_eq!(parsed.entries.len(), 3);
        assert!(parsed.sub_sitemaps.is_empty());

        let entry1 = &parsed.entries[0];
        assert_eq!(entry1.url, "https://example.com/page1");
        assert_eq!(entry1.last_modified.as_deref(), Some("2024-01-15"));
        assert_eq!(entry1.change_frequency, Some(ChangeFrequency::Weekly));
        assert_eq!(entry1.priority, Some(0.8));

        let entry2 = &parsed.entries[1];
        assert_eq!(entry2.url, "https://example.com/page2");
        assert!(entry2.last_modified.is_none());
        assert_eq!(entry2.change_frequency, Some(ChangeFrequency::Daily));

        let entry3 = &parsed.entries[2];
        assert_eq!(entry3.url, "https://example.com/page3");
        assert!(entry3.last_modified.is_none());
        assert!(entry3.change_frequency.is_none());
        assert!(entry3.priority.is_none());
    }

    #[test]
    fn test_parse_sitemap_index() {
        let content = r#"<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap-posts.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-pages.xml</loc>
  </sitemap>
</sitemapindex>"#;

        let parsed = CrawlPolicy::parse_sitemap(content).unwrap();
        assert!(parsed.is_index);
        assert!(parsed.entries.is_empty());
        assert_eq!(parsed.sub_sitemaps.len(), 2);
        assert_eq!(
            parsed.sub_sitemaps[0],
            "https://example.com/sitemap-posts.xml"
        );
        assert_eq!(
            parsed.sub_sitemaps[1],
            "https://example.com/sitemap-pages.xml"
        );
    }

    #[test]
    fn test_config_defaults_and_duration_helpers() {
        let config = CrawlPolicyConfig::default();

        assert_eq!(config.user_agent, "ai-assistant-crawler/0.1");
        assert_eq!(config.default_delay(), Duration::from_millis(1000));
        assert_eq!(config.max_delay(), Duration::from_millis(60_000));
        assert_eq!(config.robots_cache_ttl(), Duration::from_secs(86_400));
        assert_eq!(config.timeout(), Duration::from_millis(10_000));
        assert!(config.respect_robots_txt);
        assert!(config.cache_robots_txt);
    }

    #[test]
    fn test_extract_domain_and_path() {
        assert_eq!(
            CrawlPolicy::extract_domain("https://example.com/page"),
            Some("example.com".to_string())
        );
        assert_eq!(
            CrawlPolicy::extract_domain("http://sub.domain.org:8080/path"),
            Some("sub.domain.org".to_string())
        );
        assert_eq!(
            CrawlPolicy::extract_domain("https://UPPER.COM/test"),
            Some("upper.com".to_string())
        );
        assert_eq!(CrawlPolicy::extract_domain(""), None);

        assert_eq!(
            CrawlPolicy::extract_path("https://example.com/path/to/page?q=1"),
            "/path/to/page?q=1"
        );
        assert_eq!(CrawlPolicy::extract_path("https://example.com"), "/");
    }

    #[test]
    fn test_crawl_delay_lookup() {
        let content = r#"
User-agent: *
Crawl-delay: 5

User-agent: fastbot
Crawl-delay: 0.5
"#;
        let parsed = CrawlPolicy::parse_robots_txt(content);

        assert_eq!(parsed.crawl_delay("randombot"), Some(5.0));
        assert_eq!(parsed.crawl_delay("fastbot"), Some(0.5));
        assert_eq!(parsed.crawl_delay("*"), Some(5.0));
    }

    #[test]
    fn test_all_sitemaps_deduplication() {
        let content = r#"
User-agent: *
Disallow: /private/

Sitemap: https://example.com/sitemap.xml
Sitemap: https://example.com/sitemap2.xml
"#;
        let parsed = CrawlPolicy::parse_robots_txt(content);
        let sitemaps = parsed.all_sitemaps();

        assert!(sitemaps.contains(&"https://example.com/sitemap.xml"));
        assert!(sitemaps.contains(&"https://example.com/sitemap2.xml"));
    }

    #[test]
    fn test_rate_state_tracking() {
        let config = CrawlPolicyConfig {
            default_delay_ms: 50, // Small delay for testing
            ..CrawlPolicyConfig::default()
        };
        let mut policy = CrawlPolicy::new(config);

        // First call should not block significantly
        let start = Instant::now();
        policy.wait_for_rate_limit("test.com");
        let first_elapsed = start.elapsed();

        // Second call should wait approximately 50ms
        let start2 = Instant::now();
        policy.wait_for_rate_limit("test.com");
        let second_elapsed = start2.elapsed();

        // First call should be nearly instant
        assert!(first_elapsed < Duration::from_millis(100));
        // Second call should have waited at least some time
        assert!(second_elapsed >= Duration::from_millis(30));

        // Check request count
        let state = policy.rate_state.get("test.com").unwrap();
        assert_eq!(state.request_count, 2);
    }

    #[test]
    fn test_get_delay_with_robots_override() {
        let config = CrawlPolicyConfig {
            default_delay_ms: 1000,
            max_delay_ms: 5000,
            ..CrawlPolicyConfig::default()
        };
        let mut policy = CrawlPolicy::new(config);

        // Without robots.txt cached, should use default
        assert_eq!(policy.get_delay("unknown.com"), Duration::from_millis(1000));

        // Cache a robots.txt with a crawl delay
        let robots_content = "User-agent: *\nCrawl-delay: 3\n";
        let parsed = CrawlPolicy::parse_robots_txt(robots_content);
        policy
            .robots_cache
            .insert("slow.com".to_string(), (parsed, Instant::now()));

        // Should use robots.txt delay
        assert_eq!(policy.get_delay("slow.com"), Duration::from_secs(3));

        // Test max delay clamping
        let robots_huge = "User-agent: *\nCrawl-delay: 120\n";
        let parsed_huge = CrawlPolicy::parse_robots_txt(robots_huge);
        policy
            .robots_cache
            .insert("veryslow.com".to_string(), (parsed_huge, Instant::now()));

        // Should be clamped to max_delay (5s)
        assert_eq!(
            policy.get_delay("veryslow.com"),
            Duration::from_millis(5000)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transporte
// ─────────────────────────────────────────────────────────────────────────────

/// Los dos caminos de red de este modulo, probados sin red.
///
/// Antes de V473 no habia **ni uno**: lo que se probaba aqui era el parseo de
/// `robots.txt` y de los sitemaps, que son funciones puras. Y la razon no era
/// descuido ---la URL se construye aqui dentro, asi que no habia donde apuntar
/// un servidor de pruebas--- sino exactamente el hueco que cierra inyectar el
/// cliente.
#[cfg(test)]
mod transport_tests {
    use super::*;
    use crate::http_client::{CannedHttpClient, FailingHttpClient};

    const REGLAS: &str = "User-agent: *\nDisallow: /private/\nAllow: /public/\nCrawl-delay: 2\nSitemap: https://example.com/sitemap.xml\n";

    fn motor(client: Arc<dyn HttpClient>) -> CrawlPolicy {
        CrawlPolicy::new(CrawlPolicyConfig::default()).with_http_client(client)
    }

    #[test]
    fn un_2xx_manda_sus_reglas() {
        let client = Arc::new(CannedHttpClient::new(200, REGLAS));
        let mut policy = motor(client.clone());

        assert!(policy
            .is_url_allowed("https://example.com/public/a")
            .unwrap());
        assert!(!policy
            .is_url_allowed("https://example.com/private/a")
            .unwrap());
        assert_eq!(client.calls(), 1, "la segunda pregunta sale de la cache");
    }

    #[test]
    fn un_5xx_lo_prohibe_todo() {
        let client = Arc::new(CannedHttpClient::new(503, "en mantenimiento"));
        let mut policy = motor(client);

        // RFC 9309 seccion 2.3.1.4: «MUST assume complete disallow». Esto
        // contestaba `true` a las dos, porque el error se convertia en un
        // `robots.txt` vacio y un fichero vacio no prohibe nada.
        assert!(!policy.is_url_allowed("https://example.com/").unwrap());
        assert!(!policy
            .is_url_allowed("https://example.com/cualquier/cosa")
            .unwrap());
    }

    #[test]
    fn un_4xx_lo_permite_todo() {
        let client = Arc::new(CannedHttpClient::new(404, "no hay fichero"));
        let mut policy = motor(client);

        // Seccion 2.3.1.3: «the crawler MAY access any resources on the server».
        assert!(policy
            .is_url_allowed("https://example.com/cualquier/cosa")
            .unwrap());
    }

    #[test]
    fn sin_respuesta_lo_prohibe_todo() {
        let mut policy = motor(Arc::new(FailingHttpClient));

        // No es lo mismo que un 4xx: aqui no se sabe si hay reglas o no, y la
        // seccion 2.3.1.4 cubre tambien lo «unreachable».
        assert!(!policy.is_url_allowed("https://example.com/").unwrap());
    }

    #[test]
    fn una_copia_en_cache_sobrevive_a_que_el_dominio_se_caiga() {
        // Contesta una vez y a la siguiente ya se ha caido; el TTL a cero fuerza
        // a preguntar de nuevo en vez de esperar un dia.
        let client = Arc::new(CannedHttpClient::sequence(vec![
            (200, REGLAS.as_bytes().to_vec()),
            (503, b"en mantenimiento".to_vec()),
        ]));
        let mut policy =
            CrawlPolicy::new(CrawlPolicyConfig::default().with_robots_cache_ttl_secs(0))
                .with_http_client(client.clone());

        assert!(!policy
            .is_url_allowed("https://example.com/private/a")
            .unwrap());
        // Seccion 2.4: la copia sigue valiendo, asi que `/public` sigue
        // permitido en vez de caer en la prohibicion general del 5xx.
        assert!(policy
            .is_url_allowed("https://example.com/public/a")
            .unwrap());
        // La premisa del caso, afirmada: con TTL cero se vuelve a preguntar, y lo
        // que contesto la segunda vez fue la caida. Sin esta linea el test pasaba
        // tambien con un doble que sirviera dos veces el 200, o sea sin probar
        // nada de la seccion 2.4.
        assert_eq!(client.served(), vec![200, 503]);
    }

    #[test]
    fn la_cache_evita_la_segunda_peticion() {
        let client = Arc::new(CannedHttpClient::new(200, REGLAS));
        let mut policy = motor(client.clone());

        for _ in 0..3 {
            policy.is_url_allowed("https://example.com/a").unwrap();
        }
        assert_eq!(client.calls(), 1);
    }

    #[test]
    fn con_respect_robots_txt_apagado_no_se_sale_al_cable() {
        let client = Arc::new(CannedHttpClient::new(503, ""));
        let mut policy =
            CrawlPolicy::new(CrawlPolicyConfig::default().with_respect_robots_txt(false))
                .with_http_client(client.clone());

        assert!(policy
            .is_url_allowed("https://example.com/private/a")
            .unwrap());
        assert_eq!(client.calls(), 0, "no hay nada que preguntar");
    }

    #[test]
    fn la_peticion_lleva_la_url_del_dominio_el_agente_y_el_plazo_exacto() {
        let client = Arc::new(CannedHttpClient::new(200, REGLAS));
        let mut policy = CrawlPolicy::new(
            CrawlPolicyConfig::default()
                .with_user_agent("mi-rastreador/1.0".to_string())
                .with_timeout_ms(2_500),
        )
        .with_http_client(client.clone());

        policy
            .is_url_allowed("https://example.com/a/b?q=1")
            .unwrap();

        let request = client.seen();
        assert_eq!(request.url, "https://example.com/robots.txt");
        assert_eq!(
            request.headers,
            vec![("User-Agent".to_string(), "mi-rastreador/1.0".to_string())]
        );
        // El plazo sale de `config.timeout()`, que ya existia y no usaba nadie:
        // los dos fetchers lo volvian a derivar dividiendo los milisegundos
        // entre mil, con lo que 2,5 s se quedaban en 2 y medio segundo subia a 1.
        assert_eq!(request.timeout, Duration::from_millis(2_500));
    }

    #[test]
    fn un_byte_que_no_es_utf8_no_borra_las_reglas() {
        // `read_to_string().unwrap_or_default()` devolvia el fichero VACIO ante
        // un solo octeto invalido, o sea «todo permitido» por un byte suelto.
        let mut body = REGLAS.as_bytes().to_vec();
        body.extend_from_slice(b"# comentario \xff\xfe\n");
        let client = Arc::new(CannedHttpClient::raw(200, body));
        let mut policy = motor(client);

        assert!(!policy
            .is_url_allowed("https://example.com/private/a")
            .unwrap());
    }

    #[test]
    fn el_retraso_sale_del_robots_y_el_prohibido_no_inventa_ninguno() {
        let client = Arc::new(CannedHttpClient::new(200, REGLAS));
        let mut policy = motor(client);
        policy.is_url_allowed("https://example.com/a").unwrap();
        assert_eq!(policy.get_delay("example.com"), Duration::from_secs(2));

        let caido = Arc::new(CannedHttpClient::new(503, ""));
        let mut policy = motor(caido);
        policy.is_url_allowed("https://example.com/a").unwrap();
        // El robots.txt sintetico que lo prohibe todo no trae `Crawl-delay`, asi
        // que el retraso es el configurado y no un numero inventado.
        assert_eq!(
            policy.get_delay("example.com"),
            CrawlPolicyConfig::default().default_delay()
        );
    }

    #[test]
    fn los_sitemaps_declarados_salen_del_robots() {
        let client = Arc::new(CannedHttpClient::new(200, REGLAS));
        let mut policy = motor(client);

        assert_eq!(
            policy.discover_sitemaps("example.com").unwrap(),
            vec!["https://example.com/sitemap.xml".to_string()]
        );
    }

    #[test]
    fn un_sitemap_no_2xx_es_un_error_con_el_cuerpo_dentro() {
        let client = Arc::new(CannedHttpClient::new(500, "<html>se cayo la base</html>"));
        let policy = motor(client);

        let error = policy
            .fetch_sitemap("https://example.com/sitemap.xml")
            .expect_err("un 500 no es un sitemap");
        let texto = error.to_string();
        assert!(texto.contains("500"), "{texto}");
        // El cuerpo dentro del mensaje es lo que convierte «dijo que no» en un
        // diagnostico; de `ureq` 3 solo salia el numero.
        assert!(texto.contains("se cayo la base"), "{texto}");
    }

    #[test]
    fn un_sitemap_2xx_se_parsea() {
        let xml = concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
            "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n",
            "  <url><loc>https://example.com/a</loc></url>\n",
            "  <url><loc>https://example.com/b</loc></url>\n",
            "</urlset>"
        );
        let client = Arc::new(CannedHttpClient::new(200, xml));
        let policy = motor(client.clone());

        let sitemap = policy
            .fetch_sitemap("https://example.com/sitemap.xml")
            .unwrap();
        assert_eq!(sitemap.entries.len(), 2);
        assert!(!sitemap.is_index);
        assert_eq!(client.seen().url, "https://example.com/sitemap.xml");
    }
}
