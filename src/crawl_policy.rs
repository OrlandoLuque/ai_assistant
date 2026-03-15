//! Crawl policy management
//!
//! Parses robots.txt files, checks URL permissions, implements per-domain
//! rate limiting, and discovers sitemaps. Provides a complete crawl policy
//! engine suitable for respectful web crawling.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

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
}

impl CrawlPolicy {
    /// Create a new crawl policy engine with the given configuration.
    pub fn new(config: CrawlPolicyConfig) -> Self {
        Self {
            config,
            robots_cache: HashMap::new(),
            rate_state: HashMap::new(),
        }
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
    /// Returns `true` if `respect_robots_txt` is disabled.
    pub fn is_url_allowed(&mut self, url: &str) -> Result<bool> {
        if !self.config.respect_robots_txt {
            return Ok(true);
        }

        let domain = Self::extract_domain(url)
            .ok_or_else(|| anyhow!("Could not extract domain from URL: {}", url))?;
        let path = Self::extract_path(url);

        // Ensure we have robots.txt cached for this domain
        self.fetch_robots_txt(&domain)?;

        if let Some((parsed, _)) = self.robots_cache.get(&domain) {
            Ok(parsed.is_allowed(&self.config.user_agent, &path))
        } else {
            // If we failed to fetch robots.txt, allow by default
            Ok(true)
        }
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
        let url = format!("https://{}/robots.txt", domain);
        let timeout = self.config.timeout_ms / 1000;
        let timeout = if timeout == 0 { 1 } else { timeout };

        let content = match ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout))
            .build()
            .get(&url)
            .set("User-Agent", &self.config.user_agent)
            .call()
        {
            Ok(response) => response.into_string().unwrap_or_default(),
            Err(_) => {
                // If we can't fetch robots.txt, use an empty one (allow all)
                String::new()
            }
        };

        let parsed = Self::parse_robots_txt(&content);
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
    pub fn fetch_sitemap(&self, url: &str) -> Result<ParsedSitemap> {
        let timeout = self.config.timeout_ms / 1000;
        let timeout = if timeout == 0 { 1 } else { timeout };

        let response = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout))
            .build()
            .get(url)
            .set("User-Agent", &self.config.user_agent)
            .call()
            .map_err(|e| anyhow!("Failed to fetch sitemap {}: {}", url, e))?;

        let content = response
            .into_string()
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
