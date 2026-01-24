//! Web search integration
//!
//! Search the web and integrate results into AI responses.

use std::collections::HashMap;
use std::time::Duration;

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
pub enum SafeSearch {
    Off,
    Moderate,
    Strict,
}

/// Time range filter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeRange {
    Day,
    Week,
    Month,
    Year,
    All,
}

/// Search provider trait
pub trait SearchProvider: Send + Sync {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError>;
    fn name(&self) -> &str;
}

/// DuckDuckGo search provider (HTML scraping - for educational purposes)
pub struct DuckDuckGoProvider {
    base_url: String,
}

impl DuckDuckGoProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://html.duckduckgo.com/html".to_string(),
        }
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

        let response = ureq::get(&url)
            .timeout(config.timeout)
            .set("User-Agent", "Mozilla/5.0 (compatible; AIAssistant/1.0)")
            .call()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        let html = response.into_string()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        // Parse results from HTML
        let mut results = Vec::new();
        let result_pattern = regex::Regex::new(r#"class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>"#).ok();
        let snippet_pattern = regex::Regex::new(r#"class="result__snippet"[^>]*>([^<]*)</a>"#).ok();

        if let Some(re) = result_pattern {
            for (i, cap) in re.captures_iter(&html).enumerate() {
                if i >= config.max_results {
                    break;
                }

                let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
                let title = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                // Try to find snippet
                let snippet = snippet_pattern.as_ref()
                    .and_then(|re| re.captures_iter(&html).nth(i))
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str())
                    .unwrap_or("");

                // Decode DuckDuckGo redirect URL
                let actual_url = if url.contains("uddg=") {
                    url.split("uddg=")
                        .nth(1)
                        .and_then(|s| urlencoding::decode(s).ok())
                        .map(|s| s.into_owned())
                        .unwrap_or_else(|| url.to_string())
                } else {
                    url.to_string()
                };

                results.push(SearchResult::new(
                    &html_decode(title),
                    &actual_url,
                    &html_decode(snippet),
                ));
            }
        }

        Ok(results)
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
}

impl BraveSearchProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.search.brave.com/res/v1/web/search".to_string(),
        }
    }
}

impl SearchProvider for BraveSearchProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!("{}?q={}&count={}",
            self.base_url,
            urlencoding::encode(query),
            config.max_results
        );

        let response = ureq::get(&url)
            .timeout(config.timeout)
            .set("X-Subscription-Token", &self.api_key)
            .set("Accept", "application/json")
            .call()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        let text = response.into_string()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        let mut results = Vec::new();

        if let Some(web) = json.get("web").and_then(|w| w.get("results")).and_then(|r| r.as_array()) {
            for item in web {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let snippet = item.get("description").and_then(|d| d.as_str()).unwrap_or("");

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
}

impl SearXNGProvider {
    pub fn new(instance_url: &str) -> Self {
        Self {
            instance_url: instance_url.trim_end_matches('/').to_string(),
        }
    }
}

impl SearchProvider for SearXNGProvider {
    fn search(&self, query: &str, config: &SearchConfig) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!("{}/search?q={}&format=json",
            self.instance_url,
            urlencoding::encode(query)
        );

        let response = ureq::get(&url)
            .timeout(config.timeout)
            .call()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        let text = response.into_string()
            .map_err(|e| SearchError::Network(e.to_string()))?;

        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| SearchError::Parse(e.to_string()))?;

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

/// Web search manager
pub struct WebSearchManager {
    providers: Vec<Box<dyn SearchProvider>>,
    config: SearchConfig,
    cache: HashMap<String, (Vec<SearchResult>, std::time::Instant)>,
    cache_ttl: Duration,
}

impl WebSearchManager {
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
                    self.cache.insert(cache_key, (results.clone(), std::time::Instant::now()));
                    return Ok(results);
                }
                _ => continue,
            }
        }

        Err(SearchError::NoResults)
    }

    /// Search and format results for AI context
    pub fn search_for_context(&mut self, query: &str, max_chars: usize) -> Result<String, SearchError> {
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

impl Default for WebSearchManager {
    fn default() -> Self {
        let mut manager = Self::new(SearchConfig::default());
        manager.add_provider(Box::new(DuckDuckGoProvider::new()));
        manager
    }
}

/// Search error
#[derive(Debug)]
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
}
