//! Generic HTTP API client with authentication, rate limiting, pagination, and caching.
//!
//! This module provides [`DataSourceClient`], a configurable REST API client that supports
//! multiple authentication methods, automatic pagination, request rate limiting, response
//! caching, and retry with exponential backoff.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Authentication
// ============================================================================

/// Supported authentication methods for API requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// No authentication.
    None,
    /// Bearer token in the Authorization header.
    Bearer(String),
    /// API key sent as a custom header.
    ApiKey { header: String, key: String },
    /// API key sent as a query parameter.
    ApiKeyQuery { param: String, key: String },
    /// HTTP Basic authentication.
    Basic { username: String, password: String },
}

impl Default for AuthMethod {
    fn default() -> Self {
        Self::None
    }
}

// ============================================================================
// Pagination
// ============================================================================

/// Strategy for paginating through API results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaginationStrategy {
    /// No pagination - single request returns all results.
    None,
    /// Offset-based pagination with configurable parameter names.
    Offset {
        offset_param: String,
        limit_param: String,
        page_size: usize,
    },
    /// Page-number-based pagination.
    PageNumber {
        page_param: String,
        size_param: String,
        page_size: usize,
    },
    /// Cursor-based pagination.
    Cursor {
        cursor_param: String,
        cursor_field: String,
    },
}

impl Default for PaginationStrategy {
    fn default() -> Self {
        Self::None
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Policy for rate limiting outgoing requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitPolicy {
    /// Maximum number of requests allowed in the window.
    pub max_requests: usize,
    /// Duration of the rate limit window in milliseconds.
    pub window_ms: u64,
    /// Minimum delay between consecutive requests in milliseconds.
    pub min_delay_ms: u64,
}

impl Default for RateLimitPolicy {
    fn default() -> Self {
        Self {
            max_requests: 60,
            window_ms: 60_000,
            min_delay_ms: 100,
        }
    }
}

impl RateLimitPolicy {
    /// Returns the window duration.
    pub fn window(&self) -> Duration {
        Duration::from_millis(self.window_ms)
    }

    /// Returns the minimum delay between requests.
    pub fn min_delay(&self) -> Duration {
        Duration::from_millis(self.min_delay_ms)
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a [`DataSourceClient`] instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Base URL of the API (e.g., "https://api.example.com/v1").
    pub base_url: String,
    /// Authentication method to use.
    pub auth: AuthMethod,
    /// Rate limiting policy.
    pub rate_limit: RateLimitPolicy,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Additional headers to include in every request.
    pub headers: HashMap<String, String>,
    /// User-Agent header value.
    pub user_agent: String,
    /// Pagination strategy for list endpoints.
    pub pagination: PaginationStrategy,
    /// Whether response caching is enabled.
    pub cache_enabled: bool,
    /// Cache time-to-live in milliseconds.
    pub cache_ttl_ms: u64,
    /// Maximum number of entries in the cache.
    pub max_cache_entries: usize,
    /// Maximum number of retries on failure.
    pub max_retries: usize,
    /// Base delay between retries in milliseconds (doubles each attempt).
    pub retry_delay_ms: u64,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            auth: AuthMethod::default(),
            rate_limit: RateLimitPolicy::default(),
            timeout_ms: 30_000,
            headers: HashMap::new(),
            user_agent: "DataSourceClient/1.0".to_string(),
            pagination: PaginationStrategy::default(),
            cache_enabled: true,
            cache_ttl_ms: 300_000, // 5 minutes
            max_cache_entries: 1000,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

impl DataSourceConfig {
    /// Returns the request timeout as a Duration.
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    /// Returns the cache TTL as a Duration.
    pub fn cache_ttl(&self) -> Duration {
        Duration::from_millis(self.cache_ttl_ms)
    }

    /// Returns the retry delay as a Duration.
    pub fn retry_delay(&self) -> Duration {
        Duration::from_millis(self.retry_delay_ms)
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from a single API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceResponse {
    /// HTTP status code.
    pub status: u16,
    /// Raw response body as a string.
    pub body: String,
    /// Parsed JSON body, if the response was valid JSON.
    pub json: Option<serde_json::Value>,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Whether this response was served from cache.
    pub from_cache: bool,
    /// Time taken for the request in milliseconds.
    pub duration_ms: u64,
}

/// Response from a paginated list request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse {
    /// All items collected across pages.
    pub items: Vec<serde_json::Value>,
    /// Number of pages fetched.
    pub pages_fetched: usize,
    /// Total item count if reported by the API.
    pub total_count: Option<usize>,
    /// Whether more pages are available beyond what was fetched.
    pub has_more: bool,
}

// ============================================================================
// Cache Entry (internal, not serializable due to Instant)
// ============================================================================

/// Internal cache entry storing a response along with its expiry metadata.
#[derive(Debug, Clone)]
struct CacheEntry {
    response: DataSourceResponse,
    cached_at: Instant,
    ttl: Duration,
}

impl CacheEntry {
    /// Returns true if this cache entry has expired.
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }
}

// ============================================================================
// DataSourceClient
// ============================================================================

/// A generic HTTP API client with authentication, rate limiting, pagination, and caching.
///
/// # Example
///
/// ```no_run
/// use ai_assistant::data_source_client::{DataSourceClient, AuthMethod};
///
/// let mut client = DataSourceClient::from_url("https://api.example.com")
///     .with_auth(AuthMethod::Bearer("my-token".into()));
///
/// let response = client.get("/users").unwrap();
/// println!("Status: {}", response.status);
/// ```
#[derive(Debug)]
pub struct DataSourceClient {
    config: DataSourceConfig,
    cache: HashMap<String, CacheEntry>,
    last_request_time: Option<Instant>,
    request_count_in_window: usize,
    window_start: Option<Instant>,
}

impl DataSourceClient {
    /// Creates a new client with the given configuration.
    pub fn new(config: DataSourceConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            last_request_time: None,
            request_count_in_window: 0,
            window_start: None,
        }
    }

    /// Creates a new client configured for the given base URL with default settings.
    pub fn from_url(base_url: &str) -> Self {
        let config = DataSourceConfig {
            base_url: base_url.trim_end_matches('/').to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Builder method: sets the authentication method.
    pub fn with_auth(mut self, auth: AuthMethod) -> Self {
        self.config.auth = auth;
        self
    }

    /// Builder method: sets the rate limit policy.
    pub fn with_rate_limit(mut self, policy: RateLimitPolicy) -> Self {
        self.config.rate_limit = policy;
        self
    }

    // ========================================================================
    // Public API Methods
    // ========================================================================

    /// Performs a GET request to the given path.
    pub fn get(&mut self, path: &str) -> Result<DataSourceResponse> {
        self.get_with_params(path, &[])
    }

    /// Performs a GET request with query parameters.
    pub fn get_with_params(
        &mut self,
        path: &str,
        params: &[(&str, &str)],
    ) -> Result<DataSourceResponse> {
        let url = self.build_url(path, params);
        let cache_key = url.clone();

        if self.config.cache_enabled {
            if let Some(cached) = self.check_cache(&cache_key) {
                return Ok(cached);
            }
        }

        let response = self.execute_with_retry(&url, "GET", None)?;

        if self.config.cache_enabled {
            self.store_cache(&cache_key, &response);
        }

        Ok(response)
    }

    /// Performs a POST request with a JSON body.
    pub fn post_json(
        &mut self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<DataSourceResponse> {
        let url = self.build_url(path, &[]);
        self.execute_with_retry(&url, "POST", Some(body))
    }

    /// Fetches a single entity by ID from the given path.
    ///
    /// Constructs the URL as `{path}/{id}`.
    pub fn get_entity(&mut self, path: &str, id: &str) -> Result<DataSourceResponse> {
        let full_path = format!("{}/{}", path.trim_end_matches('/'), id);
        self.get(&full_path)
    }

    /// Searches the given endpoint using a `q` query parameter.
    pub fn search(&mut self, path: &str, query: &str) -> Result<DataSourceResponse> {
        self.get_with_params(path, &[("q", query)])
    }

    /// Fetches all pages from a paginated endpoint.
    ///
    /// This will keep fetching until no more results are available.
    /// Use [`list_pages`](Self::list_pages) to limit the number of pages.
    pub fn list_all(&mut self, path: &str) -> Result<PaginatedResponse> {
        self.list_pages(path, usize::MAX)
    }

    /// Fetches up to `max_pages` pages from a paginated endpoint.
    pub fn list_pages(&mut self, path: &str, max_pages: usize) -> Result<PaginatedResponse> {
        match self.config.pagination.clone() {
            PaginationStrategy::None => {
                let response = self.get(path)?;
                let items = self.extract_items(&response);
                Ok(PaginatedResponse {
                    pages_fetched: 1,
                    total_count: Some(items.len()),
                    has_more: false,
                    items,
                })
            }
            PaginationStrategy::Offset {
                offset_param,
                limit_param,
                page_size,
            } => self.paginate_offset(path, &offset_param, &limit_param, page_size, max_pages),
            PaginationStrategy::PageNumber {
                page_param,
                size_param,
                page_size,
            } => self.paginate_page_number(path, &page_param, &size_param, page_size, max_pages),
            PaginationStrategy::Cursor {
                cursor_param,
                cursor_field,
            } => self.paginate_cursor(path, &cursor_param, &cursor_field, max_pages),
        }
    }

    /// Clears all cached responses.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Returns cache statistics: (total entries, expired entries).
    pub fn cache_stats(&self) -> (usize, usize) {
        let total = self.cache.len();
        let expired = self.cache.values().filter(|e| e.is_expired()).count();
        (total, expired)
    }

    // ========================================================================
    // Pagination Helpers
    // ========================================================================

    fn paginate_offset(
        &mut self,
        path: &str,
        offset_param: &str,
        limit_param: &str,
        page_size: usize,
        max_pages: usize,
    ) -> Result<PaginatedResponse> {
        let mut all_items = Vec::new();
        let mut offset: usize = 0;
        let mut pages_fetched: usize = 0;
        let mut has_more = false;

        for _ in 0..max_pages {
            let offset_str = offset.to_string();
            let size_str = page_size.to_string();
            let params: Vec<(&str, &str)> =
                vec![(offset_param, &offset_str), (limit_param, &size_str)];
            let response = self.get_with_params(path, &params)?;
            let items = self.extract_items(&response);
            let count = items.len();
            all_items.extend(items);
            pages_fetched += 1;

            if count < page_size {
                break;
            }
            offset += page_size;
            if pages_fetched >= max_pages {
                has_more = true;
                break;
            }
        }

        Ok(PaginatedResponse {
            total_count: None,
            has_more,
            pages_fetched,
            items: all_items,
        })
    }

    fn paginate_page_number(
        &mut self,
        path: &str,
        page_param: &str,
        size_param: &str,
        page_size: usize,
        max_pages: usize,
    ) -> Result<PaginatedResponse> {
        let mut all_items = Vec::new();
        let mut page: usize = 1;
        let mut pages_fetched: usize = 0;
        let mut has_more = false;

        for _ in 0..max_pages {
            let page_str = page.to_string();
            let size_str = page_size.to_string();
            let params: Vec<(&str, &str)> = vec![(page_param, &page_str), (size_param, &size_str)];
            let response = self.get_with_params(path, &params)?;
            let items = self.extract_items(&response);
            let count = items.len();
            all_items.extend(items);
            pages_fetched += 1;

            if count < page_size {
                break;
            }
            page += 1;
            if pages_fetched >= max_pages {
                has_more = true;
                break;
            }
        }

        Ok(PaginatedResponse {
            total_count: None,
            has_more,
            pages_fetched,
            items: all_items,
        })
    }

    fn paginate_cursor(
        &mut self,
        path: &str,
        cursor_param: &str,
        cursor_field: &str,
        max_pages: usize,
    ) -> Result<PaginatedResponse> {
        let mut all_items = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages_fetched: usize = 0;
        let mut has_more = false;

        for _ in 0..max_pages {
            let response = if let Some(ref c) = cursor {
                self.get_with_params(path, &[(cursor_param, c.as_str())])?
            } else {
                self.get(path)?
            };

            // Extract next cursor from the response JSON
            let next_cursor = response
                .json
                .as_ref()
                .and_then(|j| j.get(cursor_field))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let items = self.extract_items(&response);
            all_items.extend(items);
            pages_fetched += 1;

            match next_cursor {
                Some(c) if !c.is_empty() => {
                    cursor = Some(c);
                    if pages_fetched >= max_pages {
                        has_more = true;
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(PaginatedResponse {
            total_count: None,
            has_more,
            pages_fetched,
            items: all_items,
        })
    }

    /// Extracts items from a response body. Handles both array responses and
    /// object responses with common list field names.
    fn extract_items(&self, response: &DataSourceResponse) -> Vec<serde_json::Value> {
        match &response.json {
            Some(serde_json::Value::Array(arr)) => arr.clone(),
            Some(serde_json::Value::Object(obj)) => {
                // Try common field names for list results
                for key in &["data", "items", "results", "records", "entries", "list"] {
                    if let Some(serde_json::Value::Array(arr)) = obj.get(*key) {
                        return arr.clone();
                    }
                }
                // If the object itself doesn't contain an array field, wrap it
                vec![serde_json::Value::Object(obj.clone())]
            }
            Some(other) => vec![other.clone()],
            None => Vec::new(),
        }
    }

    // ========================================================================
    // Private: Rate Limiting
    // ========================================================================

    fn apply_rate_limit(&mut self) {
        let now = Instant::now();

        // Enforce minimum delay between requests
        if let Some(last) = self.last_request_time {
            let elapsed = now.duration_since(last);
            let min_delay = self.config.rate_limit.min_delay();
            if elapsed < min_delay {
                std::thread::sleep(min_delay - elapsed);
            }
        }

        // Enforce window-based rate limiting
        let window = self.config.rate_limit.window();
        match self.window_start {
            Some(start) if now.duration_since(start) < window => {
                if self.request_count_in_window >= self.config.rate_limit.max_requests {
                    // Window is still active and we've hit the limit - sleep until window resets
                    let remaining = window - now.duration_since(start);
                    std::thread::sleep(remaining);
                    self.window_start = Some(Instant::now());
                    self.request_count_in_window = 0;
                }
            }
            Some(start) if now.duration_since(start) >= window => {
                // Window has expired, reset
                self.window_start = Some(now);
                self.request_count_in_window = 0;
            }
            None => {
                self.window_start = Some(now);
                self.request_count_in_window = 0;
            }
            _ => {}
        }

        self.request_count_in_window += 1;
        self.last_request_time = Some(Instant::now());
    }

    // ========================================================================
    // Private: URL Building
    // ========================================================================

    fn build_url(&self, path: &str, params: &[(&str, &str)]) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        let path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{}", path)
        };

        let mut url = format!("{}{}", base, path);

        // Collect all query params (including auth if ApiKeyQuery)
        let mut all_params: Vec<(&str, &str)> = params.to_vec();
        let auth_param_storage: (String, String);
        if let AuthMethod::ApiKeyQuery { ref param, ref key } = self.config.auth {
            auth_param_storage = (param.clone(), key.clone());
            all_params.push((&auth_param_storage.0, &auth_param_storage.1));
        }

        if !all_params.is_empty() {
            url.push('?');
            let query: Vec<String> = all_params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect();
            url.push_str(&query.join("&"));
        }

        url
    }

    // ========================================================================
    // Private: Authentication
    // ========================================================================

    fn apply_auth(&self, request: ureq::Request) -> ureq::Request {
        match &self.config.auth {
            AuthMethod::None => request,
            AuthMethod::Bearer(token) => request.set("Authorization", &format!("Bearer {}", token)),
            AuthMethod::ApiKey { header, key } => request.set(header, key),
            AuthMethod::ApiKeyQuery { .. } => {
                // Already handled in build_url
                request
            }
            AuthMethod::Basic { username, password } => {
                let credentials = base64_encode(&format!("{}:{}", username, password));
                request.set("Authorization", &format!("Basic {}", credentials))
            }
        }
    }

    // ========================================================================
    // Private: Caching
    // ========================================================================

    fn check_cache(&self, cache_key: &str) -> Option<DataSourceResponse> {
        if let Some(entry) = self.cache.get(cache_key) {
            if !entry.is_expired() {
                let mut response = entry.response.clone();
                response.from_cache = true;
                return Some(response);
            }
        }
        None
    }

    fn store_cache(&mut self, cache_key: &str, response: &DataSourceResponse) {
        // Evict expired entries if we're at capacity
        if self.cache.len() >= self.config.max_cache_entries {
            self.evict_expired();
        }

        // If still at capacity after eviction, remove oldest entry
        if self.cache.len() >= self.config.max_cache_entries {
            if let Some(oldest_key) = self
                .cache
                .iter()
                .min_by_key(|(_, entry)| entry.cached_at)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&oldest_key);
            }
        }

        let entry = CacheEntry {
            response: response.clone(),
            cached_at: Instant::now(),
            ttl: self.config.cache_ttl(),
        };
        self.cache.insert(cache_key.to_string(), entry);
    }

    /// Removes all expired entries from the cache.
    fn evict_expired(&mut self) {
        self.cache.retain(|_, entry| !entry.is_expired());
    }

    // ========================================================================
    // Private: Request Execution with Retry
    // ========================================================================

    fn execute_with_retry(
        &mut self,
        url: &str,
        method: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<DataSourceResponse> {
        let max_retries = self.config.max_retries;
        let base_delay = self.config.retry_delay();
        let mut last_error: Option<anyhow::Error> = None;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                // Exponential backoff: delay * 2^(attempt-1)
                let backoff = base_delay * 2u32.pow((attempt - 1) as u32);
                std::thread::sleep(backoff);
            }

            self.apply_rate_limit();

            match self.execute_request(url, method, body) {
                Ok(response) => {
                    // Retry on server errors (5xx)
                    if response.status >= 500 && attempt < max_retries {
                        last_error = Some(anyhow!(
                            "Server error: HTTP {} from {}",
                            response.status,
                            url
                        ));
                        continue;
                    }
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt >= max_retries {
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("Request failed after {} retries", max_retries)))
    }

    /// Executes a single HTTP request without retry logic.
    fn execute_request(
        &self,
        url: &str,
        method: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<DataSourceResponse> {
        let start = Instant::now();
        let timeout_secs = self.config.timeout_ms / 1000;

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs.max(1)))
            .build();

        let mut request = match method {
            "GET" => agent.get(url),
            "POST" => agent.post(url),
            "PUT" => agent.put(url),
            "DELETE" => agent.delete(url),
            "PATCH" => agent.request("PATCH", url),
            _ => return Err(anyhow!("Unsupported HTTP method: {}", method)),
        };

        // Apply user agent
        request = request.set("User-Agent", &self.config.user_agent);

        // Apply custom headers
        for (key, value) in &self.config.headers {
            request = request.set(key, value);
        }

        // Apply authentication
        request = self.apply_auth(request);

        // Execute request
        let result = if let Some(json_body) = body {
            request.send_json(json_body.clone())
        } else {
            request.call()
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(resp) => {
                let status = resp.status();
                let mut headers = HashMap::new();

                // Collect response headers
                for name in resp.headers_names() {
                    if let Some(value) = resp.header(&name) {
                        headers.insert(name, value.to_string());
                    }
                }

                let body_str = resp.into_string().unwrap_or_default();
                let json = serde_json::from_str::<serde_json::Value>(&body_str).ok();

                Ok(DataSourceResponse {
                    status,
                    body: body_str,
                    json,
                    headers,
                    from_cache: false,
                    duration_ms,
                })
            }
            Err(ureq::Error::Status(status, resp)) => {
                let mut headers = HashMap::new();
                for name in resp.headers_names() {
                    if let Some(value) = resp.header(&name) {
                        headers.insert(name, value.to_string());
                    }
                }

                let body_str = resp.into_string().unwrap_or_default();
                let json = serde_json::from_str::<serde_json::Value>(&body_str).ok();

                Ok(DataSourceResponse {
                    status,
                    body: body_str,
                    json,
                    headers,
                    from_cache: false,
                    duration_ms,
                })
            }
            Err(ureq::Error::Transport(transport)) => {
                Err(anyhow!("Transport error: {}", transport))
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Simple base64 encoder for Basic auth credentials.
fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::new();
    let mut i = 0;

    while i < bytes.len() {
        let b0 = bytes[i] as u32;
        let b1 = if i + 1 < bytes.len() {
            bytes[i + 1] as u32
        } else {
            0
        };
        let b2 = if i + 2 < bytes.len() {
            bytes[i + 2] as u32
        } else {
            0
        };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if i + 1 < bytes.len() {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if i + 2 < bytes.len() {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        i += 3;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DataSourceConfig::default();
        assert_eq!(config.timeout_ms, 30_000);
        assert!(config.cache_enabled);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.rate_limit.max_requests, 60);
        assert_eq!(config.rate_limit.window_ms, 60_000);
        assert_eq!(config.rate_limit.min_delay_ms, 100);
        assert_eq!(config.cache_ttl_ms, 300_000);
        assert_eq!(config.max_cache_entries, 1000);
        assert_eq!(config.user_agent, "DataSourceClient/1.0");
    }

    #[test]
    fn test_build_url_no_params() {
        let client = DataSourceClient::from_url("https://api.example.com/v1");
        let url = client.build_url("/users", &[]);
        assert_eq!(url, "https://api.example.com/v1/users");
    }

    #[test]
    fn test_build_url_with_params() {
        let client = DataSourceClient::from_url("https://api.example.com");
        let url = client.build_url("/search", &[("q", "hello world"), ("limit", "10")]);
        assert!(url.starts_with("https://api.example.com/search?"));
        assert!(url.contains("q=hello%20world"));
        assert!(url.contains("limit=10"));
    }

    #[test]
    fn test_build_url_with_api_key_query_auth() {
        let client = DataSourceClient::from_url("https://api.example.com").with_auth(
            AuthMethod::ApiKeyQuery {
                param: "api_key".into(),
                key: "secret123".into(),
            },
        );
        let url = client.build_url("/data", &[("page", "1")]);
        assert!(url.contains("page=1"));
        assert!(url.contains("api_key=secret123"));
    }

    #[test]
    fn test_build_url_path_normalization() {
        let client = DataSourceClient::from_url("https://api.example.com/");
        let url1 = client.build_url("/users", &[]);
        let url2 = client.build_url("users", &[]);
        assert_eq!(url1, "https://api.example.com/users");
        assert_eq!(url2, "https://api.example.com/users");
    }

    #[test]
    fn test_cache_store_and_retrieve() {
        let mut client = DataSourceClient::from_url("https://api.example.com");
        let response = DataSourceResponse {
            status: 200,
            body: r#"{"id": 1}"#.to_string(),
            json: Some(serde_json::json!({"id": 1})),
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 50,
        };

        client.store_cache("test_key", &response);

        let cached = client.check_cache("test_key");
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert!(cached.from_cache);
        assert_eq!(cached.status, 200);
        assert_eq!(cached.body, r#"{"id": 1}"#);
    }

    #[test]
    fn test_cache_miss_on_unknown_key() {
        let client = DataSourceClient::from_url("https://api.example.com");
        let cached = client.check_cache("nonexistent");
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_expiry() {
        let mut client = DataSourceClient::from_url("https://api.example.com");
        // Set TTL to 0ms so entries expire immediately
        client.config.cache_ttl_ms = 0;

        let response = DataSourceResponse {
            status: 200,
            body: "test".to_string(),
            json: None,
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 10,
        };

        client.store_cache("expire_key", &response);

        // Sleep a tiny bit to ensure expiry
        std::thread::sleep(Duration::from_millis(1));

        let cached = client.check_cache("expire_key");
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let mut client = DataSourceClient::from_url("https://api.example.com");
        client.config.cache_ttl_ms = 0; // Entries expire immediately

        let response = DataSourceResponse {
            status: 200,
            body: "data".to_string(),
            json: None,
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 5,
        };

        client.store_cache("key1", &response);
        client.store_cache("key2", &response);

        std::thread::sleep(Duration::from_millis(2));

        let (total, expired) = client.cache_stats();
        assert_eq!(total, 2);
        assert_eq!(expired, 2);
    }

    #[test]
    fn test_clear_cache() {
        let mut client = DataSourceClient::from_url("https://api.example.com");
        let response = DataSourceResponse {
            status: 200,
            body: "cached".to_string(),
            json: None,
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 1,
        };

        client.store_cache("a", &response);
        client.store_cache("b", &response);
        assert_eq!(client.cache_stats().0, 2);

        client.clear_cache();
        assert_eq!(client.cache_stats().0, 0);
    }

    #[test]
    fn test_cache_eviction_at_capacity() {
        let mut client = DataSourceClient::from_url("https://api.example.com");
        client.config.max_cache_entries = 2;

        let response = DataSourceResponse {
            status: 200,
            body: "x".to_string(),
            json: None,
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 1,
        };

        client.store_cache("first", &response);
        std::thread::sleep(Duration::from_millis(1));
        client.store_cache("second", &response);
        std::thread::sleep(Duration::from_millis(1));
        client.store_cache("third", &response);

        // Should have evicted one entry to stay at capacity
        assert!(client.cache.len() <= 2);
    }

    #[test]
    fn test_extract_items_array() {
        let client = DataSourceClient::from_url("https://example.com");
        let response = DataSourceResponse {
            status: 200,
            body: String::new(),
            json: Some(serde_json::json!([1, 2, 3])),
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 0,
        };
        let items = client.extract_items(&response);
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_extract_items_object_with_data_field() {
        let client = DataSourceClient::from_url("https://example.com");
        let response = DataSourceResponse {
            status: 200,
            body: String::new(),
            json: Some(serde_json::json!({
                "data": [{"id": 1}, {"id": 2}],
                "total": 2
            })),
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 0,
        };
        let items = client.extract_items(&response);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["id"], 1);
    }

    #[test]
    fn test_extract_items_object_with_results_field() {
        let client = DataSourceClient::from_url("https://example.com");
        let response = DataSourceResponse {
            status: 200,
            body: String::new(),
            json: Some(serde_json::json!({
                "results": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                "count": 3
            })),
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 0,
        };
        let items = client.extract_items(&response);
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode("user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64_encode("hello"), "aGVsbG8=");
        assert_eq!(base64_encode(""), "");
    }

    #[test]
    fn test_rate_limit_policy_default() {
        let policy = RateLimitPolicy::default();
        assert_eq!(policy.window(), Duration::from_secs(60));
        assert_eq!(policy.min_delay(), Duration::from_millis(100));
    }

    #[test]
    fn test_with_auth_builder() {
        let client = DataSourceClient::from_url("https://api.example.com")
            .with_auth(AuthMethod::Bearer("token123".into()));

        match &client.config.auth {
            AuthMethod::Bearer(t) => assert_eq!(t, "token123"),
            _ => panic!("Expected Bearer auth"),
        }
    }

    #[test]
    fn test_with_rate_limit_builder() {
        let policy = RateLimitPolicy {
            max_requests: 10,
            window_ms: 5000,
            min_delay_ms: 200,
        };
        let client = DataSourceClient::from_url("https://api.example.com").with_rate_limit(policy);

        assert_eq!(client.config.rate_limit.max_requests, 10);
        assert_eq!(client.config.rate_limit.window_ms, 5000);
        assert_eq!(client.config.rate_limit.min_delay_ms, 200);
    }

    #[test]
    fn test_config_serialization() {
        let config = DataSourceConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DataSourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.timeout_ms, config.timeout_ms);
        assert_eq!(deserialized.cache_ttl_ms, config.cache_ttl_ms);
        assert_eq!(deserialized.max_retries, config.max_retries);
    }

    #[test]
    fn test_response_serialization() {
        let response = DataSourceResponse {
            status: 200,
            body: r#"{"ok": true}"#.to_string(),
            json: Some(serde_json::json!({"ok": true})),
            headers: HashMap::new(),
            from_cache: false,
            duration_ms: 42,
        };
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: DataSourceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.status, 200);
        assert_eq!(deserialized.duration_ms, 42);
    }

    #[test]
    fn test_paginated_response_serialization() {
        let paginated = PaginatedResponse {
            items: vec![serde_json::json!({"id": 1}), serde_json::json!({"id": 2})],
            pages_fetched: 1,
            total_count: Some(2),
            has_more: false,
        };
        let json = serde_json::to_string(&paginated).unwrap();
        let deserialized: PaginatedResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.items.len(), 2);
        assert_eq!(deserialized.pages_fetched, 1);
        assert!(!deserialized.has_more);
    }
}
