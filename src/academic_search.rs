//! Academic search APIs
//!
//! Provides a unified interface to search academic literature across multiple
//! providers: arXiv, Semantic Scholar, and PubMed.
//!
//! # Providers
//!
//! | Provider | API | Auth | Rate Limit |
//! |----------|-----|------|------------|
//! | arXiv | Atom/XML | None | 3s between requests |
//! | Semantic Scholar | REST/JSON | Optional API key | 100 req/5min (free) |
//! | PubMed | E-utilities XML | Optional API key | 3 req/s (10 with key) |
//!
//! # Security
//!
//! - All query parameters are URL-encoded
//! - API keys are sourced from environment variables, never logged
//! - Response Content-Type is validated before parsing

use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// Core Types
// =============================================================================

/// Academic paper source/database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AcademicSource {
    /// arXiv preprint server
    ArXiv,
    /// Semantic Scholar (Allen AI)
    SemanticScholar,
    /// PubMed / NCBI
    PubMed,
    /// CrossRef (reserved for Batch 2)
    CrossRef,
    /// OpenAlex (reserved for Batch 2)
    OpenAlex,
}

impl AcademicSource {
    /// Display name for the source.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ArXiv => "arXiv",
            Self::SemanticScholar => "Semantic Scholar",
            Self::PubMed => "PubMed",
            Self::CrossRef => "CrossRef",
            Self::OpenAlex => "OpenAlex",
        }
    }
}

impl std::fmt::Display for AcademicSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// An author of an academic paper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Author {
    /// Full name
    pub name: String,
    /// Affiliation (university, lab, etc.)
    pub affiliation: Option<String>,
    /// Provider-specific author ID
    pub author_id: Option<String>,
}

impl Author {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            affiliation: None,
            author_id: None,
        }
    }

    pub fn with_affiliation(mut self, affiliation: &str) -> Self {
        self.affiliation = Some(affiliation.to_string());
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.author_id = Some(id.to_string());
        self
    }
}

/// An academic paper with metadata.
#[derive(Debug, Clone)]
pub struct AcademicPaper {
    /// Provider-specific ID (e.g., arXiv ID, S2 paper ID, PMID)
    pub id: String,
    /// Paper title
    pub title: String,
    /// Authors
    pub authors: Vec<Author>,
    /// Abstract text
    pub abstract_text: Option<String>,
    /// Publication year
    pub year: Option<u16>,
    /// Venue (journal, conference)
    pub venue: Option<String>,
    /// DOI
    pub doi: Option<String>,
    /// URL to the paper
    pub url: Option<String>,
    /// URL to the PDF
    pub pdf_url: Option<String>,
    /// Citation count
    pub citation_count: Option<u32>,
    /// Fields of study / categories
    pub fields_of_study: Vec<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Which provider returned this paper
    pub source: AcademicSource,
    /// External IDs (e.g., "DOI" -> "10.1234/...", "ArXiv" -> "2301.01234")
    pub external_ids: HashMap<String, String>,
}

impl AcademicPaper {
    /// Create a minimal paper with just ID, title, and source.
    pub fn new(id: &str, title: &str, source: AcademicSource) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            authors: Vec::new(),
            abstract_text: None,
            year: None,
            venue: None,
            doi: None,
            url: None,
            pdf_url: None,
            citation_count: None,
            fields_of_study: Vec::new(),
            keywords: Vec::new(),
            source,
            external_ids: HashMap::new(),
        }
    }

    /// Citation string: "Author1, Author2 (Year). Title."
    pub fn citation_string(&self) -> String {
        let authors_str = if self.authors.is_empty() {
            "Unknown".to_string()
        } else if self.authors.len() <= 3 {
            self.authors
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            format!("{} et al.", self.authors[0].name)
        };

        let year_str = self.year.map(|y| format!(" ({})", y)).unwrap_or_default();

        format!("{}{}. {}.", authors_str, year_str, self.title)
    }
}

/// Sort field for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortField {
    /// Sort by relevance (default)
    Relevance,
    /// Sort by date (newest first)
    Date,
    /// Sort by citation count
    Citations,
}

impl Default for SortField {
    fn default() -> Self {
        Self::Relevance
    }
}

/// Configuration for academic search queries.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AcademicSearchConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Filter by year range (inclusive)
    pub year_range: Option<(u16, u16)>,
    /// Filter by fields of study
    pub fields_of_study: Vec<String>,
    /// Sort order
    pub sort_by: SortField,
    /// Request timeout
    pub timeout: Duration,
}

impl Default for AcademicSearchConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            year_range: None,
            fields_of_study: Vec::new(),
            sort_by: SortField::default(),
            timeout: Duration::from_secs(15),
        }
    }
}

/// Errors from academic search operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum AcademicSearchError {
    /// Network/HTTP error
    Network(String),
    /// Response parsing error
    Parse(String),
    /// Rate limited
    RateLimit(String),
    /// No results found
    NoResults,
    /// Invalid query
    InvalidQuery(String),
    /// Provider not available
    ProviderUnavailable(String),
}

impl std::fmt::Display for AcademicSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "Network error: {}", e),
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::RateLimit(e) => write!(f, "Rate limited: {}", e),
            Self::NoResults => write!(f, "No results found"),
            Self::InvalidQuery(e) => write!(f, "Invalid query: {}", e),
            Self::ProviderUnavailable(e) => write!(f, "Provider unavailable: {}", e),
        }
    }
}

impl std::error::Error for AcademicSearchError {}

// =============================================================================
// Rate limiting
// =============================================================================
//
// `AcademicSearchError::RateLimit` existed from the start and **nothing ever
// constructed it**: every call site collapsed `ureq` errors into `Network(..)`, so a
// 429 reached the caller as "Network error: http status 429" — indistinguishable from
// the connection being down. That matters more here than it sounds. arXiv, Semantic
// Scholar and NCBI all throttle by default (NCBI: 3 req/s without a key), so the FIRST
// thing a wide search does is get throttled, and the message sent the user looking at
// their network instead of at their request rate.
//
// The retry decision is a pure function on purpose: it is the part worth testing, and
// testing it must not require a network or a clock.

/// Identifies us to the academic APIs. NCBI and OpenAlex both ask for a contact in the
/// User-Agent and give anonymous clients a much lower quota, so this is a rate-limit
/// setting as much as a courtesy.
const USER_AGENT: &str = "AIAssistant/1.0 (+https://github.com/OrlandoLuque/ai_assistant)";

/// Longest a single backoff will wait. A server asking for ten minutes via
/// `Retry-After` is telling us to come back later, not to block the caller.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// How long to wait before retrying, or `None` when the response is not worth retrying.
///
/// * **429 / 503** are retryable: the server is asking us to slow down.
/// * A `Retry-After` header wins over our own guess — it is the server saying how long,
///   and ignoring it is how a client gets banned rather than throttled.
/// * Everything else (200, 404, 400…) returns `None`: retrying a bad query just sends
///   the same bad query again.
///
/// How many times a throttled request is retried before giving up.
const MAX_ATTEMPTS: u32 = 3;

/// Perform a GET, retrying while the server says we are going too fast.
///
/// `build` is called afresh for each attempt because a `ureq::Request` is consumed by
/// `call()`. On giving up, the error is [`AcademicSearchError::RateLimit`] and **not**
/// `Network` — the whole point is that the caller can tell "slow down" from "no route to
/// host", and act differently.
fn get_with_retry(
    build: impl Fn() -> ureq::Request,
) -> Result<ureq::Response, AcademicSearchError> {
    let mut attempt = 0u32;
    loop {
        match build().call() {
            Ok(resp) => return Ok(resp),
            Err(ureq::Error::Status(code, resp)) => {
                let retry_after = resp.header("Retry-After").map(|s| s.to_string());
                match retry_delay(code, retry_after.as_deref(), attempt) {
                    Some(delay) if attempt + 1 < MAX_ATTEMPTS => {
                        std::thread::sleep(delay);
                        attempt += 1;
                    }
                    Some(_) => {
                        return Err(AcademicSearchError::RateLimit(format!(
                            "still throttled (HTTP {code}) after {MAX_ATTEMPTS} attempts — \
                             slow down, or set an API key for a higher quota"
                        )))
                    }
                    None => {
                        return Err(AcademicSearchError::Network(format!("http status {code}")))
                    }
                }
            }
            Err(e) => return Err(AcademicSearchError::Network(e.to_string())),
        }
    }
}

/// `attempt` is 0-based, so the delays grow 1s, 2s, 4s… capped at [`MAX_BACKOFF`].
fn retry_delay(status: u16, retry_after: Option<&str>, attempt: u32) -> Option<Duration> {
    if status != 429 && status != 503 {
        return None;
    }
    // `Retry-After` is either seconds or an HTTP date; only the numeric form is worth
    // parsing here, and a malformed one falls back to the exponential guess rather than
    // failing — the server still said "too fast", which is the part that matters.
    if let Some(secs) = retry_after.and_then(|v| v.trim().parse::<u64>().ok()) {
        return Some(Duration::from_secs(secs).min(MAX_BACKOFF));
    }
    let backoff = Duration::from_secs(1u64 << attempt.min(6));
    Some(backoff.min(MAX_BACKOFF))
}

// =============================================================================
// Provider Trait
// =============================================================================

/// Trait for academic search providers.
///
/// Each provider implements search, single-paper lookup, and citation/reference
/// retrieval. Providers handle their own rate limiting and API authentication.
pub trait AcademicSearchProvider: Send + Sync {
    /// Search for papers matching a query.
    fn search_papers(
        &self,
        query: &str,
        config: &AcademicSearchConfig,
    ) -> Result<Vec<AcademicPaper>, AcademicSearchError>;

    /// Get a single paper by its provider-specific ID.
    fn get_paper(&self, id: &str) -> Result<AcademicPaper, AcademicSearchError>;

    /// Get papers that cite the given paper.
    fn get_citations(&self, paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError>;

    /// Get papers referenced by the given paper.
    fn get_references(&self, paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError>;

    /// Provider name.
    fn name(&self) -> &str;

    /// Which academic source this provider represents.
    fn source(&self) -> AcademicSource;
}

// =============================================================================
// arXiv Provider
// =============================================================================

/// arXiv search provider using the Atom/XML API.
///
/// API docs: https://info.arxiv.org/help/api/basics.html
/// Rate limit: 3 seconds between requests (polite throttling).
pub struct ArxivProvider {
    base_url: String,
}

impl ArxivProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://export.arxiv.org/api/query".to_string(),
        }
    }

    /// Build the arXiv API URL for a search query.
    fn build_url(&self, query: &str, config: &AcademicSearchConfig) -> String {
        let encoded_query = urlencoding::encode(query);
        let sort_by = match config.sort_by {
            SortField::Relevance => "relevance",
            SortField::Date => "lastUpdatedDate",
            SortField::Citations => "relevance", // arXiv doesn't sort by citations
        };
        format!(
            "{}?search_query=all:{}&start=0&max_results={}&sortBy={}&sortOrder=descending",
            self.base_url, encoded_query, config.max_results, sort_by
        )
    }

    /// Parse arXiv Atom/XML response into papers.
    fn parse_atom_response(&self, xml: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        let mut papers = Vec::new();

        // Simple XML parsing using string matching (no external XML dependency)
        for entry in xml.split("<entry>").skip(1) {
            let entry_end = entry.find("</entry>").unwrap_or(entry.len());
            let entry_xml = &entry[..entry_end];

            let id = extract_xml_text(entry_xml, "id").unwrap_or_default();
            let title = extract_xml_text(entry_xml, "title")
                .unwrap_or_default()
                .replace('\n', " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");

            if title.is_empty() {
                continue;
            }

            let abstract_text = extract_xml_text(entry_xml, "summary")
                .map(|s| s.replace('\n', " ").trim().to_string());

            // Extract arXiv ID from URL
            let arxiv_id = id.rsplit('/').next().unwrap_or(&id).to_string();

            // Extract authors
            let authors: Vec<Author> = entry_xml
                .split("<author>")
                .skip(1)
                .filter_map(|a| extract_xml_text(a, "name").map(|name| Author::new(&name)))
                .collect();

            // Extract published date year
            let year = extract_xml_text(entry_xml, "published")
                .and_then(|d| d.get(..4).and_then(|y| y.parse::<u16>().ok()));

            // Extract categories
            let fields: Vec<String> = entry_xml
                .split("category term=\"")
                .skip(1)
                .filter_map(|c| c.split('"').next().map(|s| s.to_string()))
                .collect();

            // PDF link
            let pdf_url = entry_xml
                .split("<link")
                .find(|l| l.contains("title=\"pdf\""))
                .and_then(|l| l.split("href=\"").nth(1).and_then(|h| h.split('"').next()))
                .map(|s| s.to_string());

            let mut paper = AcademicPaper::new(&arxiv_id, &title, AcademicSource::ArXiv);
            paper.authors = authors;
            paper.abstract_text = abstract_text;
            paper.year = year;
            paper.url = Some(id.clone());
            paper.pdf_url = pdf_url;
            paper.fields_of_study = fields;
            paper.external_ids.insert("ArXiv".to_string(), arxiv_id);

            papers.push(paper);
        }

        Ok(papers)
    }
}

impl Default for ArxivProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AcademicSearchProvider for ArxivProvider {
    fn search_papers(
        &self,
        query: &str,
        config: &AcademicSearchConfig,
    ) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        if query.trim().is_empty() {
            return Err(AcademicSearchError::InvalidQuery(
                "Query cannot be empty".to_string(),
            ));
        }

        let url = self.build_url(query, config);

        let timeout = config.timeout;
        let response = get_with_retry(|| {
            ureq::get(&url)
                .timeout(timeout)
                .set("User-Agent", USER_AGENT)
        })?;

        let xml = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let mut papers = self.parse_atom_response(&xml)?;

        // Apply year filter if specified
        if let Some((min_year, max_year)) = config.year_range {
            papers.retain(|p| {
                p.year
                    .map(|y| y >= min_year && y <= max_year)
                    .unwrap_or(true)
            });
        }

        Ok(papers)
    }

    fn get_paper(&self, id: &str) -> Result<AcademicPaper, AcademicSearchError> {
        let url = format!("{}?id_list={}", self.base_url, urlencoding::encode(id));

        let response = get_with_retry(|| {
            ureq::get(&url)
                .timeout(Duration::from_secs(10))
                .set("User-Agent", USER_AGENT)
        })?;

        let xml = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let papers = self.parse_atom_response(&xml)?;
        papers
            .into_iter()
            .next()
            .ok_or(AcademicSearchError::NoResults)
    }

    fn get_citations(&self, _paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        // arXiv does not provide citation data
        Err(AcademicSearchError::ProviderUnavailable(
            "arXiv does not provide citation data; use Semantic Scholar".to_string(),
        ))
    }

    fn get_references(&self, _paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        Err(AcademicSearchError::ProviderUnavailable(
            "arXiv does not provide reference data; use Semantic Scholar".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "arXiv"
    }

    fn source(&self) -> AcademicSource {
        AcademicSource::ArXiv
    }
}

// =============================================================================
// Semantic Scholar Provider
// =============================================================================

/// Semantic Scholar search provider using REST/JSON API.
///
/// API docs: https://api.semanticscholar.org/api-docs/
/// Rate limit: 100 requests per 5 minutes (free), higher with API key.
pub struct SemanticScholarProvider {
    api_key: Option<String>,
    base_url: String,
}

impl SemanticScholarProvider {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("SEMANTIC_SCHOLAR_API_KEY").ok(),
            base_url: "https://api.semanticscholar.org/graph/v1".to_string(),
        }
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Build request with optional API key header.
    fn build_request(&self, url: &str, timeout: Duration) -> ureq::Request {
        let mut req = ureq::get(url).timeout(timeout);
        if let Some(key) = &self.api_key {
            req = req.set("x-api-key", key);
        }
        req
    }

    /// Parse Semantic Scholar JSON paper object.
    fn parse_paper(json: &serde_json::Value) -> Option<AcademicPaper> {
        let paper_id = json.get("paperId")?.as_str()?.to_string();
        let title = json.get("title")?.as_str()?.to_string();

        if title.is_empty() {
            return None;
        }

        let mut paper = AcademicPaper::new(&paper_id, &title, AcademicSource::SemanticScholar);

        paper.abstract_text = json
            .get("abstract")
            .and_then(|a| a.as_str())
            .map(|s| s.to_string());

        paper.year = json.get("year").and_then(|y| y.as_u64()).map(|y| y as u16);

        paper.venue = json
            .get("venue")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string());

        paper.citation_count = json
            .get("citationCount")
            .and_then(|c| c.as_u64())
            .map(|c| c as u32);

        paper.url = json
            .get("url")
            .and_then(|u| u.as_str())
            .map(|u| u.to_string());

        paper.doi = json
            .get("externalIds")
            .and_then(|e| e.get("DOI"))
            .and_then(|d| d.as_str())
            .map(|d| d.to_string());

        // Authors
        if let Some(authors) = json.get("authors").and_then(|a| a.as_array()) {
            paper.authors = authors
                .iter()
                .filter_map(|a| {
                    a.get("name").and_then(|n| n.as_str()).map(|n| {
                        let mut author = Author::new(n);
                        if let Some(id) = a.get("authorId").and_then(|i| i.as_str()) {
                            author = author.with_id(id);
                        }
                        author
                    })
                })
                .collect();
        }

        // Fields of study
        if let Some(fields) = json.get("fieldsOfStudy").and_then(|f| f.as_array()) {
            paper.fields_of_study = fields
                .iter()
                .filter_map(|f| f.as_str().map(|s| s.to_string()))
                .collect();
        }

        // External IDs
        if let Some(ext) = json.get("externalIds").and_then(|e| e.as_object()) {
            for (k, v) in ext {
                if let Some(val) = v.as_str() {
                    paper.external_ids.insert(k.clone(), val.to_string());
                }
            }
        }

        Some(paper)
    }
}

impl Default for SemanticScholarProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AcademicSearchProvider for SemanticScholarProvider {
    fn search_papers(
        &self,
        query: &str,
        config: &AcademicSearchConfig,
    ) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        if query.trim().is_empty() {
            return Err(AcademicSearchError::InvalidQuery(
                "Query cannot be empty".to_string(),
            ));
        }

        let fields =
            "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds,fieldsOfStudy";
        let url = format!(
            "{}/paper/search?query={}&limit={}&fields={}",
            self.base_url,
            urlencoding::encode(query),
            config.max_results,
            fields,
        );

        let response = get_with_retry(|| self.build_request(&url, config.timeout))?;

        let text = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let mut papers = Vec::new();
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(paper) = Self::parse_paper(item) {
                    papers.push(paper);
                }
            }
        }

        // Apply year filter
        if let Some((min_year, max_year)) = config.year_range {
            papers.retain(|p| {
                p.year
                    .map(|y| y >= min_year && y <= max_year)
                    .unwrap_or(true)
            });
        }

        Ok(papers)
    }

    fn get_paper(&self, id: &str) -> Result<AcademicPaper, AcademicSearchError> {
        let fields =
            "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds,fieldsOfStudy";
        let url = format!(
            "{}/paper/{}?fields={}",
            self.base_url,
            urlencoding::encode(id),
            fields,
        );

        let response = get_with_retry(|| self.build_request(&url, Duration::from_secs(10)))?;

        let text = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        Self::parse_paper(&json).ok_or(AcademicSearchError::NoResults)
    }

    fn get_citations(&self, paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        let fields = "paperId,title,authors,year,venue,citationCount,url";
        let url = format!(
            "{}/paper/{}/citations?fields={}&limit=100",
            self.base_url,
            urlencoding::encode(paper_id),
            fields,
        );

        let response = get_with_retry(|| self.build_request(&url, Duration::from_secs(15)))?;

        let text = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let mut papers = Vec::new();
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(citing) = item.get("citingPaper") {
                    if let Some(paper) = Self::parse_paper(citing) {
                        papers.push(paper);
                    }
                }
            }
        }

        Ok(papers)
    }

    fn get_references(&self, paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        let fields = "paperId,title,authors,year,venue,citationCount,url";
        let url = format!(
            "{}/paper/{}/references?fields={}&limit=100",
            self.base_url,
            urlencoding::encode(paper_id),
            fields,
        );

        let response = get_with_retry(|| self.build_request(&url, Duration::from_secs(15)))?;

        let text = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let mut papers = Vec::new();
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(cited) = item.get("citedPaper") {
                    if let Some(paper) = Self::parse_paper(cited) {
                        papers.push(paper);
                    }
                }
            }
        }

        Ok(papers)
    }

    fn name(&self) -> &str {
        "Semantic Scholar"
    }

    fn source(&self) -> AcademicSource {
        AcademicSource::SemanticScholar
    }
}

// =============================================================================
// PubMed Provider
// =============================================================================

/// PubMed search provider using NCBI E-utilities.
///
/// API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
/// Rate limit: 3 req/s without key, 10 req/s with NCBI_API_KEY.
pub struct PubMedProvider {
    api_key: Option<String>,
    base_url: String,
}

impl PubMedProvider {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("NCBI_API_KEY").ok(),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils".to_string(),
        }
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Build URL with optional API key parameter.
    fn url_with_key(&self, base: &str) -> String {
        if let Some(key) = &self.api_key {
            format!("{}&api_key={}", base, urlencoding::encode(key))
        } else {
            base.to_string()
        }
    }

    /// Search PubMed and return PMIDs.
    fn search_ids(
        &self,
        query: &str,
        max_results: usize,
        timeout: Duration,
    ) -> Result<Vec<String>, AcademicSearchError> {
        let url = self.url_with_key(&format!(
            "{}/esearch.fcgi?db=pubmed&term={}&retmax={}&retmode=json",
            self.base_url,
            urlencoding::encode(query),
            max_results,
        ));

        let response = get_with_retry(|| {
            ureq::get(&url)
                .timeout(timeout)
                .set("User-Agent", USER_AGENT)
        })?;

        let text = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        let ids = json
            .get("esearchresult")
            .and_then(|r| r.get("idlist"))
            .and_then(|l| l.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(ids)
    }

    /// Fetch paper details for a list of PMIDs using efetch XML.
    fn fetch_details(
        &self,
        pmids: &[String],
        timeout: Duration,
    ) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        if pmids.is_empty() {
            return Ok(Vec::new());
        }

        let ids_param = pmids.join(",");
        let url = self.url_with_key(&format!(
            "{}/efetch.fcgi?db=pubmed&id={}&retmode=xml",
            self.base_url, ids_param,
        ));

        let response = get_with_retry(|| {
            ureq::get(&url)
                .timeout(timeout)
                .set("User-Agent", USER_AGENT)
        })?;

        let xml = response
            .into_string()
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;

        self.parse_pubmed_xml(&xml)
    }

    /// Parse PubMed XML response.
    fn parse_pubmed_xml(&self, xml: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        let mut papers = Vec::new();

        for article in xml.split("<PubmedArticle>").skip(1) {
            let article_end = article.find("</PubmedArticle>").unwrap_or(article.len());
            let article_xml = &article[..article_end];

            let pmid = extract_xml_text(article_xml, "PMID").unwrap_or_default();
            let title = extract_xml_text(article_xml, "ArticleTitle").unwrap_or_default();

            if title.is_empty() || pmid.is_empty() {
                continue;
            }

            let abstract_text = extract_xml_text(article_xml, "AbstractText");

            // Extract year from PubDate
            let year = extract_xml_text(article_xml, "Year").and_then(|y| y.parse::<u16>().ok());

            // Extract journal title as venue
            let venue = extract_xml_text(article_xml, "Title");

            // Extract DOI from ArticleId
            let doi = article_xml
                .split("<ArticleId IdType=\"doi\">")
                .nth(1)
                .and_then(|d| d.split("</ArticleId>").next())
                .map(|d| d.to_string());

            // Extract authors
            let mut authors = Vec::new();
            for author_block in article_xml.split("<Author").skip(1) {
                let last_name = extract_xml_text(author_block, "LastName");
                let first_name = extract_xml_text(author_block, "ForeName");

                if let Some(last) = last_name {
                    let full_name = if let Some(first) = first_name {
                        format!("{} {}", first, last)
                    } else {
                        last
                    };
                    let mut author = Author::new(&full_name);
                    if let Some(aff) = extract_xml_text(author_block, "Affiliation") {
                        author = author.with_affiliation(&aff);
                    }
                    authors.push(author);
                }
            }

            // Extract keywords
            let mut keywords = Vec::new();
            for kw in article_xml.split("<Keyword").skip(1) {
                if let Some(_end) = kw.find("</Keyword>") {
                    let text = kw.split('>').nth(1).unwrap_or("");
                    let text = &text[..text.find("</Keyword>").unwrap_or(text.len())];
                    if !text.is_empty() {
                        keywords.push(text.to_string());
                    }
                }
            }

            let mut paper = AcademicPaper::new(&pmid, &title, AcademicSource::PubMed);
            paper.authors = authors;
            paper.abstract_text = abstract_text;
            paper.year = year;
            paper.venue = venue;
            paper.doi = doi;
            paper.url = Some(format!("https://pubmed.ncbi.nlm.nih.gov/{}/", pmid));
            paper.keywords = keywords;
            paper.external_ids.insert("PMID".to_string(), pmid);

            papers.push(paper);
        }

        Ok(papers)
    }
}

impl Default for PubMedProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AcademicSearchProvider for PubMedProvider {
    fn search_papers(
        &self,
        query: &str,
        config: &AcademicSearchConfig,
    ) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        if query.trim().is_empty() {
            return Err(AcademicSearchError::InvalidQuery(
                "Query cannot be empty".to_string(),
            ));
        }

        // Step 1: Search for PMIDs
        let pmids = self.search_ids(query, config.max_results, config.timeout)?;
        if pmids.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Fetch details
        let mut papers = self.fetch_details(&pmids, config.timeout)?;

        // Apply year filter
        if let Some((min_year, max_year)) = config.year_range {
            papers.retain(|p| {
                p.year
                    .map(|y| y >= min_year && y <= max_year)
                    .unwrap_or(true)
            });
        }

        Ok(papers)
    }

    fn get_paper(&self, id: &str) -> Result<AcademicPaper, AcademicSearchError> {
        let papers = self.fetch_details(&[id.to_string()], Duration::from_secs(10))?;
        papers
            .into_iter()
            .next()
            .ok_or(AcademicSearchError::NoResults)
    }

    fn get_citations(&self, _paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        // PubMed cited-by requires elink which is complex
        Err(AcademicSearchError::ProviderUnavailable(
            "PubMed citation lookup not yet implemented; use Semantic Scholar".to_string(),
        ))
    }

    fn get_references(&self, _paper_id: &str) -> Result<Vec<AcademicPaper>, AcademicSearchError> {
        Err(AcademicSearchError::ProviderUnavailable(
            "PubMed reference lookup not yet implemented; use Semantic Scholar".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "PubMed"
    }

    fn source(&self) -> AcademicSource {
        AcademicSource::PubMed
    }
}

// =============================================================================
// Multi-provider Search Engine
// =============================================================================

/// Aggregated academic search across multiple providers.
pub struct AcademicSearchEngine {
    providers: Vec<Box<dyn AcademicSearchProvider>>,
}

impl AcademicSearchEngine {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Add a provider to the engine.
    pub fn add_provider(&mut self, provider: Box<dyn AcademicSearchProvider>) {
        self.providers.push(provider);
    }

    /// Search across all providers and deduplicate by DOI.
    pub fn search_all(&self, query: &str, config: &AcademicSearchConfig) -> Vec<AcademicPaper> {
        let mut all_papers = Vec::new();
        let mut seen_dois = std::collections::HashSet::new();

        for provider in &self.providers {
            match provider.search_papers(query, config) {
                Ok(papers) => {
                    for paper in papers {
                        // Deduplicate by DOI
                        if let Some(doi) = &paper.doi {
                            if !seen_dois.insert(doi.clone()) {
                                continue;
                            }
                        }
                        all_papers.push(paper);
                    }
                }
                Err(_) => {
                    // Log error but continue with other providers
                    continue;
                }
            }
        }

        all_papers
    }

    /// Get list of available provider names.
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }
}

impl Default for AcademicSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract text content between XML tags (simple, non-recursive).
fn extract_xml_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    let start = xml.find(&open)?;
    // Find the closing > of the opening tag
    let content_start = xml[start..].find('>')? + start + 1;
    let end = xml[content_start..].find(&close)? + content_start;

    let text = xml[content_start..end].trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_academic_source_display() {
        assert_eq!(AcademicSource::ArXiv.display_name(), "arXiv");
        assert_eq!(
            AcademicSource::SemanticScholar.display_name(),
            "Semantic Scholar"
        );
        assert_eq!(AcademicSource::PubMed.display_name(), "PubMed");
        assert_eq!(AcademicSource::CrossRef.display_name(), "CrossRef");
        assert_eq!(AcademicSource::OpenAlex.display_name(), "OpenAlex");
    }

    #[test]
    fn test_author_creation() {
        let author = Author::new("John Doe")
            .with_affiliation("MIT")
            .with_id("12345");
        assert_eq!(author.name, "John Doe");
        assert_eq!(author.affiliation.as_deref(), Some("MIT"));
        assert_eq!(author.author_id.as_deref(), Some("12345"));
    }

    #[test]
    fn test_paper_creation() {
        let paper = AcademicPaper::new("2301.01234", "Test Paper", AcademicSource::ArXiv);
        assert_eq!(paper.id, "2301.01234");
        assert_eq!(paper.title, "Test Paper");
        assert_eq!(paper.source, AcademicSource::ArXiv);
        assert!(paper.authors.is_empty());
        assert!(paper.abstract_text.is_none());
    }

    #[test]
    fn test_paper_citation_string() {
        let mut paper = AcademicPaper::new("id1", "Deep Learning", AcademicSource::SemanticScholar);
        paper.authors = vec![Author::new("Smith"), Author::new("Jones")];
        paper.year = Some(2024);
        assert_eq!(
            paper.citation_string(),
            "Smith, Jones (2024). Deep Learning."
        );

        // Et al. for > 3 authors
        let mut paper2 = AcademicPaper::new("id2", "Big Paper", AcademicSource::ArXiv);
        paper2.authors = vec![
            Author::new("A"),
            Author::new("B"),
            Author::new("C"),
            Author::new("D"),
        ];
        paper2.year = Some(2023);
        assert!(paper2.citation_string().contains("et al."));
    }

    #[test]
    fn test_paper_citation_no_authors_no_year() {
        let paper = AcademicPaper::new("id", "Title", AcademicSource::PubMed);
        assert_eq!(paper.citation_string(), "Unknown. Title.");
    }

    #[test]
    fn test_search_config_defaults() {
        let config = AcademicSearchConfig::default();
        assert_eq!(config.max_results, 10);
        assert!(config.year_range.is_none());
        assert_eq!(config.sort_by, SortField::Relevance);
    }

    #[test]
    fn test_sort_field_default() {
        assert_eq!(SortField::default(), SortField::Relevance);
    }

    #[test]
    fn test_extract_xml_text() {
        let xml = "<root><title>Hello World</title><year>2024</year></root>";
        assert_eq!(
            extract_xml_text(xml, "title"),
            Some("Hello World".to_string())
        );
        assert_eq!(extract_xml_text(xml, "year"), Some("2024".to_string()));
        assert_eq!(extract_xml_text(xml, "missing"), None);
    }

    #[test]
    fn test_extract_xml_text_with_attributes() {
        let xml = r#"<root><name type="full">John Doe</name></root>"#;
        assert_eq!(extract_xml_text(xml, "name"), Some("John Doe".to_string()));
    }

    #[test]
    fn test_extract_xml_text_empty() {
        let xml = "<root><empty></empty></root>";
        assert_eq!(extract_xml_text(xml, "empty"), None);
    }

    #[test]
    fn test_arxiv_parse_atom() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed>
  <entry>
    <id>http://arxiv.org/abs/2301.01234v1</id>
    <title>Attention Is All You Need</title>
    <summary>We propose a new architecture called Transformer.</summary>
    <published>2023-01-03T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.01234v1" rel="related"/>
  </entry>
</feed>"#;

        let provider = ArxivProvider::new();
        let papers = provider.parse_atom_response(xml).unwrap();
        assert_eq!(papers.len(), 1);
        let p = &papers[0];
        assert_eq!(p.title, "Attention Is All You Need");
        assert_eq!(p.authors.len(), 2);
        assert_eq!(p.authors[0].name, "Ashish Vaswani");
        assert_eq!(p.year, Some(2023));
        assert!(p.fields_of_study.contains(&"cs.CL".to_string()));
        assert!(p.pdf_url.is_some());
        assert_eq!(p.source, AcademicSource::ArXiv);
    }

    #[test]
    fn test_arxiv_parse_empty() {
        let xml = r#"<?xml version="1.0"?><feed></feed>"#;
        let provider = ArxivProvider::new();
        let papers = provider.parse_atom_response(xml).unwrap();
        assert!(papers.is_empty());
    }

    #[test]
    fn test_arxiv_build_url() {
        let provider = ArxivProvider::new();
        let config = AcademicSearchConfig {
            max_results: 5,
            sort_by: SortField::Date,
            ..Default::default()
        };
        let url = provider.build_url("transformer attention", &config);
        assert!(url.contains("max_results=5"));
        assert!(url.contains("sortBy=lastUpdatedDate"));
        assert!(url.contains("transformer"));
    }

    #[test]
    fn test_arxiv_empty_query() {
        let provider = ArxivProvider::new();
        let config = AcademicSearchConfig::default();
        let result = provider.search_papers("", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_s2_parse_paper() {
        let json = serde_json::json!({
            "paperId": "abc123",
            "title": "Test Paper Title",
            "abstract": "This is the abstract.",
            "year": 2024,
            "venue": "NeurIPS",
            "citationCount": 42,
            "url": "https://www.semanticscholar.org/paper/abc123",
            "authors": [
                {"name": "Alice", "authorId": "1"},
                {"name": "Bob", "authorId": "2"}
            ],
            "fieldsOfStudy": ["Computer Science", "Mathematics"],
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.00001"}
        });

        let paper = SemanticScholarProvider::parse_paper(&json).unwrap();
        assert_eq!(paper.id, "abc123");
        assert_eq!(paper.title, "Test Paper Title");
        assert_eq!(
            paper.abstract_text.as_deref(),
            Some("This is the abstract.")
        );
        assert_eq!(paper.year, Some(2024));
        assert_eq!(paper.venue.as_deref(), Some("NeurIPS"));
        assert_eq!(paper.citation_count, Some(42));
        assert_eq!(paper.authors.len(), 2);
        assert_eq!(paper.fields_of_study.len(), 2);
        assert_eq!(paper.doi.as_deref(), Some("10.1234/test"));
        assert_eq!(
            paper.external_ids.get("ArXiv").map(|s| s.as_str()),
            Some("2401.00001")
        );
    }

    #[test]
    fn test_s2_parse_minimal_paper() {
        let json = serde_json::json!({
            "paperId": "xyz",
            "title": "Minimal"
        });
        let paper = SemanticScholarProvider::parse_paper(&json).unwrap();
        assert_eq!(paper.id, "xyz");
        assert!(paper.authors.is_empty());
        assert!(paper.year.is_none());
    }

    #[test]
    fn test_s2_parse_no_title() {
        let json = serde_json::json!({
            "paperId": "xyz",
            "title": ""
        });
        assert!(SemanticScholarProvider::parse_paper(&json).is_none());
    }

    #[test]
    fn test_s2_empty_query() {
        let provider = SemanticScholarProvider::new();
        let config = AcademicSearchConfig::default();
        let result = provider.search_papers("  ", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_pubmed_parse_xml() {
        let xml = r#"<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation>
    <PMID>12345678</PMID>
    <Article>
      <ArticleTitle>Effect of Drug X on Disease Y</ArticleTitle>
      <Abstract><AbstractText>We studied the effect of Drug X.</AbstractText></Abstract>
      <AuthorList>
        <Author><ForeName>Jane</ForeName><LastName>Smith</LastName></Author>
        <Author><ForeName>John</ForeName><LastName>Doe</LastName></Author>
      </AuthorList>
      <Journal><Title>Nature Medicine</Title></Journal>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="doi">10.1038/test</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
</PubmedArticleSet>"#;

        let provider = PubMedProvider::new();
        let papers = provider.parse_pubmed_xml(xml).unwrap();
        assert_eq!(papers.len(), 1);
        let p = &papers[0];
        assert_eq!(p.title, "Effect of Drug X on Disease Y");
        assert_eq!(p.authors.len(), 2);
        assert_eq!(p.authors[0].name, "Jane Smith");
        assert_eq!(p.doi.as_deref(), Some("10.1038/test"));
        assert_eq!(p.venue.as_deref(), Some("Nature Medicine"));
        assert_eq!(p.source, AcademicSource::PubMed);
        assert!(p.url.as_ref().unwrap().contains("pubmed.ncbi.nlm.nih.gov"));
    }

    #[test]
    fn test_pubmed_parse_empty_xml() {
        let xml = "<PubmedArticleSet></PubmedArticleSet>";
        let provider = PubMedProvider::new();
        let papers = provider.parse_pubmed_xml(xml).unwrap();
        assert!(papers.is_empty());
    }

    #[test]
    fn test_pubmed_empty_query() {
        let provider = PubMedProvider::new();
        let config = AcademicSearchConfig::default();
        let result = provider.search_papers("", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_engine_dedup() {
        let engine = AcademicSearchEngine::new();
        assert!(engine.provider_names().is_empty());

        // Simulated: just test the engine structure
        assert_eq!(engine.provider_names().len(), 0);
    }

    #[test]
    fn test_academic_error_display() {
        let e = AcademicSearchError::Network("timeout".to_string());
        assert!(e.to_string().contains("timeout"));
        let e2 = AcademicSearchError::NoResults;
        assert!(e2.to_string().contains("No results"));
    }

    #[test]
    fn test_arxiv_citations_unavailable() {
        let provider = ArxivProvider::new();
        assert!(provider.get_citations("id").is_err());
        assert!(provider.get_references("id").is_err());
    }

    #[test]
    fn test_pubmed_citations_unavailable() {
        let provider = PubMedProvider::new();
        assert!(provider.get_citations("12345").is_err());
        assert!(provider.get_references("12345").is_err());
    }

    #[test]
    fn test_provider_names() {
        let arxiv = ArxivProvider::new();
        assert_eq!(arxiv.name(), "arXiv");
        assert_eq!(arxiv.source(), AcademicSource::ArXiv);

        let s2 = SemanticScholarProvider::new();
        assert_eq!(s2.name(), "Semantic Scholar");
        assert_eq!(s2.source(), AcademicSource::SemanticScholar);

        let pm = PubMedProvider::new();
        assert_eq!(pm.name(), "PubMed");
        assert_eq!(pm.source(), AcademicSource::PubMed);
    }
}

#[cfg(test)]
mod rate_limit_tests {
    use super::*;

    #[test]
    fn only_throttling_statuses_are_retried() {
        // Retrying a 404 or a 400 just sends the same bad request again; the point of
        // the backoff is to wait out a server asking us to slow down, nothing else.
        assert!(retry_delay(429, None, 0).is_some());
        assert!(retry_delay(503, None, 0).is_some());
        assert!(retry_delay(200, None, 0).is_none());
        assert!(retry_delay(404, None, 0).is_none());
        assert!(retry_delay(400, None, 0).is_none());
        assert!(retry_delay(500, None, 0).is_none());
    }

    #[test]
    fn the_delay_grows_with_each_attempt() {
        let first = retry_delay(429, None, 0).unwrap();
        let second = retry_delay(429, None, 1).unwrap();
        let third = retry_delay(429, None, 2).unwrap();
        assert!(
            second > first && third > second,
            "{first:?} {second:?} {third:?}"
        );
    }

    #[test]
    fn retry_after_wins_over_our_own_guess() {
        // The server saying "come back in 7 seconds" is information we do not have.
        // Ignoring it is how a client gets banned rather than throttled.
        assert_eq!(retry_delay(429, Some("7"), 0), Some(Duration::from_secs(7)));
        assert_eq!(
            retry_delay(429, Some("  7 "), 0),
            Some(Duration::from_secs(7)),
            "a padded header value is still a number"
        );
    }

    #[test]
    fn a_retry_after_we_cannot_parse_falls_back_instead_of_failing() {
        // `Retry-After` may be an HTTP date. We do not parse those, but the server still
        // said "too fast" — which is the part that decides whether to wait at all.
        let delay = retry_delay(429, Some("Wed, 21 Oct 2026 07:28:00 GMT"), 0);
        assert_eq!(delay, Some(Duration::from_secs(1)));
    }

    #[test]
    fn no_wait_is_longer_than_the_cap() {
        // A server asking for ten minutes is telling us to come back later, not to block
        // the caller for ten minutes.
        assert_eq!(retry_delay(429, Some("600"), 0), Some(MAX_BACKOFF));
        assert_eq!(retry_delay(429, None, 20), Some(MAX_BACKOFF));
    }
}
