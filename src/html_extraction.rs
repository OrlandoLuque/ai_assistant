//! HTML extraction module
//!
//! Extracts structured data from HTML content including tables, lists, links,
//! and metadata. Uses regex-based parsing without external HTML parser dependencies.
//!
//! # Features
//!
//! - **Metadata extraction**: Title, meta tags, Open Graph, Twitter Card, Schema.org
//! - **Content extraction**: Body text with script/style stripping
//! - **Link extraction**: All anchor links with external/internal classification
//! - **List extraction**: Ordered and unordered lists
//! - **Table extraction**: Delegates to `table_extraction` module
//! - **Selector-based matching**: CSS-like selectors for element targeting
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::html_extraction::{HtmlExtractor, HtmlExtractionConfig};
//!
//! let config = HtmlExtractionConfig::default();
//! let extractor = HtmlExtractor::new(config);
//! let result = extractor.extract("<html><head><title>Test</title></head><body><p>Hello</p></body></html>", None);
//! assert_eq!(result.metadata.title, Some("Test".to_string()));
//! ```

use std::collections::HashMap;

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::table_extraction::{ExtractedTable, TableExtractor, TableExtractorConfig};

// ============================================================================
// Data structures
// ============================================================================

/// CSS-like selector for targeting HTML elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlSelector {
    /// Tag name to match (e.g., "div", "p", "article")
    pub tag: Option<String>,
    /// CSS class to match
    pub class: Option<String>,
    /// Element ID to match
    pub id: Option<String>,
    /// Attribute name-value pair to match
    pub attribute: Option<(String, String)>,
}

impl HtmlSelector {
    /// Create a new empty selector.
    pub fn new() -> Self {
        Self {
            tag: None,
            class: None,
            id: None,
            attribute: None,
        }
    }

    /// Set the tag name.
    pub fn tag(mut self, name: &str) -> Self {
        self.tag = Some(name.to_string());
        self
    }

    /// Set the class name.
    pub fn class(mut self, name: &str) -> Self {
        self.class = Some(name.to_string());
        self
    }

    /// Set the ID.
    pub fn id(mut self, name: &str) -> Self {
        self.id = Some(name.to_string());
        self
    }

    /// Parse a CSS-like selector string.
    ///
    /// Supported formats:
    /// - `"div"` - tag name
    /// - `"div.class-name"` - tag with class
    /// - `".class-name"` - class only
    /// - `"#my-id"` - ID only
    /// - `"div#my-id"` - tag with ID
    /// - `"tag[attr=value]"` - tag with attribute
    pub fn parse(selector: &str) -> Self {
        let mut result = Self::new();
        let selector = selector.trim();

        if selector.is_empty() {
            return result;
        }

        // Check for attribute selector: tag[attr=value]
        let attr_re =
            Regex::new(r"^([a-zA-Z0-9-]*)?\[([a-zA-Z0-9-]+)=([^\]]*)\]$").expect("valid regex");
        if let Some(caps) = attr_re.captures(selector) {
            let tag_part = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if !tag_part.is_empty() {
                result.tag = Some(tag_part.to_string());
            }
            let attr_name = caps[2].to_string();
            let attr_value = caps[3].trim_matches('"').trim_matches('\'').to_string();
            result.attribute = Some((attr_name, attr_value));
            return result;
        }

        // Check for ID selector: #id or tag#id
        if selector.contains('#') {
            let parts: Vec<&str> = selector.splitn(2, '#').collect();
            if !parts[0].is_empty() {
                result.tag = Some(parts[0].to_string());
            }
            // ID might also have class after it, handle simple case
            let id_part = parts[1].split('.').next().unwrap_or(parts[1]);
            result.id = Some(id_part.to_string());
            return result;
        }

        // Check for class selector: .class or tag.class
        if selector.contains('.') {
            let parts: Vec<&str> = selector.splitn(2, '.').collect();
            if !parts[0].is_empty() {
                result.tag = Some(parts[0].to_string());
            }
            result.class = Some(parts[1].to_string());
            return result;
        }

        // Simple tag name
        result.tag = Some(selector.to_string());
        result
    }
}

impl Default for HtmlSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a parsed HTML element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlElement {
    /// Tag name
    pub tag: String,
    /// Element attributes
    pub attributes: HashMap<String, String>,
    /// Text content (tags stripped)
    pub text: String,
    /// Inner HTML content (with tags)
    pub inner_html: String,
    /// Child elements
    pub children: Vec<HtmlElement>,
}

/// Metadata extracted from an HTML document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlMetadata {
    /// Page title from <title> tag
    pub title: Option<String>,
    /// Meta description
    pub description: Option<String>,
    /// Meta keywords
    pub keywords: Vec<String>,
    /// Author meta tag
    pub author: Option<String>,
    /// Canonical URL
    pub canonical_url: Option<String>,
    /// Document language
    pub language: Option<String>,
    /// Open Graph metadata
    pub opengraph: HashMap<String, String>,
    /// Twitter Card metadata
    pub twitter_card: HashMap<String, String>,
    /// Schema.org JSON-LD blocks
    pub schema_org: Vec<String>,
    /// Favicon URL
    pub favicon: Option<String>,
    /// RSS/Atom feed URLs
    pub feed_urls: Vec<String>,
}

impl Default for HtmlMetadata {
    fn default() -> Self {
        Self {
            title: None,
            description: None,
            keywords: Vec::new(),
            author: None,
            canonical_url: None,
            language: None,
            opengraph: HashMap::new(),
            twitter_card: HashMap::new(),
            schema_org: Vec::new(),
            favicon: None,
            feed_urls: Vec::new(),
        }
    }
}

/// Rule for domain-specific content extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRule {
    /// Domain this rule applies to (e.g., "example.com")
    pub domain: String,
    /// Selector for main content area
    pub content_selector: HtmlSelector,
    /// Selector for the title element
    pub title_selector: Option<HtmlSelector>,
    /// Selector for the author element
    pub author_selector: Option<HtmlSelector>,
    /// Selector for the date element
    pub date_selector: Option<HtmlSelector>,
    /// Selectors for elements to exclude (ads, nav, etc.)
    pub exclude_selectors: Vec<HtmlSelector>,
}

/// Configuration for HTML extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlExtractionConfig {
    /// Domain-specific extraction rules
    pub rules: Vec<ExtractionRule>,
    /// Whether to extract metadata
    pub extract_metadata: bool,
    /// Whether to extract tables
    pub extract_tables: bool,
    /// Whether to extract lists
    pub extract_lists: bool,
    /// Whether to extract links
    pub extract_links: bool,
    /// Whether to strip script and style elements
    pub strip_scripts: bool,
}

impl Default for HtmlExtractionConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            extract_metadata: true,
            extract_tables: true,
            extract_lists: true,
            extract_links: true,
            strip_scripts: true,
        }
    }
}

/// An HTML list (ordered or unordered).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlList {
    /// Whether this is an ordered list (<ol>) or unordered (<ul>)
    pub ordered: bool,
    /// List items
    pub items: Vec<String>,
}

/// A hyperlink extracted from HTML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlLink {
    /// Link URL (resolved if base_url provided)
    pub url: String,
    /// Link text content
    pub text: String,
    /// Title attribute
    pub title: Option<String>,
    /// Whether link points to external domain
    pub is_external: bool,
}

/// Result of HTML extraction containing all extracted data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlExtractionResult {
    /// Extracted metadata
    pub metadata: HtmlMetadata,
    /// Main text content
    pub text_content: String,
    /// Extracted tables
    pub tables: Vec<ExtractedTable>,
    /// Extracted lists
    pub lists: Vec<HtmlList>,
    /// Extracted links
    pub links: Vec<HtmlLink>,
    /// Elements matched by selector rules
    pub matched_elements: Vec<HtmlElement>,
}

// ============================================================================
// HTML Extractor
// ============================================================================

/// Main HTML extraction engine.
///
/// Provides methods to extract metadata, text, links, lists, and tables
/// from HTML content using regex-based parsing.
pub struct HtmlExtractor {
    config: HtmlExtractionConfig,
}

impl HtmlExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: HtmlExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract all structured data from HTML content.
    ///
    /// This is the main entry point that runs all extraction steps based
    /// on configuration flags.
    pub fn extract(&self, html: &str, source_url: Option<&str>) -> HtmlExtractionResult {
        let metadata = if self.config.extract_metadata {
            self.extract_metadata(html)
        } else {
            HtmlMetadata::default()
        };

        let text_content = self.extract_text(html);

        let tables = if self.config.extract_tables {
            let table_config = TableExtractorConfig {
                detect_html: true,
                ..TableExtractorConfig::default()
            };
            let table_extractor = TableExtractor::new(table_config);
            table_extractor.extract_html_tables(html)
        } else {
            Vec::new()
        };

        let lists = if self.config.extract_lists {
            self.extract_lists(html)
        } else {
            Vec::new()
        };

        let links = if self.config.extract_links {
            self.extract_links(html, source_url)
        } else {
            Vec::new()
        };

        // Apply extraction rules for matched elements
        let mut matched_elements = Vec::new();
        if let Some(url) = source_url {
            let domain = extract_domain_from_url(url);
            for rule in &self.config.rules {
                if rule.domain == domain || domain.ends_with(&format!(".{}", rule.domain)) {
                    let elements = self.select(html, &rule.content_selector);
                    matched_elements.extend(elements);
                }
            }
        }

        HtmlExtractionResult {
            metadata,
            text_content,
            tables,
            lists,
            links,
            matched_elements,
        }
    }

    /// Extract metadata from HTML head section.
    ///
    /// Extracts title, meta tags, Open Graph, Twitter Card, Schema.org,
    /// favicon, feed URLs, and document language.
    pub fn extract_metadata(&self, html: &str) -> HtmlMetadata {
        let mut meta = HtmlMetadata::default();

        // Title
        let title_re = Regex::new(r"(?is)<title[^>]*>(.*?)</title>").expect("valid regex");
        if let Some(caps) = title_re.captures(html) {
            let title = strip_tags(&caps[1]).trim().to_string();
            if !title.is_empty() {
                meta.title = Some(decode_html_entities(&title));
            }
        }

        // Meta tags (both name="..." content="..." and content="..." name="..." orderings)
        let meta_re = Regex::new(
            r#"(?is)<meta\s+(?:[^>]*?\s+)?(?:name|property)=["']([^"']*)["']\s+(?:[^>]*?\s+)?content=["']([^"']*)["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        let meta_rev_re = Regex::new(
            r#"(?is)<meta\s+(?:[^>]*?\s+)?content=["']([^"']*)["']\s+(?:[^>]*?\s+)?(?:name|property)=["']([^"']*)["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");

        let mut meta_tags: HashMap<String, String> = HashMap::new();

        for caps in meta_re.captures_iter(html) {
            let name = caps[1].to_lowercase();
            let content = caps[2].to_string();
            meta_tags.insert(name, content);
        }
        for caps in meta_rev_re.captures_iter(html) {
            let content = caps[1].to_string();
            let name = caps[2].to_lowercase();
            meta_tags.entry(name).or_insert(content);
        }

        // Standard meta tags
        if let Some(desc) = meta_tags.get("description") {
            meta.description = Some(decode_html_entities(desc));
        }
        if let Some(kw) = meta_tags.get("keywords") {
            meta.keywords = kw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(author) = meta_tags.get("author") {
            meta.author = Some(decode_html_entities(author));
        }

        // Open Graph
        for (key, value) in &meta_tags {
            if let Some(og_key) = key.strip_prefix("og:") {
                meta.opengraph
                    .insert(og_key.to_string(), decode_html_entities(value));
            }
        }

        // Twitter Card
        for (key, value) in &meta_tags {
            if let Some(tw_key) = key.strip_prefix("twitter:") {
                meta.twitter_card
                    .insert(tw_key.to_string(), decode_html_entities(value));
            }
        }

        // Canonical URL
        let canonical_re = Regex::new(
            r#"(?is)<link[^>]*\s+rel=["']canonical["'][^>]*\s+href=["']([^"']*)["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        let canonical_rev_re = Regex::new(
            r#"(?is)<link[^>]*\s+href=["']([^"']*)["'][^>]*\s+rel=["']canonical["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        if let Some(caps) = canonical_re.captures(html) {
            meta.canonical_url = Some(caps[1].to_string());
        } else if let Some(caps) = canonical_rev_re.captures(html) {
            meta.canonical_url = Some(caps[1].to_string());
        }

        // Language
        let lang_re =
            Regex::new(r#"(?is)<html[^>]*\s+lang=["']([^"']*)["']"#).expect("valid regex");
        if let Some(caps) = lang_re.captures(html) {
            meta.language = Some(caps[1].to_string());
        }

        // Schema.org JSON-LD
        let schema_re =
            Regex::new(r#"(?is)<script[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>"#)
                .expect("valid regex");
        for caps in schema_re.captures_iter(html) {
            let json = caps[1].trim().to_string();
            if !json.is_empty() {
                meta.schema_org.push(json);
            }
        }

        // Favicon
        let favicon_re = Regex::new(
            r#"(?is)<link[^>]*\s+rel=["'](?:shortcut\s+)?icon["'][^>]*\s+href=["']([^"']*)["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        let favicon_rev_re = Regex::new(
            r#"(?is)<link[^>]*\s+href=["']([^"']*)["'][^>]*\s+rel=["'](?:shortcut\s+)?icon["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        if let Some(caps) = favicon_re.captures(html) {
            meta.favicon = Some(caps[1].to_string());
        } else if let Some(caps) = favicon_rev_re.captures(html) {
            meta.favicon = Some(caps[1].to_string());
        }

        // Feed URLs (RSS/Atom)
        let feed_re = Regex::new(
            r#"(?is)<link[^>]*\s+type=["']application/(?:rss|atom)\+xml["'][^>]*\s+href=["']([^"']*)["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        let feed_rev_re = Regex::new(
            r#"(?is)<link[^>]*\s+href=["']([^"']*)["'][^>]*\s+type=["']application/(?:rss|atom)\+xml["'][^>]*/?\s*>"#,
        )
        .expect("valid regex");
        for caps in feed_re.captures_iter(html) {
            meta.feed_urls.push(caps[1].to_string());
        }
        for caps in feed_rev_re.captures_iter(html) {
            let url = caps[1].to_string();
            if !meta.feed_urls.contains(&url) {
                meta.feed_urls.push(url);
            }
        }

        meta
    }

    /// Select elements matching a CSS-like selector.
    pub fn select(&self, html: &str, selector: &HtmlSelector) -> Vec<HtmlElement> {
        let matches = self.match_elements(html, selector);
        let mut elements = Vec::new();

        for (_start, _end, content) in matches {
            // Parse the matched element
            if let Some(element) = self.parse_element(&content) {
                elements.push(element);
            }
        }

        elements
    }

    /// Extract the main text content from HTML, stripped of all tags.
    ///
    /// Removes scripts and styles first, then strips all HTML tags
    /// and normalizes whitespace.
    pub fn extract_text(&self, html: &str) -> String {
        let cleaned = if self.config.strip_scripts {
            self.strip_scripts_and_styles(html)
        } else {
            html.to_string()
        };

        // Try to extract body content first
        let body_re = Regex::new(r"(?is)<body[^>]*>(.*?)</body>").expect("valid regex");
        let body_content = if let Some(caps) = body_re.captures(&cleaned) {
            caps[1].to_string()
        } else {
            cleaned.clone()
        };

        let text = strip_tags(&body_content);
        normalize_whitespace(&text)
    }

    /// Extract all hyperlinks from HTML.
    ///
    /// If `base_url` is provided, relative URLs will be resolved against it.
    /// Links are classified as external if their domain differs from the base URL domain.
    pub fn extract_links(&self, html: &str, base_url: Option<&str>) -> Vec<HtmlLink> {
        let link_re = Regex::new(r#"(?is)<a\s+([^>]*)>(.*?)</a>"#).expect("valid regex");
        let mut links = Vec::new();
        let base_domain = base_url
            .map(|u| extract_domain_from_url(u))
            .unwrap_or_default();

        for caps in link_re.captures_iter(html) {
            let attrs_str = &caps[1];
            let text = strip_tags(&caps[2]).trim().to_string();

            // Extract href
            let href = extract_attribute(attrs_str, "href");
            if href.is_none() {
                continue;
            }
            let href = href.expect("href verified above");

            // Skip anchors and javascript
            if href.starts_with('#') || href.starts_with("javascript:") {
                continue;
            }

            // Resolve URL
            let resolved = if let Some(base) = base_url {
                resolve_url(&href, base)
            } else {
                href.clone()
            };

            // Extract title
            let title = extract_attribute(attrs_str, "title");

            // Determine if external
            let link_domain = extract_domain_from_url(&resolved);
            let is_external = if base_domain.is_empty() {
                false
            } else {
                !link_domain.is_empty() && link_domain != base_domain
            };

            links.push(HtmlLink {
                url: resolved,
                text,
                title,
                is_external,
            });
        }

        links
    }

    /// Extract all ordered and unordered lists from HTML.
    pub fn extract_lists(&self, html: &str) -> Vec<HtmlList> {
        let ul_re = Regex::new(r"(?is)<ul[^>]*>(.*?)</ul>").expect("valid regex");
        let ol_re = Regex::new(r"(?is)<ol[^>]*>(.*?)</ol>").expect("valid regex");
        let li_re = Regex::new(r"(?is)<li[^>]*>(.*?)</li>").expect("valid regex");
        let mut lists = Vec::new();

        // Extract unordered lists
        for caps in ul_re.captures_iter(html) {
            let list_content = &caps[1];
            let items: Vec<String> = li_re
                .captures_iter(list_content)
                .map(|li| strip_tags(&li[1]).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !items.is_empty() {
                lists.push(HtmlList {
                    ordered: false,
                    items,
                });
            }
        }

        // Extract ordered lists
        for caps in ol_re.captures_iter(html) {
            let list_content = &caps[1];
            let items: Vec<String> = li_re
                .captures_iter(list_content)
                .map(|li| strip_tags(&li[1]).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !items.is_empty() {
                lists.push(HtmlList {
                    ordered: true,
                    items,
                });
            }
        }

        lists
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    /// Remove <script> and <style> elements and their content.
    fn strip_scripts_and_styles(&self, html: &str) -> String {
        let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("valid regex");
        let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("valid regex");
        let noscript_re = Regex::new(r"(?is)<noscript[^>]*>.*?</noscript>").expect("valid regex");

        let result = script_re.replace_all(html, "");
        let result = style_re.replace_all(&result, "");
        let result = noscript_re.replace_all(&result, "");
        result.to_string()
    }

    /// Find elements matching a selector, returning (start, end, content) tuples.
    fn match_elements(&self, html: &str, selector: &HtmlSelector) -> Vec<(usize, usize, String)> {
        let tag_pattern = selector.tag.as_deref().unwrap_or("[a-zA-Z][a-zA-Z0-9]*");

        // Build attribute constraints
        let mut attr_patterns = Vec::new();

        if let Some(ref class) = selector.class {
            attr_patterns.push(format!(
                r#"[^>]*\s+class=["'][^"']*\b{}\b[^"']*["']"#,
                regex::escape(class)
            ));
        }

        if let Some(ref id) = selector.id {
            attr_patterns.push(format!(r#"[^>]*\s+id=["']{}["']"#, regex::escape(id)));
        }

        if let Some((ref attr_name, ref attr_value)) = selector.attribute {
            attr_patterns.push(format!(
                r#"[^>]*\s+{}=["']{}["']"#,
                regex::escape(attr_name),
                regex::escape(attr_value)
            ));
        }

        let attrs_regex = if attr_patterns.is_empty() {
            "[^>]*".to_string()
        } else {
            // Combine attribute patterns - element must match all
            attr_patterns.join("")
        };

        // Build the opening tag regex
        let open_pattern = format!(
            r"(?is)<({tag})({attrs})\s*/?\s*>",
            tag = tag_pattern,
            attrs = attrs_regex,
        );

        let open_re = match Regex::new(&open_pattern) {
            Ok(re) => re,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();

        for open_match in open_re.find_iter(html) {
            let start = open_match.start();
            let tag_name = open_re
                .captures(&html[start..])
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_lowercase())
                .unwrap_or_default();

            // Check if it's a self-closing tag
            if open_match.as_str().ends_with("/>") {
                results.push((start, open_match.end(), open_match.as_str().to_string()));
                continue;
            }

            // Find the matching closing tag (handling nesting)
            if let Some(end) = find_closing_tag(html, open_match.end(), &tag_name) {
                let full = &html[start..end];
                results.push((start, end, full.to_string()));
            }
        }

        results
    }

    /// Parse an element string into an HtmlElement.
    fn parse_element(&self, html: &str) -> Option<HtmlElement> {
        let open_re = Regex::new(r"(?is)^<([a-zA-Z][a-zA-Z0-9]*)\s*([^>]*)>").expect("valid regex");

        let caps = open_re.captures(html)?;
        let tag = caps[1].to_lowercase();
        let attrs_str = &caps[2];

        // Parse attributes
        let attr_re =
            Regex::new(r#"([a-zA-Z][a-zA-Z0-9_-]*)=["']([^"']*)["']"#).expect("valid regex");
        let mut attributes = HashMap::new();
        for attr_caps in attr_re.captures_iter(attrs_str) {
            attributes.insert(attr_caps[1].to_lowercase(), attr_caps[2].to_string());
        }

        // Extract inner HTML
        let close_tag = format!("</{}>", tag);
        let inner_start = caps.get(0).map(|m| m.end()).unwrap_or(0);
        let inner_html = if let Some(close_pos) = html.to_lowercase().rfind(&close_tag) {
            html[inner_start..close_pos].to_string()
        } else {
            html[inner_start..].to_string()
        };

        let text = strip_tags(&inner_html).trim().to_string();

        // Parse direct children (simplified - first level only)
        let child_re = Regex::new(r"(?is)<([a-zA-Z][a-zA-Z0-9]*)\s*[^>]*>").expect("valid regex");
        let mut children = Vec::new();
        for child_match in child_re.captures_iter(&inner_html) {
            let child_tag = child_match[1].to_lowercase();
            let child_start = child_match.get(0).expect("capture group 0").start();
            if let Some(child_end) = find_closing_tag(
                &inner_html,
                child_match.get(0).expect("capture group 0").end(),
                &child_tag,
            ) {
                let child_html = &inner_html[child_start..child_end];
                if let Some(child_elem) = self.parse_element(child_html) {
                    children.push(child_elem);
                }
            }
        }

        Some(HtmlElement {
            tag,
            attributes,
            text,
            inner_html,
            children,
        })
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract an attribute value from an HTML attributes string.
fn extract_attribute(attrs: &str, attr_name: &str) -> Option<String> {
    let pattern = format!(r#"(?i){}=["']([^"']*)["']"#, regex::escape(attr_name));
    let re = Regex::new(&pattern).ok()?;
    re.captures(attrs).map(|c| c[1].to_string())
}

/// Strip all HTML tags from a string, leaving only text content.
fn strip_tags(html: &str) -> String {
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    tag_re.replace_all(html, "").to_string()
}

/// Normalize whitespace: collapse multiple spaces/newlines into single spaces.
fn normalize_whitespace(text: &str) -> String {
    let ws_re = Regex::new(r"\s+").expect("valid regex");
    ws_re.replace_all(text.trim(), " ").to_string()
}

/// Resolve a relative URL against a base URL.
fn resolve_url(relative: &str, base: &str) -> String {
    // Already absolute
    if relative.starts_with("http://") || relative.starts_with("https://") {
        return relative.to_string();
    }

    // Protocol-relative
    if relative.starts_with("//") {
        let protocol = if base.starts_with("https") {
            "https:"
        } else {
            "http:"
        };
        return format!("{}{}", protocol, relative);
    }

    // Extract base components
    let base_parts: Vec<&str> = base.splitn(4, '/').collect();
    if base_parts.len() < 3 {
        return relative.to_string();
    }

    let scheme_and_host = format!("{}//{}", base_parts[0], base_parts[2]);

    // Absolute path
    if relative.starts_with('/') {
        return format!("{}{}", scheme_and_host, relative);
    }

    // Relative path - resolve against base directory
    let base_path = if base_parts.len() > 3 {
        let path = base_parts[3..].join("/");
        if let Some(last_slash) = path.rfind('/') {
            format!("/{}/", &path[..last_slash])
        } else {
            "/".to_string()
        }
    } else {
        "/".to_string()
    };

    format!("{}{}{}", scheme_and_host, base_path, relative)
}

/// Extract the domain from a URL.
fn extract_domain_from_url(url: &str) -> String {
    url.split("//")
        .nth(1)
        .and_then(|s| s.split('/').next())
        .and_then(|s| s.split(':').next())
        .unwrap_or("")
        .to_lowercase()
}

/// Find the position of the closing tag, handling nesting.
fn find_closing_tag(html: &str, start: usize, tag: &str) -> Option<usize> {
    let open_re = Regex::new(&format!(r"(?is)<{}\b[^>]*>", regex::escape(tag))).ok()?;
    let close_re = Regex::new(&format!(r"(?is)</{}\s*>", regex::escape(tag))).ok()?;

    let mut depth = 1;
    let mut pos = start;

    while depth > 0 && pos < html.len() {
        let next_open = open_re.find_at(html, pos);
        let next_close = close_re.find_at(html, pos);

        match (next_open, next_close) {
            (_, None) => return None,
            (None, Some(close_match)) => {
                depth -= 1;
                if depth == 0 {
                    return Some(close_match.end());
                }
                pos = close_match.end();
            }
            (Some(open_match), Some(close_match)) => {
                if open_match.start() < close_match.start() {
                    depth += 1;
                    pos = open_match.end();
                } else {
                    depth -= 1;
                    if depth == 0 {
                        return Some(close_match.end());
                    }
                    pos = close_match.end();
                }
            }
        }
    }

    None
}

/// Decode common HTML entities.
fn decode_html_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/")
        .replace("&hellip;", "...")
        .replace("&mdash;", "\u{2014}")
        .replace("&ndash;", "\u{2013}")
        .replace("&copy;", "\u{00A9}")
        .replace("&reg;", "\u{00AE}")
        .replace("&trade;", "\u{2122}")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_parse_tag() {
        let sel = HtmlSelector::parse("div");
        assert_eq!(sel.tag, Some("div".to_string()));
        assert_eq!(sel.class, None);
        assert_eq!(sel.id, None);
    }

    #[test]
    fn test_selector_parse_class() {
        let sel = HtmlSelector::parse("div.content");
        assert_eq!(sel.tag, Some("div".to_string()));
        assert_eq!(sel.class, Some("content".to_string()));

        let sel2 = HtmlSelector::parse(".sidebar");
        assert_eq!(sel2.tag, None);
        assert_eq!(sel2.class, Some("sidebar".to_string()));
    }

    #[test]
    fn test_selector_parse_id() {
        let sel = HtmlSelector::parse("#main");
        assert_eq!(sel.tag, None);
        assert_eq!(sel.id, Some("main".to_string()));

        let sel2 = HtmlSelector::parse("section#content");
        assert_eq!(sel2.tag, Some("section".to_string()));
        assert_eq!(sel2.id, Some("content".to_string()));
    }

    #[test]
    fn test_selector_parse_attribute() {
        let sel = HtmlSelector::parse("input[type=text]");
        assert_eq!(sel.tag, Some("input".to_string()));
        assert_eq!(
            sel.attribute,
            Some(("type".to_string(), "text".to_string()))
        );

        let sel2 = HtmlSelector::parse("[data-role=nav]");
        assert_eq!(sel2.tag, None);
        assert_eq!(
            sel2.attribute,
            Some(("data-role".to_string(), "nav".to_string()))
        );
    }

    #[test]
    fn test_extract_metadata_title() {
        let html = r#"<html><head><title>My Page Title</title></head><body></body></html>"#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.title, Some("My Page Title".to_string()));
    }

    #[test]
    fn test_extract_metadata_meta_tags() {
        let html = r#"
            <html>
            <head>
                <meta name="description" content="A test page">
                <meta name="keywords" content="rust, html, parser">
                <meta name="author" content="Test Author">
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.description, Some("A test page".to_string()));
        assert_eq!(
            meta.keywords,
            vec!["rust".to_string(), "html".to_string(), "parser".to_string()]
        );
        assert_eq!(meta.author, Some("Test Author".to_string()));
    }

    #[test]
    fn test_extract_metadata_opengraph() {
        let html = r#"
            <html>
            <head>
                <meta property="og:title" content="OG Title">
                <meta property="og:description" content="OG Description">
                <meta property="og:image" content="https://example.com/img.jpg">
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.opengraph.get("title"), Some(&"OG Title".to_string()));
        assert_eq!(
            meta.opengraph.get("description"),
            Some(&"OG Description".to_string())
        );
        assert_eq!(
            meta.opengraph.get("image"),
            Some(&"https://example.com/img.jpg".to_string())
        );
    }

    #[test]
    fn test_extract_metadata_language_and_favicon() {
        let html = r#"
            <html lang="en-US">
            <head>
                <link rel="icon" href="/favicon.ico">
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.language, Some("en-US".to_string()));
        assert_eq!(meta.favicon, Some("/favicon.ico".to_string()));
    }

    #[test]
    fn test_extract_metadata_schema_org() {
        let html = r#"
            <html>
            <head>
                <script type="application/ld+json">
                {"@type": "WebPage", "name": "Test"}
                </script>
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.schema_org.len(), 1);
        assert!(meta.schema_org[0].contains("WebPage"));
    }

    #[test]
    fn test_extract_links() {
        let html = r#"
            <body>
                <a href="https://example.com/page1" title="Page One">Page 1</a>
                <a href="/about">About Us</a>
                <a href="https://other.com/ext">External</a>
            </body>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let links = extractor.extract_links(html, Some("https://example.com/"));

        assert_eq!(links.len(), 3);
        assert_eq!(links[0].url, "https://example.com/page1");
        assert_eq!(links[0].text, "Page 1");
        assert_eq!(links[0].title, Some("Page One".to_string()));
        assert!(!links[0].is_external);

        assert_eq!(links[1].url, "https://example.com/about");
        assert!(!links[1].is_external);

        assert_eq!(links[2].url, "https://other.com/ext");
        assert!(links[2].is_external);
    }

    #[test]
    fn test_extract_lists() {
        let html = r#"
            <body>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                    <li>Item 3</li>
                </ul>
                <ol>
                    <li>First</li>
                    <li>Second</li>
                </ol>
            </body>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let lists = extractor.extract_lists(html);

        assert_eq!(lists.len(), 2);
        assert!(!lists[0].ordered);
        assert_eq!(lists[0].items, vec!["Item 1", "Item 2", "Item 3"]);
        assert!(lists[1].ordered);
        assert_eq!(lists[1].items, vec!["First", "Second"]);
    }

    #[test]
    fn test_extract_text_strips_scripts() {
        let html = r#"
            <html>
            <head><script>var x = 1;</script></head>
            <body>
                <p>Hello World</p>
                <script>console.log('hidden');</script>
                <style>.x { color: red; }</style>
                <p>Visible text</p>
            </body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let text = extractor.extract_text(html);

        assert!(text.contains("Hello World"));
        assert!(text.contains("Visible text"));
        assert!(!text.contains("var x"));
        assert!(!text.contains("console.log"));
        assert!(!text.contains("color: red"));
    }

    #[test]
    fn test_full_extraction() {
        let html = r#"
            <html lang="en">
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
            </head>
            <body>
                <h1>Welcome</h1>
                <p>Hello World</p>
                <ul>
                    <li>One</li>
                    <li>Two</li>
                </ul>
                <a href="https://example.com">Example</a>
                <table>
                    <tr><th>Name</th><th>Value</th></tr>
                    <tr><td>A</td><td>1</td></tr>
                </table>
            </body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let result = extractor.extract(html, Some("https://mysite.com/page"));

        assert_eq!(result.metadata.title, Some("Test Page".to_string()));
        assert_eq!(result.metadata.language, Some("en".to_string()));
        assert!(result.text_content.contains("Hello World"));
        assert_eq!(result.lists.len(), 1);
        assert_eq!(result.lists[0].items, vec!["One", "Two"]);
        assert_eq!(result.links.len(), 1);
        assert!(result.links[0].is_external);
        assert_eq!(result.tables.len(), 1);
        assert_eq!(
            result.tables[0].headers,
            vec!["Name".to_string(), "Value".to_string()]
        );
    }

    #[test]
    fn test_resolve_url() {
        assert_eq!(
            resolve_url("/path/page", "https://example.com/old/page"),
            "https://example.com/path/page"
        );
        assert_eq!(
            resolve_url("https://other.com/abs", "https://example.com/"),
            "https://other.com/abs"
        );
        assert_eq!(
            resolve_url("//cdn.example.com/js/app.js", "https://example.com/"),
            "https://cdn.example.com/js/app.js"
        );
        assert_eq!(
            resolve_url("relative.html", "https://example.com/dir/page.html"),
            "https://example.com/dir/relative.html"
        );
    }

    #[test]
    fn test_strip_tags() {
        assert_eq!(strip_tags("<p>Hello</p>"), "Hello");
        assert_eq!(strip_tags("<a href='x'>link</a> text"), "link text");
        assert_eq!(strip_tags("no tags here"), "no tags here");
        assert_eq!(strip_tags("<div><span>nested</span></div>"), "nested");
    }

    #[test]
    fn test_html_entities_decoded() {
        let html = r#"<html><head><title>Tom &amp; Jerry</title></head><body></body></html>"#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.title, Some("Tom & Jerry".to_string()));
    }

    #[test]
    fn test_select_by_class() {
        let html = r#"
            <div class="container">
                <p class="highlight">Important text</p>
                <p>Normal text</p>
                <p class="highlight extra">Also important</p>
            </div>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let selector = HtmlSelector::parse("p.highlight");
        let elements = extractor.select(html, &selector);

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].text, "Important text");
        assert_eq!(elements[1].text, "Also important");
    }

    #[test]
    fn test_select_by_id() {
        let html = r#"
            <div>
                <section id="main">
                    <p>Main content here</p>
                </section>
                <section id="sidebar">
                    <p>Sidebar content</p>
                </section>
            </div>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let selector = HtmlSelector::parse("section#main");
        let elements = extractor.select(html, &selector);

        assert_eq!(elements.len(), 1);
        assert!(elements[0].text.contains("Main content here"));
    }

    #[test]
    fn test_extract_tables() {
        let html = r#"
            <table>
                <caption>Test Table</caption>
                <thead>
                    <tr><th>Col1</th><th>Col2</th></tr>
                </thead>
                <tbody>
                    <tr><td>A</td><td>B</td></tr>
                    <tr><td>C</td><td>D</td></tr>
                </tbody>
            </table>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let result = extractor.extract(html, None);

        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0].caption, Some("Test Table".to_string()));
        assert_eq!(
            result.tables[0].headers,
            vec!["Col1".to_string(), "Col2".to_string()]
        );
        assert_eq!(result.tables[0].rows.len(), 2);
        assert_eq!(result.tables[0].cell(0, 0), Some("A"));
        assert_eq!(result.tables[0].cell(0, 1), Some("B"));
        assert_eq!(result.tables[0].cell(1, 0), Some("C"));
        assert_eq!(result.tables[0].cell(1, 1), Some("D"));
    }

    #[test]
    fn test_extraction_config_flags() {
        let html = r#"
            <html>
            <head><title>Test</title></head>
            <body>
                <ul><li>item</li></ul>
                <a href="http://x.com">link</a>
            </body>
            </html>
        "#;

        let config = HtmlExtractionConfig {
            rules: Vec::new(),
            extract_metadata: false,
            extract_tables: false,
            extract_lists: false,
            extract_links: false,
            strip_scripts: true,
        };
        let extractor = HtmlExtractor::new(config);
        let result = extractor.extract(html, None);

        assert_eq!(result.metadata.title, None);
        assert!(result.tables.is_empty());
        assert!(result.lists.is_empty());
        assert!(result.links.is_empty());
    }

    #[test]
    fn test_feed_urls_extraction() {
        let html = r#"
            <html>
            <head>
                <link type="application/rss+xml" href="/feed.xml">
                <link type="application/atom+xml" href="/atom.xml">
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(meta.feed_urls.len(), 2);
        assert!(meta.feed_urls.contains(&"/feed.xml".to_string()));
        assert!(meta.feed_urls.contains(&"/atom.xml".to_string()));
    }

    #[test]
    fn test_twitter_card_extraction() {
        let html = r#"
            <html>
            <head>
                <meta name="twitter:card" content="summary_large_image">
                <meta name="twitter:site" content="@example">
                <meta name="twitter:title" content="Twitter Title">
            </head>
            <body></body>
            </html>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let meta = extractor.extract_metadata(html);
        assert_eq!(
            meta.twitter_card.get("card"),
            Some(&"summary_large_image".to_string())
        );
        assert_eq!(meta.twitter_card.get("site"), Some(&"@example".to_string()));
        assert_eq!(
            meta.twitter_card.get("title"),
            Some(&"Twitter Title".to_string())
        );
    }

    #[test]
    fn test_links_skip_javascript_and_anchors() {
        let html = r##"
            <body>
                <a href="javascript:void(0)">JS Link</a>
                <a href="#section">Anchor</a>
                <a href="/real-page">Real Link</a>
            </body>
        "##;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let links = extractor.extract_links(html, Some("https://example.com/"));
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].url, "https://example.com/real-page");
    }

    #[test]
    fn test_strip_scripts_and_styles() {
        let html = r#"
            <p>Before</p>
            <script>alert('evil');</script>
            <p>Middle</p>
            <style>.hidden { display: none; }</style>
            <p>After</p>
        "#;
        let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
        let result = extractor.strip_scripts_and_styles(html);
        assert!(!result.contains("alert"));
        assert!(!result.contains("display: none"));
        assert!(result.contains("Before"));
        assert!(result.contains("Middle"));
        assert!(result.contains("After"));
    }

    #[test]
    fn test_extraction_rule_application() {
        let html = r#"
            <html>
            <body>
                <article class="post">
                    <h1>Title</h1>
                    <nav>Skip this</nav>
                    <p>Article content here</p>
                </article>
                <aside>Sidebar</aside>
            </body>
            </html>
        "#;

        let rule = ExtractionRule {
            domain: "example.com".to_string(),
            content_selector: HtmlSelector::parse("article.post"),
            title_selector: None,
            author_selector: None,
            date_selector: None,
            exclude_selectors: vec![HtmlSelector::parse("nav")],
        };

        let config = HtmlExtractionConfig {
            rules: vec![rule],
            extract_metadata: true,
            extract_tables: false,
            extract_lists: false,
            extract_links: false,
            strip_scripts: true,
        };

        let extractor = HtmlExtractor::new(config);
        let result = extractor.extract(html, Some("https://example.com/post/1"));

        assert!(!result.matched_elements.is_empty());
        assert!(result.matched_elements[0].text.contains("Article content"));
    }
}
