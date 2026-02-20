//! RSS/Atom feed monitoring
//!
//! Parses RSS 2.0 and Atom feeds using regex-based XML extraction and monitors
//! feeds for new entries. Supports automatic format detection, date parsing with
//! multiple format variants, and stateful tracking of seen entries.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry parsed from a feed (RSS item or Atom entry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEntry {
    /// Title of the entry.
    pub title: String,
    /// URL/link of the entry.
    pub url: String,
    /// Short summary or description.
    pub summary: String,
    /// Full content if available.
    pub content: Option<String>,
    /// Publication date.
    pub published: Option<DateTime<Utc>>,
    /// Last update date.
    pub updated: Option<DateTime<Utc>>,
    /// Author name.
    pub author: Option<String>,
    /// Category tags.
    pub categories: Vec<String>,
    /// Unique identifier (guid or id).
    pub id: String,
}

/// Metadata about the feed itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedMetadata {
    /// Title of the feed.
    pub title: String,
    /// Description of the feed.
    pub description: String,
    /// URL of the feed itself.
    pub feed_url: String,
    /// URL of the associated website.
    pub site_url: String,
    /// Language code (e.g. "en-us").
    pub language: Option<String>,
    /// Last build or update date.
    pub last_build_date: Option<DateTime<Utc>>,
    /// Detected feed format.
    pub format: FeedFormat,
}

/// Feed format type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedFormat {
    /// RSS 2.0 format.
    Rss2,
    /// Atom format.
    Atom,
    /// Could not be determined.
    Unknown,
}

/// A fully parsed feed containing metadata and entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFeed {
    /// Feed-level metadata.
    pub metadata: FeedMetadata,
    /// Parsed entries.
    pub entries: Vec<FeedEntry>,
    /// Number of entries parsed.
    pub entry_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the feed monitor.
///
/// Durations are stored as integer milliseconds (u64) for reliable serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedMonitorConfig {
    /// Map of feed name to feed URL.
    pub feeds: HashMap<String, String>,
    /// How often to check feeds (seconds).
    pub check_interval_secs: u64,
    /// Maximum entries to keep per feed.
    pub max_entries_per_feed: usize,
    /// User agent string for HTTP requests.
    pub user_agent: String,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for FeedMonitorConfig {
    fn default() -> Self {
        Self {
            feeds: HashMap::new(),
            check_interval_secs: 3600,
            max_entries_per_feed: 100,
            user_agent: "feed-monitor/1.0".to_string(),
            timeout_ms: 30000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent state for the feed monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedMonitorState {
    /// Last time each feed was checked.
    pub last_checked: HashMap<String, DateTime<Utc>>,
    /// Known entry IDs per feed.
    pub known_entries: HashMap<String, Vec<String>>,
    /// Most recent entry date per feed.
    pub last_entry_date: HashMap<String, DateTime<Utc>>,
}

impl Default for FeedMonitorState {
    fn default() -> Self {
        Self {
            last_checked: HashMap::new(),
            known_entries: HashMap::new(),
            last_entry_date: HashMap::new(),
        }
    }
}

/// Result of checking a single feed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedCheckResult {
    /// Name of the feed that was checked.
    pub feed_name: String,
    /// Entries that are new since the last check.
    pub new_entries: Vec<FeedEntry>,
    /// Total number of entries in the feed.
    pub total_entries: usize,
    /// Whether the check succeeded.
    pub success: bool,
    /// Error message if the check failed.
    pub error: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Feed Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless feed parser with regex-based XML extraction.
///
/// All methods are associated functions (no `self`). Supports RSS 2.0 and Atom
/// with automatic format detection.
pub struct FeedParser;

impl FeedParser {
    /// Parse feed content, auto-detecting the format.
    pub fn parse(content: &str) -> Result<ParsedFeed> {
        let format = Self::detect_format(content);
        match format {
            FeedFormat::Rss2 => Self::parse_rss(content),
            FeedFormat::Atom => Self::parse_atom(content),
            FeedFormat::Unknown => Err(anyhow!("Unable to detect feed format")),
        }
    }

    /// Detect whether content is RSS 2.0, Atom, or unknown.
    pub fn detect_format(content: &str) -> FeedFormat {
        let lower = content.to_lowercase();
        if lower.contains("<rss") {
            FeedFormat::Rss2
        } else if lower.contains("<feed") {
            FeedFormat::Atom
        } else {
            FeedFormat::Unknown
        }
    }

    /// Parse RSS 2.0 feed content.
    fn parse_rss(content: &str) -> Result<ParsedFeed> {
        // Extract the <channel> block
        let channel = Self::extract_tag_content(content, "channel")
            .ok_or_else(|| anyhow!("No <channel> element found in RSS feed"))?;

        // Parse channel metadata
        let title = Self::extract_tag_content(&channel, "title").unwrap_or_default();
        let description = Self::extract_tag_content(&channel, "description").unwrap_or_default();
        let link = Self::extract_tag_content(&channel, "link").unwrap_or_default();
        let language = Self::extract_tag_content(&channel, "language");
        let last_build_date =
            Self::extract_tag_content(&channel, "lastBuildDate").and_then(|d| Self::parse_date(&d));

        let metadata = FeedMetadata {
            title,
            description,
            feed_url: String::new(),
            site_url: link,
            language,
            last_build_date,
            format: FeedFormat::Rss2,
        };

        // Parse items
        let items_xml = Self::extract_all_tag_content(content, "item");
        let mut entries = Vec::new();

        for item_xml in &items_xml {
            let item_title = Self::extract_tag_content(item_xml, "title").unwrap_or_default();
            let item_link = Self::extract_tag_content(item_xml, "link").unwrap_or_default();
            let item_description =
                Self::extract_tag_content(item_xml, "description").unwrap_or_default();
            let pub_date =
                Self::extract_tag_content(item_xml, "pubDate").and_then(|d| Self::parse_date(&d));
            let guid =
                Self::extract_tag_content(item_xml, "guid").unwrap_or_else(|| item_link.clone());
            let author = Self::extract_tag_content(item_xml, "author")
                .or_else(|| Self::extract_tag_content(item_xml, "dc:creator"));
            let content_encoded = Self::extract_tag_content(item_xml, "content:encoded");

            let categories = Self::extract_all_tag_content(item_xml, "category");

            entries.push(FeedEntry {
                title: item_title,
                url: item_link,
                summary: item_description,
                content: content_encoded,
                published: pub_date,
                updated: None,
                author,
                categories,
                id: guid,
            });
        }

        let entry_count = entries.len();
        Ok(ParsedFeed {
            metadata,
            entries,
            entry_count,
        })
    }

    /// Parse Atom feed content.
    fn parse_atom(content: &str) -> Result<ParsedFeed> {
        // Parse feed-level metadata
        let title = Self::extract_tag_content(content, "title").unwrap_or_default();
        let subtitle = Self::extract_tag_content(content, "subtitle").unwrap_or_default();

        // Extract feed link (site link with rel="alternate" or first link)
        let site_url = Self::extract_atom_link(content, Some("alternate"))
            .or_else(|| Self::extract_atom_link(content, None))
            .unwrap_or_default();

        let feed_url = Self::extract_atom_link(content, Some("self")).unwrap_or_default();

        let updated_str = Self::extract_tag_content(content, "updated");
        let last_build_date = updated_str.and_then(|d| Self::parse_date(&d));

        let metadata = FeedMetadata {
            title,
            description: subtitle,
            feed_url,
            site_url,
            language: None,
            last_build_date,
            format: FeedFormat::Atom,
        };

        // Parse entries
        let entries_xml = Self::extract_all_tag_content(content, "entry");
        let mut entries = Vec::new();

        for entry_xml in &entries_xml {
            let entry_title = Self::extract_tag_content(entry_xml, "title").unwrap_or_default();
            let entry_link = Self::extract_atom_link(entry_xml, Some("alternate"))
                .or_else(|| Self::extract_atom_link(entry_xml, None))
                .or_else(|| Self::extract_atom_link_href(entry_xml))
                .unwrap_or_default();
            let summary = Self::extract_tag_content(entry_xml, "summary").unwrap_or_default();
            let content = Self::extract_tag_content(entry_xml, "content");
            let published = Self::extract_tag_content(entry_xml, "published")
                .and_then(|d| Self::parse_date(&d));
            let updated =
                Self::extract_tag_content(entry_xml, "updated").and_then(|d| Self::parse_date(&d));
            let id =
                Self::extract_tag_content(entry_xml, "id").unwrap_or_else(|| entry_link.clone());

            // Author: <author><name>...</name></author>
            let author = Self::extract_tag_content(entry_xml, "author")
                .and_then(|a| Self::extract_tag_content(&a, "name"));

            let categories = Self::extract_atom_categories(entry_xml);

            entries.push(FeedEntry {
                title: entry_title,
                url: entry_link,
                summary,
                content,
                published,
                updated,
                author,
                categories,
                id,
            });
        }

        let entry_count = entries.len();
        Ok(ParsedFeed {
            metadata,
            entries,
            entry_count,
        })
    }

    /// Extract the text content of the first occurrence of a given XML tag.
    ///
    /// Handles both `<tag>content</tag>` and `<tag attr="val">content</tag>`.
    fn extract_tag_content(xml: &str, tag: &str) -> Option<String> {
        let pattern = format!(
            r"(?si)<{tag}(?:\s[^>]*)?>(.+?)</{tag}>",
            tag = regex::escape(tag)
        );
        let re = Regex::new(&pattern).ok()?;
        re.captures(xml).map(|c| {
            let text = c
                .get(1)
                .expect("capture group 1")
                .as_str()
                .trim()
                .to_string();
            // Strip CDATA wrapping if present
            strip_cdata(&text)
        })
    }

    /// Extract text content of all occurrences of a given XML tag.
    fn extract_all_tag_content(xml: &str, tag: &str) -> Vec<String> {
        let pattern = format!(
            r"(?si)<{tag}(?:\s[^>]*)?>(.+?)</{tag}>",
            tag = regex::escape(tag)
        );
        let re = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        re.captures_iter(xml)
            .map(|c| {
                let text = c
                    .get(1)
                    .expect("capture group 1")
                    .as_str()
                    .trim()
                    .to_string();
                strip_cdata(&text)
            })
            .collect()
    }

    /// Extract the value of a named attribute from a tag string.
    fn extract_attribute(tag_content: &str, attr: &str) -> Option<String> {
        let pattern = format!(
            r#"(?i){attr}\s*=\s*["']([^"']+)["']"#,
            attr = regex::escape(attr)
        );
        let re = Regex::new(&pattern).ok()?;
        re.captures(tag_content)
            .map(|c| c.get(1).expect("capture group 1").as_str().to_string())
    }

    /// Parse a date string trying multiple common formats.
    fn parse_date(date_str: &str) -> Option<DateTime<Utc>> {
        let trimmed = date_str.trim();

        // Try RFC 3339 / ISO 8601 (most common in Atom)
        if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
            return Some(dt.with_timezone(&Utc));
        }

        // Try RFC 2822 (most common in RSS)
        if let Ok(dt) = DateTime::parse_from_rfc2822(trimmed) {
            return Some(dt.with_timezone(&Utc));
        }

        // Try ISO 8601 variants without timezone
        let iso_formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ];

        for fmt in &iso_formats {
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(trimmed, fmt) {
                return Some(DateTime::from_naive_utc_and_offset(dt, Utc));
            }
            // Also try as a date-only format
            if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, fmt) {
                let dt = d.and_hms_opt(0, 0, 0)?;
                return Some(DateTime::from_naive_utc_and_offset(dt, Utc));
            }
        }

        None
    }

    /// Extract href from an Atom <link> element, optionally filtering by rel attribute.
    fn extract_atom_link(xml: &str, rel: Option<&str>) -> Option<String> {
        let pattern = r#"(?si)<link\s([^>]*?)/?>"#;
        let re = Regex::new(pattern).ok()?;

        for caps in re.captures_iter(xml) {
            let attrs = caps.get(1)?.as_str();
            if let Some(expected_rel) = rel {
                if let Some(found_rel) = Self::extract_attribute(attrs, "rel") {
                    if found_rel.to_lowercase() == expected_rel.to_lowercase() {
                        return Self::extract_attribute(attrs, "href");
                    }
                }
            } else {
                return Self::extract_attribute(attrs, "href");
            }
        }
        None
    }

    /// Fallback: extract href from any <link> tag.
    fn extract_atom_link_href(xml: &str) -> Option<String> {
        let pattern = r#"(?si)<link\s[^>]*href\s*=\s*["']([^"']+)["'][^>]*/?\s*>"#;
        let re = Regex::new(pattern).ok()?;
        re.captures(xml)
            .map(|c| c.get(1).expect("capture group 1").as_str().to_string())
    }

    /// Extract category terms from Atom <category term="..."/> elements.
    fn extract_atom_categories(xml: &str) -> Vec<String> {
        let pattern = r#"(?si)<category\s[^>]*term\s*=\s*["']([^"']+)["'][^>]*/?\s*>"#;
        let re = match Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        re.captures_iter(xml)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }
}

/// Strip CDATA wrapping from a string if present.
fn strip_cdata(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.starts_with("<![CDATA[") && trimmed.ends_with("]]>") {
        trimmed[9..trimmed.len() - 3].to_string()
    } else {
        trimmed.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feed Monitor
// ─────────────────────────────────────────────────────────────────────────────

/// Monitors multiple feeds for new entries and tracks state.
pub struct FeedMonitor {
    /// Monitor configuration.
    config: FeedMonitorConfig,
    /// Persistent monitor state.
    state: FeedMonitorState,
}

impl FeedMonitor {
    /// Create a new feed monitor with the given config.
    pub fn new(config: FeedMonitorConfig) -> Self {
        Self {
            config,
            state: FeedMonitorState::default(),
        }
    }

    /// Add a feed to monitor.
    pub fn add_feed(&mut self, name: &str, url: &str) {
        self.config.feeds.insert(name.to_string(), url.to_string());
    }

    /// Remove a feed from monitoring.
    pub fn remove_feed(&mut self, name: &str) {
        self.config.feeds.remove(name);
        self.state.last_checked.remove(name);
        self.state.known_entries.remove(name);
        self.state.last_entry_date.remove(name);
    }

    /// Check a single feed for new entries.
    pub fn check_feed(&mut self, name: &str) -> Result<FeedCheckResult> {
        let url = self
            .config
            .feeds
            .get(name)
            .ok_or_else(|| anyhow!("Feed '{}' not found", name))?
            .clone();

        let result = match self.fetch_feed(&url) {
            Ok(content) => {
                match FeedParser::parse(&content) {
                    Ok(parsed) => {
                        let known = self
                            .state
                            .known_entries
                            .entry(name.to_string())
                            .or_insert_with(Vec::new);

                        // Find new entries by comparing IDs
                        let new_entries: Vec<FeedEntry> = parsed
                            .entries
                            .iter()
                            .filter(|e| !known.contains(&e.id))
                            .cloned()
                            .collect();

                        // Update known entries (respecting max limit)
                        for entry in &new_entries {
                            known.push(entry.id.clone());
                        }

                        // Trim to max entries per feed
                        let max = self.config.max_entries_per_feed;
                        if known.len() > max {
                            let drain_count = known.len() - max;
                            known.drain(0..drain_count);
                        }

                        // Update last entry date
                        if let Some(newest_date) = parsed
                            .entries
                            .iter()
                            .filter_map(|e| e.published.or(e.updated))
                            .max()
                        {
                            self.state
                                .last_entry_date
                                .insert(name.to_string(), newest_date);
                        }

                        // Update last checked timestamp
                        self.state.last_checked.insert(name.to_string(), Utc::now());

                        FeedCheckResult {
                            feed_name: name.to_string(),
                            new_entries,
                            total_entries: parsed.entry_count,
                            success: true,
                            error: None,
                        }
                    }
                    Err(e) => FeedCheckResult {
                        feed_name: name.to_string(),
                        new_entries: Vec::new(),
                        total_entries: 0,
                        success: false,
                        error: Some(format!("Parse error: {}", e)),
                    },
                }
            }
            Err(e) => FeedCheckResult {
                feed_name: name.to_string(),
                new_entries: Vec::new(),
                total_entries: 0,
                success: false,
                error: Some(format!("Fetch error: {}", e)),
            },
        };

        Ok(result)
    }

    /// Check all configured feeds for new entries.
    pub fn check_all(&mut self) -> Vec<FeedCheckResult> {
        let feed_names: Vec<String> = self.config.feeds.keys().cloned().collect();
        let mut results = Vec::new();

        for name in feed_names {
            match self.check_feed(&name) {
                Ok(result) => results.push(result),
                Err(e) => results.push(FeedCheckResult {
                    feed_name: name,
                    new_entries: Vec::new(),
                    total_entries: 0,
                    success: false,
                    error: Some(format!("{}", e)),
                }),
            }
        }

        results
    }

    /// Get a reference to the current state.
    pub fn state(&self) -> &FeedMonitorState {
        &self.state
    }

    /// Restore monitor state from a previous session.
    pub fn restore_state(&mut self, state: FeedMonitorState) {
        self.state = state;
    }

    /// Export the current state as a JSON string.
    pub fn export_state(&self) -> String {
        serde_json::to_string_pretty(&self.state).unwrap_or_else(|_| "{}".to_string())
    }

    /// Import state from a JSON string.
    pub fn import_state(&mut self, json: &str) -> Result<()> {
        let state: FeedMonitorState =
            serde_json::from_str(json).map_err(|e| anyhow!("Failed to parse state JSON: {}", e))?;
        self.state = state;
        Ok(())
    }

    /// Export state to internal binary format (bincode+gzip when feature enabled).
    #[cfg(feature = "binary-storage")]
    pub fn export_state_bytes(&self) -> Result<Vec<u8>> {
        crate::internal_storage::serialize_internal(&self.state)
    }

    /// Import state from internal binary format (auto-detects binary or JSON).
    #[cfg(feature = "binary-storage")]
    pub fn import_state_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.state = crate::internal_storage::deserialize_internal(bytes)?;
        Ok(())
    }

    /// Get a list of all configured feeds as (name, url) pairs.
    pub fn feeds(&self) -> Vec<(&str, &str)> {
        self.config
            .feeds
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    /// Fetch feed content from a URL using ureq.
    fn fetch_feed(&self, url: &str) -> Result<String> {
        let timeout = std::time::Duration::from_millis(self.config.timeout_ms);

        let agent = ureq::AgentBuilder::new()
            .timeout_connect(timeout)
            .timeout_read(timeout)
            .user_agent(&self.config.user_agent)
            .build();

        let response = agent
            .get(url)
            .call()
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        let body = response
            .into_string()
            .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

        Ok(body)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RSS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Blog</title>
    <link>https://example.com</link>
    <description>A test blog feed</description>
    <language>en-us</language>
    <lastBuildDate>Mon, 20 Jan 2025 12:00:00 +0000</lastBuildDate>
    <item>
      <title>First Post</title>
      <link>https://example.com/first</link>
      <description>This is the first post</description>
      <pubDate>Sun, 19 Jan 2025 10:00:00 +0000</pubDate>
      <guid>https://example.com/first</guid>
      <author>alice@example.com</author>
      <category>Tech</category>
      <category>Rust</category>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/second</link>
      <description>This is the second post</description>
      <pubDate>Mon, 20 Jan 2025 08:00:00 +0000</pubDate>
      <guid>https://example.com/second</guid>
      <category>News</category>
    </item>
  </channel>
</rss>"#;

    const SAMPLE_ATOM: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <subtitle>A test atom feed</subtitle>
  <link href="https://atom.example.com" rel="alternate"/>
  <link href="https://atom.example.com/feed.xml" rel="self"/>
  <updated>2025-01-20T12:00:00Z</updated>
  <entry>
    <title>Atom Entry One</title>
    <link href="https://atom.example.com/entry1" rel="alternate"/>
    <id>urn:uuid:entry-001</id>
    <published>2025-01-18T09:00:00Z</published>
    <updated>2025-01-18T10:00:00Z</updated>
    <summary>Summary of entry one</summary>
    <content type="html">Full content of entry one</content>
    <author><name>Bob</name></author>
    <category term="Programming"/>
    <category term="Atom"/>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <link href="https://atom.example.com/entry2" rel="alternate"/>
    <id>urn:uuid:entry-002</id>
    <published>2025-01-19T14:30:00Z</published>
    <summary>Summary of entry two</summary>
    <author><name>Carol</name></author>
    <category term="Updates"/>
  </entry>
</feed>"#;

    #[test]
    fn test_parse_rss_feed() {
        let parsed = FeedParser::parse(SAMPLE_RSS).expect("Failed to parse RSS");

        // Check metadata
        assert_eq!(parsed.metadata.format, FeedFormat::Rss2);
        assert_eq!(parsed.metadata.title, "Test Blog");
        assert_eq!(parsed.metadata.description, "A test blog feed");
        assert_eq!(parsed.metadata.site_url, "https://example.com");
        assert_eq!(parsed.metadata.language.as_deref(), Some("en-us"));
        assert!(parsed.metadata.last_build_date.is_some());

        // Check entries
        assert_eq!(parsed.entry_count, 2);
        assert_eq!(parsed.entries.len(), 2);

        let first = &parsed.entries[0];
        assert_eq!(first.title, "First Post");
        assert_eq!(first.url, "https://example.com/first");
        assert_eq!(first.summary, "This is the first post");
        assert_eq!(first.id, "https://example.com/first");
        assert!(first.published.is_some());
        assert_eq!(first.author.as_deref(), Some("alice@example.com"));
        assert_eq!(first.categories, vec!["Tech", "Rust"]);

        let second = &parsed.entries[1];
        assert_eq!(second.title, "Second Post");
        assert_eq!(second.categories, vec!["News"]);
    }

    #[test]
    fn test_parse_atom_feed() {
        let parsed = FeedParser::parse(SAMPLE_ATOM).expect("Failed to parse Atom");

        // Check metadata
        assert_eq!(parsed.metadata.format, FeedFormat::Atom);
        assert_eq!(parsed.metadata.title, "Atom Test Feed");
        assert_eq!(parsed.metadata.description, "A test atom feed");
        assert_eq!(parsed.metadata.site_url, "https://atom.example.com");
        assert_eq!(
            parsed.metadata.feed_url,
            "https://atom.example.com/feed.xml"
        );
        assert!(parsed.metadata.last_build_date.is_some());

        // Check entries
        assert_eq!(parsed.entry_count, 2);

        let first = &parsed.entries[0];
        assert_eq!(first.title, "Atom Entry One");
        assert_eq!(first.url, "https://atom.example.com/entry1");
        assert_eq!(first.summary, "Summary of entry one");
        assert_eq!(first.content.as_deref(), Some("Full content of entry one"));
        assert_eq!(first.id, "urn:uuid:entry-001");
        assert!(first.published.is_some());
        assert!(first.updated.is_some());
        assert_eq!(first.author.as_deref(), Some("Bob"));
        assert_eq!(first.categories, vec!["Programming", "Atom"]);

        let second = &parsed.entries[1];
        assert_eq!(second.title, "Atom Entry Two");
        assert_eq!(second.author.as_deref(), Some("Carol"));
        assert_eq!(second.categories, vec!["Updates"]);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(FeedParser::detect_format(SAMPLE_RSS), FeedFormat::Rss2);
        assert_eq!(FeedParser::detect_format(SAMPLE_ATOM), FeedFormat::Atom);
        assert_eq!(
            FeedParser::detect_format("just some text"),
            FeedFormat::Unknown
        );
        assert_eq!(
            FeedParser::detect_format("<RSS version='2.0'>"),
            FeedFormat::Rss2
        );
        assert_eq!(
            FeedParser::detect_format("<FEED xmlns='...'>"),
            FeedFormat::Atom
        );
    }

    #[test]
    fn test_date_parsing() {
        // RFC 2822
        let rfc2822 = FeedParser::parse_date("Mon, 20 Jan 2025 12:00:00 +0000");
        assert!(rfc2822.is_some());

        // RFC 3339
        let rfc3339 = FeedParser::parse_date("2025-01-20T12:00:00Z");
        assert!(rfc3339.is_some());

        // ISO 8601 without timezone
        let iso = FeedParser::parse_date("2025-01-20T12:00:00");
        assert!(iso.is_some());

        // Date only
        let date_only = FeedParser::parse_date("2025-01-20");
        assert!(date_only.is_some());

        // Invalid
        let invalid = FeedParser::parse_date("not a date");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_feed_monitor_state_management() {
        let config = FeedMonitorConfig {
            feeds: HashMap::new(),
            ..Default::default()
        };
        let mut monitor = FeedMonitor::new(config);

        // Add and remove feeds
        monitor.add_feed("test", "https://example.com/feed.xml");
        assert_eq!(monitor.feeds().len(), 1);

        monitor.add_feed("test2", "https://example2.com/feed.xml");
        assert_eq!(monitor.feeds().len(), 2);

        monitor.remove_feed("test");
        assert_eq!(monitor.feeds().len(), 1);

        // State export/import round-trip
        let state_json = monitor.export_state();
        assert!(state_json.contains("{"));

        let mut monitor2 = FeedMonitor::new(FeedMonitorConfig::default());
        monitor2
            .import_state(&state_json)
            .expect("Failed to import state");
    }

    #[test]
    fn test_new_entry_detection() {
        let config = FeedMonitorConfig::default();
        let mut monitor = FeedMonitor::new(config);
        monitor.add_feed("test", "https://example.com/rss");

        // Simulate first check: mark some entries as known
        monitor.state.known_entries.insert(
            "test".to_string(),
            vec!["https://example.com/first".to_string()],
        );

        // Parse the RSS and check which entries are new
        let parsed = FeedParser::parse(SAMPLE_RSS).unwrap();
        let known = monitor.state.known_entries.get("test").unwrap();
        let new_entries: Vec<&FeedEntry> = parsed
            .entries
            .iter()
            .filter(|e| !known.contains(&e.id))
            .collect();

        // Only the second entry should be new
        assert_eq!(new_entries.len(), 1);
        assert_eq!(new_entries[0].title, "Second Post");
    }

    #[test]
    fn test_cdata_stripping() {
        let with_cdata = r#"<rss version="2.0">
  <channel>
    <title><![CDATA[CDATA Title]]></title>
    <link>https://example.com</link>
    <description><![CDATA[A <b>bold</b> description]]></description>
    <item>
      <title><![CDATA[Entry with CDATA]]></title>
      <link>https://example.com/cdata</link>
      <description><![CDATA[Some <em>html</em> content]]></description>
      <guid>cdata-entry-1</guid>
    </item>
  </channel>
</rss>"#;

        let parsed = FeedParser::parse(with_cdata).expect("Failed to parse CDATA RSS");
        assert_eq!(parsed.metadata.title, "CDATA Title");
        assert_eq!(parsed.metadata.description, "A <b>bold</b> description");
        assert_eq!(parsed.entries[0].title, "Entry with CDATA");
        assert_eq!(parsed.entries[0].summary, "Some <em>html</em> content");
    }

    #[test]
    fn test_unknown_format_error() {
        let result = FeedParser::parse("This is not a feed at all");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unable to detect feed format"));
    }

    #[test]
    fn test_config_defaults() {
        let config = FeedMonitorConfig::default();
        assert_eq!(config.check_interval_secs, 3600);
        assert_eq!(config.max_entries_per_feed, 100);
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.feeds.is_empty());
    }

    #[test]
    fn test_restore_state() {
        let config = FeedMonitorConfig::default();
        let mut monitor = FeedMonitor::new(config);

        let mut state = FeedMonitorState::default();
        state.last_checked.insert("feed1".to_string(), Utc::now());
        state.known_entries.insert(
            "feed1".to_string(),
            vec!["id-1".to_string(), "id-2".to_string()],
        );

        monitor.restore_state(state.clone());

        assert!(monitor.state().last_checked.contains_key("feed1"));
        assert_eq!(monitor.state().known_entries.get("feed1").unwrap().len(), 2);
    }
}
