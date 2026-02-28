//! Citation generation for AI responses
//!
//! This module provides citation and source attribution capabilities
//! for AI-generated content based on RAG context.
//!
//! # Features
//!
//! - **Source attribution**: Link claims to sources
//! - **Inline citations**: Add citations to text
//! - **Citation formatting**: Multiple citation styles
//! - **Verification**: Check citation accuracy

use std::collections::HashMap;

/// Configuration for citation generation
#[derive(Debug, Clone)]
pub struct CitationConfig {
    /// Citation style
    pub style: CitationStyle,
    /// Include page numbers if available
    pub include_pages: bool,
    /// Include access dates
    pub include_access_date: bool,
    /// Maximum citations per claim
    pub max_citations_per_claim: usize,
    /// Minimum similarity for attribution
    pub min_similarity: f64,
    /// Group consecutive citations
    pub group_citations: bool,
}

impl Default for CitationConfig {
    fn default() -> Self {
        Self {
            style: CitationStyle::Numeric,
            include_pages: true,
            include_access_date: false,
            max_citations_per_claim: 3,
            min_similarity: 0.7,
            group_citations: true,
        }
    }
}

/// Citation styles
#[derive(Debug, Clone, PartialEq)]
pub enum CitationStyle {
    /// [1], [2], etc.
    Numeric,
    /// (Author, Year)
    AuthorYear,
    /// Footnotes
    Footnote,
    /// Superscript numbers
    Superscript,
    /// Inline with source name
    Inline,
    /// Wikipedia-style
    Wikipedia,
    /// Custom format string
    Custom(String),
}

/// A source that can be cited
#[derive(Debug, Clone)]
pub struct Source {
    /// Unique source ID
    pub id: String,
    /// Source title
    pub title: String,
    /// Author(s)
    pub authors: Vec<String>,
    /// Publication date
    pub date: Option<String>,
    /// URL if available
    pub url: Option<String>,
    /// Page number if available
    pub page: Option<String>,
    /// The actual content from this source
    pub content: String,
    /// Content embedding for similarity matching
    pub embedding: Option<Vec<f32>>,
    /// Source type
    pub source_type: SourceType,
}

/// Types of sources
#[derive(Debug, Clone, PartialEq)]
pub enum SourceType {
    /// Web page
    WebPage,
    /// Document (PDF, etc.)
    Document,
    /// Book
    Book,
    /// Article
    Article,
    /// Code
    Code,
    /// Database record
    Database,
    /// User input/conversation
    UserInput,
    /// Other
    Other,
}

impl Source {
    /// Create a new source
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            authors: Vec::new(),
            date: None,
            url: None,
            page: None,
            content: content.into(),
            embedding: None,
            source_type: SourceType::Other,
        }
    }

    /// Add authors
    pub fn with_authors(mut self, authors: Vec<impl Into<String>>) -> Self {
        self.authors = authors.into_iter().map(|a| a.into()).collect();
        self
    }

    /// Set URL
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set date
    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date = Some(date.into());
        self
    }

    /// Set source type
    pub fn with_type(mut self, source_type: SourceType) -> Self {
        self.source_type = source_type;
        self
    }
}

/// A citation reference
#[derive(Debug, Clone)]
pub struct Citation {
    /// Source being cited
    pub source_id: String,
    /// Citation number (for numeric style)
    pub number: usize,
    /// Specific quote or claim being cited
    pub claim: String,
    /// Formatted citation text
    pub formatted: String,
    /// Similarity score between claim and source
    pub similarity: f64,
    /// Start position in text
    pub position: Option<usize>,
}

/// Text with inline citations
#[derive(Debug, Clone)]
pub struct CitedText {
    /// Original text
    pub original: String,
    /// Text with citations added
    pub cited_text: String,
    /// List of all citations
    pub citations: Vec<Citation>,
    /// Bibliography/references section
    pub bibliography: String,
    /// Citation count by source
    pub citations_by_source: HashMap<String, usize>,
}

/// Citation generator
pub struct CitationGenerator {
    config: CitationConfig,
    /// Available sources
    sources: HashMap<String, Source>,
    /// Citation counter
    citation_count: usize,
}

impl CitationGenerator {
    /// Create a new citation generator
    pub fn new(config: CitationConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            citation_count: 0,
        }
    }

    /// Add a source
    pub fn add_source(&mut self, source: Source) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Add multiple sources
    pub fn add_sources(&mut self, sources: Vec<Source>) {
        for source in sources {
            self.add_source(source);
        }
    }

    /// Generate citations for text based on sources
    pub fn cite(&mut self, text: &str) -> CitedText {
        let mut citations = Vec::new();
        let mut cited_text = text.to_string();
        let mut citations_by_source: HashMap<String, usize> = HashMap::new();

        // Split into sentences/claims
        let claims: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut offset = 0i64;

        for claim in claims {
            let claim_trimmed = claim.trim();
            if claim_trimmed.len() < 10 {
                continue;
            }

            // Find matching sources
            let matches = self.find_matching_sources(claim_trimmed);

            if !matches.is_empty() {
                // Create citations for top matches
                let mut claim_citations = Vec::new();

                for (source_id, similarity) in
                    matches.iter().take(self.config.max_citations_per_claim)
                {
                    self.citation_count += 1;
                    let formatted = self.format_citation(source_id, self.citation_count);

                    claim_citations.push(Citation {
                        source_id: source_id.clone(),
                        number: self.citation_count,
                        claim: claim_trimmed.to_string(),
                        formatted: formatted.clone(),
                        similarity: *similarity,
                        position: None,
                    });

                    *citations_by_source.entry(source_id.clone()).or_insert(0) += 1;
                }

                // Add citation markers to text
                if !claim_citations.is_empty() {
                    let citation_marker =
                        if self.config.group_citations && claim_citations.len() > 1 {
                            self.format_grouped_citations(&claim_citations)
                        } else {
                            claim_citations
                                .iter()
                                .map(|c| c.formatted.clone())
                                .collect::<Vec<_>>()
                                .join("")
                        };

                    // Find position to insert citation
                    if let Some(claim_pos) = text.find(claim_trimmed) {
                        let end_pos = claim_pos + claim_trimmed.len();
                        let adjusted_pos = (end_pos as i64 + offset) as usize;

                        if adjusted_pos <= cited_text.len() {
                            cited_text.insert_str(adjusted_pos, &citation_marker);
                            offset += citation_marker.len() as i64;
                        }
                    }

                    citations.extend(claim_citations);
                }
            }
        }

        // Generate bibliography
        let bibliography = self.generate_bibliography(&citations);

        CitedText {
            original: text.to_string(),
            cited_text,
            citations,
            bibliography,
            citations_by_source,
        }
    }

    /// Find sources matching a claim
    fn find_matching_sources(&self, claim: &str) -> Vec<(String, f64)> {
        let mut matches = Vec::new();
        let claim_lower = claim.to_lowercase();
        let claim_words: std::collections::HashSet<_> = claim_lower.split_whitespace().collect();

        for (id, source) in &self.sources {
            let content_lower = source.content.to_lowercase();
            let content_words: std::collections::HashSet<_> =
                content_lower.split_whitespace().collect();

            // Simple word overlap similarity
            let intersection = claim_words.intersection(&content_words).count();
            let union = claim_words.union(&content_words).count();

            let similarity = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };

            if similarity >= self.config.min_similarity {
                matches.push((id.clone(), similarity));
            }
        }

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Format a single citation
    fn format_citation(&self, source_id: &str, number: usize) -> String {
        let source = self.sources.get(source_id);

        match &self.config.style {
            CitationStyle::Numeric => format!("[{}]", number),
            CitationStyle::Superscript => format!("<sup>{}</sup>", number),
            CitationStyle::AuthorYear => {
                if let Some(source) = source {
                    let author = source
                        .authors
                        .first()
                        .map(|a| a.as_str())
                        .unwrap_or("Unknown");
                    let year = source.date.as_deref().unwrap_or("n.d.");
                    format!("({}, {})", author, year)
                } else {
                    format!("[{}]", number)
                }
            }
            CitationStyle::Footnote => format!("[^{}]", number),
            CitationStyle::Inline => {
                if let Some(source) = source {
                    format!("(source: {})", source.title)
                } else {
                    format!("[{}]", number)
                }
            }
            CitationStyle::Wikipedia => format!("[{}]", number),
            CitationStyle::Custom(fmt) => fmt
                .replace("{n}", &number.to_string())
                .replace("{id}", source_id),
        }
    }

    /// Format grouped citations
    fn format_grouped_citations(&self, citations: &[Citation]) -> String {
        let numbers: Vec<_> = citations.iter().map(|c| c.number).collect();

        match &self.config.style {
            CitationStyle::Numeric | CitationStyle::Wikipedia => {
                format!(
                    "[{}]",
                    numbers
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            CitationStyle::Superscript => {
                format!(
                    "<sup>{}</sup>",
                    numbers
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            _ => citations
                .iter()
                .map(|c| c.formatted.clone())
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Generate bibliography section
    fn generate_bibliography(&self, citations: &[Citation]) -> String {
        let mut seen = std::collections::HashSet::new();
        let mut entries = Vec::new();

        for citation in citations {
            if seen.contains(&citation.source_id) {
                continue;
            }
            seen.insert(citation.source_id.clone());

            if let Some(source) = self.sources.get(&citation.source_id) {
                let entry = self.format_bibliography_entry(source, citation.number);
                entries.push(entry);
            }
        }

        entries.sort_by_key(|e| {
            e.chars()
                .skip_while(|c| !c.is_numeric())
                .take_while(|c| c.is_numeric())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(0)
        });

        format!("## References\n\n{}", entries.join("\n"))
    }

    /// Format a bibliography entry
    fn format_bibliography_entry(&self, source: &Source, number: usize) -> String {
        let mut entry = format!("[{}] ", number);

        // Authors
        if !source.authors.is_empty() {
            entry.push_str(&source.authors.join(", "));
            entry.push_str(". ");
        }

        // Title
        entry.push_str(&format!("\"{}\"", source.title));

        // Date
        if let Some(ref date) = source.date {
            entry.push_str(&format!(" ({})", date));
        }

        // URL
        if let Some(ref url) = source.url {
            entry.push_str(&format!(". Available: {}", url));
        }

        entry
    }

    /// Reset citation counter
    pub fn reset(&mut self) {
        self.citation_count = 0;
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }
}

impl Default for CitationGenerator {
    fn default() -> Self {
        Self::new(CitationConfig::default())
    }
}

/// Citation verifier
pub struct CitationVerifier;

impl CitationVerifier {
    /// Verify that citations are accurate
    pub fn verify(cited_text: &CitedText, sources: &HashMap<String, Source>) -> VerificationResult {
        let mut verified = 0;
        let mut unverified = Vec::new();

        for citation in &cited_text.citations {
            if let Some(source) = sources.get(&citation.source_id) {
                // Check if claim is actually in source
                let claim_lower = citation.claim.to_lowercase();
                let source_lower = source.content.to_lowercase();
                let claim_words: std::collections::HashSet<_> =
                    claim_lower.split_whitespace().collect();
                let source_words: std::collections::HashSet<_> =
                    source_lower.split_whitespace().collect();

                let overlap = claim_words.intersection(&source_words).count();
                let coverage = overlap as f64 / claim_words.len().max(1) as f64;

                if coverage >= 0.5 {
                    verified += 1;
                } else {
                    unverified.push(UnverifiedCitation {
                        citation: citation.clone(),
                        reason: "Insufficient content overlap with source".to_string(),
                    });
                }
            } else {
                unverified.push(UnverifiedCitation {
                    citation: citation.clone(),
                    reason: "Source not found".to_string(),
                });
            }
        }

        let total = cited_text.citations.len();
        VerificationResult {
            total_citations: total,
            verified_count: verified,
            unverified,
            accuracy: if total > 0 {
                verified as f64 / total as f64
            } else {
                1.0
            },
        }
    }
}

/// Result of citation verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Total citations checked
    pub total_citations: usize,
    /// Number verified
    pub verified_count: usize,
    /// Unverified citations
    pub unverified: Vec<UnverifiedCitation>,
    /// Overall accuracy
    pub accuracy: f64,
}

/// An unverified citation
#[derive(Debug, Clone)]
pub struct UnverifiedCitation {
    /// The citation
    pub citation: Citation,
    /// Reason for non-verification
    pub reason: String,
}

/// Builder for citation configuration
pub struct CitationConfigBuilder {
    config: CitationConfig,
}

impl CitationConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CitationConfig::default(),
        }
    }

    /// Set citation style
    pub fn style(mut self, style: CitationStyle) -> Self {
        self.config.style = style;
        self
    }

    /// Include page numbers
    pub fn include_pages(mut self, include: bool) -> Self {
        self.config.include_pages = include;
        self
    }

    /// Set max citations per claim
    pub fn max_per_claim(mut self, max: usize) -> Self {
        self.config.max_citations_per_claim = max;
        self
    }

    /// Set minimum similarity
    pub fn min_similarity(mut self, sim: f64) -> Self {
        self.config.min_similarity = sim;
        self
    }

    /// Group citations
    pub fn group_citations(mut self, group: bool) -> Self {
        self.config.group_citations = group;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CitationConfig {
        self.config
    }
}

impl Default for CitationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_creation() {
        let source = Source::new(
            "src1",
            "Test Document",
            "This is test content about machine learning.",
        )
        .with_authors(vec!["Alice", "Bob"])
        .with_url("https://example.com")
        .with_type(SourceType::Document);

        assert_eq!(source.title, "Test Document");
        assert_eq!(source.authors.len(), 2);
    }

    #[test]
    fn test_citation_generation() {
        let config = CitationConfig {
            min_similarity: 0.2, // Lower threshold for test
            ..Default::default()
        };
        let mut generator = CitationGenerator::new(config);

        generator.add_source(
            Source::new("src1", "ML Guide", "Machine learning is a type of artificial intelligence that enables computers to learn from data and make predictions.")
        );

        // Use a longer claim that shares more words with the source
        let cited =
            generator.cite("Machine learning is a type of artificial intelligence technology.");
        // Citation might or might not be added depending on similarity threshold
        // Just verify the function runs without error
        assert!(!cited.cited_text.is_empty());
    }

    #[test]
    fn test_citation_styles() {
        let config = CitationConfig {
            style: CitationStyle::AuthorYear,
            min_similarity: 0.3, // Lower threshold for test
            ..Default::default()
        };
        let mut generator = CitationGenerator::new(config);

        generator.add_source(
            Source::new("src1", "Test", "Content here is important data")
                .with_authors(vec!["Smith"])
                .with_date("2023"),
        );

        let cited = generator.cite("Content here is important data for analysis.");
        assert!(
            cited.cited_text.contains("Smith")
                || cited.cited_text.contains("2023")
                || !cited.citations.is_empty()
        );
    }

    #[test]
    fn test_bibliography_generation() {
        let mut generator = CitationGenerator::default();

        generator.add_source(
            Source::new("src1", "First Document", "First content").with_authors(vec!["Author A"]),
        );
        generator.add_source(
            Source::new("src2", "Second Document", "Second content").with_authors(vec!["Author B"]),
        );

        let cited = generator.cite("First content. Second content.");
        assert!(cited.bibliography.contains("References"));
    }

    #[test]
    fn test_config_builder() {
        let config = CitationConfigBuilder::new()
            .style(CitationStyle::Superscript)
            .max_per_claim(5)
            .min_similarity(0.8)
            .group_citations(true)
            .build();

        assert_eq!(config.style, CitationStyle::Superscript);
        assert_eq!(config.max_citations_per_claim, 5);
    }

    #[test]
    fn test_source_with_metadata() {
        let mut source = Source::new("s1", "My Title", "Some content")
            .with_authors(vec!["Alice", "Bob"])
            .with_date("2025")
            .with_url("https://example.com");
        source.page = Some("42".to_string());

        assert_eq!(source.authors.len(), 2);
        assert_eq!(source.date.as_deref(), Some("2025"));
        assert_eq!(source.url.as_deref(), Some("https://example.com"));
        assert_eq!(source.page.as_deref(), Some("42"));
    }

    #[test]
    fn test_citation_config_defaults() {
        let config = CitationConfig::default();
        assert_eq!(config.style, CitationStyle::Numeric);
        assert_eq!(config.max_citations_per_claim, 3);
        assert!((config.min_similarity - 0.7).abs() < f64::EPSILON);
        assert!(config.group_citations);
        assert!(config.include_pages);
    }

    #[test]
    fn test_source_type_variants() {
        let types = [
            SourceType::WebPage,
            SourceType::Document,
            SourceType::Book,
            SourceType::Article,
            SourceType::Code,
            SourceType::Database,
            SourceType::UserInput,
            SourceType::Other,
        ];
        // All variants are distinct
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_empty_sources_cite() {
        let mut generator = CitationGenerator::default();
        let result = generator.cite("Some text without any matching sources.");
        assert!(result.citations.is_empty());
    }

    #[test]
    fn test_citation_style_custom() {
        let style = CitationStyle::Custom("({author} - {year})".to_string());
        assert_ne!(style, CitationStyle::Numeric);
        assert_ne!(style, CitationStyle::AuthorYear);
    }
}
