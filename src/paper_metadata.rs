//! Paper metadata extraction
//!
//! Extracts structured metadata (title, authors, abstract, sections) from
//! academic paper text. Uses heuristic section detection based on common
//! academic paper structures.

use std::collections::HashMap;

// =============================================================================
// Core Types
// =============================================================================

/// Academic paper section types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SectionType {
    /// Abstract
    Abstract,
    /// Introduction
    Introduction,
    /// Related work / literature review
    RelatedWork,
    /// Methodology / methods
    Methodology,
    /// Results / experiments
    Results,
    /// Discussion
    Discussion,
    /// Conclusion
    Conclusion,
    /// References / bibliography
    References,
    /// Appendix
    Appendix,
    /// Other / unclassified section
    Other,
}

impl SectionType {
    /// Classify a section heading into a type.
    pub fn from_heading(heading: &str) -> Self {
        let lower = heading.to_lowercase();
        let lower = lower.trim();

        if lower.contains("abstract") {
            Self::Abstract
        } else if lower.contains("introduction") {
            Self::Introduction
        } else if lower.contains("related work")
            || lower.contains("literature review")
            || lower.contains("background")
            || lower.contains("prior work")
        {
            Self::RelatedWork
        } else if lower.contains("method")
            || lower.contains("approach")
            || lower.contains("framework")
            || lower.contains("model")
            || lower.contains("architecture")
        {
            Self::Methodology
        } else if lower.contains("result")
            || lower.contains("experiment")
            || lower.contains("evaluation")
            || lower.contains("finding")
        {
            Self::Results
        } else if lower.contains("discussion") {
            Self::Discussion
        } else if lower.contains("conclusion")
            || lower.contains("summary")
            || lower.contains("future work")
        {
            Self::Conclusion
        } else if lower.contains("reference") || lower.contains("bibliography") {
            Self::References
        } else if lower.contains("appendix") || lower.contains("supplementa") {
            Self::Appendix
        } else {
            Self::Other
        }
    }

    /// Display name for the section type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Abstract => "Abstract",
            Self::Introduction => "Introduction",
            Self::RelatedWork => "Related Work",
            Self::Methodology => "Methodology",
            Self::Results => "Results",
            Self::Discussion => "Discussion",
            Self::Conclusion => "Conclusion",
            Self::References => "References",
            Self::Appendix => "Appendix",
            Self::Other => "Other",
        }
    }
}

impl std::fmt::Display for SectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// A section of an academic paper.
#[derive(Debug, Clone)]
pub struct PaperSection {
    /// Section heading
    pub title: String,
    /// Section content (plain text)
    pub content: String,
    /// Nesting level (1 = top-level, 2 = subsection, etc.)
    pub level: u8,
    /// Detected section type
    pub section_type: SectionType,
}

impl PaperSection {
    /// Create a new section.
    pub fn new(title: &str, content: &str, level: u8) -> Self {
        let section_type = SectionType::from_heading(title);
        Self {
            title: title.to_string(),
            content: content.to_string(),
            level,
            section_type,
        }
    }

    /// Word count of the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// Extracted metadata from an academic paper.
#[derive(Debug, Clone)]
pub struct PaperMetadata {
    /// Paper title
    pub title: Option<String>,
    /// Authors
    pub authors: Vec<super::academic_search::Author>,
    /// Abstract text
    pub abstract_text: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// DOI
    pub doi: Option<String>,
    /// Publication year
    pub year: Option<u16>,
    /// Detected sections
    pub sections: Vec<PaperSection>,
    /// Raw reference strings
    pub references_raw: Vec<String>,
    /// Estimated page count
    pub page_count: usize,
    /// Confidence in the extraction (0.0 - 1.0)
    pub extraction_confidence: f64,
    /// Additional metadata fields
    pub extra: HashMap<String, String>,
}

impl PaperMetadata {
    /// Create empty metadata.
    pub fn empty() -> Self {
        Self {
            title: None,
            authors: Vec::new(),
            abstract_text: None,
            keywords: Vec::new(),
            doi: None,
            year: None,
            sections: Vec::new(),
            references_raw: Vec::new(),
            page_count: 0,
            extraction_confidence: 0.0,
            extra: HashMap::new(),
        }
    }

    /// Whether the extraction found meaningful content.
    pub fn has_content(&self) -> bool {
        self.title.is_some() || !self.sections.is_empty() || self.abstract_text.is_some()
    }

    /// Total word count across all sections.
    pub fn total_word_count(&self) -> usize {
        self.sections.iter().map(|s| s.word_count()).sum()
    }

    /// Get sections of a specific type.
    pub fn sections_of_type(&self, section_type: SectionType) -> Vec<&PaperSection> {
        self.sections
            .iter()
            .filter(|s| s.section_type == section_type)
            .collect()
    }

    /// Summary string: "Title (Year) by Authors"
    pub fn summary(&self) -> String {
        let title = self.title.as_deref().unwrap_or("Untitled");
        let year = self.year.map(|y| format!(" ({})", y)).unwrap_or_default();
        let authors = if self.authors.is_empty() {
            String::new()
        } else if self.authors.len() <= 3 {
            format!(
                " by {}",
                self.authors
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            format!(" by {} et al.", self.authors[0].name)
        };

        format!("{}{}{}", title, year, authors)
    }
}

// =============================================================================
// Extractor
// =============================================================================

/// Configuration for metadata extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Whether to extract sections
    pub extract_sections: bool,
    /// Whether to extract references
    pub extract_references: bool,
    /// Maximum text length to process (bytes)
    pub max_text_length: usize,
    /// Minimum confidence to report
    pub min_confidence: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            extract_sections: true,
            extract_references: true,
            max_text_length: 5_000_000, // 5 MB
            min_confidence: 0.0,
        }
    }
}

/// Paper metadata extractor.
///
/// Uses heuristics to detect paper structure from plain text. Works best with
/// well-structured academic papers but degrades gracefully on other text.
pub struct PaperMetadataExtractor {
    config: ExtractionConfig,
}

impl PaperMetadataExtractor {
    /// Create a new extractor with default config.
    pub fn new() -> Self {
        Self {
            config: ExtractionConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract metadata from plain text.
    pub fn extract(&self, text: &str) -> PaperMetadata {
        let text = if text.len() > self.config.max_text_length {
            &text[..self.config.max_text_length]
        } else {
            text
        };

        let mut metadata = PaperMetadata::empty();
        let lines: Vec<&str> = text.lines().collect();

        if lines.is_empty() {
            return metadata;
        }

        // Estimate page count (rough: ~3000 chars per page)
        metadata.page_count = (text.len() / 3000).max(1);

        let mut confidence_points: f64 = 0.0;
        let max_points: f64 = 5.0;

        // Extract title (first non-empty line, typically short and without period)
        if let Some(title) = self.extract_title(&lines) {
            metadata.title = Some(title);
            confidence_points += 1.0;
        }

        // Extract abstract
        if let Some(abstract_text) = self.extract_abstract(text) {
            metadata.abstract_text = Some(abstract_text);
            confidence_points += 1.0;
        }

        // Extract DOI
        if let Some(doi) = self.extract_doi(text) {
            metadata.doi = Some(doi);
            confidence_points += 0.5;
        }

        // Extract year
        if let Some(year) = self.extract_year(text) {
            metadata.year = Some(year);
            confidence_points += 0.5;
        }

        // Extract keywords
        metadata.keywords = self.extract_keywords(text);
        if !metadata.keywords.is_empty() {
            confidence_points += 0.5;
        }

        // Extract sections
        if self.config.extract_sections {
            metadata.sections = self.extract_sections(text);
            if !metadata.sections.is_empty() {
                confidence_points += 1.0;
            }
        }

        // Extract references
        if self.config.extract_references {
            metadata.references_raw = self.extract_references(text);
            if !metadata.references_raw.is_empty() {
                confidence_points += 0.5;
            }
        }

        metadata.extraction_confidence = (confidence_points / max_points).min(1.0);
        metadata
    }

    /// Extract title from first lines.
    fn extract_title(&self, lines: &[&str]) -> Option<String> {
        // Title is usually the first non-empty line that:
        // - Doesn't start with common metadata prefixes
        // - Is relatively short (< 200 chars)
        // - Doesn't end with a period (usually)
        for line in lines.iter().take(10) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.len() > 200 {
                continue;
            }
            // Skip common metadata lines
            let lower = trimmed.to_lowercase();
            if lower.starts_with("abstract")
                || lower.starts_with("keywords")
                || lower.starts_with("doi:")
                || lower.starts_with("http")
                || lower.starts_with("arxiv:")
                || lower.starts_with("published")
                || lower.starts_with("submitted")
                || lower.starts_with("copyright")
            {
                continue;
            }
            return Some(trimmed.to_string());
        }
        None
    }

    /// Extract abstract text.
    fn extract_abstract(&self, text: &str) -> Option<String> {
        // Look for "Abstract" heading followed by text
        for marker in &[
            "abstract\n",
            "abstract.",
            "abstract\r\n",
            "abstract —",
            "abstract:",
        ] {
            if let Some((_, marker_end)) = crate::text_util::find_ci_range(text, marker) {
                // Abstract ends at next section heading or double newline
                let content = &text[marker_end..];
                let end = self.find_section_break(content);
                let abstract_text = content[..end].trim().to_string();
                if !abstract_text.is_empty() && abstract_text.len() > 20 {
                    return Some(abstract_text);
                }
            }
        }
        None
    }

    /// Extract DOI from text.
    fn extract_doi(&self, text: &str) -> Option<String> {
        // Pattern: 10.XXXX/...
        for line in text.lines() {
            let line = line.trim();
            if let Some(pos) = line.find("10.") {
                let doi_start = pos;
                let remaining = &line[doi_start..];
                // DOI ends at whitespace, comma, or end of line
                let end = remaining
                    .find(|c: char| {
                        c.is_whitespace() || c == ',' || c == ';' || c == '>' || c == ')'
                    })
                    .unwrap_or(remaining.len());
                let doi = &remaining[..end];
                // Validate: must have at least one slash
                if doi.contains('/') && doi.len() > 7 {
                    return Some(doi.to_string());
                }
            }
        }
        None
    }

    /// Extract publication year.
    fn extract_year(&self, text: &str) -> Option<u16> {
        // Look for 4-digit year in common patterns
        let first_chunk = if text.len() > 2000 {
            &text[..2000]
        } else {
            text
        };

        for line in first_chunk.lines() {
            let lower = line.to_lowercase();
            if lower.contains("published")
                || lower.contains("submitted")
                || lower.contains("accepted")
                || lower.contains("copyright")
                || lower.contains("©")
            {
                // Extract 4-digit year (2000-2099)
                for word in line.split(|c: char| !c.is_ascii_digit()) {
                    if word.len() == 4 {
                        if let Ok(y) = word.parse::<u16>() {
                            if (2000..=2099).contains(&y) {
                                return Some(y);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract keywords.
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        for marker in &["keywords:", "keywords —", "key words:", "index terms:"] {
            // Case-insensitive match on the ORIGINAL text (valid byte offset;
            // a `to_lowercase()` copy drifts → slice could panic off a boundary).
            if let Some((_, marker_end)) = crate::text_util::find_ci_range(text, marker) {
                let content = &text[marker_end..];
                // Keywords line ends at newline
                let end = content.find('\n').unwrap_or(content.len());
                let kw_line = content[..end].trim();

                // Split by comma or semicolon
                let keywords: Vec<String> = kw_line
                    .split(|c: char| c == ',' || c == ';')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty() && k.len() < 100)
                    .collect();

                if !keywords.is_empty() {
                    return keywords;
                }
            }
        }
        Vec::new()
    }

    /// Extract sections from text.
    fn extract_sections(&self, text: &str) -> Vec<PaperSection> {
        let mut sections = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        let mut current_heading: Option<String> = None;
        let mut current_content = String::new();
        let mut current_level: u8 = 1;

        for line in &lines {
            let trimmed = line.trim();

            // Detect section headings:
            // - Lines that are all caps and short
            // - Lines starting with numbers like "1." "2.1"
            // - Lines matching known section names
            if self.is_section_heading(trimmed) {
                // Save previous section
                if let Some(heading) = current_heading.take() {
                    let content = current_content.trim().to_string();
                    if !content.is_empty() {
                        sections.push(PaperSection::new(&heading, &content, current_level));
                    }
                }
                current_heading = Some(trimmed.to_string());
                current_content.clear();

                // Determine level from numbering
                current_level = if trimmed.contains('.') && trimmed.len() < 50 {
                    let dots = trimmed
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.')
                        .filter(|c| *c == '.')
                        .count();
                    (dots as u8 + 1).min(3)
                } else {
                    1
                };
            } else if current_heading.is_some() {
                current_content.push_str(trimmed);
                current_content.push('\n');
            }
        }

        // Save last section
        if let Some(heading) = current_heading {
            let content = current_content.trim().to_string();
            if !content.is_empty() {
                sections.push(PaperSection::new(&heading, &content, current_level));
            }
        }

        sections
    }

    /// Check if a line looks like a section heading.
    fn is_section_heading(&self, line: &str) -> bool {
        if line.is_empty() || line.len() > 100 {
            return false;
        }

        let lower = line.to_lowercase();

        // Known section names
        let known = [
            "abstract",
            "introduction",
            "related work",
            "background",
            "methodology",
            "methods",
            "approach",
            "model",
            "architecture",
            "results",
            "experiments",
            "evaluation",
            "discussion",
            "conclusion",
            "conclusions",
            "summary",
            "future work",
            "references",
            "bibliography",
            "appendix",
            "acknowledgments",
            "acknowledgements",
        ];
        for name in &known {
            if lower == *name || lower.ends_with(name) {
                return true;
            }
        }

        // Numbered sections: "1. Introduction", "2.1 Related Work"
        let first_char = line.chars().next().unwrap_or(' ');
        if first_char.is_ascii_digit() {
            let after_number = line
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.')
                .trim();
            if !after_number.is_empty() && after_number.len() < 60 {
                // Numbered heading — check it doesn't look like a sentence
                if !after_number.ends_with('.') || after_number.len() < 40 {
                    return true;
                }
            }
        }

        // ALL CAPS headings (at least 3 chars, mostly uppercase)
        if line.len() >= 3 && line.len() <= 50 {
            let upper_count = line.chars().filter(|c| c.is_uppercase()).count();
            let alpha_count = line.chars().filter(|c| c.is_alphabetic()).count();
            if alpha_count > 2 && upper_count as f64 / alpha_count as f64 > 0.8 {
                return true;
            }
        }

        false
    }

    /// Extract raw reference strings.
    fn extract_references(&self, text: &str) -> Vec<String> {
        // Find the References/Bibliography section case-insensitively on the
        // ORIGINAL text, so the offset is valid there (a `to_lowercase()` copy
        // drifts and could slice off a UTF-8 boundary → panic).
        let ref_start = if let Some((_, end)) =
            crate::text_util::rfind_ci_range(text, "\nreferences\n")
        {
            end
        } else if let Some((_, end)) = crate::text_util::rfind_ci_range(text, "\nreferences\r\n") {
            end
        } else if let Some((_, end)) = crate::text_util::rfind_ci_range(text, "\nbibliography\n") {
            end
        } else {
            return Vec::new();
        };

        let ref_text = &text[ref_start..];
        let mut references = Vec::new();
        let mut current_ref = String::new();

        for line in ref_text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !current_ref.is_empty() {
                    references.push(current_ref.trim().to_string());
                    current_ref.clear();
                }
                continue;
            }

            // New reference starts with [N], (N), or a number followed by period
            let starts_new = trimmed.starts_with('[')
                || (trimmed
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                    && (trimmed.contains(". ") || trimmed.contains("] ")));

            if starts_new && !current_ref.is_empty() {
                references.push(current_ref.trim().to_string());
                current_ref.clear();
            }
            current_ref.push_str(trimmed);
            current_ref.push(' ');

            // Safety limit
            if references.len() >= 500 {
                break;
            }
        }

        if !current_ref.is_empty() {
            references.push(current_ref.trim().to_string());
        }

        references
    }

    /// Find the next section break (double newline or next heading).
    fn find_section_break(&self, text: &str) -> usize {
        // Double newline
        if let Some(pos) = text.find("\n\n") {
            return pos;
        }
        if let Some(pos) = text.find("\r\n\r\n") {
            return pos;
        }
        text.len().min(2000) // Cap at 2000 chars if no break found
    }
}

impl Default for PaperMetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_type_from_heading() {
        assert_eq!(SectionType::from_heading("Abstract"), SectionType::Abstract);
        assert_eq!(
            SectionType::from_heading("1. Introduction"),
            SectionType::Introduction
        );
        assert_eq!(
            SectionType::from_heading("Related Work"),
            SectionType::RelatedWork
        );
        assert_eq!(
            SectionType::from_heading("2.1 Methodology"),
            SectionType::Methodology
        );
        assert_eq!(SectionType::from_heading("Results"), SectionType::Results);
        assert_eq!(
            SectionType::from_heading("Discussion"),
            SectionType::Discussion
        );
        assert_eq!(
            SectionType::from_heading("Conclusion"),
            SectionType::Conclusion
        );
        assert_eq!(
            SectionType::from_heading("References"),
            SectionType::References
        );
        assert_eq!(
            SectionType::from_heading("Appendix A"),
            SectionType::Appendix
        );
        assert_eq!(
            SectionType::from_heading("Acknowledgments"),
            SectionType::Other
        );
    }

    #[test]
    fn test_section_type_display() {
        assert_eq!(SectionType::Abstract.display_name(), "Abstract");
        assert_eq!(SectionType::Introduction.display_name(), "Introduction");
        assert_eq!(SectionType::RelatedWork.display_name(), "Related Work");
    }

    #[test]
    fn test_paper_section_creation() {
        let section = PaperSection::new(
            "Introduction",
            "This paper presents a novel approach to...",
            1,
        );
        assert_eq!(section.title, "Introduction");
        assert_eq!(section.section_type, SectionType::Introduction);
        assert_eq!(section.level, 1);
        assert!(section.word_count() > 0);
    }

    #[test]
    fn test_paper_metadata_empty() {
        let meta = PaperMetadata::empty();
        assert!(meta.title.is_none());
        assert!(!meta.has_content());
        assert_eq!(meta.total_word_count(), 0);
        assert_eq!(meta.extraction_confidence, 0.0);
    }

    #[test]
    fn test_paper_metadata_summary() {
        use super::super::academic_search::Author;
        let mut meta = PaperMetadata::empty();
        meta.title = Some("Deep Learning Survey".to_string());
        meta.year = Some(2024);
        meta.authors = vec![Author::new("Smith"), Author::new("Jones")];
        assert_eq!(
            meta.summary(),
            "Deep Learning Survey (2024) by Smith, Jones"
        );
    }

    #[test]
    fn test_paper_metadata_summary_et_al() {
        use super::super::academic_search::Author;
        let mut meta = PaperMetadata::empty();
        meta.title = Some("Big Paper".to_string());
        meta.authors = vec![
            Author::new("A"),
            Author::new("B"),
            Author::new("C"),
            Author::new("D"),
        ];
        assert!(meta.summary().contains("et al."));
    }

    #[test]
    fn test_extract_title() {
        let text = "Attention Is All You Need\n\nAbstract\n\nWe propose...";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert_eq!(meta.title.as_deref(), Some("Attention Is All You Need"));
    }

    #[test]
    fn test_extract_abstract() {
        let text = "Title\n\nAbstract\nWe propose a new model called Transformer that relies entirely on attention.\n\nIntroduction\nRecent work...";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert!(meta.abstract_text.is_some());
        assert!(meta.abstract_text.as_ref().unwrap().contains("Transformer"));
    }

    #[test]
    fn test_extract_doi() {
        let text = "Title\nDOI: 10.1234/test.2024.001\nAbstract\nContent";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert_eq!(meta.doi.as_deref(), Some("10.1234/test.2024.001"));
    }

    #[test]
    fn test_extract_keywords() {
        let text =
            "Title\nKeywords: deep learning, transformers, attention, NLP\n\nAbstract\nContent";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert_eq!(meta.keywords.len(), 4);
        assert!(meta.keywords.contains(&"deep learning".to_string()));
        assert!(meta.keywords.contains(&"NLP".to_string()));
    }

    #[test]
    fn test_extract_sections() {
        let text = "Title\n\nAbstract\nWe present...\n\nIntroduction\nRecent advances in AI have led to...\n\nMethods\nOur approach uses...\n\nResults\nWe observe significant improvements...\n\nConclusion\nIn this paper we showed...";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert!(!meta.sections.is_empty());
        assert!(meta.has_content());
    }

    #[test]
    fn test_extract_year() {
        let text = "Title\nPublished in 2024\nAbstract\nContent";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert_eq!(meta.year, Some(2024));
    }

    #[test]
    fn test_extract_references() {
        let text = "Content\n\nReferences\n[1] Smith et al. Deep Learning. 2024.\n[2] Jones. Neural Nets. 2023.\n";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert_eq!(meta.references_raw.len(), 2);
    }

    #[test]
    fn test_extraction_confidence() {
        let text = "Attention Is All You Need\nPublished in 2024\nDOI: 10.1234/test\nKeywords: attention, transformer\n\nAbstract\nWe propose a new model.\n\nIntroduction\nContent here.\n\nReferences\n[1] Ref one.";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        assert!(meta.extraction_confidence > 0.5);
    }

    #[test]
    fn test_sections_of_type() {
        let text =
            "Title\n\nIntroduction\nFirst intro.\n\nMethods\nOur methods.\n\nResults\nOur results.";
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(text);
        let intros = meta.sections_of_type(SectionType::Introduction);
        assert!(intros.len() <= 1);
    }

    #[test]
    fn test_is_section_heading_numbered() {
        let extractor = PaperMetadataExtractor::new();
        assert!(extractor.is_section_heading("1. Introduction"));
        assert!(extractor.is_section_heading("2.1 Related Work"));
        assert!(!extractor.is_section_heading("")); // empty
        assert!(!extractor.is_section_heading("This is a long sentence that is definitely not a heading and should not match the heuristic at all because it is way too long to be a heading."));
    }

    #[test]
    fn test_is_section_heading_all_caps() {
        let extractor = PaperMetadataExtractor::new();
        assert!(extractor.is_section_heading("INTRODUCTION"));
        assert!(extractor.is_section_heading("RESULTS AND DISCUSSION"));
    }

    #[test]
    fn test_empty_text_extraction() {
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract("");
        assert!(!meta.has_content());
        assert_eq!(meta.extraction_confidence, 0.0);
    }

    #[test]
    fn test_extraction_config() {
        let config = ExtractionConfig {
            extract_sections: false,
            extract_references: false,
            ..Default::default()
        };
        let extractor = PaperMetadataExtractor::with_config(config);
        let text =
            "Title\n\nAbstract\nContent\n\nIntroduction\nMore content\n\nReferences\n[1] Ref.";
        let meta = extractor.extract(text);
        assert!(meta.sections.is_empty());
        assert!(meta.references_raw.is_empty());
    }

    #[test]
    fn test_page_count_estimation() {
        let text = "x".repeat(9000); // ~3 pages
        let extractor = PaperMetadataExtractor::new();
        let meta = extractor.extract(&text);
        assert_eq!(meta.page_count, 3);
    }
}
