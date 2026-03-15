//! Core types for document parsing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DocumentFormat {
    Epub,
    Docx,
    Odt,
    Html,
    Pdf,
    PlainText,
    Csv,
    Email,
    Image,
    Pptx,
    Xlsx,
}

/// A section extracted from a parsed document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    /// Optional title of the section (e.g. heading text).
    pub title: Option<String>,
    /// The textual content of this section.
    pub content: String,
    /// Heading level (1-6 for HTML headings, 0 if unknown).
    pub level: u8,
    /// Zero-based index of this section in the document.
    pub index: usize,
}

/// Content structure extracted from a single PDF page.
///
/// Separates the main text from auxiliary content like footnotes,
/// captions, and sidebars using heuristic-based detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageContent {
    /// The main body text after removing footnotes, captions, etc.
    pub main_text: String,
    /// Footnotes detected at the bottom of the page
    pub footnotes: Vec<String>,
    /// Figure/table captions (text near "Figure X" or "Table X")
    pub captions: Vec<String>,
    /// Sidebar-like content (indented or boxed text blocks)
    pub sidebars: Vec<String>,
    /// Page number (1-indexed)
    pub page_number: usize,
}

impl PageContent {
    /// Check if the page has any auxiliary content
    pub fn has_auxiliary_content(&self) -> bool {
        !self.footnotes.is_empty() || !self.captions.is_empty() || !self.sidebars.is_empty()
    }

    /// Get all auxiliary content combined
    pub fn all_auxiliary(&self) -> Vec<&str> {
        let mut aux = Vec::new();
        for f in &self.footnotes {
            aux.push(f.as_str());
        }
        for c in &self.captions {
            aux.push(c.as_str());
        }
        for s in &self.sidebars {
            aux.push(s.as_str());
        }
        aux
    }
}

/// A table extracted from a document (primarily PDF).
///
/// Tables are detected using heuristics on text patterns:
/// - Multiple whitespace-separated columns
/// - Consistent column alignment across rows
/// - Numeric or short text content in cells
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfTable {
    /// Optional caption or header found near the table
    pub caption: Option<String>,
    /// Column headers (if detected)
    pub headers: Vec<String>,
    /// Table rows, each row is a vector of cell values
    pub rows: Vec<Vec<String>>,
    /// Page number where the table was found (1-indexed, for PDFs)
    pub page: Option<usize>,
    /// Confidence score (0.0-1.0) of the table detection
    pub confidence: f32,
    /// Original raw text of the table region (for debugging)
    pub raw_text: String,
}

/// Metadata extracted from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title.
    pub title: Option<String>,
    /// List of authors.
    pub authors: Vec<String>,
    /// Language code (e.g. "en", "es").
    pub language: Option<String>,
    /// Publication or creation date.
    pub date: Option<String>,
    /// Document description or summary.
    pub description: Option<String>,
    /// Publisher name.
    pub publisher: Option<String>,
    /// Any extra metadata key-value pairs.
    pub extra: HashMap<String, String>,
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            title: None,
            authors: Vec::new(),
            language: None,
            date: None,
            description: None,
            publisher: None,
            extra: HashMap::new(),
        }
    }
}

/// A fully parsed document with text, metadata, and sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    /// The full extracted plain text (tables removed if extract_tables is enabled).
    pub text: String,
    /// Extracted metadata.
    pub metadata: DocumentMetadata,
    /// Extracted sections.
    pub sections: Vec<DocumentSection>,
    /// Tables extracted from the document (primarily for PDFs).
    #[serde(default)]
    pub tables: Vec<PdfTable>,
    /// The format that was parsed.
    pub format: DocumentFormat,
    /// Original file path, if parsed from a file.
    pub source_path: Option<String>,
    /// Total character count of the extracted text.
    pub char_count: usize,
    /// Total word count of the extracted text.
    pub word_count: usize,
}

impl ParsedDocument {
    /// Get the text content of a section by index.
    pub fn section_text(&self, index: usize) -> Option<&str> {
        self.sections.get(index).map(|s| s.content.as_str())
    }

    /// Get a list of all section titles (only those that have one).
    pub fn section_titles(&self) -> Vec<&str> {
        self.sections
            .iter()
            .filter_map(|s| s.title.as_deref())
            .collect()
    }
}

/// Configuration for the document parser.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DocumentParserConfig {
    /// Whether to preserve paragraph breaks as double newlines.
    pub preserve_paragraphs: bool,
    /// Whether to attempt metadata extraction.
    pub extract_metadata: bool,
    /// Whether to extract sections (headings + content).
    pub extract_sections: bool,
    /// Whether to extract tables (primarily for PDFs).
    /// When enabled, detected tables are removed from the text and stored separately.
    pub extract_tables: bool,
    /// Maximum input size in bytes (0 = unlimited).
    pub max_size_bytes: usize,
    /// Whether to strip HTML/XML tags from content.
    pub strip_tags: bool,
    /// Whether to normalize whitespace (collapse runs, trim lines).
    pub normalize_whitespace: bool,
}

impl Default for DocumentParserConfig {
    fn default() -> Self {
        Self {
            preserve_paragraphs: true,
            extract_metadata: true,
            extract_sections: true,
            extract_tables: false, // Disabled by default (adds processing overhead)
            max_size_bytes: 50 * 1024 * 1024, // 50 MB
            strip_tags: true,
            normalize_whitespace: true,
        }
    }
}
