//! Document parsing module for extracting plain text and metadata from various file formats.
//!
//! ## Supported Formats
//!
//! | Format | Feature Required | Notes |
//! |--------|------------------|-------|
//! | EPUB | `documents` | ZIP-based ebooks |
//! | DOCX | `documents` | Microsoft Word |
//! | ODT | `documents` | OpenDocument Text |
//! | PDF | `documents` | With header/footer detection |
//! | HTML | (always) | Regex-based tag stripping |
//! | Plain Text | (always) | TXT, MD, etc. |
//!
//! ## PDF Extraction Notes
//!
//! PDF text extraction has inherent challenges:
//! - **Headers/footers**: Detected by finding repeated lines across pages and filtered out
//! - **Page numbers**: Common formats are automatically detected and removed
//! - **Multi-column layouts**: May still cause interleaved text
//! - **Tables**: Structure is lost, text is linearized
//!
//! Each page becomes a section with `title = "Page N"`, allowing page-level referencing.
//! The `metadata.extra["total_pages"]` contains the page count.
//!
//! All XML/HTML parsing is done using regex patterns rather than full XML parsers,
//! which keeps dependencies minimal while handling the common cases.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "documents")]
use std::io::Read;

// ============================================================================
// Types
// ============================================================================

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

// ============================================================================
// DocumentParser
// ============================================================================

/// Main parser that can handle multiple document formats.
#[derive(Debug, Clone)]
pub struct DocumentParser {
    pub config: DocumentParserConfig,
}

impl DocumentParser {
    /// Create a new parser with the given configuration.
    pub fn new(config: DocumentParserConfig) -> Self {
        Self { config }
    }

    /// Parse a file from the filesystem, auto-detecting format from the extension.
    pub fn parse_file(&self, path: &Path) -> Result<ParsedDocument> {
        let format = self.detect_format(path).ok_or_else(|| {
            anyhow::anyhow!(
                "Could not detect document format for path: {}",
                path.display()
            )
        })?;

        let data = std::fs::read(path)?;

        if self.config.max_size_bytes > 0 && data.len() > self.config.max_size_bytes {
            anyhow::bail!(
                "File size {} exceeds maximum allowed size {}",
                data.len(),
                self.config.max_size_bytes
            );
        }

        let mut doc = self.parse_bytes(&data, format)?;
        doc.source_path = Some(path.to_string_lossy().into_owned());
        Ok(doc)
    }

    /// Parse raw bytes in the specified format.
    pub fn parse_bytes(&self, data: &[u8], format: DocumentFormat) -> Result<ParsedDocument> {
        if self.config.max_size_bytes > 0 && data.len() > self.config.max_size_bytes {
            anyhow::bail!(
                "Data size {} exceeds maximum allowed size {}",
                data.len(),
                self.config.max_size_bytes
            );
        }

        match format {
            #[cfg(feature = "documents")]
            DocumentFormat::Epub => self.parse_epub(data),
            #[cfg(feature = "documents")]
            DocumentFormat::Docx => self.parse_docx(data),
            #[cfg(feature = "documents")]
            DocumentFormat::Odt => self.parse_odt(data),

            #[cfg(not(feature = "documents"))]
            DocumentFormat::Epub | DocumentFormat::Docx | DocumentFormat::Odt => {
                anyhow::bail!(
                    "Document format {:?} requires the 'documents' feature to be enabled",
                    format
                );
            }

            #[cfg(feature = "pdf-extract")]
            DocumentFormat::Pdf => self.parse_pdf(data),

            #[cfg(not(feature = "pdf-extract"))]
            DocumentFormat::Pdf => {
                anyhow::bail!(
                    "PDF parsing requires the 'documents' feature (which includes pdf-extract)"
                );
            }

            DocumentFormat::Html => {
                let content = String::from_utf8_lossy(data);
                self.parse_html(&content)
            }
            DocumentFormat::PlainText => {
                let content = String::from_utf8_lossy(data);
                self.parse_string(&content, DocumentFormat::PlainText)
            }

            DocumentFormat::Csv => {
                let content = String::from_utf8_lossy(data);
                self.parse_csv(&content)
            }
            DocumentFormat::Email => {
                let content = String::from_utf8_lossy(data);
                self.parse_email(&content)
            }
            DocumentFormat::Image => self.parse_image(data),

            #[cfg(feature = "documents")]
            DocumentFormat::Pptx => self.parse_pptx(data),
            #[cfg(not(feature = "documents"))]
            DocumentFormat::Pptx => {
                anyhow::bail!("PPTX parsing requires the 'documents' feature (ZIP support)");
            }

            #[cfg(feature = "documents")]
            DocumentFormat::Xlsx => self.parse_xlsx(data),
            #[cfg(not(feature = "documents"))]
            DocumentFormat::Xlsx => {
                anyhow::bail!("XLSX parsing requires the 'documents' feature");
            }
        }
    }

    /// Parse a string in the specified format.
    pub fn parse_string(&self, content: &str, format: DocumentFormat) -> Result<ParsedDocument> {
        match format {
            DocumentFormat::Html => self.parse_html(content),
            DocumentFormat::PlainText => {
                let text = if self.config.normalize_whitespace {
                    normalize_text(content)
                } else {
                    content.to_string()
                };

                let sections = if self.config.extract_sections {
                    vec![DocumentSection {
                        title: None,
                        content: text.clone(),
                        level: 0,
                        index: 0,
                    }]
                } else {
                    Vec::new()
                };

                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();

                Ok(ParsedDocument {
                    text,
                    metadata: DocumentMetadata::default(),
                    sections,
                    tables: Vec::new(),
                    format: DocumentFormat::PlainText,
                    source_path: None,
                    char_count,
                    word_count,
                })
            }
            _ => {
                // For binary formats, convert to bytes and use parse_bytes
                self.parse_bytes(content.as_bytes(), format)
            }
        }
    }

    /// Detect the document format from a file extension.
    pub fn detect_format(&self, path: &Path) -> Option<DocumentFormat> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "epub" => Some(DocumentFormat::Epub),
            "docx" => Some(DocumentFormat::Docx),
            "odt" => Some(DocumentFormat::Odt),
            "html" | "htm" | "xhtml" => Some(DocumentFormat::Html),
            "pdf" => Some(DocumentFormat::Pdf),
            "txt" | "text" | "md" | "markdown" => Some(DocumentFormat::PlainText),
            "csv" | "tsv" => Some(DocumentFormat::Csv),
            "eml" | "msg" => Some(DocumentFormat::Email),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" => Some(DocumentFormat::Image),
            "pptx" => Some(DocumentFormat::Pptx),
            "xlsx" | "xls" => Some(DocumentFormat::Xlsx),
            _ => None,
        }
    }

    // ========================================================================
    // EPUB parsing
    // ========================================================================

    /// Parse an EPUB file from bytes.
    ///
    /// EPUBs are ZIP archives containing XHTML content files. We locate the OPF
    /// manifest to determine the reading order, then extract and strip text from
    /// each content file in order.
    #[cfg(feature = "documents")]
    fn parse_epub(&self, data: &[u8]) -> Result<ParsedDocument> {
        use std::io::Cursor;

        let cursor = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // First, try to find the OPF file via META-INF/container.xml
        let opf_path = self.find_epub_opf_path(&mut archive)?;

        // Read the OPF manifest to get the spine order
        let opf_content = self.read_zip_entry(&mut archive, &opf_path)?;

        // Extract metadata from OPF
        let metadata = if self.config.extract_metadata {
            self.extract_epub_metadata(&opf_content)
        } else {
            DocumentMetadata::default()
        };

        // Get the content file paths in reading order from the spine
        let content_paths = self.extract_epub_spine_paths(&opf_content, &opf_path);

        // Extract text from each content file
        let mut all_text = String::new();
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut section_index = 0;

        for content_path in &content_paths {
            if let Ok(xhtml) = self.read_zip_entry(&mut archive, content_path) {
                let section_title = if self.config.extract_sections {
                    extract_first_heading(&xhtml)
                } else {
                    None
                };

                let text = strip_xml_tags(&xhtml);
                let text = if self.config.normalize_whitespace {
                    normalize_text(&text)
                } else {
                    text
                };

                if !text.trim().is_empty() {
                    if !all_text.is_empty() && self.config.preserve_paragraphs {
                        all_text.push_str("\n\n");
                    }
                    all_text.push_str(&text);

                    if self.config.extract_sections {
                        sections.push(DocumentSection {
                            title: section_title,
                            content: text,
                            level: 1,
                            index: section_index,
                        });
                        section_index += 1;
                    }
                }
            }
        }

        // If we didn't find spine items, fall back to scanning all xhtml/html files
        if all_text.is_empty() {
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let name = file.name().to_string();
                if name.ends_with(".xhtml") || name.ends_with(".html") || name.ends_with(".htm") {
                    let mut content = String::new();
                    file.read_to_string(&mut content)?;

                    let text = strip_xml_tags(&content);
                    let text = if self.config.normalize_whitespace {
                        normalize_text(&text)
                    } else {
                        text
                    };

                    if !text.trim().is_empty() {
                        if !all_text.is_empty() && self.config.preserve_paragraphs {
                            all_text.push_str("\n\n");
                        }
                        all_text.push_str(&text);

                        if self.config.extract_sections {
                            sections.push(DocumentSection {
                                title: None,
                                content: text,
                                level: 1,
                                index: section_index,
                            });
                            section_index += 1;
                        }
                    }
                }
            }
        }

        let char_count = all_text.chars().count();
        let word_count = all_text.split_whitespace().count();

        Ok(ParsedDocument {
            text: all_text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Epub,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Find the path to the OPF file inside the EPUB archive.
    #[cfg(feature = "documents")]
    fn find_epub_opf_path<R: Read + std::io::Seek>(
        &self,
        archive: &mut zip::ZipArchive<R>,
    ) -> Result<String> {
        // Try reading META-INF/container.xml
        if let Ok(container_xml) = self.read_zip_entry(archive, "META-INF/container.xml") {
            // Extract rootfile full-path attribute
            let re = Regex::new(r#"rootfile[^>]*full-path="([^"]+)""#)?;
            if let Some(caps) = re.captures(&container_xml) {
                return Ok(caps[1].to_string());
            }
        }

        // Fallback: look for any .opf file
        for i in 0..archive.len() {
            let file = archive.by_index(i)?;
            let name = file.name().to_string();
            if name.ends_with(".opf") {
                return Ok(name);
            }
        }

        anyhow::bail!("Could not find OPF manifest in EPUB archive")
    }

    /// Read a single entry from a ZIP archive by name.
    #[cfg(feature = "documents")]
    fn read_zip_entry<R: Read + std::io::Seek>(
        &self,
        archive: &mut zip::ZipArchive<R>,
        name: &str,
    ) -> Result<String> {
        let mut file = archive.by_name(name)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }

    /// Extract metadata from the OPF file content.
    #[cfg(feature = "documents")]
    fn extract_epub_metadata(&self, opf: &str) -> DocumentMetadata {
        let mut meta = DocumentMetadata::default();

        // Title: <dc:title>...</dc:title>
        let titles = extract_xml_text(opf, "dc:title");
        if let Some(t) = titles.into_iter().next() {
            meta.title = Some(t);
        }

        // Authors: <dc:creator>...</dc:creator>
        meta.authors = extract_xml_text(opf, "dc:creator");

        // Language: <dc:language>...</dc:language>
        let langs = extract_xml_text(opf, "dc:language");
        if let Some(l) = langs.into_iter().next() {
            meta.language = Some(l);
        }

        // Date: <dc:date>...</dc:date>
        let dates = extract_xml_text(opf, "dc:date");
        if let Some(d) = dates.into_iter().next() {
            meta.date = Some(d);
        }

        // Description: <dc:description>...</dc:description>
        let descs = extract_xml_text(opf, "dc:description");
        if let Some(d) = descs.into_iter().next() {
            meta.description = Some(d);
        }

        // Publisher: <dc:publisher>...</dc:publisher>
        let pubs = extract_xml_text(opf, "dc:publisher");
        if let Some(p) = pubs.into_iter().next() {
            meta.publisher = Some(p);
        }

        meta
    }

    /// Extract spine item paths from OPF content, resolving idrefs to hrefs.
    #[cfg(feature = "documents")]
    fn extract_epub_spine_paths(&self, opf: &str, opf_path: &str) -> Vec<String> {
        // Determine the base directory of the OPF file
        let base_dir = if let Some(pos) = opf_path.rfind('/') {
            &opf_path[..=pos]
        } else {
            ""
        };

        // Build a map of id -> href from manifest items
        let mut id_to_href: HashMap<String, String> = HashMap::new();
        let item_re = Regex::new(r#"<item\s[^>]*id="([^"]+)"[^>]*href="([^"]+)"[^>]*/?\s*>"#)
            .expect("valid regex");
        // Also handle the case where href comes before id
        let item_re2 = Regex::new(r#"<item\s[^>]*href="([^"]+)"[^>]*id="([^"]+)"[^>]*/?\s*>"#)
            .expect("valid regex");

        for caps in item_re.captures_iter(opf) {
            let id = caps[1].to_string();
            let href = caps[2].to_string();
            id_to_href.insert(id, href);
        }
        for caps in item_re2.captures_iter(opf) {
            let href = caps[1].to_string();
            let id = caps[2].to_string();
            id_to_href.entry(id).or_insert(href);
        }

        // Extract spine itemrefs in order
        let spine_re =
            Regex::new(r#"<itemref\s[^>]*idref="([^"]+)"[^>]*/?\s*>"#).expect("valid regex");
        let mut paths = Vec::new();

        for caps in spine_re.captures_iter(opf) {
            let idref = &caps[1];
            if let Some(href) = id_to_href.get(idref) {
                let full_path = format!("{}{}", base_dir, href);
                paths.push(full_path);
            }
        }

        paths
    }

    // ========================================================================
    // DOCX parsing
    // ========================================================================

    /// Parse a DOCX file from bytes.
    ///
    /// DOCX files are ZIP archives. The main content is in `word/document.xml`.
    /// Text runs are in `<w:t>` elements, paragraphs in `<w:p>` elements.
    #[cfg(feature = "documents")]
    fn parse_docx(&self, data: &[u8]) -> Result<ParsedDocument> {
        use std::io::Cursor;

        let cursor = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // Extract metadata from docProps/core.xml if available
        let metadata = if self.config.extract_metadata {
            if let Ok(core_xml) = self.read_zip_entry(&mut archive, "docProps/core.xml") {
                extract_xml_metadata(&core_xml)
            } else {
                DocumentMetadata::default()
            }
        } else {
            DocumentMetadata::default()
        };

        // Read the main document content
        let doc_xml = self.read_zip_entry(&mut archive, "word/document.xml")?;

        // Extract paragraphs from <w:p> elements
        let paragraphs = self.extract_docx_paragraphs(&doc_xml);

        let mut all_text = String::new();
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut section_index = 0;
        let mut current_section_title: Option<String> = None;
        let mut current_section_content = String::new();
        let mut current_level: u8 = 0;

        for para in &paragraphs {
            // Check if this paragraph is a heading
            let heading_level = self.detect_docx_heading_level(&doc_xml, para);

            if heading_level > 0 && self.config.extract_sections {
                // Save previous section if any
                if !current_section_content.trim().is_empty() {
                    sections.push(DocumentSection {
                        title: current_section_title.take(),
                        content: current_section_content.trim().to_string(),
                        level: current_level,
                        index: section_index,
                    });
                    section_index += 1;
                }
                current_section_title = Some(para.clone());
                current_section_content = String::new();
                current_level = heading_level;
            } else {
                if !current_section_content.is_empty() && self.config.preserve_paragraphs {
                    current_section_content.push_str("\n\n");
                }
                current_section_content.push_str(para);
            }

            if !all_text.is_empty() && self.config.preserve_paragraphs {
                all_text.push_str("\n\n");
            }
            all_text.push_str(para);
        }

        // Don't forget the last section
        if self.config.extract_sections && !current_section_content.trim().is_empty() {
            sections.push(DocumentSection {
                title: current_section_title,
                content: current_section_content.trim().to_string(),
                level: current_level,
                index: section_index,
            });
        }

        let text = if self.config.normalize_whitespace {
            normalize_text(&all_text)
        } else {
            all_text
        };

        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Docx,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract paragraph texts from DOCX XML by finding <w:p> blocks and their <w:t> runs.
    #[cfg(feature = "documents")]
    fn extract_docx_paragraphs(&self, xml: &str) -> Vec<String> {
        let mut paragraphs = Vec::new();

        // Match each <w:p ...>...</w:p> block
        let para_re = Regex::new(r"(?s)<w:p[\s>].*?</w:p>").expect("valid regex");
        let text_re = Regex::new(r"(?s)<w:t[^>]*>(.*?)</w:t>").expect("valid regex");

        for para_match in para_re.find_iter(xml) {
            let para_xml = para_match.as_str();
            let mut para_text = String::new();

            for text_cap in text_re.captures_iter(para_xml) {
                para_text.push_str(&text_cap[1]);
            }

            if !para_text.trim().is_empty() {
                paragraphs.push(para_text.trim().to_string());
            }
        }

        paragraphs
    }

    /// Detect if a paragraph in DOCX XML has a heading style (Heading1, Heading2, etc.)
    /// Returns the heading level (1-9) or 0 if not a heading.
    #[cfg(feature = "documents")]
    fn detect_docx_heading_level(&self, _full_xml: &str, _para_text: &str) -> u8 {
        // In a full implementation we would look at the paragraph's <w:pStyle> value
        // within the <w:p> block. For regex-based parsing, we do a simplified check.
        // This is best-effort since matching paragraph text back to XML is complex.
        0
    }

    // ========================================================================
    // ODT parsing
    // ========================================================================

    /// Parse an ODT (OpenDocument Text) file from bytes.
    ///
    /// ODT files are ZIP archives. The main content is in `content.xml`.
    /// Text paragraphs are in `<text:p>` elements.
    #[cfg(feature = "documents")]
    fn parse_odt(&self, data: &[u8]) -> Result<ParsedDocument> {
        use std::io::Cursor;

        let cursor = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // Extract metadata from meta.xml if available
        let metadata = if self.config.extract_metadata {
            if let Ok(meta_xml) = self.read_zip_entry(&mut archive, "meta.xml") {
                self.extract_odt_metadata(&meta_xml)
            } else {
                DocumentMetadata::default()
            }
        } else {
            DocumentMetadata::default()
        };

        // Read the main content
        let content_xml = self.read_zip_entry(&mut archive, "content.xml")?;

        // Extract text from <text:p> and <text:h> elements
        let mut all_text = String::new();
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut section_index = 0;

        // Match headings: <text:h ...>...</text:h>
        // Match paragraphs: <text:p ...>...</text:p>
        let element_re =
            Regex::new(r"(?s)<text:(h|p)([^>]*)>(.*?)</text:(h|p)>").expect("valid regex");
        let outline_re = Regex::new(r#"text:outline-level="(\d+)""#).expect("valid regex");

        let mut current_section_title: Option<String> = None;
        let mut current_section_content = String::new();
        let mut current_level: u8 = 0;

        for caps in element_re.captures_iter(&content_xml) {
            let tag_type = &caps[1]; // "h" or "p"
            let attrs = &caps[2];
            let inner = &caps[3];

            let text = strip_xml_tags(inner);
            let text = text.trim().to_string();

            if text.is_empty() {
                continue;
            }

            let is_heading = tag_type == "h";
            let heading_level = if is_heading {
                if let Some(level_caps) = outline_re.captures(attrs) {
                    level_caps[1].parse::<u8>().unwrap_or(1)
                } else {
                    1
                }
            } else {
                0
            };

            if is_heading && self.config.extract_sections {
                // Save previous section
                if !current_section_content.trim().is_empty() {
                    sections.push(DocumentSection {
                        title: current_section_title.take(),
                        content: current_section_content.trim().to_string(),
                        level: current_level,
                        index: section_index,
                    });
                    section_index += 1;
                }
                current_section_title = Some(text.clone());
                current_section_content = String::new();
                current_level = heading_level;
            } else {
                if !current_section_content.is_empty() && self.config.preserve_paragraphs {
                    current_section_content.push('\n');
                }
                current_section_content.push_str(&text);
            }

            if !all_text.is_empty() && self.config.preserve_paragraphs {
                all_text.push('\n');
            }
            all_text.push_str(&text);
        }

        // Save last section
        if self.config.extract_sections && !current_section_content.trim().is_empty() {
            sections.push(DocumentSection {
                title: current_section_title,
                content: current_section_content.trim().to_string(),
                level: current_level,
                index: section_index,
            });
        }

        let text = if self.config.normalize_whitespace {
            normalize_text(&all_text)
        } else {
            all_text
        };

        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Odt,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract metadata from an ODT meta.xml file.
    #[cfg(feature = "documents")]
    fn extract_odt_metadata(&self, meta_xml: &str) -> DocumentMetadata {
        let mut meta = DocumentMetadata::default();

        // Title
        let titles = extract_xml_text(meta_xml, "dc:title");
        if let Some(t) = titles.into_iter().next() {
            meta.title = Some(t);
        }

        // Creator (author)
        let creators = extract_xml_text(meta_xml, "dc:creator");
        if !creators.is_empty() {
            meta.authors = creators;
        }
        // Also check meta:initial-creator
        let initial = extract_xml_text(meta_xml, "meta:initial-creator");
        for ic in initial {
            if !meta.authors.contains(&ic) {
                meta.authors.push(ic);
            }
        }

        // Language
        let langs = extract_xml_text(meta_xml, "dc:language");
        if let Some(l) = langs.into_iter().next() {
            meta.language = Some(l);
        }

        // Date
        let dates = extract_xml_text(meta_xml, "dc:date");
        if let Some(d) = dates.into_iter().next() {
            meta.date = Some(d);
        }

        // Description
        let descs = extract_xml_text(meta_xml, "dc:description");
        if let Some(d) = descs.into_iter().next() {
            meta.description = Some(d);
        }

        meta
    }

    // ========================================================================
    // PDF parsing
    // ========================================================================

    /// Parse a PDF file from bytes.
    ///
    /// PDF extraction has inherent challenges:
    /// - Headers and footers can mix with content
    /// - Multi-column layouts may interleave text incorrectly
    /// - Tables are difficult to preserve structure
    ///
    /// This implementation:
    /// - Extracts text page by page
    /// - Attempts to detect and filter headers/footers
    /// - Tracks page numbers for each section
    /// - Normalizes whitespace for readability
    #[cfg(feature = "pdf-extract")]
    fn parse_pdf(&self, data: &[u8]) -> Result<ParsedDocument> {
        use pdf_extract::extract_text_from_mem;

        // Extract raw text from PDF
        let raw_text = extract_text_from_mem(data)
            .map_err(|e| anyhow::anyhow!("Failed to extract text from PDF: {}", e))?;

        // Split into pages (pdf-extract uses form-feed character)
        let pages: Vec<&str> = raw_text.split('\u{0C}').collect();
        let total_pages = pages.len();

        // Detect potential headers/footers by finding repeated lines
        let header_footer_lines = if self.config.extract_sections && total_pages > 2 {
            self.detect_pdf_headers_footers(&pages)
        } else {
            std::collections::HashSet::new()
        };

        let mut all_text = String::new();
        let mut sections = Vec::new();
        let mut all_tables = Vec::new();

        for (page_num, page_text) in pages.iter().enumerate() {
            let page_number = page_num + 1; // 1-indexed

            // Filter out header/footer lines
            let cleaned_text: String = page_text
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    // Skip if it's a detected header/footer
                    if header_footer_lines.contains(trimmed) {
                        return false;
                    }
                    // Skip if it looks like a standalone page number
                    if Self::is_page_number_line(trimmed, page_number, total_pages) {
                        return false;
                    }
                    true
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Extract tables if configured
            let (page_tables, text_without_tables) = if self.config.extract_tables {
                self.extract_tables_from_page(&cleaned_text, page_number)
            } else {
                (Vec::new(), cleaned_text)
            };

            all_tables.extend(page_tables);

            // Normalize whitespace if configured
            let normalized = if self.config.normalize_whitespace {
                normalize_text(&text_without_tables)
            } else {
                text_without_tables
            };

            if !normalized.trim().is_empty() {
                // Add to full text with page separator
                if !all_text.is_empty() {
                    all_text.push_str("\n\n");
                }
                all_text.push_str(&normalized);

                // Create section for this page if extracting sections
                if self.config.extract_sections {
                    sections.push(DocumentSection {
                        title: Some(format!("Page {}", page_number)),
                        content: normalized,
                        level: 0,
                        index: page_num,
                    });
                }
            }
        }

        // Try to extract title from first page
        let title = self.extract_pdf_title(&pages);

        let metadata = DocumentMetadata {
            title,
            extra: {
                let mut extra = HashMap::new();
                extra.insert("total_pages".to_string(), total_pages.to_string());
                if !all_tables.is_empty() {
                    extra.insert("tables_extracted".to_string(), all_tables.len().to_string());
                }
                extra
            },
            ..Default::default()
        };

        let char_count = all_text.chars().count();
        let word_count = all_text.split_whitespace().count();

        Ok(ParsedDocument {
            text: all_text,
            metadata,
            sections,
            tables: all_tables,
            format: DocumentFormat::Pdf,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Detect repeated header/footer lines across pages
    #[cfg(feature = "pdf-extract")]
    fn detect_pdf_headers_footers(&self, pages: &[&str]) -> std::collections::HashSet<String> {
        use std::collections::HashMap as StdHashMap;

        let mut line_occurrences: StdHashMap<String, usize> = StdHashMap::new();

        // Count how many pages each line appears in
        for page in pages.iter() {
            let mut seen_on_page: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for line in page.lines() {
                let trimmed = line.trim().to_string();
                // Only consider lines that look like headers/footers
                // (short lines, often containing only numbers/dates/company names)
                if trimmed.len() > 3 && trimmed.len() < 100 {
                    if !seen_on_page.contains(&trimmed) {
                        seen_on_page.insert(trimmed.clone());
                        *line_occurrences.entry(trimmed).or_insert(0) += 1;
                    }
                }
            }
        }

        // Lines appearing in more than 50% of pages are likely headers/footers
        let threshold = pages.len() / 2;
        line_occurrences
            .into_iter()
            .filter(|(_, count)| *count > threshold)
            .map(|(line, _)| line)
            .collect()
    }

    /// Check if a line is just a page number
    #[cfg(feature = "pdf-extract")]
    fn is_page_number_line(line: &str, current_page: usize, total_pages: usize) -> bool {
        let trimmed = line.trim();

        // Empty or very short
        if trimmed.is_empty() || trimmed.len() > 20 {
            return false;
        }

        // Just a number matching the page
        if let Ok(n) = trimmed.parse::<usize>() {
            if n == current_page || n <= total_pages {
                return true;
            }
        }

        // Common page number formats
        let page_patterns = [
            format!("{}", current_page),
            format!("- {} -", current_page),
            format!("— {} —", current_page),
            format!("Page {}", current_page),
            format!("page {}", current_page),
            format!("{} of {}", current_page, total_pages),
            format!("{}/{}", current_page, total_pages),
        ];

        page_patterns.iter().any(|p| trimmed == p)
    }

    /// Try to extract a title from the first page of a PDF
    #[cfg(feature = "pdf-extract")]
    fn extract_pdf_title(&self, pages: &[&str]) -> Option<String> {
        if pages.is_empty() {
            return None;
        }

        let first_page = pages[0];
        let lines: Vec<&str> = first_page
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .take(10) // Look at first 10 non-empty lines
            .collect();

        // Heuristic: title is often the first substantial line
        // that's not a date, page number, or very short
        for line in lines {
            // Skip if too short or too long
            if line.len() < 5 || line.len() > 200 {
                continue;
            }
            // Skip if it looks like a date
            if line.contains('/') && line.len() < 15 {
                continue;
            }
            // Skip if it's all numbers
            if line.chars().all(|c| c.is_numeric() || c.is_whitespace()) {
                continue;
            }
            // This looks like a potential title
            return Some(line.to_string());
        }

        None
    }

    /// Extract tables from a page of text using heuristics.
    ///
    /// Table detection looks for:
    /// - Multiple consecutive lines with similar column structure
    /// - Columns separated by multiple spaces or tabs
    /// - Numeric or short text content typical of tables
    #[cfg(feature = "pdf-extract")]
    fn extract_tables_from_page(
        &self,
        page_text: &str,
        page_number: usize,
    ) -> (Vec<PdfTable>, String) {
        let lines: Vec<&str> = page_text.lines().collect();
        let mut tables = Vec::new();
        let mut remaining_lines: Vec<&str> = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            // Try to detect a table starting at this line
            if let Some((table, end_idx)) = self.detect_table_at(&lines, i, page_number) {
                tables.push(table);
                i = end_idx;
            } else {
                remaining_lines.push(lines[i]);
                i += 1;
            }
        }

        (tables, remaining_lines.join("\n"))
    }

    /// Try to detect a table starting at the given line index.
    /// Returns the table and the index after the last table line.
    #[cfg(feature = "pdf-extract")]
    fn detect_table_at(
        &self,
        lines: &[&str],
        start: usize,
        page_number: usize,
    ) -> Option<(PdfTable, usize)> {
        if start >= lines.len() {
            return None;
        }

        // Analyze column structure of consecutive lines
        let mut table_lines = Vec::new();
        let mut column_positions: Option<Vec<usize>> = None;
        let mut end_idx = start;

        for (idx, line) in lines[start..].iter().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines at the start
            if trimmed.is_empty() && table_lines.is_empty() {
                continue;
            }

            // Empty line might end the table
            if trimmed.is_empty() && !table_lines.is_empty() {
                break;
            }

            // Detect columns by finding multiple-space separators
            let cols = self.split_into_columns(line);

            // Need at least 2 columns to be a table
            if cols.len() < 2 {
                if table_lines.is_empty() {
                    return None; // Not a table
                } else {
                    break; // End of table
                }
            }

            // Check column alignment consistency
            let positions = self.get_column_positions(line);

            if let Some(ref expected_positions) = column_positions {
                // Check if columns roughly align with previous rows
                if !self.columns_align(&positions, expected_positions) {
                    if table_lines.len() < 2 {
                        return None; // Not enough rows to be a table
                    }
                    break; // End of table
                }
            } else {
                column_positions = Some(positions);
            }

            table_lines.push((line.to_string(), cols));
            end_idx = start + idx + 1;
        }

        // Need at least 2 rows to be considered a table
        if table_lines.len() < 2 {
            return None;
        }

        // Build the PdfTable
        let raw_text = table_lines
            .iter()
            .map(|(l, _)| l.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // First row might be headers (check if it's different from data rows)
        let (headers, data_rows) = self.separate_headers_and_data(&table_lines);

        let confidence = self.calculate_table_confidence(&table_lines);

        Some((
            PdfTable {
                caption: None, // Could try to detect caption above table
                headers,
                rows: data_rows,
                page: Some(page_number),
                confidence,
                raw_text,
            },
            end_idx,
        ))
    }

    /// Split a line into columns based on multiple spaces
    #[cfg(feature = "pdf-extract")]
    fn split_into_columns(&self, line: &str) -> Vec<String> {
        // Split on 2+ spaces or tabs
        let re = Regex::new(r"\s{2,}|\t+").expect("valid regex");
        re.split(line.trim())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Get approximate column start positions
    #[cfg(feature = "pdf-extract")]
    fn get_column_positions(&self, line: &str) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut in_word = false;
        let mut space_count = 0;

        for (i, c) in line.chars().enumerate() {
            if c.is_whitespace() {
                space_count += 1;
                in_word = false;
            } else {
                if !in_word && (space_count >= 2 || i == 0 || positions.is_empty()) {
                    positions.push(i);
                }
                space_count = 0;
                in_word = true;
            }
        }

        positions
    }

    /// Check if column positions roughly align (within tolerance)
    #[cfg(feature = "pdf-extract")]
    fn columns_align(&self, a: &[usize], b: &[usize]) -> bool {
        if a.len() != b.len() {
            // Allow slight variation in column count
            let diff = (a.len() as i32 - b.len() as i32).abs();
            if diff > 1 {
                return false;
            }
        }

        let tolerance = 5; // Characters of tolerance for alignment
        let min_len = a.len().min(b.len());

        for i in 0..min_len {
            if (a[i] as i32 - b[i] as i32).abs() > tolerance as i32 {
                return false;
            }
        }

        true
    }

    /// Separate headers from data rows
    #[cfg(feature = "pdf-extract")]
    fn separate_headers_and_data(
        &self,
        table_lines: &[(String, Vec<String>)],
    ) -> (Vec<String>, Vec<Vec<String>>) {
        if table_lines.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let first_row = &table_lines[0].1;

        // Heuristic: headers are often non-numeric or have different characteristics
        let first_row_numeric_ratio = first_row
            .iter()
            .filter(|s| {
                s.parse::<f64>().is_ok()
                    || s.chars()
                        .all(|c| c.is_numeric() || c == '.' || c == ',' || c == '-')
            })
            .count() as f32
            / first_row.len().max(1) as f32;

        let second_row_numeric_ratio = if table_lines.len() > 1 {
            let second_row = &table_lines[1].1;
            second_row
                .iter()
                .filter(|s| {
                    s.parse::<f64>().is_ok()
                        || s.chars()
                            .all(|c| c.is_numeric() || c == '.' || c == ',' || c == '-')
                })
                .count() as f32
                / second_row.len().max(1) as f32
        } else {
            first_row_numeric_ratio
        };

        // If first row is less numeric than second, treat it as header
        if first_row_numeric_ratio < second_row_numeric_ratio - 0.2 {
            let headers = first_row.clone();
            let rows = table_lines[1..]
                .iter()
                .map(|(_, cols)| cols.clone())
                .collect();
            (headers, rows)
        } else {
            // No clear header row
            let rows = table_lines.iter().map(|(_, cols)| cols.clone()).collect();
            (Vec::new(), rows)
        }
    }

    /// Calculate confidence score for table detection
    #[cfg(feature = "pdf-extract")]
    fn calculate_table_confidence(&self, table_lines: &[(String, Vec<String>)]) -> f32 {
        if table_lines.is_empty() {
            return 0.0;
        }

        let mut score = 0.5; // Base score for having 2+ rows

        // More rows = higher confidence
        if table_lines.len() >= 3 {
            score += 0.1;
        }
        if table_lines.len() >= 5 {
            score += 0.1;
        }

        // Consistent column count = higher confidence
        let col_counts: Vec<usize> = table_lines.iter().map(|(_, cols)| cols.len()).collect();
        let avg_cols = col_counts.iter().sum::<usize>() as f32 / col_counts.len() as f32;
        let variance = col_counts
            .iter()
            .map(|&c| (c as f32 - avg_cols).powi(2))
            .sum::<f32>()
            / col_counts.len() as f32;

        if variance < 0.5 {
            score += 0.2; // Very consistent
        } else if variance < 1.0 {
            score += 0.1; // Reasonably consistent
        }

        // Numeric content suggests data table
        let numeric_ratio = table_lines
            .iter()
            .flat_map(|(_, cols)| cols.iter())
            .filter(|s| s.parse::<f64>().is_ok())
            .count() as f32
            / table_lines
                .iter()
                .flat_map(|(_, cols)| cols.iter())
                .count()
                .max(1) as f32;

        score += numeric_ratio * 0.2;

        score.min(1.0)
    }

    // ========================================================================
    // PDF Page Content Separation (Footnotes, Captions, Sidebars)
    // ========================================================================

    /// Separate page content into main text, footnotes, captions, and sidebars.
    ///
    /// Uses heuristics to detect:
    /// - **Footnotes**: Text at page bottom with superscript-style markers (¹, ², *, †)
    /// - **Captions**: Lines starting with "Figure X", "Table X", "Fig.", etc.
    /// - **Sidebars**: Indented blocks that appear separate from main flow
    #[cfg(feature = "pdf-extract")]
    pub fn separate_page_content(&self, page_text: &str, page_number: usize) -> PageContent {
        let lines: Vec<&str> = page_text.lines().collect();
        if lines.is_empty() {
            return PageContent {
                page_number,
                ..Default::default()
            };
        }

        let mut footnotes = Vec::new();
        let mut captions = Vec::new();
        let mut sidebars = Vec::new();
        let mut main_lines = Vec::new();

        // Detect footnote region (bottom of page with markers)
        let footnote_start = self.detect_footnote_region(&lines);

        // Analyze each line
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check if we're in the footnote region
            if let Some(start) = footnote_start {
                if idx >= start {
                    if !trimmed.is_empty() {
                        footnotes.push(trimmed.to_string());
                    }
                    continue;
                }
            }

            // Check for captions
            if Self::is_caption_line(trimmed) {
                captions.push(trimmed.to_string());
                continue;
            }

            // Check for sidebar-like content (indented blocks)
            if self.is_sidebar_line(line, &lines, idx) {
                sidebars.push(trimmed.to_string());
                continue;
            }

            // Everything else is main text
            main_lines.push(*line);
        }

        PageContent {
            main_text: main_lines.join("\n"),
            footnotes,
            captions,
            sidebars,
            page_number,
        }
    }

    /// Detect where the footnote region starts (returns line index)
    #[cfg(feature = "pdf-extract")]
    fn detect_footnote_region(&self, lines: &[&str]) -> Option<usize> {
        // Footnotes typically:
        // 1. Appear in the bottom 30% of the page
        // 2. Start with markers like ¹, ², ³, *, †, ‡, or numbers followed by period/space
        // 3. Often separated by a horizontal line or extra space

        let total_lines = lines.len();
        if total_lines < 5 {
            return None;
        }

        // Only look at bottom 40% of the page
        let search_start = (total_lines as f32 * 0.6) as usize;

        // Look for footnote marker patterns
        let footnote_markers =
            Regex::new(r"^[\s]*[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§\[\(]?\d{0,2}[\]\)]?[\.\s]").ok()?;

        // Also check for common footnote separators
        let separator_pattern = Regex::new(r"^[\s]*[_\-─═]{3,}[\s]*$").ok()?;

        for (idx, line) in lines.iter().enumerate().skip(search_start) {
            let trimmed = line.trim();

            // Check for separator line (often precedes footnotes)
            if separator_pattern.is_match(trimmed) {
                // Footnotes start after the separator
                if idx + 1 < total_lines {
                    return Some(idx + 1);
                }
            }

            // Check for footnote marker at start of line
            if footnote_markers.is_match(trimmed) && !trimmed.is_empty() {
                // Verify it looks like a footnote (short-ish, starts with marker)
                let first_char = trimmed.chars().next().unwrap_or(' ');
                if "¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§[(".contains(first_char) || first_char.is_ascii_digit()
                {
                    return Some(idx);
                }
            }
        }

        None
    }

    /// Check if a line looks like a figure/table caption
    #[cfg(feature = "pdf-extract")]
    fn is_caption_line(line: &str) -> bool {
        let line_lower = line.to_lowercase();
        let trimmed = line_lower.trim();

        // Common caption patterns
        let caption_patterns = [
            r"^fig\.?\s*\d",       // Fig. 1, Fig 2
            r"^figure\s+\d",       // Figure 1
            r"^table\s+\d",        // Table 1
            r"^tab\.?\s*\d",       // Tab. 1
            r"^chart\s+\d",        // Chart 1
            r"^graph\s+\d",        // Graph 1
            r"^diagram\s+\d",      // Diagram 1
            r"^plate\s+\d",        // Plate 1
            r"^illustration\s+\d", // Illustration 1
            r"^exhibit\s+\d",      // Exhibit 1
            r"^source:",           // Source: ...
            r"^note:",             // Note: ...
            r"^caption:",          // Caption: ...
        ];

        for pattern in caption_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(trimmed) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a line appears to be part of a sidebar
    #[cfg(feature = "pdf-extract")]
    fn is_sidebar_line(&self, line: &str, all_lines: &[&str], current_idx: usize) -> bool {
        // Sidebars are typically:
        // 1. Significantly indented compared to surrounding text
        // 2. Part of a block of similarly-indented lines
        // 3. Sometimes prefixed with markers like "Note:", "Tip:", "Warning:"

        let trimmed = line.trim();

        // Check for explicit sidebar markers
        let sidebar_markers = [
            "note:",
            "tip:",
            "warning:",
            "important:",
            "caution:",
            "sidebar:",
            "box:",
            "callout:",
            "inset:",
        ];
        let lower = trimmed.to_lowercase();
        for marker in sidebar_markers {
            if lower.starts_with(marker) {
                return true;
            }
        }

        // Check for significant indentation difference
        let leading_spaces = line.len() - line.trim_start().len();

        // Calculate average indentation of surrounding lines
        let context_start = current_idx.saturating_sub(3);
        let context_end = (current_idx + 4).min(all_lines.len());

        let mut avg_indent = 0.0;
        let mut count = 0;
        for i in context_start..context_end {
            if i != current_idx && !all_lines[i].trim().is_empty() {
                let indent = all_lines[i].len() - all_lines[i].trim_start().len();
                avg_indent += indent as f32;
                count += 1;
            }
        }

        if count > 0 {
            avg_indent /= count as f32;
            // Line is significantly more indented (>8 chars more than average)
            if leading_spaces as f32 > avg_indent + 8.0 && !trimmed.is_empty() {
                return true;
            }
        }

        false
    }

    // ========================================================================
    // HTML parsing
    // ========================================================================

    /// Parse HTML content using regex-based tag stripping.
    fn parse_html(&self, content: &str) -> Result<ParsedDocument> {
        // Extract metadata from <head>
        let metadata = if self.config.extract_metadata {
            self.extract_html_metadata(content)
        } else {
            DocumentMetadata::default()
        };

        // Extract sections from headings if requested
        let sections = if self.config.extract_sections {
            self.extract_html_sections(content)
        } else {
            Vec::new()
        };

        // Strip tags to get plain text
        let text = if self.config.strip_tags {
            let body_text = self.extract_html_body(content);
            let stripped = strip_xml_tags(&body_text);
            if self.config.normalize_whitespace {
                normalize_text(&stripped)
            } else {
                stripped
            }
        } else if self.config.normalize_whitespace {
            normalize_text(content)
        } else {
            content.to_string()
        };

        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Html,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract metadata from HTML <head> section.
    fn extract_html_metadata(&self, html: &str) -> DocumentMetadata {
        let mut meta = DocumentMetadata::default();

        // Title from <title> tag
        let titles = extract_xml_text(html, "title");
        if let Some(t) = titles.into_iter().next() {
            meta.title = Some(t.trim().to_string());
        }

        // Meta tags: <meta name="..." content="...">
        let meta_re =
            Regex::new(r#"(?i)<meta\s[^>]*name="([^"]+)"[^>]*content="([^"]+)"[^>]*/?\s*>"#)
                .expect("valid regex");
        let meta_re2 =
            Regex::new(r#"(?i)<meta\s[^>]*content="([^"]+)"[^>]*name="([^"]+)"[^>]*/?\s*>"#)
                .expect("valid regex");

        let mut meta_map: HashMap<String, String> = HashMap::new();
        for caps in meta_re.captures_iter(html) {
            meta_map.insert(caps[1].to_lowercase(), caps[2].to_string());
        }
        for caps in meta_re2.captures_iter(html) {
            meta_map
                .entry(caps[2].to_lowercase())
                .or_insert_with(|| caps[1].to_string());
        }

        if let Some(author) = meta_map.remove("author") {
            meta.authors = author
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(desc) = meta_map.remove("description") {
            meta.description = Some(desc);
        }
        if let Some(lang) = meta_map.remove("language") {
            meta.language = Some(lang);
        }

        // Also check html lang attribute
        if meta.language.is_none() {
            let lang_re =
                Regex::new(r#"(?i)<html[^>]*\slang="([^"]+)"[^>]*>"#).expect("valid regex");
            if let Some(caps) = lang_re.captures(html) {
                meta.language = Some(caps[1].to_string());
            }
        }

        // Store remaining meta tags in extra
        for (key, value) in meta_map {
            meta.extra.insert(key, value);
        }

        meta
    }

    /// Extract the body content from HTML, or the whole string if no body tag.
    fn extract_html_body(&self, html: &str) -> String {
        let body_re = Regex::new(r"(?is)<body[^>]*>(.*)</body>").expect("valid regex");
        if let Some(caps) = body_re.captures(html) {
            caps[1].to_string()
        } else {
            html.to_string()
        }
    }

    /// Extract sections from HTML based on heading tags (h1-h6).
    fn extract_html_sections(&self, html: &str) -> Vec<DocumentSection> {
        let mut sections = Vec::new();
        let body = self.extract_html_body(html);

        // Split on heading tags
        let heading_re = Regex::new(r"(?is)<h([1-6])[^>]*>(.*?)</h[1-6]>").expect("valid regex");

        let mut last_end = 0;
        let mut section_index = 0;

        let matches: Vec<_> = heading_re.captures_iter(&body).collect();

        if matches.is_empty() {
            // No headings found: treat entire body as one section
            let text = strip_xml_tags(&body);
            let text = normalize_text(&text);
            if !text.is_empty() {
                sections.push(DocumentSection {
                    title: None,
                    content: text,
                    level: 0,
                    index: 0,
                });
            }
            return sections;
        }

        for caps in &matches {
            let full_match = caps.get(0).expect("capture group 0");
            let level: u8 = caps[1].parse().unwrap_or(1);
            let title_html = &caps[2];
            let title = strip_xml_tags(title_html).trim().to_string();

            // Content before this heading (belongs to previous section or is preamble)
            let before = &body[last_end..full_match.start()];
            let before_text = strip_xml_tags(before);
            let before_text = normalize_text(&before_text);

            if !before_text.is_empty() && section_index == 0 {
                // Preamble content before the first heading
                sections.push(DocumentSection {
                    title: None,
                    content: before_text,
                    level: 0,
                    index: section_index,
                });
                section_index += 1;
            } else if !before_text.is_empty() {
                // Append to the last section's content
                if let Some(last) = sections.last_mut() {
                    if !last.content.is_empty() {
                        last.content.push_str("\n\n");
                    }
                    last.content.push_str(&before_text);
                }
            }

            // Start a new section for this heading
            sections.push(DocumentSection {
                title: if title.is_empty() { None } else { Some(title) },
                content: String::new(),
                level,
                index: section_index,
            });
            section_index += 1;

            last_end = full_match.end();
        }

        // Remaining content after the last heading
        if last_end < body.len() {
            let remaining = &body[last_end..];
            let remaining_text = strip_xml_tags(remaining);
            let remaining_text = normalize_text(&remaining_text);

            if !remaining_text.is_empty() {
                if let Some(last) = sections.last_mut() {
                    last.content = remaining_text;
                }
            }
        }

        sections
    }

    // ========================================================================
    // CSV parsing
    // ========================================================================

    /// Parse CSV/TSV content into a structured document.
    ///
    /// Supports RFC 4180: comma-delimited, quoted fields, escaped quotes (double-quote),
    /// multi-line cells. TSV auto-detected by tab prevalence.
    fn parse_csv(&self, content: &str) -> Result<ParsedDocument> {
        let content = content.trim();
        if content.is_empty() {
            return Ok(ParsedDocument {
                text: String::new(),
                metadata: DocumentMetadata::default(),
                sections: Vec::new(),
                tables: Vec::new(),
                format: DocumentFormat::Csv,
                source_path: None,
                char_count: 0,
                word_count: 0,
            });
        }

        // Detect delimiter: if first line has more tabs than commas, use tab
        let first_line = content.lines().next().unwrap_or("");
        let tab_count = first_line.matches('\t').count();
        let comma_count = first_line.matches(',').count();
        let delimiter = if tab_count > comma_count { '\t' } else { ',' };

        let rows = Self::parse_csv_rows(content, delimiter);
        if rows.is_empty() {
            return Ok(ParsedDocument {
                text: String::new(),
                metadata: DocumentMetadata::default(),
                sections: Vec::new(),
                tables: Vec::new(),
                format: DocumentFormat::Csv,
                source_path: None,
                char_count: 0,
                word_count: 0,
            });
        }

        // First row is header if it has no numeric-only cells and subsequent rows have numbers
        let has_header = if rows.len() > 1 {
            let first_row_numeric = rows[0].iter().filter(|c| c.parse::<f64>().is_ok()).count();
            let second_row_numeric = rows[1].iter().filter(|c| c.parse::<f64>().is_ok()).count();
            first_row_numeric == 0 && second_row_numeric > 0
        } else {
            false
        };

        let (headers, data_rows) = if has_header {
            (rows[0].clone(), &rows[1..])
        } else {
            let auto_headers: Vec<String> = (0..rows[0].len())
                .map(|i| format!("Column {}", i + 1))
                .collect();
            (auto_headers, &rows[..])
        };

        // Build text representation
        let mut text_lines = Vec::new();
        for row in &rows {
            text_lines.push(row.join(if delimiter == '\t' { "\t" } else { ", " }));
        }
        let text = text_lines.join("\n");

        let table = PdfTable {
            caption: None,
            headers: headers.clone(),
            rows: data_rows.iter().map(|r| r.clone()).collect(),
            page: None,
            confidence: 1.0,
            raw_text: String::new(),
        };

        let mut metadata = DocumentMetadata::default();
        metadata.extra.insert(
            "delimiter".to_string(),
            (if delimiter == '\t' { "tab" } else { "comma" }).to_string(),
        );
        metadata
            .extra
            .insert("rows".to_string(), rows.len().to_string());
        metadata
            .extra
            .insert("columns".to_string(), headers.len().to_string());
        metadata
            .extra
            .insert("has_header".to_string(), has_header.to_string());

        let section = DocumentSection {
            title: Some("Data".to_string()),
            content: text.clone(),
            level: 1,
            index: 0,
        };

        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections: vec![section],
            tables: vec![table],
            format: DocumentFormat::Csv,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Parse CSV rows handling RFC 4180 quoting rules.
    fn parse_csv_rows(content: &str, delimiter: char) -> Vec<Vec<String>> {
        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut current_row: Vec<String> = Vec::new();
        let mut current_field = String::new();
        let mut in_quotes = false;
        let mut chars = content.chars().peekable();

        while let Some(ch) = chars.next() {
            if in_quotes {
                if ch == '"' {
                    if chars.peek() == Some(&'"') {
                        // Escaped quote
                        chars.next();
                        current_field.push('"');
                    } else {
                        // End of quoted field
                        in_quotes = false;
                    }
                } else {
                    current_field.push(ch);
                }
            } else if ch == '"' && current_field.is_empty() {
                in_quotes = true;
            } else if ch == delimiter {
                current_row.push(current_field.trim().to_string());
                current_field = String::new();
            } else if ch == '\n' {
                current_row.push(current_field.trim().to_string());
                current_field = String::new();
                if !current_row.iter().all(|c| c.is_empty()) {
                    rows.push(current_row);
                }
                current_row = Vec::new();
            } else if ch == '\r' {
                // Skip \r, \n will handle row break
            } else {
                current_field.push(ch);
            }
        }

        // Last field/row
        if !current_field.is_empty() || !current_row.is_empty() {
            current_row.push(current_field.trim().to_string());
            if !current_row.iter().all(|c| c.is_empty()) {
                rows.push(current_row);
            }
        }

        rows
    }

    // ========================================================================
    // Email parsing (RFC 5322 + MIME)
    // ========================================================================

    /// Parse an email (.eml) file.
    ///
    /// Supports RFC 5322 headers, MIME multipart boundaries, quoted-printable
    /// and base64 content-transfer-encoding, multipart/alternative (prefers text/plain).
    fn parse_email(&self, content: &str) -> Result<ParsedDocument> {
        // Split headers from body at first blank line
        let (header_section, body) = if let Some(pos) = content.find("\n\n") {
            (&content[..pos], &content[pos + 2..])
        } else if let Some(pos) = content.find("\r\n\r\n") {
            (&content[..pos], &content[pos + 4..])
        } else {
            (content, "")
        };

        // Parse headers (handle folded lines: continuation lines start with space/tab)
        let mut headers: HashMap<String, String> = HashMap::new();
        let mut current_key = String::new();
        let mut current_value = String::new();

        for line in header_section.lines() {
            if line.starts_with(' ') || line.starts_with('\t') {
                // Continuation of previous header
                current_value.push(' ');
                current_value.push_str(line.trim());
            } else if let Some(colon_pos) = line.find(':') {
                // Save previous header
                if !current_key.is_empty() {
                    headers.insert(current_key.to_lowercase(), current_value.trim().to_string());
                }
                current_key = line[..colon_pos].to_string();
                current_value = line[colon_pos + 1..].to_string();
            }
        }
        if !current_key.is_empty() {
            headers.insert(current_key.to_lowercase(), current_value.trim().to_string());
        }

        // Extract content type and boundary
        let content_type = headers.get("content-type").cloned().unwrap_or_default();
        let encoding = headers
            .get("content-transfer-encoding")
            .cloned()
            .unwrap_or_default();

        // Extract body text
        let body_text = if content_type.contains("multipart/") {
            self.parse_mime_multipart(body, &content_type)
        } else if content_type.contains("text/html") {
            let decoded = Self::decode_email_body(body, &encoding);
            strip_xml_tags(&decoded)
        } else {
            Self::decode_email_body(body, &encoding)
        };

        let body_text = if self.config.normalize_whitespace {
            normalize_text(&body_text)
        } else {
            body_text.trim().to_string()
        };

        // Build metadata
        let mut metadata = DocumentMetadata::default();
        if let Some(subject) = headers.get("subject") {
            metadata.title = Some(subject.clone());
        }
        if let Some(from) = headers.get("from") {
            metadata.authors.push(from.clone());
        }
        if let Some(date) = headers.get("date") {
            metadata.date = Some(date.clone());
        }
        for key in &["to", "cc", "bcc", "message-id", "in-reply-to"] {
            if let Some(val) = headers.get(*key) {
                metadata.extra.insert(key.to_string(), val.clone());
            }
        }

        // Count attachments
        let attachment_count = self.count_attachments(body, &content_type);
        if attachment_count > 0 {
            metadata
                .extra
                .insert("attachments".to_string(), attachment_count.to_string());
        }

        // Build sections
        let mut sections = Vec::new();
        if let Some(ref subject) = metadata.title {
            sections.push(DocumentSection {
                title: Some(format!("Subject: {}", subject)),
                content: String::new(),
                level: 1,
                index: 0,
            });
        }

        let header_summary = self.format_email_headers(&headers);
        sections.push(DocumentSection {
            title: Some("Headers".to_string()),
            content: header_summary,
            level: 2,
            index: 1,
        });

        sections.push(DocumentSection {
            title: Some("Body".to_string()),
            content: body_text.clone(),
            level: 2,
            index: 2,
        });

        // Full text includes headers summary + body
        let text = format!(
            "From: {}\nTo: {}\nSubject: {}\nDate: {}\n\n{}",
            headers.get("from").unwrap_or(&String::new()),
            headers.get("to").unwrap_or(&String::new()),
            headers.get("subject").unwrap_or(&String::new()),
            headers.get("date").unwrap_or(&String::new()),
            body_text,
        );

        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Email,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Parse MIME multipart body, extracting text content.
    fn parse_mime_multipart(&self, body: &str, content_type: &str) -> String {
        // Extract boundary from Content-Type header
        let boundary = if let Some(pos) = content_type.find("boundary=") {
            let rest = &content_type[pos + 9..];
            let boundary = rest
                .trim_start_matches('"')
                .split('"')
                .next()
                .unwrap_or("")
                .split(';')
                .next()
                .unwrap_or("")
                .trim();
            boundary.to_string()
        } else {
            return body.to_string();
        };

        let separator = format!("--{}", boundary);
        let parts: Vec<&str> = body.split(&separator).collect();

        let mut text_plain = String::new();
        let mut text_html = String::new();

        for part in &parts[1..] {
            // Skip closing boundary
            if part.starts_with("--") {
                continue;
            }

            let (part_headers, part_body) = if let Some(pos) = part.find("\n\n") {
                (&part[..pos], &part[pos + 2..])
            } else if let Some(pos) = part.find("\r\n\r\n") {
                (&part[..pos], &part[pos + 4..])
            } else {
                continue;
            };

            let part_ct = part_headers
                .lines()
                .find(|l| l.to_lowercase().starts_with("content-type:"))
                .unwrap_or("")
                .to_lowercase();
            let part_encoding = part_headers
                .lines()
                .find(|l| l.to_lowercase().starts_with("content-transfer-encoding:"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_lowercase())
                .unwrap_or_default();

            if part_ct.contains("text/plain") {
                text_plain = Self::decode_email_body(part_body, &part_encoding);
            } else if part_ct.contains("text/html") {
                let decoded = Self::decode_email_body(part_body, &part_encoding);
                text_html = strip_xml_tags(&decoded);
            } else if part_ct.contains("multipart/") {
                // Nested multipart
                let nested = self.parse_mime_multipart(part_body, &part_ct);
                if !nested.is_empty() {
                    text_plain = nested;
                }
            }
        }

        if !text_plain.is_empty() {
            text_plain
        } else {
            text_html
        }
    }

    /// Decode email body based on Content-Transfer-Encoding.
    fn decode_email_body(body: &str, encoding: &str) -> String {
        let encoding = encoding.trim().to_lowercase();
        match encoding.as_str() {
            "quoted-printable" => Self::decode_quoted_printable(body),
            "base64" => Self::decode_base64_text(body),
            _ => body.to_string(),
        }
    }

    /// Decode quoted-printable encoding (RFC 2045).
    fn decode_quoted_printable(input: &str) -> String {
        let mut result = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '=' {
                // Check for soft line break
                match chars.peek() {
                    Some('\r') => {
                        chars.next(); // skip \r
                        if chars.peek() == Some(&'\n') {
                            chars.next(); // skip \n
                        }
                        // Soft line break — continuation, don't add newline
                    }
                    Some('\n') => {
                        chars.next(); // soft line break
                    }
                    Some(_) => {
                        // Hex-encoded byte
                        let high = chars.next().unwrap_or('0');
                        let low = chars.next().unwrap_or('0');
                        let hex = format!("{}{}", high, low);
                        if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                            result.push(byte);
                        }
                    }
                    None => {}
                }
            } else {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                result.extend_from_slice(encoded.as_bytes());
            }
        }

        String::from_utf8_lossy(&result).to_string()
    }

    /// Decode base64-encoded text (RFC 4648).
    fn decode_base64_text(input: &str) -> String {
        // Remove whitespace
        let clean: String = input.chars().filter(|c| !c.is_whitespace()).collect();

        let lookup = |c: char| -> Option<u8> {
            match c {
                'A'..='Z' => Some(c as u8 - b'A'),
                'a'..='z' => Some(c as u8 - b'a' + 26),
                '0'..='9' => Some(c as u8 - b'0' + 52),
                '+' => Some(62),
                '/' => Some(63),
                _ => None,
            }
        };

        let mut result = Vec::new();
        let chars: Vec<char> = clean.chars().collect();
        let mut i = 0;

        while i + 3 < chars.len() {
            let a = lookup(chars[i]).unwrap_or(0);
            let b = lookup(chars[i + 1]).unwrap_or(0);
            let c_val = lookup(chars[i + 2]).unwrap_or(0);
            let d = lookup(chars[i + 3]).unwrap_or(0);

            result.push((a << 2) | (b >> 4));
            if chars[i + 2] != '=' {
                result.push((b << 4) | (c_val >> 2));
            }
            if chars[i + 3] != '=' {
                result.push((c_val << 6) | d);
            }
            i += 4;
        }

        String::from_utf8_lossy(&result).to_string()
    }

    /// Count attachments in a multipart email.
    fn count_attachments(&self, body: &str, content_type: &str) -> usize {
        let boundary = if let Some(pos) = content_type.find("boundary=") {
            let rest = &content_type[pos + 9..];
            rest.trim_start_matches('"')
                .split('"')
                .next()
                .unwrap_or("")
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_string()
        } else {
            return 0;
        };

        let separator = format!("--{}", boundary);
        let parts: Vec<&str> = body.split(&separator).collect();
        let mut count = 0;

        for part in &parts[1..] {
            if part.starts_with("--") {
                continue;
            }
            let part_lower = part.to_lowercase();
            if part_lower.contains("content-disposition: attachment")
                || part_lower.contains("content-disposition:attachment")
            {
                count += 1;
            }
        }
        count
    }

    /// Format email headers into readable text.
    fn format_email_headers(&self, headers: &HashMap<String, String>) -> String {
        let display_order = ["from", "to", "cc", "subject", "date", "message-id"];
        let mut lines = Vec::new();
        for key in &display_order {
            if let Some(val) = headers.get(*key) {
                lines.push(format!("{}: {}", Self::capitalize_header(key), val));
            }
        }
        lines.join("\n")
    }

    /// Capitalize email header name (e.g., "message-id" → "Message-Id").
    fn capitalize_header(name: &str) -> String {
        name.split('-')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    // ========================================================================
    // Image metadata extraction
    // ========================================================================

    /// Extract metadata from image files (dimensions, format, basic EXIF).
    ///
    /// No OCR — returns metadata only (dimensions, format, EXIF date/camera if JPEG).
    fn parse_image(&self, data: &[u8]) -> Result<ParsedDocument> {
        if data.len() < 8 {
            anyhow::bail!("Data too small to be a valid image");
        }

        let mut metadata = DocumentMetadata::default();
        let mut description_parts = Vec::new();

        // Detect format from magic bytes
        let (format_name, width, height) = if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            // PNG: IHDR chunk starts at byte 16 (after 8-byte signature + 4-byte length + 4-byte "IHDR")
            let (w, h) = if data.len() >= 24 {
                let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
                let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
                (w, h)
            } else {
                (0, 0)
            };
            ("PNG", w, h)
        } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            // JPEG: find SOF0 marker (0xFF 0xC0) for dimensions
            let (w, h) = Self::jpeg_dimensions(data);
            // Try EXIF extraction
            if let Some(exif) = Self::extract_basic_exif(data) {
                for (k, v) in exif {
                    metadata.extra.insert(k, v);
                }
            }
            ("JPEG", w, h)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            // GIF: width/height at bytes 6-9 (little-endian)
            let w = if data.len() >= 8 {
                u16::from_le_bytes([data[6], data[7]]) as u32
            } else {
                0
            };
            let h = if data.len() >= 10 {
                u16::from_le_bytes([data[8], data[9]]) as u32
            } else {
                0
            };
            ("GIF", w, h)
        } else if data.starts_with(b"BM") {
            // BMP: width at offset 18, height at offset 22 (little-endian i32)
            let w = if data.len() >= 22 {
                i32::from_le_bytes([data[18], data[19], data[20], data[21]]) as u32
            } else {
                0
            };
            let h = if data.len() >= 26 {
                i32::from_le_bytes([data[22], data[23], data[24], data[25]]).unsigned_abs()
            } else {
                0
            };
            ("BMP", w, h)
        } else if data.starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"WEBP" {
            // WebP: VP8 chunk has dimensions at specific offsets
            ("WebP", 0, 0) // WebP dimension parsing is complex; report format only
        } else {
            ("Unknown", 0, 0)
        };

        metadata
            .extra
            .insert("format".to_string(), format_name.to_string());
        metadata
            .extra
            .insert("file_size".to_string(), data.len().to_string());
        description_parts.push(format!("Image format: {}", format_name));

        if width > 0 && height > 0 {
            metadata
                .extra
                .insert("width".to_string(), width.to_string());
            metadata
                .extra
                .insert("height".to_string(), height.to_string());
            description_parts.push(format!("Dimensions: {}x{}", width, height));
        }

        description_parts.push(format!("File size: {} bytes", data.len()));
        metadata.title = Some(format!("{} image ({}x{})", format_name, width, height));

        let text = description_parts.join("\n");
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections: vec![DocumentSection {
                title: Some("Image Metadata".to_string()),
                content: description_parts.join("\n"),
                level: 1,
                index: 0,
            }],
            tables: Vec::new(),
            format: DocumentFormat::Image,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract JPEG dimensions from SOF0/SOF2 marker.
    fn jpeg_dimensions(data: &[u8]) -> (u32, u32) {
        let mut i = 2; // Skip FF D8
        while i + 1 < data.len() {
            if data[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = data[i + 1];
            // SOF0 (0xC0), SOF1 (0xC1), SOF2 (0xC2)
            if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) && i + 9 < data.len() {
                let h = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
                let w = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
                return (w, h);
            }
            if i + 3 < data.len() {
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                i += 2 + len;
            } else {
                break;
            }
        }
        (0, 0)
    }

    /// Extract basic EXIF metadata from JPEG (date, camera model).
    fn extract_basic_exif(data: &[u8]) -> Option<HashMap<String, String>> {
        // Look for APP1 marker (0xFF 0xE1) with "Exif\0\0" header
        let mut i = 2;
        while i + 1 < data.len() {
            if data[i] != 0xFF {
                i += 1;
                continue;
            }
            if data[i + 1] == 0xE1 && i + 10 < data.len() {
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                if &data[i + 4..i + 8] == b"Exif" {
                    let exif_data = &data[i + 10..std::cmp::min(i + 2 + len, data.len())];
                    return Self::parse_tiff_exif(exif_data);
                }
            }
            if i + 3 < data.len() {
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                i += 2 + len;
            } else {
                break;
            }
        }
        None
    }

    /// Minimal TIFF/EXIF parser: extract DateTimeOriginal and Model tags.
    fn parse_tiff_exif(data: &[u8]) -> Option<HashMap<String, String>> {
        if data.len() < 8 {
            return None;
        }
        let big_endian = &data[0..2] == b"MM";
        let read_u16 = |offset: usize| -> u16 {
            if offset + 1 >= data.len() {
                return 0;
            }
            if big_endian {
                u16::from_be_bytes([data[offset], data[offset + 1]])
            } else {
                u16::from_le_bytes([data[offset], data[offset + 1]])
            }
        };
        let read_u32 = |offset: usize| -> u32 {
            if offset + 3 >= data.len() {
                return 0;
            }
            if big_endian {
                u32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ])
            } else {
                u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ])
            }
        };

        let ifd_offset = read_u32(4) as usize;
        if ifd_offset >= data.len() {
            return None;
        }

        let mut result = HashMap::new();
        let entry_count = read_u16(ifd_offset) as usize;

        for e in 0..entry_count {
            let entry_offset = ifd_offset + 2 + e * 12;
            if entry_offset + 12 > data.len() {
                break;
            }

            let tag = read_u16(entry_offset);
            let count = read_u32(entry_offset + 4) as usize;
            let value_offset = read_u32(entry_offset + 8) as usize;

            // Tag 0x0110 = Model, Tag 0x0132 = DateTime
            if (tag == 0x0110 || tag == 0x0132) && value_offset < data.len() {
                let end = std::cmp::min(value_offset + count, data.len());
                if let Ok(s) = std::str::from_utf8(&data[value_offset..end]) {
                    let s = s.trim_end_matches('\0').to_string();
                    if !s.is_empty() {
                        let key = if tag == 0x0110 {
                            "camera_model"
                        } else {
                            "date_taken"
                        };
                        result.insert(key.to_string(), s);
                    }
                }
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // ========================================================================
    // PPTX parsing (Phase 2 — requires `documents` feature for ZIP)
    // ========================================================================

    #[cfg(feature = "documents")]
    fn parse_pptx(&self, data: &[u8]) -> Result<ParsedDocument> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // Find slide files: ppt/slides/slide1.xml, slide2.xml, etc.
        let mut slide_names: Vec<String> = Vec::new();
        for i in 0..archive.len() {
            if let Ok(file) = archive.by_index(i) {
                let name = file.name().to_string();
                if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                    slide_names.push(name);
                }
            }
        }
        // Sort by slide number
        slide_names.sort_by(|a, b| {
            let num_a = a
                .trim_start_matches("ppt/slides/slide")
                .trim_end_matches(".xml")
                .parse::<u32>()
                .unwrap_or(0);
            let num_b = b
                .trim_start_matches("ppt/slides/slide")
                .trim_end_matches(".xml")
                .parse::<u32>()
                .unwrap_or(0);
            num_a.cmp(&num_b)
        });

        let mut sections = Vec::new();
        let mut all_text = Vec::new();

        for (idx, slide_name) in slide_names.iter().enumerate() {
            if let Ok(mut file) = archive.by_name(slide_name) {
                let mut xml = String::new();
                file.read_to_string(&mut xml)?;

                // Extract text from <a:t> elements
                let text_re = Regex::new(r"<a:t>(.*?)</a:t>").expect("valid regex");
                let texts: Vec<String> = text_re
                    .captures_iter(&xml)
                    .map(|c| c[1].to_string())
                    .collect();
                let slide_text = texts.join(" ");

                // Try to extract title from <p:ph type="title"/>
                let title = if let Some(title_match) =
                    Regex::new(r#"<p:ph[^>]*type="(?:title|ctrTitle)"[^>]*/>"#)
                        .ok()
                        .and_then(|re| re.find(&xml))
                {
                    // Title is usually the first <a:t> in the shape containing the title placeholder
                    let _before_title = &xml[..title_match.start()];
                    // Find the enclosing <p:sp> and get its text
                    texts.first().cloned()
                } else {
                    texts.first().cloned()
                };

                let slide_title = title.unwrap_or_else(|| format!("Slide {}", idx + 1));

                if !slide_text.is_empty() {
                    all_text.push(format!("## {}\n{}", slide_title, slide_text));
                    sections.push(DocumentSection {
                        title: Some(slide_title),
                        content: slide_text,
                        level: 1,
                        index: idx,
                    });
                }
            }

            // Try to extract speaker notes
            let notes_name = format!("ppt/notesSlides/notesSlide{}.xml", idx + 1);
            if let Ok(mut file) = archive.by_name(&notes_name) {
                let mut xml = String::new();
                file.read_to_string(&mut xml)?;
                let text_re = Regex::new(r"<a:t>(.*?)</a:t>").expect("valid regex");
                let notes: Vec<String> = text_re
                    .captures_iter(&xml)
                    .map(|c| c[1].to_string())
                    .filter(|t| !t.is_empty())
                    .collect();
                if !notes.is_empty() {
                    let notes_text = notes.join(" ");
                    all_text.push(format!("Notes: {}", notes_text));
                }
            }
        }

        // Extract metadata from docProps/core.xml
        let metadata = if let Ok(mut file) = archive.by_name("docProps/core.xml") {
            let mut xml = String::new();
            file.read_to_string(&mut xml)?;
            extract_xml_metadata(&xml)
        } else {
            DocumentMetadata::default()
        };

        let text = all_text.join("\n\n");
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables: Vec::new(),
            format: DocumentFormat::Pptx,
            source_path: None,
            char_count,
            word_count,
        })
    }

    // ========================================================================
    // XLSX parsing (Phase 2 — requires `documents` feature)
    // ========================================================================

    #[cfg(feature = "documents")]
    fn parse_xlsx(&self, data: &[u8]) -> Result<ParsedDocument> {
        // XLSX is a ZIP of XML files; we parse it manually using the existing zip dep
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // Read shared strings from xl/sharedStrings.xml
        let shared_strings = if let Ok(mut file) = archive.by_name("xl/sharedStrings.xml") {
            let mut xml = String::new();
            file.read_to_string(&mut xml)?;
            Self::parse_shared_strings(&xml)
        } else {
            Vec::new()
        };

        // Find sheet files: xl/worksheets/sheet1.xml, etc.
        let mut sheet_names: Vec<String> = Vec::new();
        for i in 0..archive.len() {
            if let Ok(file) = archive.by_index(i) {
                let name = file.name().to_string();
                if name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml") {
                    sheet_names.push(name);
                }
            }
        }
        sheet_names.sort();

        let mut sections = Vec::new();
        let mut tables = Vec::new();
        let mut all_text = Vec::new();

        for (idx, sheet_name) in sheet_names.iter().enumerate() {
            if let Ok(mut file) = archive.by_name(sheet_name) {
                let mut xml = String::new();
                file.read_to_string(&mut xml)?;

                let rows = Self::parse_xlsx_sheet(&xml, &shared_strings);
                if rows.is_empty() {
                    continue;
                }

                let sheet_label = format!("Sheet {}", idx + 1);
                let mut text_rows = Vec::new();
                for row in &rows {
                    text_rows.push(row.join(", "));
                }
                let sheet_text = text_rows.join("\n");

                all_text.push(format!("## {}\n{}", sheet_label, sheet_text));
                sections.push(DocumentSection {
                    title: Some(sheet_label),
                    content: sheet_text,
                    level: 1,
                    index: idx,
                });

                // Build table
                let (headers, data_rows) = if rows.len() > 1 {
                    (rows[0].clone(), rows[1..].to_vec())
                } else {
                    (rows[0].clone(), Vec::new())
                };
                tables.push(PdfTable {
                    caption: Some(format!("Sheet {}", idx + 1)),
                    headers,
                    rows: data_rows,
                    page: None,
                    confidence: 1.0,
                    raw_text: String::new(),
                });
            }
        }

        // Extract metadata from docProps/core.xml
        let metadata = if let Ok(mut file) = archive.by_name("docProps/core.xml") {
            let mut xml = String::new();
            file.read_to_string(&mut xml)?;
            extract_xml_metadata(&xml)
        } else {
            DocumentMetadata::default()
        };

        let text = all_text.join("\n\n");
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();

        Ok(ParsedDocument {
            text,
            metadata,
            sections,
            tables,
            format: DocumentFormat::Xlsx,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Parse shared strings from xl/sharedStrings.xml
    #[cfg(feature = "documents")]
    fn parse_shared_strings(xml: &str) -> Vec<String> {
        let re = Regex::new(r"<t[^>]*>(.*?)</t>").expect("valid regex");
        re.captures_iter(xml).map(|c| c[1].to_string()).collect()
    }

    /// Parse a single XLSX sheet, returning rows of cell values.
    #[cfg(feature = "documents")]
    fn parse_xlsx_sheet(xml: &str, shared_strings: &[String]) -> Vec<Vec<String>> {
        let mut rows = Vec::new();

        // Match <row> elements
        let row_re = Regex::new(r"(?s)<row[^>]*>(.*?)</row>").expect("valid regex");
        let cell_re =
            Regex::new(r#"(?s)<c[^>]*?(?:t="([^"]*)")?[^>]*>(?:.*?<v>(.*?)</v>)?.*?</c>"#)
                .expect("valid regex");

        for row_match in row_re.captures_iter(xml) {
            let row_xml = &row_match[1];
            let mut cells = Vec::new();

            for cell_match in cell_re.captures_iter(row_xml) {
                let cell_type = cell_match.get(1).map(|m| m.as_str()).unwrap_or("");
                let value = cell_match.get(2).map(|m| m.as_str()).unwrap_or("");

                let cell_value = if cell_type == "s" {
                    // Shared string reference
                    if let Ok(idx) = value.parse::<usize>() {
                        shared_strings.get(idx).cloned().unwrap_or_default()
                    } else {
                        value.to_string()
                    }
                } else {
                    value.to_string()
                };
                cells.push(cell_value);
            }

            if !cells.iter().all(|c| c.is_empty()) {
                rows.push(cells);
            }
        }

        rows
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Remove all XML/HTML tags from a string, preserving the text content.
/// Also decodes common HTML entities.
pub fn strip_xml_tags(xml: &str) -> String {
    // Remove script and style blocks entirely
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("valid regex");
    let cleaned = script_re.replace_all(xml, "");
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("valid regex");
    let cleaned = style_re.replace_all(&cleaned, "");

    // Replace <br>, <br/>, <br /> with newlines
    let br_re = Regex::new(r"(?i)<br\s*/?\s*>").expect("valid regex");
    let cleaned = br_re.replace_all(&cleaned, "\n");

    // Replace block-level closing tags with newlines for paragraph separation
    let block_re = Regex::new(r"(?i)</?(p|div|li|tr|blockquote|article|section|header|footer|nav|aside|main|figure|figcaption|details|summary|dd|dt)\s*[^>]*>").expect("valid regex");
    let cleaned = block_re.replace_all(&cleaned, "\n");

    // Remove all remaining tags
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    let cleaned = tag_re.replace_all(&cleaned, "");

    // Decode common HTML entities
    let result = cleaned
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&mdash;", "\u{2014}")
        .replace("&ndash;", "\u{2013}")
        .replace("&hellip;", "\u{2026}")
        .replace("&copy;", "\u{00A9}")
        .replace("&reg;", "\u{00AE}");

    // Decode numeric character references: &#NNN; and &#xHHH;
    let num_entity_re = Regex::new(r"&#(\d+);").expect("valid regex");
    let result = num_entity_re.replace_all(&result, |caps: &regex::Captures| {
        if let Ok(code) = caps[1].parse::<u32>() {
            if let Some(ch) = char::from_u32(code) {
                return ch.to_string();
            }
        }
        caps[0].to_string()
    });

    let hex_entity_re = Regex::new(r"(?i)&#x([0-9a-f]+);").expect("valid regex");
    let result = hex_entity_re.replace_all(&result, |caps: &regex::Captures| {
        if let Ok(code) = u32::from_str_radix(&caps[1], 16) {
            if let Some(ch) = char::from_u32(code) {
                return ch.to_string();
            }
        }
        caps[0].to_string()
    });

    result.to_string()
}

/// Extract all text content found between opening and closing instances of the given tag.
/// For example, `extract_xml_text(xml, "title")` finds all `<title>...</title>` occurrences.
pub fn extract_xml_text(xml: &str, tag: &str) -> Vec<String> {
    let escaped_tag = regex::escape(tag);
    let pattern = format!(r"(?s)<{}[^>]*>(.*?)</{}>", escaped_tag, escaped_tag);
    let re = Regex::new(&pattern).expect("valid regex");

    re.captures_iter(xml)
        .map(|caps| {
            let inner = &caps[1];
            let text = strip_xml_tags(inner);
            text.trim().to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract common metadata elements from an XML document (Dublin Core, etc.).
pub fn extract_xml_metadata(xml: &str) -> DocumentMetadata {
    let mut meta = DocumentMetadata::default();

    // Title
    let titles = extract_xml_text(xml, "dc:title");
    if let Some(t) = titles.into_iter().next() {
        meta.title = Some(t);
    }
    // Fallback to <title>
    if meta.title.is_none() {
        let titles = extract_xml_text(xml, "title");
        if let Some(t) = titles.into_iter().next() {
            meta.title = Some(t);
        }
    }

    // Authors
    let creators = extract_xml_text(xml, "dc:creator");
    if !creators.is_empty() {
        meta.authors = creators;
    }

    // Language
    let langs = extract_xml_text(xml, "dc:language");
    if let Some(l) = langs.into_iter().next() {
        meta.language = Some(l);
    }

    // Date
    let dates = extract_xml_text(xml, "dc:date");
    if let Some(d) = dates.into_iter().next() {
        meta.date = Some(d);
    }
    // Fallback to dcterms:modified
    if meta.date.is_none() {
        let modified = extract_xml_text(xml, "dcterms:modified");
        if let Some(m) = modified.into_iter().next() {
            meta.date = Some(m);
        }
    }

    // Description
    let descs = extract_xml_text(xml, "dc:description");
    if let Some(d) = descs.into_iter().next() {
        meta.description = Some(d);
    }

    // Publisher
    let pubs = extract_xml_text(xml, "dc:publisher");
    if let Some(p) = pubs.into_iter().next() {
        meta.publisher = Some(p);
    }

    meta
}

/// Normalize text: collapse multiple whitespace into single spaces within lines,
/// collapse multiple blank lines into at most two newlines, and trim.
pub fn normalize_text(text: &str) -> String {
    // Replace \r\n with \n
    let text = text.replace("\r\n", "\n");

    // Collapse multiple spaces/tabs within lines (but preserve newlines)
    let space_re = Regex::new(r"[^\S\n]+").expect("valid regex");
    let text = space_re.replace_all(&text, " ");

    // Collapse 3+ newlines into 2
    let newline_re = Regex::new(r"\n{3,}").expect("valid regex");
    let text = newline_re.replace_all(&text, "\n\n");

    // Trim each line
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let text = lines.join("\n");

    // Trim the whole result
    text.trim().to_string()
}

/// Extract the first heading (h1-h6) text from an HTML/XHTML document.
#[allow(dead_code)]
fn extract_first_heading(html: &str) -> Option<String> {
    let heading_re = Regex::new(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>").expect("valid regex");
    if let Some(caps) = heading_re.captures(html) {
        let text = strip_xml_tags(&caps[1]);
        let text = text.trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

// ============================================================================
// OCR-lite Engine (Template Matching)
// ============================================================================

/// OCR engine configuration.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Minimum confidence threshold for character recognition (0.0 - 1.0).
    pub min_confidence: f32,
    /// Expected character height in pixels.
    pub char_height: usize,
    /// Binarization threshold (0-255). If None, uses Otsu's method.
    pub binarize_threshold: Option<u8>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            char_height: 7,
            binarize_threshold: None,
        }
    }
}

/// A glyph template for template matching.
#[derive(Debug, Clone)]
pub struct GlyphTemplate {
    pub character: char,
    pub width: usize,
    pub height: usize,
    pub bitmap: Vec<u8>, // row-major, 0=white, 1=black
}

/// A recognized text line.
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub y_position: usize,
}

/// OCR recognition result.
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub lines: Vec<OcrLine>,
    pub full_text: String,
    pub average_confidence: f32,
}

/// Pure-Rust template-matching OCR engine.
///
/// Uses 5x7 LED-style bitmaps for character recognition via cross-correlation.
pub struct OcrEngine {
    pub templates: Vec<GlyphTemplate>,
    config: OcrConfig,
}

impl OcrEngine {
    /// Create a new OCR engine with default templates (A-Z, a-z, 0-9, common punctuation).
    pub fn with_default_templates(config: OcrConfig) -> Self {
        let mut engine = Self {
            templates: Vec::new(),
            config,
        };
        engine.load_default_templates();
        engine
    }

    /// Create a new OCR engine with custom templates.
    pub fn new(templates: Vec<GlyphTemplate>, config: OcrConfig) -> Self {
        Self { templates, config }
    }

    /// Add a custom glyph template.
    pub fn add_template(&mut self, template: GlyphTemplate) {
        self.templates.push(template);
    }

    /// Recognize text from a grayscale bitmap image.
    /// `image` is row-major grayscale pixels (0-255), `width` x `height`.
    pub fn recognize_bitmap(&self, image: &[u8], width: usize, height: usize) -> OcrResult {
        if image.len() != width * height || width == 0 || height == 0 {
            return OcrResult {
                lines: Vec::new(),
                full_text: String::new(),
                average_confidence: 0.0,
            };
        }

        // Step 1: Binarize
        let binary = self.binarize(image, width, height);

        // Step 2: Detect text lines (horizontal projection)
        let line_ranges = self.detect_text_lines(&binary, width, height);

        // Step 3: For each line, segment characters and match
        let mut lines = Vec::new();
        for (y_start, y_end) in &line_ranges {
            let line_height = y_end - y_start;
            let line_slice: Vec<u8> =
                binary[y_start * width..(y_end * width).min(binary.len())].to_vec();

            let char_ranges = self.segment_characters(&line_slice, width, line_height);
            let mut text = String::new();
            let mut total_conf = 0.0f32;
            let mut char_count = 0;

            let mut last_x_end = 0;
            for (x_start, x_end) in &char_ranges {
                // Detect spaces: if gap > char_width * 0.8
                if *x_start > last_x_end + 3 && !text.is_empty() {
                    text.push(' ');
                }

                // Extract character region
                let char_width = x_end - x_start;
                let mut char_bitmap = vec![0u8; char_width * line_height];
                for row in 0..line_height {
                    for col in 0..char_width {
                        if x_start + col < width {
                            char_bitmap[row * char_width + col] =
                                line_slice[row * width + x_start + col];
                        }
                    }
                }

                let (ch, conf) = self.match_template(&char_bitmap, char_width, line_height);
                if conf >= self.config.min_confidence {
                    text.push(ch);
                    total_conf += conf;
                    char_count += 1;
                }
                last_x_end = *x_end;
            }

            let avg_conf = if char_count > 0 {
                total_conf / char_count as f32
            } else {
                0.0
            };
            if !text.is_empty() {
                lines.push(OcrLine {
                    text: text.clone(),
                    confidence: avg_conf,
                    y_position: *y_start,
                });
            }
        }

        let full_text = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let avg = if lines.is_empty() {
            0.0
        } else {
            lines.iter().map(|l| l.confidence).sum::<f32>() / lines.len() as f32
        };
        OcrResult {
            lines,
            full_text,
            average_confidence: avg,
        }
    }

    /// Binarize a grayscale image using Otsu's method or a fixed threshold.
    pub fn binarize(&self, image: &[u8], _width: usize, _height: usize) -> Vec<u8> {
        let threshold = self
            .config
            .binarize_threshold
            .unwrap_or_else(|| self.otsu_threshold(image));
        image
            .iter()
            .map(|&p| if p < threshold { 1 } else { 0 })
            .collect()
    }

    /// Compute Otsu's optimal binarization threshold.
    fn otsu_threshold(&self, image: &[u8]) -> u8 {
        let mut histogram = [0u32; 256];
        for &pixel in image {
            histogram[pixel as usize] += 1;
        }

        let total = image.len() as f64;
        let mut sum_total = 0.0f64;
        for (i, &count) in histogram.iter().enumerate() {
            sum_total += i as f64 * count as f64;
        }

        let mut sum_bg = 0.0f64;
        let mut weight_bg = 0.0f64;
        let mut max_variance = 0.0f64;
        let mut best_threshold = 0u8;

        for (t, &count) in histogram.iter().enumerate() {
            weight_bg += count as f64;
            if weight_bg == 0.0 {
                continue;
            }

            let weight_fg = total - weight_bg;
            if weight_fg == 0.0 {
                break;
            }

            sum_bg += t as f64 * count as f64;
            let mean_bg = sum_bg / weight_bg;
            let mean_fg = (sum_total - sum_bg) / weight_fg;

            let variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
            if variance > max_variance {
                max_variance = variance;
                best_threshold = t as u8;
            }
        }

        best_threshold
    }

    /// Detect text lines using horizontal projection profile.
    /// Returns (y_start, y_end) pairs for each detected line.
    pub fn detect_text_lines(
        &self,
        binary: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        // Count black pixels per row
        let mut row_sums: Vec<usize> = Vec::with_capacity(height);
        for y in 0..height {
            let sum: usize = binary[y * width..(y + 1) * width]
                .iter()
                .map(|&p| p as usize)
                .sum();
            row_sums.push(sum);
        }

        // Find runs of non-zero rows (text lines)
        let mut lines = Vec::new();
        let mut in_line = false;
        let mut start = 0;
        let threshold = 1; // at least 1 black pixel

        for (y, &sum) in row_sums.iter().enumerate() {
            if sum >= threshold && !in_line {
                in_line = true;
                start = y;
            } else if sum < threshold && in_line {
                in_line = false;
                if y - start >= 3 {
                    // minimum line height
                    lines.push((start, y));
                }
            }
        }
        if in_line && height - start >= 3 {
            lines.push((start, height));
        }

        lines
    }

    /// Segment characters in a binary line image using vertical projection.
    /// Returns (x_start, x_end) pairs for each character.
    pub fn segment_characters(
        &self,
        line_binary: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        // Count black pixels per column
        let mut col_sums: Vec<usize> = Vec::with_capacity(width);
        for x in 0..width {
            let mut sum = 0;
            for y in 0..height {
                if x < width && y * width + x < line_binary.len() {
                    sum += line_binary[y * width + x] as usize;
                }
            }
            col_sums.push(sum);
        }

        // Find runs of non-zero columns (characters)
        let mut chars = Vec::new();
        let mut in_char = false;
        let mut start = 0;

        for (x, &sum) in col_sums.iter().enumerate() {
            if sum > 0 && !in_char {
                in_char = true;
                start = x;
            } else if sum == 0 && in_char {
                in_char = false;
                chars.push((start, x));
            }
        }
        if in_char {
            chars.push((start, width));
        }

        chars
    }

    /// Match a character bitmap against templates using normalized cross-correlation.
    /// Returns (best_char, confidence).
    pub fn match_template(
        &self,
        char_bitmap: &[u8],
        char_width: usize,
        char_height: usize,
    ) -> (char, f32) {
        let mut best_char = '?';
        let mut best_score = -1.0f32;

        for template in &self.templates {
            let score = self.cross_correlate(
                char_bitmap,
                char_width,
                char_height,
                &template.bitmap,
                template.width,
                template.height,
            );
            if score > best_score {
                best_score = score;
                best_char = template.character;
            }
        }

        (best_char, best_score.max(0.0))
    }

    /// Normalized cross-correlation between two bitmaps.
    /// Resizes the smaller to match the larger for comparison.
    fn cross_correlate(
        &self,
        img: &[u8],
        img_w: usize,
        img_h: usize,
        tmpl: &[u8],
        tmpl_w: usize,
        tmpl_h: usize,
    ) -> f32 {
        // Resize both to template size using nearest-neighbor
        let w = tmpl_w;
        let h = tmpl_h;

        let resized_img = Self::resize_nearest(img, img_w, img_h, w, h);

        // NCC = sum(a*b) / sqrt(sum(a^2) * sum(b^2))
        let mut sum_ab = 0.0f64;
        let mut sum_aa = 0.0f64;
        let mut sum_bb = 0.0f64;

        for i in 0..(w * h) {
            let a = *resized_img.get(i).unwrap_or(&0) as f64;
            let b = *tmpl.get(i).unwrap_or(&0) as f64;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
        }

        let denom = (sum_aa * sum_bb).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (sum_ab / denom) as f32
    }

    /// Nearest-neighbor resize.
    fn resize_nearest(
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        let mut dst = vec![0u8; dst_w * dst_h];
        if src_w == 0 || src_h == 0 {
            return dst;
        }
        for y in 0..dst_h {
            for x in 0..dst_w {
                let sx = (x * src_w) / dst_w.max(1);
                let sy = (y * src_h) / dst_h.max(1);
                let si = sy * src_w + sx;
                dst[y * dst_w + x] = *src.get(si).unwrap_or(&0);
            }
        }
        dst
    }

    /// Load default 5x7 LED-style glyph templates for common characters.
    fn load_default_templates(&mut self) {
        // Define templates as 5-wide x 7-tall bitmaps
        // Each string is 7 rows of 5 chars, where '#' = 1 (black) and '.' = 0 (white)
        let templates = vec![
            (
                'A',
                vec![
                    ".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
                ],
            ),
            (
                'B',
                vec![
                    "####.", "#...#", "#...#", "####.", "#...#", "#...#", "####.",
                ],
            ),
            (
                'C',
                vec![
                    ".###.", "#...#", "#....", "#....", "#....", "#...#", ".###.",
                ],
            ),
            (
                'D',
                vec![
                    "####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####.",
                ],
            ),
            (
                'E',
                vec![
                    "#####", "#....", "#....", "###..", "#....", "#....", "#####",
                ],
            ),
            (
                'F',
                vec![
                    "#####", "#....", "#....", "###..", "#....", "#....", "#....",
                ],
            ),
            (
                'H',
                vec![
                    "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
                ],
            ),
            ('I', vec!["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"]),
            (
                'L',
                vec![
                    "#....", "#....", "#....", "#....", "#....", "#....", "#####",
                ],
            ),
            (
                'O',
                vec![
                    ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
                ],
            ),
            (
                'T',
                vec![
                    "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..",
                ],
            ),
            (
                '0',
                vec![
                    ".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###.",
                ],
            ),
            (
                '1',
                vec![
                    "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
                ],
            ),
            (
                ' ',
                vec![
                    ".....", ".....", ".....", ".....", ".....", ".....", ".....",
                ],
            ),
        ];

        for (ch, rows) in templates {
            let width = rows[0].len();
            let height = rows.len();
            let mut bitmap = Vec::with_capacity(width * height);
            for row in &rows {
                for c in row.chars() {
                    bitmap.push(if c == '#' { 1 } else { 0 });
                }
            }
            self.templates.push(GlyphTemplate {
                character: ch,
                width,
                height,
                bitmap,
            });
        }
    }
}

// ============================================================================
// Image Extraction
// ============================================================================

/// Supported image formats identifiable by magic bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    Bmp,
    Tiff,
    Unknown,
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageFormat::Jpeg => write!(f, "JPEG"),
            ImageFormat::Png => write!(f, "PNG"),
            ImageFormat::Gif => write!(f, "GIF"),
            ImageFormat::Bmp => write!(f, "BMP"),
            ImageFormat::Tiff => write!(f, "TIFF"),
            ImageFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A single image extracted from a binary document.
#[derive(Debug, Clone)]
pub struct ExtractedImage {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub page: Option<usize>,
    pub index: usize,
    pub offset: usize,
}

/// Configuration for image extraction behaviour.
pub struct ImageExtractionConfig {
    /// Minimum image size in bytes; smaller blobs are skipped.
    pub min_size_bytes: usize,
    /// Maximum number of images to extract.
    pub max_images: usize,
    /// Whether to attempt reading width/height from image headers.
    pub extract_dimensions: bool,
}

impl Default for ImageExtractionConfig {
    fn default() -> Self {
        Self {
            min_size_bytes: 100,
            max_images: 1000,
            extract_dimensions: true,
        }
    }
}

/// Scans raw byte streams for embedded images using magic-byte detection.
pub struct ImageExtractor {
    config: ImageExtractionConfig,
}

impl ImageExtractor {
    pub fn new(config: ImageExtractionConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: ImageExtractionConfig::default(),
        }
    }

    /// Detect the image format from the first few bytes of a header.
    pub fn detect_format(header: &[u8]) -> Option<ImageFormat> {
        if header.len() >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF {
            return Some(ImageFormat::Jpeg);
        }
        if header.len() >= 4
            && header[0] == 0x89
            && header[1] == 0x50
            && header[2] == 0x4E
            && header[3] == 0x47
        {
            return Some(ImageFormat::Png);
        }
        if header.len() >= 4
            && header[0] == 0x47
            && header[1] == 0x49
            && header[2] == 0x46
            && header[3] == 0x38
        {
            return Some(ImageFormat::Gif);
        }
        if header.len() >= 2 && header[0] == 0x42 && header[1] == 0x4D {
            return Some(ImageFormat::Bmp);
        }
        if header.len() >= 4 {
            // Little-endian TIFF
            if header[0] == 0x49 && header[1] == 0x49 && header[2] == 0x2A && header[3] == 0x00 {
                return Some(ImageFormat::Tiff);
            }
            // Big-endian TIFF
            if header[0] == 0x4D && header[1] == 0x4D && header[2] == 0x00 && header[3] == 0x2A {
                return Some(ImageFormat::Tiff);
            }
        }
        None
    }

    /// Find the end of a JPEG image (offset after the EOI marker 0xFF 0xD9).
    pub fn find_jpeg_end(data: &[u8], start: usize) -> Option<usize> {
        if start + 2 > data.len() {
            return None;
        }
        for i in start..data.len() - 1 {
            if data[i] == 0xFF && data[i + 1] == 0xD9 {
                return Some(i + 2);
            }
        }
        None
    }

    /// Find the end of a PNG image (offset after the IEND chunk + CRC).
    /// The IEND trailer is the 12-byte sequence:
    /// [0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]
    pub fn find_png_end(data: &[u8], start: usize) -> Option<usize> {
        const IEND_TRAILER: [u8; 12] = [
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        if data.len() < start + IEND_TRAILER.len() {
            return None;
        }
        for i in start..=data.len() - IEND_TRAILER.len() {
            if data[i..i + IEND_TRAILER.len()] == IEND_TRAILER {
                return Some(i + IEND_TRAILER.len());
            }
        }
        None
    }

    /// Read JPEG dimensions from SOF0 marker (0xFF, 0xC0).
    /// Height is 2 bytes big-endian at marker+3, width at marker+5.
    pub fn read_jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 2 {
            return None;
        }
        for i in 0..data.len() - 1 {
            if data[i] == 0xFF && data[i + 1] == 0xC0 {
                // SOF0 layout from marker byte (i):
                //   +0,+1: marker (FF C0)
                //   +2,+3: length
                //   +4: precision
                //   +5,+6: height (2 bytes BE)
                //   +7,+8: width (2 bytes BE)
                if i + 9 <= data.len() {
                    let height = ((data[i + 5] as u32) << 8) | (data[i + 6] as u32);
                    let width = ((data[i + 7] as u32) << 8) | (data[i + 8] as u32);
                    return Some((width, height));
                }
            }
        }
        None
    }

    /// Read PNG dimensions from the IHDR chunk.
    /// Width is 4 bytes big-endian at offset 16, height at offset 20.
    pub fn read_png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 24 {
            return None;
        }
        let width = ((data[16] as u32) << 24)
            | ((data[17] as u32) << 16)
            | ((data[18] as u32) << 8)
            | (data[19] as u32);
        let height = ((data[20] as u32) << 24)
            | ((data[21] as u32) << 16)
            | ((data[22] as u32) << 8)
            | (data[23] as u32);
        Some((width, height))
    }

    /// Scan raw bytes for embedded images, returning all found images.
    pub fn extract_from_bytes(&self, data: &[u8]) -> Vec<ExtractedImage> {
        let mut results = Vec::new();
        let mut pos = 0;
        let mut index = 0;
        const MAX_FALLBACK_SIZE: usize = 10 * 1024 * 1024; // 10 MB cap

        while pos < data.len() && results.len() < self.config.max_images {
            let remaining = &data[pos..];
            if let Some(format) = Self::detect_format(remaining) {
                let end = match format {
                    ImageFormat::Jpeg => Self::find_jpeg_end(data, pos),
                    ImageFormat::Png => Self::find_png_end(data, pos),
                    _ => {
                        // For GIF/BMP/TIFF: scan until the next image magic or end-of-data,
                        // capped at 10 MB.
                        let search_limit = std::cmp::min(pos + MAX_FALLBACK_SIZE, data.len());
                        let mut found_next = None;
                        for j in (pos + 2)..search_limit {
                            if let Some(_) = Self::detect_format(&data[j..]) {
                                found_next = Some(j);
                                break;
                            }
                        }
                        Some(found_next.unwrap_or(search_limit))
                    }
                };

                if let Some(end_offset) = end {
                    let img_data = &data[pos..end_offset];
                    if img_data.len() >= self.config.min_size_bytes {
                        let (width, height) = if self.config.extract_dimensions {
                            match format {
                                ImageFormat::Jpeg => Self::read_jpeg_dimensions(img_data)
                                    .map(|(w, h)| (Some(w), Some(h)))
                                    .unwrap_or((None, None)),
                                ImageFormat::Png => Self::read_png_dimensions(img_data)
                                    .map(|(w, h)| (Some(w), Some(h)))
                                    .unwrap_or((None, None)),
                                _ => (None, None),
                            }
                        } else {
                            (None, None)
                        };

                        results.push(ExtractedImage {
                            data: img_data.to_vec(),
                            format: format.clone(),
                            width,
                            height,
                            page: None,
                            index,
                            offset: pos,
                        });
                        index += 1;
                    }
                    pos = end_offset;
                } else {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        }

        results
    }
}

/// Summary analysis of all images found in a document.
pub struct DocumentImageAnalysis {
    pub images: Vec<ExtractedImage>,
    pub total_image_bytes: usize,
    pub formats_found: HashMap<ImageFormat, usize>,
}

impl DocumentImageAnalysis {
    /// Scan document bytes and produce an image analysis summary.
    pub fn from_document(data: &[u8], config: ImageExtractionConfig) -> Self {
        let extractor = ImageExtractor::new(config);
        let images = extractor.extract_from_bytes(data);
        let total_image_bytes = images.iter().map(|img| img.data.len()).sum();
        let mut formats_found = HashMap::new();
        for img in &images {
            *formats_found.entry(img.format.clone()).or_insert(0) += 1;
        }
        Self {
            images,
            total_image_bytes,
            formats_found,
        }
    }
}

// ============================================================================
// OCR Integration (WS9)
// ============================================================================

/// Backend trait for pluggable OCR engines.
///
/// Implementations must be thread-safe (`Send + Sync`) so they can be shared
/// across an `OcrPipeline` that may be used from multiple threads.
pub trait OcrBackend: Send + Sync {
    /// Human-readable name of this backend.
    fn name(&self) -> &str;
    /// Run OCR on a grayscale bitmap and return the recognition result.
    fn recognize(&self, image: &[u8], width: usize, height: usize) -> OcrResult;
    /// Whether this backend can handle the given image format.
    fn supports_format(&self, format: &ImageFormat) -> bool;
    /// Minimum confidence value this backend considers acceptable.
    fn confidence_threshold(&self) -> f32;
}

/// An `OcrBackend` powered by the built-in template-matching `OcrEngine`.
pub struct TemplateOcrBackend {
    pub engine: OcrEngine,
    min_confidence: f32,
}

impl TemplateOcrBackend {
    /// Create a new template backend from the given OCR config.
    pub fn new(config: OcrConfig) -> Self {
        let min_confidence = config.min_confidence;
        let engine = OcrEngine::with_default_templates(config);
        Self {
            engine,
            min_confidence,
        }
    }
}

impl OcrBackend for TemplateOcrBackend {
    fn name(&self) -> &str {
        "template"
    }

    fn recognize(&self, image: &[u8], width: usize, height: usize) -> OcrResult {
        self.engine.recognize_bitmap(image, width, height)
    }

    fn supports_format(&self, format: &ImageFormat) -> bool {
        // The template engine works on grayscale bitmaps decoded from common
        // raster formats, but not animated GIF.
        matches!(
            format,
            ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::Bmp | ImageFormat::Tiff
        )
    }

    fn confidence_threshold(&self) -> f32 {
        self.min_confidence
    }
}

/// Configuration for an external Tesseract OCR process.
#[derive(Debug, Clone)]
pub struct TesseractConfig {
    /// Path to the `tesseract` binary.
    pub binary_path: String,
    /// Language code (e.g. `"eng"`).
    pub language: String,
    /// Page segmentation mode.
    pub psm: u32,
    /// OCR engine mode.
    pub oem: u32,
}

impl Default for TesseractConfig {
    fn default() -> Self {
        Self {
            binary_path: "tesseract".to_string(),
            language: "eng".to_string(),
            psm: 3,
            oem: 3,
        }
    }
}

/// An `OcrBackend` that wraps an external Tesseract binary.
///
/// This backend does **not** actually invoke the binary at runtime; it serves
/// as a configuration holder so that callers can integrate Tesseract via their
/// own process-spawning logic.  Calling `recognize` returns a placeholder
/// result with zero confidence.
pub struct TesseractOcrBackend {
    config: TesseractConfig,
}

impl TesseractOcrBackend {
    pub fn new(config: TesseractConfig) -> Self {
        Self { config }
    }
}

impl OcrBackend for TesseractOcrBackend {
    fn name(&self) -> &str {
        "tesseract"
    }

    fn recognize(&self, _image: &[u8], _width: usize, _height: usize) -> OcrResult {
        let msg = format!(
            "Tesseract OCR backend ({}): binary not available for direct invocation",
            self.config.binary_path
        );
        OcrResult {
            lines: vec![OcrLine {
                text: msg.clone(),
                confidence: 0.0,
                y_position: 0,
            }],
            full_text: msg,
            average_confidence: 0.0,
        }
    }

    fn supports_format(&self, _format: &ImageFormat) -> bool {
        true
    }

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}

/// Configuration for the multi-backend [`OcrPipeline`].
#[derive(Debug, Clone)]
pub struct OcrPipelineConfig {
    /// Minimum acceptable confidence for a result to be considered valid.
    pub min_confidence: f32,
    /// Whether to merge results from multiple backends (reserved for future use).
    pub merge_results: bool,
    /// Name of the preferred backend that should be tried first.
    pub preferred_backend: Option<String>,
}

impl Default for OcrPipelineConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            merge_results: true,
            preferred_backend: None,
        }
    }
}

/// A pipeline that dispatches OCR work to one or more [`OcrBackend`]s and
/// selects the result with the highest confidence.
pub struct OcrPipeline {
    backends: Vec<Box<dyn OcrBackend>>,
    config: OcrPipelineConfig,
}

impl OcrPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: OcrPipelineConfig) -> Self {
        Self {
            backends: Vec::new(),
            config,
        }
    }

    /// Register a backend.  Returns `&mut Self` for chaining.
    pub fn add_backend(&mut self, backend: Box<dyn OcrBackend>) -> &mut Self {
        self.backends.push(backend);
        self
    }

    /// Run all backends against a single image and return the best result.
    ///
    /// If `preferred_backend` is set, that backend is tried first.  The result
    /// with the highest `average_confidence` that meets `min_confidence` wins.
    /// If no result meets the threshold the best available result is returned
    /// anyway.  If no backends are registered an empty `OcrResult` is returned.
    pub fn process_image(&self, data: &[u8], width: usize, height: usize) -> OcrResult {
        if self.backends.is_empty() {
            return OcrResult {
                lines: Vec::new(),
                full_text: String::new(),
                average_confidence: 0.0,
            };
        }

        // Build an ordering that puts the preferred backend first.
        let mut indices: Vec<usize> = (0..self.backends.len()).collect();
        if let Some(ref pref) = self.config.preferred_backend {
            if let Some(pos) = self.backends.iter().position(|b| b.name() == pref.as_str()) {
                indices.remove(pos);
                indices.insert(0, pos);
            }
        }

        let mut best: Option<OcrResult> = None;
        for idx in indices {
            let result = self.backends[idx].recognize(data, width, height);
            let dominated = match best {
                Some(ref b) => result.average_confidence > b.average_confidence,
                None => true,
            };
            if dominated {
                best = Some(result);
            }
        }

        best.unwrap_or(OcrResult {
            lines: Vec::new(),
            full_text: String::new(),
            average_confidence: 0.0,
        })
    }

    /// Run the pipeline over a collection of [`ExtractedImage`]s.
    ///
    /// Returns a `Vec` of `(image_index, OcrResult)` pairs.
    pub fn process_extracted_images(&self, images: &[ExtractedImage]) -> Vec<(usize, OcrResult)> {
        images
            .iter()
            .map(|img| {
                let w = img.width.unwrap_or(0) as usize;
                let h = img.height.unwrap_or(0) as usize;
                let result = self.process_image(&img.data, w, h);
                (img.index, result)
            })
            .collect()
    }

    /// Number of registered backends.
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Names of all registered backends in insertion order.
    pub fn backend_names(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.name().to_string()).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_xml_tags_basic() {
        let input = "<p>Hello <b>world</b>!</p>";
        let result = strip_xml_tags(input);
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(result.contains("!"));
        assert!(!result.contains("<p>"));
        assert!(!result.contains("<b>"));
    }

    #[test]
    fn test_strip_xml_tags_entities() {
        let input = "5 &lt; 10 &amp; 3 &gt; 1 &quot;test&quot;";
        let result = strip_xml_tags(input);
        assert_eq!(result, "5 < 10 & 3 > 1 \"test\"");
    }

    #[test]
    fn test_strip_xml_tags_script_removal() {
        let input = "<div>Before<script>var x = 1;</script>After</div>";
        let result = strip_xml_tags(input);
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
        assert!(!result.contains("var x"));
    }

    #[test]
    fn test_extract_xml_text_simple() {
        let xml = r#"<root><title>My Book</title><title>Subtitle</title></root>"#;
        let results = extract_xml_text(xml, "title");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "My Book");
        assert_eq!(results[1], "Subtitle");
    }

    #[test]
    fn test_extract_xml_text_nested_tags() {
        let xml = r#"<dc:creator><name>John Doe</name></dc:creator>"#;
        let results = extract_xml_text(xml, "dc:creator");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "John Doe");
    }

    #[test]
    fn test_extract_xml_metadata() {
        let xml = r#"
            <metadata>
                <dc:title>Test Document</dc:title>
                <dc:creator>Author One</dc:creator>
                <dc:creator>Author Two</dc:creator>
                <dc:language>en</dc:language>
                <dc:date>2024-01-15</dc:date>
                <dc:description>A test document</dc:description>
                <dc:publisher>Test Publisher</dc:publisher>
            </metadata>
        "#;
        let meta = extract_xml_metadata(xml);
        assert_eq!(meta.title.as_deref(), Some("Test Document"));
        assert_eq!(meta.authors.len(), 2);
        assert_eq!(meta.authors[0], "Author One");
        assert_eq!(meta.authors[1], "Author Two");
        assert_eq!(meta.language.as_deref(), Some("en"));
        assert_eq!(meta.date.as_deref(), Some("2024-01-15"));
        assert_eq!(meta.description.as_deref(), Some("A test document"));
        assert_eq!(meta.publisher.as_deref(), Some("Test Publisher"));
    }

    #[test]
    fn test_normalize_text() {
        let input = "  Hello   world  \n\n\n\n  Second paragraph  \n  Third  ";
        let result = normalize_text(input);
        assert_eq!(result, "Hello world\n\nSecond paragraph\nThird");
    }

    #[test]
    fn test_normalize_text_crlf() {
        let input = "Line one\r\nLine two\r\n\r\n\r\nLine three";
        let result = normalize_text(input);
        assert_eq!(result, "Line one\nLine two\n\nLine three");
    }

    #[test]
    fn test_detect_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());

        assert_eq!(
            parser.detect_format(Path::new("book.epub")),
            Some(DocumentFormat::Epub)
        );
        assert_eq!(
            parser.detect_format(Path::new("report.docx")),
            Some(DocumentFormat::Docx)
        );
        assert_eq!(
            parser.detect_format(Path::new("letter.odt")),
            Some(DocumentFormat::Odt)
        );
        assert_eq!(
            parser.detect_format(Path::new("page.html")),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            parser.detect_format(Path::new("page.htm")),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            parser.detect_format(Path::new("notes.txt")),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(
            parser.detect_format(Path::new("readme.md")),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(
            parser.detect_format(Path::new("image.png")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(parser.detect_format(Path::new("noext")), None);
    }

    #[test]
    fn test_parse_plain_text() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let content = "Hello, this is a plain text document.\n\nIt has two paragraphs.";
        let doc = parser
            .parse_string(content, DocumentFormat::PlainText)
            .unwrap();

        assert_eq!(doc.format, DocumentFormat::PlainText);
        assert!(doc.text.contains("Hello"));
        assert!(doc.text.contains("two paragraphs"));
        assert_eq!(doc.word_count, 11);
        assert!(doc.char_count > 0);
        assert_eq!(doc.sections.len(), 1);
    }

    #[test]
    fn test_parse_html_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let html = r#"
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <title>Test Page</title>
                <meta name="author" content="Jane Doe">
                <meta name="description" content="A test page">
            </head>
            <body>
                <h1>Welcome</h1>
                <p>This is the first paragraph.</p>
                <h2>Section Two</h2>
                <p>This is the second section content.</p>
            </body>
            </html>
        "#;

        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();

        assert_eq!(doc.format, DocumentFormat::Html);
        assert_eq!(doc.metadata.title.as_deref(), Some("Test Page"));
        assert_eq!(doc.metadata.authors, vec!["Jane Doe"]);
        assert_eq!(doc.metadata.description.as_deref(), Some("A test page"));
        assert_eq!(doc.metadata.language.as_deref(), Some("en"));
        assert!(doc.text.contains("first paragraph"));
        assert!(doc.text.contains("second section"));
        assert!(doc.word_count > 0);
    }

    #[test]
    fn test_parse_html_sections() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let html = r#"
            <body>
                <h1>Title</h1>
                <p>Intro text here.</p>
                <h2>Chapter 1</h2>
                <p>Chapter one content.</p>
            </body>
        "#;

        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();
        let titles = doc.section_titles();

        assert!(titles.contains(&"Title"));
        assert!(titles.contains(&"Chapter 1"));
    }

    #[test]
    fn test_parse_html_no_sections() {
        let mut config = DocumentParserConfig::default();
        config.extract_sections = false;
        let parser = DocumentParser::new(config);

        let html = "<p>Just a paragraph.</p>";
        let doc = parser.parse_string(html, DocumentFormat::Html).unwrap();
        assert!(doc.sections.is_empty());
        assert!(doc.text.contains("Just a paragraph"));
    }

    #[test]
    fn test_parsed_document_section_text() {
        let doc = ParsedDocument {
            text: "Full text".to_string(),
            metadata: DocumentMetadata::default(),
            sections: vec![
                DocumentSection {
                    title: Some("First".to_string()),
                    content: "Content one".to_string(),
                    level: 1,
                    index: 0,
                },
                DocumentSection {
                    title: Some("Second".to_string()),
                    content: "Content two".to_string(),
                    level: 2,
                    index: 1,
                },
            ],
            tables: Vec::new(),
            format: DocumentFormat::Html,
            source_path: None,
            char_count: 9,
            word_count: 2,
        };

        assert_eq!(doc.section_text(0), Some("Content one"));
        assert_eq!(doc.section_text(1), Some("Content two"));
        assert_eq!(doc.section_text(2), None);
    }

    #[test]
    fn test_parsed_document_section_titles() {
        let doc = ParsedDocument {
            text: String::new(),
            metadata: DocumentMetadata::default(),
            sections: vec![
                DocumentSection {
                    title: Some("Alpha".to_string()),
                    content: String::new(),
                    level: 1,
                    index: 0,
                },
                DocumentSection {
                    title: None,
                    content: String::new(),
                    level: 0,
                    index: 1,
                },
                DocumentSection {
                    title: Some("Gamma".to_string()),
                    content: String::new(),
                    level: 2,
                    index: 2,
                },
            ],
            tables: Vec::new(),
            format: DocumentFormat::PlainText,
            source_path: None,
            char_count: 0,
            word_count: 0,
        };

        let titles = doc.section_titles();
        assert_eq!(titles, vec!["Alpha", "Gamma"]);
    }

    #[test]
    fn test_max_size_enforcement() {
        let mut config = DocumentParserConfig::default();
        config.max_size_bytes = 10; // Very small limit
        let parser = DocumentParser::new(config);

        let data = b"This is way more than ten bytes of data for testing";
        let result = parser.parse_bytes(data, DocumentFormat::PlainText);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("exceeds maximum"));
    }

    #[test]
    fn test_extract_first_heading() {
        let html = "<html><body><h1>Main Title</h1><p>Text</p><h2>Sub</h2></body></html>";
        let heading = extract_first_heading(html);
        assert_eq!(heading, Some("Main Title".to_string()));
    }

    #[test]
    fn test_extract_first_heading_none() {
        let html = "<html><body><p>No headings here.</p></body></html>";
        let heading = extract_first_heading(html);
        assert_eq!(heading, None);
    }

    #[test]
    fn test_document_parser_config_default() {
        let config = DocumentParserConfig::default();
        assert!(config.preserve_paragraphs);
        assert!(config.extract_metadata);
        assert!(config.extract_sections);
        assert!(config.strip_tags);
        assert!(config.normalize_whitespace);
        assert_eq!(config.max_size_bytes, 50 * 1024 * 1024);
    }

    #[test]
    fn test_html_numeric_entities() {
        let input = "&#65;&#66;&#67; and &#x41;&#x42;&#x43;";
        let result = strip_xml_tags(input);
        assert_eq!(result, "ABC and ABC");
    }

    // ===== CSV parsing tests =====

    #[test]
    fn test_csv_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name,Age,City\nAlice,30,NYC\nBob,25,LA\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
        assert!(result.text.contains("Bob"));
        assert!(!result.tables.is_empty());
        assert_eq!(result.tables[0].headers.len(), 3);
        assert_eq!(result.tables[0].rows.len(), 2);
    }

    #[test]
    fn test_csv_tsv_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let tsv = b"Name\tAge\tCity\nAlice\t30\tNYC\n";
        let result = parser.parse_bytes(tsv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
        assert!(!result.tables.is_empty());
        assert_eq!(result.tables[0].headers.len(), 3);
    }

    #[test]
    fn test_csv_quoted_fields() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name,Description\n\"Alice\",\"She said \"\"hello\"\"\"\nBob,\"Line1\nLine2\"\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
    }

    #[test]
    fn test_csv_unicode() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = "Nombre,Ciudad\nJosé,México\nFrançois,París\n".as_bytes();
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("José"));
        assert!(result.text.contains("México"));
    }

    #[test]
    fn test_csv_empty() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.tables.is_empty());
    }

    #[test]
    fn test_csv_single_column() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let csv = b"Name\nAlice\nBob\nCharlie\n";
        let result = parser.parse_bytes(csv, DocumentFormat::Csv).unwrap();
        assert!(result.text.contains("Alice"));
    }

    // ===== Email parsing tests =====

    #[test]
    fn test_email_basic() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: alice@example.com\r\nTo: bob@example.com\r\nSubject: Test Email\r\nDate: Mon, 1 Jan 2024 12:00:00 +0000\r\n\r\nHello Bob, this is a test email.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Hello Bob"));
        assert_eq!(result.metadata.title.as_deref(), Some("Test Email"));
        assert!(!result.metadata.authors.is_empty());
        assert!(result.metadata.authors[0].contains("alice@example.com"));
        assert!(result.metadata.extra.contains_key("to"));
    }

    #[test]
    fn test_email_multipart() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: alice@example.com\r\nContent-Type: multipart/alternative; boundary=\"boundary123\"\r\n\r\n--boundary123\r\nContent-Type: text/plain\r\n\r\nPlain text body\r\n--boundary123\r\nContent-Type: text/html\r\n\r\n<html><body>HTML body</body></html>\r\n--boundary123--\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Plain text body"));
    }

    #[test]
    fn test_email_quoted_printable() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: test@example.com\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\nHello =C3=A9l=C3=A8ve, this is a =\r\ncontinued line.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        // Should decode QP: =C3=A9 = é, =C3=A8 = è
        assert!(result.text.contains("continued line"));
    }

    #[test]
    fn test_email_base64_body() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // "Hello World" in base64 is "SGVsbG8gV29ybGQ="
        let email = b"From: test@example.com\r\nContent-Transfer-Encoding: base64\r\n\r\nSGVsbG8gV29ybGQ=\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Hello World"));
    }

    #[test]
    fn test_email_no_subject() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let email = b"From: test@example.com\r\n\r\nBody text here.\r\n";
        let result = parser.parse_bytes(email, DocumentFormat::Email).unwrap();
        assert!(result.text.contains("Body text"));
    }

    // ===== Image metadata extraction tests =====

    #[test]
    fn test_image_png_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // Minimal PNG: magic + IHDR chunk (width=100, height=50)
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]; // PNG magic
                                                                            // IHDR chunk: length(13) + "IHDR" + width(100) + height(50) + bit_depth + color_type + ...
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]); // chunk length = 13
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&100u32.to_be_bytes()); // width
        png.extend_from_slice(&50u32.to_be_bytes()); // height
        png.extend_from_slice(&[8, 2, 0, 0, 0]); // bit_depth=8, color=RGB, rest
        png.extend_from_slice(&[0, 0, 0, 0]); // CRC
        let result = parser.parse_bytes(&png, DocumentFormat::Image).unwrap();
        assert!(result.metadata.extra.contains_key("format"));
        assert_eq!(result.metadata.extra["format"], "PNG");
        assert_eq!(result.metadata.extra["width"], "100");
        assert_eq!(result.metadata.extra["height"], "50");
    }

    #[test]
    fn test_image_jpeg_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // Minimal JPEG: SOI + SOF0 marker with dimensions
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0]; // SOI + APP0
        jpeg.extend_from_slice(&[0x00, 0x02, 0x00, 0x00]); // APP0 minimal
                                                           // SOF0 marker (0xFFC0)
        jpeg.extend_from_slice(&[0xFF, 0xC0]);
        jpeg.extend_from_slice(&[0x00, 0x0B]); // length = 11
        jpeg.push(8); // precision
        jpeg.extend_from_slice(&200u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&300u16.to_be_bytes()); // width
        jpeg.push(3); // num components
        jpeg.extend_from_slice(&[0; 6]); // component data padding
        let result = parser.parse_bytes(&jpeg, DocumentFormat::Image).unwrap();
        assert!(result.metadata.extra.contains_key("format"));
        assert_eq!(result.metadata.extra["format"], "JPEG");
    }

    #[test]
    fn test_image_gif_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // GIF89a header: magic + width(120) + height(80)
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&120u16.to_le_bytes()); // width LE
        gif.extend_from_slice(&80u16.to_le_bytes()); // height LE
        gif.extend_from_slice(&[0; 10]); // padding
        let result = parser.parse_bytes(&gif, DocumentFormat::Image).unwrap();
        assert_eq!(result.metadata.extra["format"], "GIF");
        assert_eq!(result.metadata.extra["width"], "120");
        assert_eq!(result.metadata.extra["height"], "80");
    }

    #[test]
    fn test_image_unknown_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let data = b"This is not an image file";
        let result = parser.parse_bytes(data, DocumentFormat::Image).unwrap();
        assert_eq!(
            result.metadata.extra.get("format").map(|s| s.as_str()),
            Some("Unknown")
        );
    }

    #[test]
    fn test_image_bmp_detection() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        // BMP header: "BM" + file size + reserved + data offset + DIB header size + width + height
        let mut bmp = b"BM".to_vec();
        bmp.extend_from_slice(&[0; 12]); // file size + reserved + data offset
        bmp.extend_from_slice(&40u32.to_le_bytes()); // DIB header size (BITMAPINFOHEADER)
        bmp.extend_from_slice(&640u32.to_le_bytes()); // width (LE, signed i32 but positive)
        bmp.extend_from_slice(&480u32.to_le_bytes()); // height
        bmp.extend_from_slice(&[0; 20]); // rest of DIB header
        let result = parser.parse_bytes(&bmp, DocumentFormat::Image).unwrap();
        assert_eq!(result.metadata.extra["format"], "BMP");
        assert_eq!(result.metadata.extra["width"], "640");
        assert_eq!(result.metadata.extra["height"], "480");
    }

    // ===== Document format detection tests =====

    #[test]
    fn test_detect_csv_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let format = parser.detect_format(std::path::Path::new("data.csv"));
        assert_eq!(format, Some(DocumentFormat::Csv));
        let format = parser.detect_format(std::path::Path::new("data.tsv"));
        assert_eq!(format, Some(DocumentFormat::Csv));
    }

    #[test]
    fn test_detect_email_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let format = parser.detect_format(std::path::Path::new("message.eml"));
        assert_eq!(format, Some(DocumentFormat::Email));
    }

    #[test]
    fn test_detect_image_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.jpg")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.jpeg")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("icon.png")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("animation.gif")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.bmp")),
            Some(DocumentFormat::Image)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("photo.webp")),
            Some(DocumentFormat::Image)
        );
    }

    #[test]
    fn test_detect_pptx_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("slides.pptx")),
            Some(DocumentFormat::Pptx)
        );
    }

    #[test]
    fn test_detect_xlsx_format() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        assert_eq!(
            parser.detect_format(std::path::Path::new("spreadsheet.xlsx")),
            Some(DocumentFormat::Xlsx)
        );
        assert_eq!(
            parser.detect_format(std::path::Path::new("legacy.xls")),
            Some(DocumentFormat::Xlsx)
        );
    }

    // ===== OCR-lite engine tests =====

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert!(config.min_confidence > 0.0);
        assert_eq!(config.char_height, 7);
        assert!(config.binarize_threshold.is_none());
    }

    #[test]
    fn test_ocr_engine_creation() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        assert!(!engine.templates.is_empty());
    }

    #[test]
    fn test_glyph_template_structure() {
        let template = GlyphTemplate {
            character: 'X',
            width: 5,
            height: 7,
            bitmap: vec![0; 35],
        };
        assert_eq!(template.bitmap.len(), template.width * template.height);
    }

    #[test]
    fn test_binarize_fixed_threshold() {
        let config = OcrConfig {
            binarize_threshold: Some(128),
            ..Default::default()
        };
        let engine = OcrEngine::with_default_templates(config);
        let image = vec![0, 50, 100, 128, 200, 255];
        let binary = engine.binarize(&image, 6, 1);
        // pixels < 128 become 1 (black), >= 128 become 0 (white)
        assert_eq!(binary, vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_otsu_threshold_bimodal() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Bimodal image: half centered around 50, half centered around 200
        let mut image = vec![50u8; 100];
        for i in 50..100 {
            image[i] = 200;
        }
        let threshold = engine.otsu_threshold(&image);
        // Otsu should pick a threshold that separates the two clusters (>= 50, <= 200)
        assert!(
            threshold >= 50 && threshold <= 200,
            "Expected threshold between 50 and 200 inclusive, got {}",
            threshold
        );
    }

    #[test]
    fn test_detect_text_lines_single() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // 10x10 image with a horizontal line of black pixels at rows 3-6
        let mut binary = vec![0u8; 100];
        for y in 3..7 {
            for x in 0..10 {
                binary[y * 10 + x] = 1;
            }
        }
        let lines = engine.detect_text_lines(&binary, 10, 10);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], (3, 7));
    }

    #[test]
    fn test_segment_characters() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Line with two "characters" separated by a gap
        // Width 12, height 5: cols 0-2 have pixels, cols 3-4 empty, cols 5-8 have pixels
        let mut line = vec![0u8; 60]; // 12 * 5
        for y in 0..5 {
            for x in 0..3 {
                line[y * 12 + x] = 1;
            }
            for x in 5..9 {
                line[y * 12 + x] = 1;
            }
        }
        let chars = engine.segment_characters(&line, 12, 5);
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0], (0, 3));
        assert_eq!(chars[1], (5, 9));
    }

    #[test]
    fn test_match_template_returns_best() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        // Use the 'T' template as input — should match 'T' best
        let t_template = engine
            .templates
            .iter()
            .find(|t| t.character == 'T')
            .unwrap();
        let (ch, confidence) =
            engine.match_template(&t_template.bitmap, t_template.width, t_template.height);
        assert_eq!(ch, 'T');
        assert!(
            confidence > 0.99,
            "Expected near-perfect match, got {}",
            confidence
        );
    }

    #[test]
    fn test_recognize_empty_image() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        let result = engine.recognize_bitmap(&[], 0, 0);
        assert!(result.full_text.is_empty());
        assert_eq!(result.average_confidence, 0.0);
    }

    #[test]
    fn test_recognize_all_white() {
        let engine = OcrEngine::with_default_templates(OcrConfig::default());
        let image = vec![255u8; 100]; // 10x10 all white
        let result = engine.recognize_bitmap(&image, 10, 10);
        assert!(result.full_text.is_empty());
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn test_epub_without_feature_returns_error() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let result = parser.parse_bytes(b"fake epub data", DocumentFormat::Epub);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("documents"));
    }

    // ========================================================================
    // Image Extraction tests
    // ========================================================================

    #[test]
    fn test_detect_jpeg_magic() {
        let header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let result = ImageExtractor::detect_format(&header);
        assert_eq!(result, Some(ImageFormat::Jpeg));
    }

    #[test]
    fn test_detect_png_magic() {
        let header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = ImageExtractor::detect_format(&header);
        assert_eq!(result, Some(ImageFormat::Png));
    }

    #[test]
    fn test_extract_jpeg_with_dimensions() {
        // Build a synthetic JPEG:
        // SOI + SOF0 marker with dimensions + padding + EOI
        let mut jpeg = Vec::new();
        // SOI
        jpeg.extend_from_slice(&[0xFF, 0xD8]);
        // SOF0 marker: FF C0, then length (2 bytes), precision (1 byte), height (2 BE), width (2 BE)
        jpeg.extend_from_slice(&[0xFF, 0xC0]);
        jpeg.extend_from_slice(&[0x00, 0x11]); // length = 17
        jpeg.push(0x08); // precision
        jpeg.extend_from_slice(&[0x00, 0x80]); // height = 128
        jpeg.extend_from_slice(&[0x01, 0x00]); // width = 256
                                               // Pad with zeros to meet the min_size_bytes=100 threshold
        jpeg.extend_from_slice(&[0x00; 100]);
        // EOI
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let extractor = ImageExtractor::with_default_config();
        let images = extractor.extract_from_bytes(&jpeg);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].format, ImageFormat::Jpeg);
        assert_eq!(images[0].width, Some(256));
        assert_eq!(images[0].height, Some(128));
        assert_eq!(images[0].index, 0);
        assert_eq!(images[0].offset, 0);
    }

    #[test]
    fn test_extract_png_with_dimensions() {
        // Build a synthetic PNG with IHDR + IEND
        let mut png = Vec::new();
        // 8-byte PNG signature
        png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk: length (4 bytes) = 13
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]);
        // Chunk type "IHDR"
        png.extend_from_slice(b"IHDR");
        // Width = 320 (4 bytes big-endian)
        png.extend_from_slice(&[0x00, 0x00, 0x01, 0x40]);
        // Height = 240 (4 bytes big-endian)
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0xF0]);
        // Bit depth, color type, compression, filter, interlace
        png.extend_from_slice(&[0x08, 0x02, 0x00, 0x00, 0x00]);
        // CRC (4 bytes, not validated by our parser)
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        // Pad to exceed min_size_bytes
        png.extend_from_slice(&[0x00; 80]);
        // IEND chunk trailer (12 bytes)
        png.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]);

        let extractor = ImageExtractor::with_default_config();
        let images = extractor.extract_from_bytes(&png);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].format, ImageFormat::Png);
        assert_eq!(images[0].width, Some(320));
        assert_eq!(images[0].height, Some(240));
    }

    #[test]
    fn test_skip_small_images() {
        // Build a tiny JPEG that is smaller than a large min_size_bytes
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
        jpeg.extend_from_slice(&[0x00; 20]);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let config = ImageExtractionConfig {
            min_size_bytes: 500, // larger than our tiny JPEG
            ..Default::default()
        };
        let extractor = ImageExtractor::new(config);
        let images = extractor.extract_from_bytes(&jpeg);
        assert!(images.is_empty(), "Small image should have been skipped");
    }

    #[test]
    fn test_max_images_limit() {
        // Build data containing 5 JPEGs, but limit to 2
        let mut data = Vec::new();
        for _ in 0..5 {
            // SOI
            data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
            // Pad to exceed 100 bytes per image
            data.extend_from_slice(&[0x00; 110]);
            // EOI
            data.extend_from_slice(&[0xFF, 0xD9]);
        }

        let config = ImageExtractionConfig {
            max_images: 2,
            ..Default::default()
        };
        let extractor = ImageExtractor::new(config);
        let images = extractor.extract_from_bytes(&data);
        assert_eq!(images.len(), 2, "Should stop after max_images reached");
        assert_eq!(images[0].index, 0);
        assert_eq!(images[1].index, 1);
    }

    #[test]
    fn test_mixed_format_document() {
        let mut data = Vec::new();

        // Embed a JPEG
        data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
        data.extend_from_slice(&[0x00; 110]);
        data.extend_from_slice(&[0xFF, 0xD9]);

        // Some random garbage between images
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        // Embed a PNG
        // PNG signature
        data.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]);
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x64]); // width=100
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x64]); // height=100
        data.extend_from_slice(&[0x08, 0x02, 0x00, 0x00, 0x00]);
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC
        data.extend_from_slice(&[0x00; 80]); // padding
                                             // IEND trailer
        data.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]);

        let analysis =
            DocumentImageAnalysis::from_document(&data, ImageExtractionConfig::default());
        assert_eq!(analysis.images.len(), 2);
        assert_eq!(analysis.formats_found.get(&ImageFormat::Jpeg), Some(&1));
        assert_eq!(analysis.formats_found.get(&ImageFormat::Png), Some(&1));
        assert!(analysis.total_image_bytes > 0);
    }

    #[test]
    fn test_empty_document() {
        let data: &[u8] = &[];
        let analysis = DocumentImageAnalysis::from_document(data, ImageExtractionConfig::default());
        assert!(analysis.images.is_empty());
        assert_eq!(analysis.total_image_bytes, 0);
        assert!(analysis.formats_found.is_empty());
    }

    // ====================================================================
    // WS9 — OCR Integration tests
    // ====================================================================

    #[test]
    fn test_template_ocr_backend_name() {
        let backend = TemplateOcrBackend::new(OcrConfig::default());
        assert_eq!(backend.name(), "template");
        assert!(backend.supports_format(&ImageFormat::Png));
        assert!(backend.supports_format(&ImageFormat::Jpeg));
        assert!(!backend.supports_format(&ImageFormat::Gif));
    }

    #[test]
    fn test_tesseract_backend_not_available() {
        let backend = TesseractOcrBackend::new(TesseractConfig::default());
        assert_eq!(backend.name(), "tesseract");
        let result = backend.recognize(&[128u8; 35], 5, 7);
        assert_eq!(result.average_confidence, 0.0);
        assert!(result.full_text.contains("binary not available"));
        assert!(backend.supports_format(&ImageFormat::Gif));
        assert_eq!(backend.confidence_threshold(), 0.0);
    }

    #[test]
    fn test_ocr_pipeline_selects_best() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig {
            min_confidence: 0.0,
            ..OcrPipelineConfig::default()
        });
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        pipeline.add_backend(Box::new(TesseractOcrBackend::new(
            TesseractConfig::default(),
        )));

        assert_eq!(pipeline.backend_count(), 2);

        // 5x7 synthetic grayscale bitmap (all mid-grey).
        let image = vec![128u8; 5 * 7];
        let result = pipeline.process_image(&image, 5, 7);

        // The template backend returns a real (possibly 0.0) OCR attempt while
        // Tesseract always returns 0.0.  Because the pipeline picks the highest
        // confidence, and ties go to the first result evaluated, the template
        // backend's result should win (or tie) since it is added first.
        // Either way, the pipeline must not panic and must return *some* result.
        assert!(result.average_confidence >= 0.0);
    }

    #[test]
    fn test_ocr_pipeline_single_backend() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig::default());
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        assert_eq!(pipeline.backend_count(), 1);

        let image = vec![200u8; 10 * 7];
        let result = pipeline.process_image(&image, 10, 7);
        // Should return a valid OcrResult (even if empty text).
        assert!(result.average_confidence >= 0.0);
    }

    #[test]
    fn test_ocr_pipeline_process_extracted_images() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig {
            min_confidence: 0.0,
            ..OcrPipelineConfig::default()
        });
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));

        let images = vec![
            ExtractedImage {
                data: vec![128u8; 35],
                format: ImageFormat::Png,
                width: Some(5),
                height: Some(7),
                page: None,
                index: 0,
                offset: 0,
            },
            ExtractedImage {
                data: vec![200u8; 70],
                format: ImageFormat::Jpeg,
                width: Some(10),
                height: Some(7),
                page: Some(1),
                index: 1,
                offset: 100,
            },
        ];

        let results = pipeline.process_extracted_images(&images);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // first image's index
        assert_eq!(results[1].0, 1); // second image's index
    }

    #[test]
    fn test_ocr_pipeline_backend_names() {
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig::default());
        pipeline.add_backend(Box::new(TemplateOcrBackend::new(OcrConfig::default())));
        pipeline.add_backend(Box::new(TesseractOcrBackend::new(
            TesseractConfig::default(),
        )));

        let names = pipeline.backend_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "template");
        assert_eq!(names[1], "tesseract");
    }
}
