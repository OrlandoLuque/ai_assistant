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
        let item_re =
            Regex::new(r#"<item\s[^>]*id="([^"]+)"[^>]*href="([^"]+)"[^>]*/?\s*>"#).unwrap();
        // Also handle the case where href comes before id
        let item_re2 =
            Regex::new(r#"<item\s[^>]*href="([^"]+)"[^>]*id="([^"]+)"[^>]*/?\s*>"#).unwrap();

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
        let spine_re = Regex::new(r#"<itemref\s[^>]*idref="([^"]+)"[^>]*/?\s*>"#).unwrap();
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
        let para_re = Regex::new(r"(?s)<w:p[\s>].*?</w:p>").unwrap();
        let text_re = Regex::new(r"(?s)<w:t[^>]*>(.*?)</w:t>").unwrap();

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
            Regex::new(r"(?s)<text:(h|p)([^>]*)>(.*?)</text:(h|p)>").unwrap();
        let outline_re = Regex::new(r#"text:outline-level="(\d+)""#).unwrap();

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
            let mut seen_on_page: std::collections::HashSet<String> = std::collections::HashSet::new();

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
    fn extract_tables_from_page(&self, page_text: &str, page_number: usize) -> (Vec<PdfTable>, String) {
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
    fn detect_table_at(&self, lines: &[&str], start: usize, page_number: usize) -> Option<(PdfTable, usize)> {
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
        let raw_text = table_lines.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join("\n");

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
        let re = Regex::new(r"\s{2,}|\t+").unwrap();
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
    fn separate_headers_and_data(&self, table_lines: &[(String, Vec<String>)]) -> (Vec<String>, Vec<Vec<String>>) {
        if table_lines.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let first_row = &table_lines[0].1;

        // Heuristic: headers are often non-numeric or have different characteristics
        let first_row_numeric_ratio = first_row.iter()
            .filter(|s| s.parse::<f64>().is_ok() || s.chars().all(|c| c.is_numeric() || c == '.' || c == ',' || c == '-'))
            .count() as f32 / first_row.len().max(1) as f32;

        let second_row_numeric_ratio = if table_lines.len() > 1 {
            let second_row = &table_lines[1].1;
            second_row.iter()
                .filter(|s| s.parse::<f64>().is_ok() || s.chars().all(|c| c.is_numeric() || c == '.' || c == ',' || c == '-'))
                .count() as f32 / second_row.len().max(1) as f32
        } else {
            first_row_numeric_ratio
        };

        // If first row is less numeric than second, treat it as header
        if first_row_numeric_ratio < second_row_numeric_ratio - 0.2 {
            let headers = first_row.clone();
            let rows = table_lines[1..].iter().map(|(_, cols)| cols.clone()).collect();
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
        let variance = col_counts.iter().map(|&c| (c as f32 - avg_cols).powi(2)).sum::<f32>() / col_counts.len() as f32;

        if variance < 0.5 {
            score += 0.2; // Very consistent
        } else if variance < 1.0 {
            score += 0.1; // Reasonably consistent
        }

        // Numeric content suggests data table
        let numeric_ratio = table_lines.iter()
            .flat_map(|(_, cols)| cols.iter())
            .filter(|s| s.parse::<f64>().is_ok())
            .count() as f32 / table_lines.iter().flat_map(|(_, cols)| cols.iter()).count().max(1) as f32;

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
        let footnote_markers = Regex::new(
            r"^[\s]*[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§\[\(]?\d{0,2}[\]\)]?[\.\s]"
        ).ok()?;

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
                if "¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§[(".contains(first_char) || first_char.is_ascii_digit() {
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
            r"^fig\.?\s*\d",           // Fig. 1, Fig 2
            r"^figure\s+\d",           // Figure 1
            r"^table\s+\d",            // Table 1
            r"^tab\.?\s*\d",           // Tab. 1
            r"^chart\s+\d",            // Chart 1
            r"^graph\s+\d",            // Graph 1
            r"^diagram\s+\d",          // Diagram 1
            r"^plate\s+\d",            // Plate 1
            r"^illustration\s+\d",     // Illustration 1
            r"^exhibit\s+\d",          // Exhibit 1
            r"^source:",               // Source: ...
            r"^note:",                 // Note: ...
            r"^caption:",              // Caption: ...
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
        let sidebar_markers = ["note:", "tip:", "warning:", "important:", "caution:",
                               "sidebar:", "box:", "callout:", "inset:"];
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
                .unwrap();
        let meta_re2 =
            Regex::new(r#"(?i)<meta\s[^>]*content="([^"]+)"[^>]*name="([^"]+)"[^>]*/?\s*>"#)
                .unwrap();

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
            let lang_re = Regex::new(r#"(?i)<html[^>]*\slang="([^"]+)"[^>]*>"#).unwrap();
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
        let body_re = Regex::new(r"(?is)<body[^>]*>(.*)</body>").unwrap();
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
        let heading_re = Regex::new(r"(?is)<h([1-6])[^>]*>(.*?)</h[1-6]>").unwrap();

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
            let full_match = caps.get(0).unwrap();
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
}

// ============================================================================
// Helper functions
// ============================================================================

/// Remove all XML/HTML tags from a string, preserving the text content.
/// Also decodes common HTML entities.
pub fn strip_xml_tags(xml: &str) -> String {
    // Remove script and style blocks entirely
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let cleaned = script_re.replace_all(xml, "");
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let cleaned = style_re.replace_all(&cleaned, "");

    // Replace <br>, <br/>, <br /> with newlines
    let br_re = Regex::new(r"(?i)<br\s*/?\s*>").unwrap();
    let cleaned = br_re.replace_all(&cleaned, "\n");

    // Replace block-level closing tags with newlines for paragraph separation
    let block_re = Regex::new(r"(?i)</?(p|div|li|tr|blockquote|article|section|header|footer|nav|aside|main|figure|figcaption|details|summary|dd|dt)\s*[^>]*>").unwrap();
    let cleaned = block_re.replace_all(&cleaned, "\n");

    // Remove all remaining tags
    let tag_re = Regex::new(r"<[^>]+>").unwrap();
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
    let num_entity_re = Regex::new(r"&#(\d+);").unwrap();
    let result = num_entity_re.replace_all(&result, |caps: &regex::Captures| {
        if let Ok(code) = caps[1].parse::<u32>() {
            if let Some(ch) = char::from_u32(code) {
                return ch.to_string();
            }
        }
        caps[0].to_string()
    });

    let hex_entity_re = Regex::new(r"(?i)&#x([0-9a-f]+);").unwrap();
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
    let re = Regex::new(&pattern).unwrap();

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
    let space_re = Regex::new(r"[^\S\n]+").unwrap();
    let text = space_re.replace_all(&text, " ");

    // Collapse 3+ newlines into 2
    let newline_re = Regex::new(r"\n{3,}").unwrap();
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
    let heading_re = Regex::new(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>").unwrap();
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
        assert_eq!(parser.detect_format(Path::new("image.png")), None);
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

    #[cfg(not(feature = "documents"))]
    #[test]
    fn test_epub_without_feature_returns_error() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let result = parser.parse_bytes(b"fake epub data", DocumentFormat::Epub);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("documents"));
    }
}
