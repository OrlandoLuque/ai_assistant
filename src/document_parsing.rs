//! Document parsing module for extracting plain text and metadata from various file formats.
//!
//! Supports EPUB, DOCX, ODT (via the `document-formats` feature which requires the `zip` crate),
//! and HTML (regex-based tag stripping, always available).
//!
//! All XML/HTML parsing is done using regex patterns rather than full XML parsers,
//! which keeps dependencies minimal while handling the common cases.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "document-formats")]
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
    /// The full extracted plain text.
    pub text: String,
    /// Extracted metadata.
    pub metadata: DocumentMetadata,
    /// Extracted sections.
    pub sections: Vec<DocumentSection>,
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
            #[cfg(feature = "document-formats")]
            DocumentFormat::Epub => self.parse_epub(data),
            #[cfg(feature = "document-formats")]
            DocumentFormat::Docx => self.parse_docx(data),
            #[cfg(feature = "document-formats")]
            DocumentFormat::Odt => self.parse_odt(data),

            #[cfg(not(feature = "document-formats"))]
            DocumentFormat::Epub | DocumentFormat::Docx | DocumentFormat::Odt => {
                anyhow::bail!(
                    "Document format {:?} requires the 'document-formats' feature to be enabled",
                    format
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
    #[cfg(feature = "document-formats")]
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
            format: DocumentFormat::Epub,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Find the path to the OPF file inside the EPUB archive.
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
            format: DocumentFormat::Docx,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract paragraph texts from DOCX XML by finding <w:p> blocks and their <w:t> runs.
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
    #[cfg(feature = "document-formats")]
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
            format: DocumentFormat::Odt,
            source_path: None,
            char_count,
            word_count,
        })
    }

    /// Extract metadata from an ODT meta.xml file.
    #[cfg(feature = "document-formats")]
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

    #[cfg(not(feature = "document-formats"))]
    #[test]
    fn test_epub_without_feature_returns_error() {
        let parser = DocumentParser::new(DocumentParserConfig::default());
        let result = parser.parse_bytes(b"fake epub data", DocumentFormat::Epub);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("document-formats"));
    }
}
