//! Main DocumentParser implementation.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use regex::Regex;

#[cfg(feature = "documents")]
use std::io::Read;

use super::types::*;
use super::xml_helpers::*;

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
            format!("\u{2014} {} \u{2014}", current_page),
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
    /// - **Footnotes**: Text at page bottom with superscript-style markers (1, 2, *, dagger)
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
        // 2. Start with markers like 1, 2, 3, *, dagger, double dagger, or numbers followed by period/space
        // 3. Often separated by a horizontal line or extra space

        let total_lines = lines.len();
        if total_lines < 5 {
            return None;
        }

        // Only look at bottom 40% of the page
        let search_start = (total_lines as f32 * 0.6) as usize;

        // Look for footnote marker patterns
        let footnote_markers =
            Regex::new(r"^[\s]*[\u{00b9}\u{00b2}\u{00b3}\u{2074}\u{2075}\u{2076}\u{2077}\u{2078}\u{2079}\u{2070}*\u{2020}\u{2021}\u{00a7}\[\(]?\d{0,2}[\]\)]?[\.\s]").ok()?;

        // Also check for common footnote separators
        let separator_pattern = Regex::new(r"^[\s]*[_\-\u{2500}\u{2550}]{3,}[\s]*$").ok()?;

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
                let first_char = trimmed.chars().next().unwrap_or(' ');
                if "\u{00b9}\u{00b2}\u{00b3}\u{2074}\u{2075}\u{2076}\u{2077}\u{2078}\u{2079}\u{2070}*\u{2020}\u{2021}\u{00a7}[(".contains(first_char) || first_char.is_ascii_digit()
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
                        // Soft line break -- continuation, don't add newline
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

    /// Capitalize email header name (e.g., "message-id" -> "Message-Id").
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
    /// No OCR -- returns metadata only (dimensions, format, EXIF date/camera if JPEG).
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
    // PPTX parsing (Phase 2 -- requires `documents` feature for ZIP)
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
    // XLSX parsing (Phase 2 -- requires `documents` feature)
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
