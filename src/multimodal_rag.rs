//! Multi-modal RAG pipeline supporting text + image retrieval.
//!
//! Provides a pipeline for ingesting, indexing, and retrieving multi-modal
//! content (text, images, tables, code) with modality-aware scoring and
//! weighted retrieval.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// ModalityType
// ---------------------------------------------------------------------------

/// The modality (content type) of a chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModalityType {
    Text,
    Image,
    Table,
    Code,
    Mixed,
}

impl fmt::Display for ModalityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModalityType::Text => write!(f, "Text"),
            ModalityType::Image => write!(f, "Image"),
            ModalityType::Table => write!(f, "Table"),
            ModalityType::Code => write!(f, "Code"),
            ModalityType::Mixed => write!(f, "Mixed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers (Jaccard similarity, same pattern as reranker.rs)
// ---------------------------------------------------------------------------

/// Convert a string to a set of normalised word tokens.
fn word_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity between two word sets.
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let union_size = a.union(b).count();
    if union_size == 0 {
        return 0.0;
    }
    let intersection_size = a.intersection(b).count();
    intersection_size as f64 / union_size as f64
}

// ---------------------------------------------------------------------------
// MultiModalChunk
// ---------------------------------------------------------------------------

/// A single chunk of multi-modal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalChunk {
    /// The textual content (or code, table markup, etc.).
    pub content: String,
    /// What kind of content this chunk represents.
    pub modality: ModalityType,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// Path or URL to an image (for `Image` chunks).
    pub image_ref: Option<String>,
    /// Caption / alt text describing an image.
    pub caption: Option<String>,
    /// Relevance score in the range 0.0 to 1.0.
    pub score: f64,
    /// Source document identifier.
    pub source: Option<String>,
}

impl MultiModalChunk {
    /// Create a text chunk.
    pub fn text(content: &str) -> Self {
        Self {
            content: content.to_string(),
            modality: ModalityType::Text,
            metadata: HashMap::new(),
            image_ref: None,
            caption: None,
            score: 0.0,
            source: None,
        }
    }

    /// Create an image chunk.
    pub fn image(image_ref: &str, caption: &str) -> Self {
        Self {
            content: String::new(),
            modality: ModalityType::Image,
            metadata: HashMap::new(),
            image_ref: Some(image_ref.to_string()),
            caption: Some(caption.to_string()),
            score: 0.0,
            source: None,
        }
    }

    /// Create a table chunk.
    pub fn table(content: &str) -> Self {
        Self {
            content: content.to_string(),
            modality: ModalityType::Table,
            metadata: HashMap::new(),
            image_ref: None,
            caption: None,
            score: 0.0,
            source: None,
        }
    }

    /// Create a code chunk, optionally tagged with a language.
    pub fn code(content: &str, language: Option<&str>) -> Self {
        let mut metadata = HashMap::new();
        if let Some(lang) = language {
            metadata.insert("language".to_string(), lang.to_string());
        }
        Self {
            content: content.to_string(),
            modality: ModalityType::Code,
            metadata,
            image_ref: None,
            caption: None,
            score: 0.0,
            source: None,
        }
    }

    /// Builder: attach a source identifier.
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Builder: attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// MultiModalDocument
// ---------------------------------------------------------------------------

/// A document composed of multiple multi-modal sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDocument {
    /// Optional document title.
    pub title: Option<String>,
    /// Ordered list of sections (chunks).
    pub sections: Vec<MultiModalChunk>,
    /// Document-level metadata.
    pub metadata: HashMap<String, String>,
}

impl MultiModalDocument {
    /// Create an empty document.
    pub fn new() -> Self {
        Self {
            title: None,
            sections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the document title (builder).
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Append a section.
    pub fn add_section(&mut self, chunk: MultiModalChunk) {
        self.sections.push(chunk);
    }

    /// Create a document from plain text, splitting on blank lines into
    /// `Text` chunks (one per paragraph).
    pub fn from_text(text: &str) -> Self {
        let mut doc = Self::new();
        for para in text.split("\n\n") {
            let trimmed = para.trim();
            if !trimmed.is_empty() {
                doc.sections.push(MultiModalChunk::text(trimmed));
            }
        }
        doc
    }

    /// Create a single-image document.
    pub fn from_image(ref_path: &str, caption: &str) -> Self {
        let mut doc = Self::new();
        doc.sections.push(MultiModalChunk::image(ref_path, caption));
        doc
    }

    /// Parse simple HTML, extracting text content and `<img>` tags.
    ///
    /// This is a lightweight, regex-free parser that handles common cases.
    pub fn from_html(html: &str) -> Self {
        let mut doc = Self::new();
        let mut remaining = html;

        while !remaining.is_empty() {
            // Look for the next <img tag
            if let Some(img_start) = remaining.find("<img") {
                // Text before the <img> tag
                let before = &remaining[..img_start];
                let text = strip_tags(before).trim().to_string();
                if !text.is_empty() {
                    doc.sections.push(MultiModalChunk::text(&text));
                }

                // Parse the <img> tag
                if let Some(img_end) = remaining[img_start..].find('>') {
                    let img_tag = &remaining[img_start..img_start + img_end + 1];
                    let src = extract_attr(img_tag, "src").unwrap_or_default();
                    let alt = extract_attr(img_tag, "alt").unwrap_or_default();
                    if !src.is_empty() {
                        doc.sections.push(MultiModalChunk::image(&src, &alt));
                    }
                    remaining = &remaining[img_start + img_end + 1..];
                } else {
                    // Malformed tag, skip past it
                    remaining = &remaining[img_start + 4..];
                }
            } else {
                // No more <img> tags — rest is text
                let text = strip_tags(remaining).trim().to_string();
                if !text.is_empty() {
                    doc.sections.push(MultiModalChunk::text(&text));
                }
                break;
            }
        }

        doc
    }

    /// Return references to text-only sections.
    pub fn text_sections(&self) -> Vec<&MultiModalChunk> {
        self.sections
            .iter()
            .filter(|c| c.modality == ModalityType::Text)
            .collect()
    }

    /// Return references to image-only sections.
    pub fn image_sections(&self) -> Vec<&MultiModalChunk> {
        self.sections
            .iter()
            .filter(|c| c.modality == ModalityType::Image)
            .collect()
    }

    /// Concatenate all textual content (text chunks + image captions).
    pub fn all_text(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for section in &self.sections {
            if !section.content.is_empty() {
                parts.push(section.content.clone());
            }
            if let Some(ref caption) = section.caption {
                if !caption.is_empty() {
                    parts.push(caption.clone());
                }
            }
        }
        parts.join("\n")
    }
}

impl Default for MultiModalDocument {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HTML helpers (lightweight, no external deps)
// ---------------------------------------------------------------------------

/// Strip HTML tags from a string, returning only text content.
fn strip_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
            result.push(' '); // replace tag with space
        } else if !in_tag {
            result.push(ch);
        }
    }
    // Collapse multiple spaces
    let mut prev_space = false;
    let mut collapsed = String::with_capacity(result.len());
    for ch in result.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    collapsed
}

/// Extract the value of an HTML attribute from a tag string.
fn extract_attr(tag: &str, attr_name: &str) -> Option<String> {
    // Look for attr_name= followed by a quoted value
    let search = format!("{}=", attr_name);
    let pos = tag.find(&search)?;
    let after_eq = &tag[pos + search.len()..];
    let trimmed = after_eq.trim_start();

    if trimmed.starts_with('"') {
        let content = &trimmed[1..];
        let end = content.find('"')?;
        Some(content[..end].to_string())
    } else if trimmed.starts_with('\'') {
        let content = &trimmed[1..];
        let end = content.find('\'')?;
        Some(content[..end].to_string())
    } else {
        // Unquoted value — take until whitespace or >
        let end = trimmed.find(|c: char| c.is_whitespace() || c == '>').unwrap_or(trimmed.len());
        Some(trimmed[..end].to_string())
    }
}

// ---------------------------------------------------------------------------
// MultiModalConfig
// ---------------------------------------------------------------------------

/// Configuration for multi-modal retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MultiModalConfig {
    /// Weight for text chunks (0.0 to 1.0). Default: 0.7.
    pub text_weight: f64,
    /// Weight for image chunks (0.0 to 1.0). Default: 0.3.
    pub image_weight: f64,
    /// Maximum number of results to return. Default: 5.
    pub top_k: usize,
    /// Minimum relevance score threshold. Default: 0.1.
    pub min_score: f64,
    /// Which modalities to include in search results.
    pub include_modalities: Vec<ModalityType>,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            text_weight: 0.7,
            image_weight: 0.3,
            top_k: 5,
            min_score: 0.1,
            include_modalities: vec![
                ModalityType::Text,
                ModalityType::Image,
                ModalityType::Table,
                ModalityType::Code,
                ModalityType::Mixed,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// MultiModalRetriever
// ---------------------------------------------------------------------------

/// Retrieves and scores multi-modal chunks against a text query.
pub struct MultiModalRetriever {
    config: MultiModalConfig,
}

impl MultiModalRetriever {
    /// Create a retriever with the given configuration.
    pub fn new(config: MultiModalConfig) -> Self {
        Self { config }
    }

    /// Create a retriever with default settings.
    pub fn with_defaults() -> Self {
        Self::new(MultiModalConfig::default())
    }

    /// Retrieve the most relevant chunks for `query` from the given pool.
    pub fn retrieve(&self, query: &str, chunks: &[MultiModalChunk]) -> Vec<MultiModalChunk> {
        let mut scored: Vec<MultiModalChunk> = chunks
            .iter()
            .filter(|c| self.config.include_modalities.contains(&c.modality))
            .map(|c| {
                let raw_score = match c.modality {
                    ModalityType::Image => self.score_image_chunk(query, c),
                    _ => self.score_text_chunk(query, c),
                };
                let mut result = c.clone();
                result.score = raw_score;
                result
            })
            .filter(|c| c.score >= self.config.min_score)
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(self.config.top_k);
        scored
    }

    /// Score a text-like chunk (Text, Table, Code, Mixed) via Jaccard
    /// overlap, then apply the text weight.
    fn score_text_chunk(&self, query: &str, chunk: &MultiModalChunk) -> f64 {
        let query_words = word_set(query);
        let chunk_words = word_set(&chunk.content);
        let sim = jaccard_similarity(&query_words, &chunk_words);
        sim * self.config.text_weight
    }

    /// Score an image chunk by comparing the query against the caption text,
    /// then apply the image weight.
    fn score_image_chunk(&self, query: &str, chunk: &MultiModalChunk) -> f64 {
        let caption_text = chunk.caption.as_deref().unwrap_or("");
        let query_words = word_set(query);
        let caption_words = word_set(caption_text);
        let sim = jaccard_similarity(&query_words, &caption_words);
        sim * self.config.image_weight
    }
}

// ---------------------------------------------------------------------------
// MultiModalResult
// ---------------------------------------------------------------------------

/// The result of a multi-modal retrieval query.
#[derive(Debug, Clone)]
pub struct MultiModalResult {
    /// All results, sorted by score (descending).
    pub chunks: Vec<MultiModalChunk>,
    /// Text-only results.
    pub text_chunks: Vec<MultiModalChunk>,
    /// Image-only results.
    pub image_chunks: Vec<MultiModalChunk>,
    /// The original query string.
    pub query: String,
    /// Total number of candidate chunks before filtering.
    pub total_candidates: usize,
}

impl MultiModalResult {
    /// Return references to the top `n` text chunks.
    pub fn top_text(&self, n: usize) -> Vec<&MultiModalChunk> {
        self.text_chunks.iter().take(n).collect()
    }

    /// Return references to the top `n` image chunks.
    pub fn top_images(&self, n: usize) -> Vec<&MultiModalChunk> {
        self.image_chunks.iter().take(n).collect()
    }

    /// Whether the result set contains any images.
    pub fn has_images(&self) -> bool {
        !self.image_chunks.is_empty()
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        let total = self.chunks.len();
        let text_count = self.text_chunks.len();
        let image_count = self.image_chunks.len();
        format!(
            "Found {} results: {} text, {} images",
            total, text_count, image_count
        )
    }
}

// ---------------------------------------------------------------------------
// MultiModalPipeline
// ---------------------------------------------------------------------------

/// End-to-end pipeline for ingesting and querying multi-modal content.
pub struct MultiModalPipeline {
    chunks: Vec<MultiModalChunk>,
    _config: MultiModalConfig,
    retriever: MultiModalRetriever,
}

impl MultiModalPipeline {
    /// Create a pipeline with the given configuration.
    pub fn new(config: MultiModalConfig) -> Self {
        let retriever = MultiModalRetriever::new(config.clone());
        Self {
            chunks: Vec::new(),
            _config: config,
            retriever,
        }
    }

    /// Create a pipeline with default settings.
    pub fn with_defaults() -> Self {
        Self::new(MultiModalConfig::default())
    }

    /// Add a text chunk to the pipeline.
    pub fn add_text(&mut self, content: &str, source: Option<&str>) {
        let mut chunk = MultiModalChunk::text(content);
        if let Some(src) = source {
            chunk.source = Some(src.to_string());
        }
        self.chunks.push(chunk);
    }

    /// Add an image chunk to the pipeline.
    pub fn add_image(&mut self, image_ref: &str, caption: &str, source: Option<&str>) {
        let mut chunk = MultiModalChunk::image(image_ref, caption);
        if let Some(src) = source {
            chunk.source = Some(src.to_string());
        }
        self.chunks.push(chunk);
    }

    /// Add all sections from a `MultiModalDocument`.
    pub fn add_document(&mut self, doc: MultiModalDocument) {
        for section in doc.sections {
            self.chunks.push(section);
        }
    }

    /// Add a batch of pre-built chunks.
    pub fn add_chunks(&mut self, chunks: Vec<MultiModalChunk>) {
        self.chunks.extend(chunks);
    }

    /// Query the pipeline and return scored results.
    pub fn query(&self, query: &str) -> MultiModalResult {
        let total_candidates = self.chunks.len();
        let results = self.retriever.retrieve(query, &self.chunks);

        let text_chunks: Vec<MultiModalChunk> = results
            .iter()
            .filter(|c| c.modality == ModalityType::Text)
            .cloned()
            .collect();

        let image_chunks: Vec<MultiModalChunk> = results
            .iter()
            .filter(|c| c.modality == ModalityType::Image)
            .cloned()
            .collect();

        MultiModalResult {
            chunks: results,
            text_chunks,
            image_chunks,
            query: query.to_string(),
            total_candidates,
        }
    }

    /// Number of chunks currently stored.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Remove all chunks.
    pub fn clear(&mut self) {
        self.chunks.clear();
    }
}

// ---------------------------------------------------------------------------
// ImageCaptionExtractor
// ---------------------------------------------------------------------------

/// Utilities for extracting human-readable captions from various sources.
pub struct ImageCaptionExtractor;

impl ImageCaptionExtractor {
    /// Derive a caption from a filename by replacing separators with spaces
    /// and stripping the extension.
    ///
    /// Example: `"sunset_over_ocean.jpg"` -> `Some("sunset over ocean")`
    pub fn from_filename(filename: &str) -> Option<String> {
        // Strip directory components
        let name = filename
            .rsplit(|c| c == '/' || c == '\\')
            .next()
            .unwrap_or(filename);

        // Strip extension (rfind to handle names like "archive.tar.gz")
        let stem = match name.rfind('.') {
            Some(pos) if pos > 0 => &name[..pos],
            _ => name,
        };

        // Reject empty stems and hidden/dot-only files (e.g. ".hidden")
        if stem.is_empty() || stem.starts_with('.') {
            return None;
        }

        // Replace separators with spaces
        let caption: String = stem
            .chars()
            .map(|c| if c == '_' || c == '-' { ' ' } else { c })
            .collect();

        let trimmed = caption.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    /// Clean up alt text, returning `None` if it is empty or a common
    /// placeholder value.
    pub fn from_alt_text(alt: &str) -> Option<String> {
        let trimmed = alt.trim();
        if trimmed.is_empty() {
            return None;
        }

        // Filter out common placeholder alt texts
        let lower = trimmed.to_lowercase();
        let placeholders = ["image", "photo", "picture", "img", "placeholder", "untitled", ""];
        if placeholders.contains(&lower.as_str()) {
            return None;
        }

        Some(trimmed.to_string())
    }

    /// Look for description-like keys in metadata and return the first match.
    pub fn from_metadata(metadata: &HashMap<String, String>) -> Option<String> {
        let keys = ["description", "caption", "alt", "title"];
        for key in &keys {
            if let Some(value) = metadata.get(*key) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. ModalityType Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", ModalityType::Text), "Text");
        assert_eq!(format!("{}", ModalityType::Image), "Image");
        assert_eq!(format!("{}", ModalityType::Table), "Table");
        assert_eq!(format!("{}", ModalityType::Code), "Code");
        assert_eq!(format!("{}", ModalityType::Mixed), "Mixed");
    }

    // -----------------------------------------------------------------------
    // 2. MultiModalChunk::text
    // -----------------------------------------------------------------------

    #[test]
    fn test_text_chunk_creation() {
        let chunk = MultiModalChunk::text("Hello world");
        assert_eq!(chunk.content, "Hello world");
        assert_eq!(chunk.modality, ModalityType::Text);
        assert!(chunk.image_ref.is_none());
        assert!(chunk.caption.is_none());
        assert_eq!(chunk.score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 3. MultiModalChunk::image
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_chunk_creation() {
        let chunk = MultiModalChunk::image("/imgs/cat.jpg", "A fluffy cat");
        assert_eq!(chunk.modality, ModalityType::Image);
        assert_eq!(chunk.image_ref.as_deref(), Some("/imgs/cat.jpg"));
        assert_eq!(chunk.caption.as_deref(), Some("A fluffy cat"));
        assert!(chunk.content.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. with_metadata and with_source
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_with_metadata() {
        let chunk = MultiModalChunk::text("data")
            .with_source("doc1.md")
            .with_metadata("author", "Alice");

        assert_eq!(chunk.source.as_deref(), Some("doc1.md"));
        assert_eq!(chunk.metadata.get("author").map(|s| s.as_str()), Some("Alice"));
    }

    // -----------------------------------------------------------------------
    // 5. MultiModalDocument::from_text
    // -----------------------------------------------------------------------

    #[test]
    fn test_document_from_text() {
        let doc = MultiModalDocument::from_text("First paragraph.\n\nSecond paragraph.\n\nThird.");
        assert_eq!(doc.sections.len(), 3);
        assert_eq!(doc.sections[0].modality, ModalityType::Text);
        assert_eq!(doc.sections[0].content, "First paragraph.");
        assert_eq!(doc.sections[2].content, "Third.");
    }

    // -----------------------------------------------------------------------
    // 6. MultiModalDocument::from_html
    // -----------------------------------------------------------------------

    #[test]
    fn test_document_from_html() {
        let html = r#"<p>Hello world</p><img src="photo.jpg" alt="A photo"><p>More text</p>"#;
        let doc = MultiModalDocument::from_html(html);

        // Should have text, image, text
        assert!(doc.sections.len() >= 3, "Expected >= 3 sections, got {}", doc.sections.len());

        // First section should be text
        assert_eq!(doc.sections[0].modality, ModalityType::Text);
        assert!(doc.sections[0].content.contains("Hello"));

        // Second section should be an image
        assert_eq!(doc.sections[1].modality, ModalityType::Image);
        assert_eq!(doc.sections[1].image_ref.as_deref(), Some("photo.jpg"));
        assert_eq!(doc.sections[1].caption.as_deref(), Some("A photo"));

        // Third section should be text
        assert_eq!(doc.sections[2].modality, ModalityType::Text);
        assert!(doc.sections[2].content.contains("More text"));
    }

    // -----------------------------------------------------------------------
    // 7. text_sections filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_document_text_sections() {
        let mut doc = MultiModalDocument::new();
        doc.add_section(MultiModalChunk::text("Hello"));
        doc.add_section(MultiModalChunk::image("a.png", "caption"));
        doc.add_section(MultiModalChunk::text("World"));

        let text = doc.text_sections();
        assert_eq!(text.len(), 2);
        assert_eq!(text[0].content, "Hello");
        assert_eq!(text[1].content, "World");
    }

    // -----------------------------------------------------------------------
    // 8. image_sections filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_document_image_sections() {
        let mut doc = MultiModalDocument::new();
        doc.add_section(MultiModalChunk::text("txt"));
        doc.add_section(MultiModalChunk::image("a.png", "cap1"));
        doc.add_section(MultiModalChunk::image("b.jpg", "cap2"));

        let imgs = doc.image_sections();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].image_ref.as_deref(), Some("a.png"));
    }

    // -----------------------------------------------------------------------
    // 9. Retriever — text only
    // -----------------------------------------------------------------------

    #[test]
    fn test_retriever_text_only() {
        let retriever = MultiModalRetriever::with_defaults();

        let chunks = vec![
            MultiModalChunk::text("Rust programming language"),
            MultiModalChunk::text("Python data science"),
            MultiModalChunk::text("cooking recipes for dinner"),
        ];

        let results = retriever.retrieve("Rust programming", &chunks);
        assert!(!results.is_empty());
        // The Rust chunk should have the highest score
        assert!(
            results[0].content.contains("Rust"),
            "Expected Rust chunk first, got: {}",
            results[0].content
        );
    }

    // -----------------------------------------------------------------------
    // 10. Retriever — mixed modalities
    // -----------------------------------------------------------------------

    #[test]
    fn test_retriever_mixed() {
        let retriever = MultiModalRetriever::with_defaults();

        let chunks = vec![
            MultiModalChunk::text("beautiful sunset over the ocean"),
            MultiModalChunk::image("sunset.jpg", "sunset over the ocean waves"),
            MultiModalChunk::text("rainy day in the city"),
        ];

        let results = retriever.retrieve("sunset ocean", &chunks);
        assert!(!results.is_empty());
        // Both the text and image about sunset should be returned
        let has_text = results.iter().any(|c| c.modality == ModalityType::Text && c.content.contains("sunset"));
        let has_img = results.iter().any(|c| c.modality == ModalityType::Image);
        assert!(has_text, "Should include text chunk about sunset");
        assert!(has_img, "Should include image chunk about sunset");
    }

    // -----------------------------------------------------------------------
    // 11. Image scored by caption
    // -----------------------------------------------------------------------

    #[test]
    fn test_retriever_image_caption_matching() {
        let config = MultiModalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let retriever = MultiModalRetriever::new(config);

        let chunks = vec![
            MultiModalChunk::image("a.jpg", "cat sitting on a mat"),
            MultiModalChunk::image("b.jpg", "car racing on track"),
        ];

        let results = retriever.retrieve("cat mat", &chunks);
        assert!(!results.is_empty());
        // The cat image should score higher
        assert_eq!(
            results[0].image_ref.as_deref(),
            Some("a.jpg"),
            "Cat image should rank first"
        );
        assert!(results[0].score > results.last().unwrap().score);
    }

    // -----------------------------------------------------------------------
    // 12. Modality weighting
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_weighting() {
        // With very high text weight, text should dominate even if Jaccard is the same
        let config = MultiModalConfig {
            text_weight: 1.0,
            image_weight: 0.1,
            min_score: 0.0,
            top_k: 10,
            ..Default::default()
        };
        let retriever = MultiModalRetriever::new(config);

        // Both have the same words, but text weight is 10x image weight
        let chunks = vec![
            MultiModalChunk::text("sunset ocean waves"),
            MultiModalChunk::image("x.jpg", "sunset ocean waves"),
        ];

        let results = retriever.retrieve("sunset ocean waves", &chunks);
        assert!(results.len() == 2);
        // Text should score higher due to weight
        assert_eq!(results[0].modality, ModalityType::Text);
        assert!(results[0].score > results[1].score);
    }

    // -----------------------------------------------------------------------
    // 13. Pipeline end-to-end
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_add_and_query() {
        let mut pipeline = MultiModalPipeline::with_defaults();

        pipeline.add_text("Rust is a systems programming language", Some("doc1"));
        pipeline.add_text("Python is great for data science", Some("doc2"));
        pipeline.add_image("rust_logo.png", "the Rust programming language logo", Some("doc1"));

        assert_eq!(pipeline.chunk_count(), 3);

        let result = pipeline.query("Rust programming");
        assert!(!result.chunks.is_empty());
        assert_eq!(result.query, "Rust programming");
        assert_eq!(result.total_candidates, 3);

        // The Rust text should rank high
        assert!(
            result.chunks[0].content.contains("Rust") || result.chunks[0].caption.as_deref().map_or(false, |c| c.contains("Rust")),
            "Top result should be about Rust"
        );

        pipeline.clear();
        assert_eq!(pipeline.chunk_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 14. MultiModalResult::summary()
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_summary() {
        let result = MultiModalResult {
            chunks: vec![
                MultiModalChunk::text("a"),
                MultiModalChunk::text("b"),
                MultiModalChunk::image("x.jpg", "cap"),
            ],
            text_chunks: vec![MultiModalChunk::text("a"), MultiModalChunk::text("b")],
            image_chunks: vec![MultiModalChunk::image("x.jpg", "cap")],
            query: "test".to_string(),
            total_candidates: 10,
        };

        let summary = result.summary();
        assert_eq!(summary, "Found 3 results: 2 text, 1 images");
        assert!(result.has_images());
        assert_eq!(result.top_text(1).len(), 1);
        assert_eq!(result.top_images(5).len(), 1);
    }

    // -----------------------------------------------------------------------
    // 15. ImageCaptionExtractor::from_filename
    // -----------------------------------------------------------------------

    #[test]
    fn test_caption_from_filename() {
        assert_eq!(
            ImageCaptionExtractor::from_filename("sunset_over_ocean.jpg"),
            Some("sunset over ocean".to_string())
        );
        assert_eq!(
            ImageCaptionExtractor::from_filename("my-photo.png"),
            Some("my photo".to_string())
        );
        assert_eq!(
            ImageCaptionExtractor::from_filename("/path/to/hello_world.webp"),
            Some("hello world".to_string())
        );
        assert_eq!(ImageCaptionExtractor::from_filename(".hidden"), None);

        // from_alt_text
        assert_eq!(
            ImageCaptionExtractor::from_alt_text("A beautiful sunset"),
            Some("A beautiful sunset".to_string())
        );
        assert_eq!(ImageCaptionExtractor::from_alt_text("image"), None);
        assert_eq!(ImageCaptionExtractor::from_alt_text(""), None);

        // from_metadata
        let mut meta = HashMap::new();
        meta.insert("description".to_string(), "A test image".to_string());
        assert_eq!(
            ImageCaptionExtractor::from_metadata(&meta),
            Some("A test image".to_string())
        );

        let empty_meta: HashMap<String, String> = HashMap::new();
        assert_eq!(ImageCaptionExtractor::from_metadata(&empty_meta), None);
    }
}
