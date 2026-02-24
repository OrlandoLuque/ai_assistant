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

mod types;
mod parser;
pub(crate) mod xml_helpers;
mod ocr_engine;
mod image_extraction;
mod ocr_pipeline;
#[cfg(test)]
mod tests;

// Re-export all public types so they remain accessible as document_parsing::TypeName

// From types.rs
pub use types::{
    DocumentFormat, DocumentSection, PageContent, PdfTable,
    DocumentMetadata, ParsedDocument, DocumentParserConfig,
};

// From parser.rs
pub use parser::DocumentParser;

// From xml_helpers.rs
pub use xml_helpers::{
    strip_xml_tags, extract_xml_text, extract_xml_metadata, normalize_text,
};

// From ocr_engine.rs
pub use ocr_engine::{
    OcrConfig, GlyphTemplate, OcrLine, OcrResult, OcrEngine,
};

// From image_extraction.rs
pub use image_extraction::{
    ImageFormat, ExtractedImage, ImageExtractionConfig,
    ImageExtractor, DocumentImageAnalysis,
};

// From ocr_pipeline.rs
pub use ocr_pipeline::{
    OcrBackend, TemplateOcrBackend, TesseractConfig,
    TesseractOcrBackend, OcrPipelineConfig, OcrPipeline,
};
