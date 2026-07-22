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

mod image_extraction;
mod ocr_engine;
mod ocr_pipeline;
mod parser;
#[cfg(test)]
mod tests;
mod types;
pub(crate) mod xml_helpers;

// Re-export all public types so they remain accessible as document_parsing::TypeName

// From types.rs
pub use types::{
    DocumentFormat, DocumentMetadata, DocumentParserConfig, DocumentSection, PageContent,
    ParsedDocument, PdfTable,
};

// From parser.rs
/// Build a minimal valid single-page PDF containing the given text — a CI-safe,
/// offline golden fixture for the PDF path (used by the test harness).
#[cfg(feature = "pdf-extract")]
pub use parser::make_minimal_pdf;
pub use parser::DocumentParser;

// From xml_helpers.rs
pub use xml_helpers::{extract_xml_metadata, extract_xml_text, normalize_text, strip_xml_tags};

// From ocr_engine.rs
pub use ocr_engine::{GlyphTemplate, OcrConfig, OcrEngine, OcrLine, OcrResult};

// From image_extraction.rs
pub use image_extraction::{
    DocumentImageAnalysis, ExtractedImage, ImageExtractionConfig, ImageExtractor, ImageFormat,
};

// From ocr_pipeline.rs
pub use ocr_pipeline::{
    OcrBackend, OcrPipeline, OcrPipelineConfig, TemplateOcrBackend, TesseractConfig,
    TesseractOcrBackend,
};
