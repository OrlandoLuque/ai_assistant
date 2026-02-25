//! Container-based document creation and conversion pipeline.
//!
//! Uses Docker containers with Pandoc, LibreOffice, and wkhtmltopdf to create
//! PDFs, DOCX, PPTX, and other document formats from Markdown/HTML content.
//!
//! ## Key types
//!
//! - [`DocumentPipeline`] — Orchestrates container-based document creation
//! - [`CreateRequest`] — Describes what to create (format, content, template)
//! - [`ConversionResult`] — Output bytes, format, and metadata
//! - [`PipelineConfig`] — Timeout, temp directory, container image overrides
//!
//! ## Feature flags
//!
//! Requires the `containers` feature flag (not in `full` by default).
//! Needs Docker installed and running on the host.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::container_executor::{ContainerError, ContainerExecutor, CreateOptions};
use crate::shared_folder::SharedFolder;

// =============================================================================
// Output format
// =============================================================================

/// Output document format.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Pdf,
    Docx,
    Pptx,
    Xlsx,
    Odt,
    Html,
    Latex,
    Epub,
    Png, // for diagram/chart rendering
    Svg,
}

impl OutputFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            OutputFormat::Pdf => "pdf",
            OutputFormat::Docx => "docx",
            OutputFormat::Pptx => "pptx",
            OutputFormat::Xlsx => "xlsx",
            OutputFormat::Odt => "odt",
            OutputFormat::Html => "html",
            OutputFormat::Latex => "tex",
            OutputFormat::Epub => "epub",
            OutputFormat::Png => "png",
            OutputFormat::Svg => "svg",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pdf" => Some(OutputFormat::Pdf),
            "docx" | "word" => Some(OutputFormat::Docx),
            "pptx" | "powerpoint" => Some(OutputFormat::Pptx),
            "xlsx" | "excel" => Some(OutputFormat::Xlsx),
            "odt" => Some(OutputFormat::Odt),
            "html" | "htm" => Some(OutputFormat::Html),
            "latex" | "tex" => Some(OutputFormat::Latex),
            "epub" => Some(OutputFormat::Epub),
            "png" => Some(OutputFormat::Png),
            "svg" => Some(OutputFormat::Svg),
            _ => None,
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            OutputFormat::Pdf => "application/pdf",
            OutputFormat::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            OutputFormat::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            OutputFormat::Xlsx => {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            OutputFormat::Odt => "application/vnd.oasis.opendocument.text",
            OutputFormat::Html => "text/html",
            OutputFormat::Latex => "application/x-latex",
            OutputFormat::Epub => "application/epub+zip",
            OutputFormat::Png => "image/png",
            OutputFormat::Svg => "image/svg+xml",
        }
    }
}

// =============================================================================
// Source format
// =============================================================================

/// Source content format hint.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceFormat {
    Markdown,
    Html,
    Latex,
    Csv,
    Json,
    PlainText,
    Rst, // reStructuredText
}

impl SourceFormat {
    /// Get the file extension for this source format.
    pub fn extension(&self) -> &str {
        match self {
            SourceFormat::Markdown => "md",
            SourceFormat::Html => "html",
            SourceFormat::Latex => "tex",
            SourceFormat::Csv => "csv",
            SourceFormat::Json => "json",
            SourceFormat::PlainText => "txt",
            SourceFormat::Rst => "rst",
        }
    }

    /// Get the pandoc input format string.
    pub fn pandoc_format(&self) -> &str {
        match self {
            SourceFormat::Markdown => "markdown",
            SourceFormat::Html => "html",
            SourceFormat::Latex => "latex",
            SourceFormat::Csv => "csv",
            SourceFormat::Json => "json",
            SourceFormat::PlainText => "plain",
            SourceFormat::Rst => "rst",
        }
    }
}

// =============================================================================
// Document request
// =============================================================================

/// Request to create a document.
#[derive(Debug, Clone)]
pub struct DocumentRequest {
    /// Source content (text).
    pub content: String,
    /// Source format.
    pub source_format: SourceFormat,
    /// Desired output format.
    pub output_format: OutputFormat,
    /// Output filename (within shared folder, without extension — extension added automatically).
    pub output_name: String,
    /// Optional CSS stylesheet for HTML/PDF.
    pub stylesheet: Option<String>,
    /// Extra pandoc flags.
    pub extra_args: Vec<String>,
    /// Document metadata (title, author, date, etc.).
    pub metadata: HashMap<String, String>,
}

impl DocumentRequest {
    /// Create a simple request with Markdown source and the given output format.
    pub fn new(content: impl Into<String>, format: OutputFormat) -> Self {
        Self {
            content: content.into(),
            source_format: SourceFormat::Markdown,
            output_format: format,
            output_name: "document".into(),
            stylesheet: None,
            extra_args: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the output filename (without extension).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    /// Set the source format.
    pub fn with_source_format(mut self, format: SourceFormat) -> Self {
        self.source_format = format;
        self
    }

    /// Add a metadata key-value pair (title, author, date, etc.).
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set a CSS stylesheet for HTML/PDF output.
    pub fn with_stylesheet(mut self, css: impl Into<String>) -> Self {
        self.stylesheet = Some(css.into());
        self
    }
}

// =============================================================================
// Document result
// =============================================================================

/// Result of document creation.
#[derive(Debug, Clone)]
pub struct DocumentResult {
    /// Path to the generated file (in shared folder).
    pub output_path: PathBuf,
    /// Filename only.
    pub filename: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Container execution log.
    pub log: String,
    /// Duration of generation.
    pub duration: Duration,
    /// Output format.
    pub format: OutputFormat,
}

// =============================================================================
// Document errors
// =============================================================================

/// Errors from document pipeline operations.
#[derive(Debug)]
pub enum DocumentError {
    /// Error originating from the container executor.
    ContainerError(ContainerError),
    /// I/O error (reading/writing files in shared folder).
    IoError(std::io::Error),
    /// Conversion failed (pandoc exited with error).
    ConversionFailed(String),
    /// Unsupported source/output format combination.
    UnsupportedConversion(String),
    /// Expected output file was not found after conversion.
    OutputNotFound(String),
}

impl std::fmt::Display for DocumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocumentError::ContainerError(e) => write!(f, "Container error: {}", e),
            DocumentError::IoError(e) => write!(f, "I/O error: {}", e),
            DocumentError::ConversionFailed(msg) => write!(f, "Conversion failed: {}", msg),
            DocumentError::UnsupportedConversion(msg) => {
                write!(f, "Unsupported conversion: {}", msg)
            }
            DocumentError::OutputNotFound(name) => {
                write!(f, "Output file not found: {}", name)
            }
        }
    }
}

impl std::error::Error for DocumentError {}

impl From<ContainerError> for DocumentError {
    fn from(e: ContainerError) -> Self {
        DocumentError::ContainerError(e)
    }
}

impl From<std::io::Error> for DocumentError {
    fn from(e: std::io::Error) -> Self {
        DocumentError::IoError(e)
    }
}

// =============================================================================
// Pipeline configuration
// =============================================================================

/// Configuration for the document pipeline.
#[derive(Debug, Clone)]
pub struct DocumentPipelineConfig {
    /// Docker image with document tools (default: "pandoc/extra:latest" — has pandoc + LaTeX).
    pub image: String,
    /// Timeout for document generation.
    pub timeout: Duration,
    /// Reuse container across operations (keep-alive between calls).
    pub reuse_container: bool,
}

impl Default for DocumentPipelineConfig {
    fn default() -> Self {
        Self {
            image: "pandoc/extra:latest".into(),
            timeout: Duration::from_secs(120),
            reuse_container: true,
        }
    }
}

// =============================================================================
// Document pipeline
// =============================================================================

/// Document creation pipeline using Docker containers.
///
/// Creates documents (PDF, DOCX, etc.) from source content (Markdown, HTML, etc.)
/// using pandoc and other tools running inside Docker containers.
///
/// Output files are placed in a shared folder that can be synced to cloud storage.
///
/// # Example
///
/// ```rust,no_run
/// use ai_assistant::document_pipeline::*;
/// use ai_assistant::shared_folder::SharedFolder;
/// use ai_assistant::container_executor::ContainerExecutor;
/// use std::sync::{Arc, RwLock};
///
/// let executor = Arc::new(RwLock::new(ContainerExecutor::default()));
/// let folder = SharedFolder::temp().unwrap();
/// let config = DocumentPipelineConfig::default();
/// let mut pipeline = DocumentPipeline::new(config, executor, folder);
///
/// let request = DocumentRequest::new("# Hello\nWorld", OutputFormat::Pdf)
///     .with_name("report")
///     .with_metadata("title", "My Report")
///     .with_metadata("author", "AI Assistant");
///
/// let result = pipeline.create(&request).unwrap();
/// println!("Created {} ({} bytes)", result.filename, result.size_bytes);
/// ```
pub struct DocumentPipeline {
    config: DocumentPipelineConfig,
    executor: Arc<RwLock<ContainerExecutor>>,
    shared_folder: SharedFolder,
    container_id: Option<String>,
}

impl DocumentPipeline {
    /// Create a new document pipeline.
    pub fn new(
        config: DocumentPipelineConfig,
        executor: Arc<RwLock<ContainerExecutor>>,
        shared_folder: SharedFolder,
    ) -> Self {
        Self {
            config,
            executor,
            shared_folder,
            container_id: None,
        }
    }

    /// Create a document from a request.
    ///
    /// Writes the source content to the shared folder, runs pandoc inside a Docker
    /// container, and returns a `DocumentResult` with the generated file info.
    pub fn create(&mut self, request: &DocumentRequest) -> Result<DocumentResult, DocumentError> {
        // 1. Write source content to shared folder as input file
        let input_filename = format!(
            "input_{}.{}",
            request.output_name,
            request.source_format.extension()
        );
        self.shared_folder
            .put_file(&input_filename, request.content.as_bytes())
            .map_err(|e| {
                DocumentError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

        // 2. Write optional stylesheet
        if let Some(ref css) = request.stylesheet {
            self.shared_folder
                .put_file("style.css", css.as_bytes())
                .map_err(|e| {
                    DocumentError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
        }

        // 3. Ensure container is running
        let container_id = self.ensure_container()?;

        // 4. Build pandoc command
        let output_filename = format!(
            "{}.{}",
            request.output_name,
            request.output_format.extension()
        );
        let cmd = Self::build_pandoc_command(request, &input_filename, &output_filename);
        let cmd_strs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();

        // 5. Execute
        let start = std::time::Instant::now();
        let result = {
            let exec = self.executor.read().map_err(|_| {
                DocumentError::ConversionFailed("executor lock poisoned".into())
            })?;
            exec.exec(&container_id, &cmd_strs, self.config.timeout)
                .map_err(DocumentError::ContainerError)?
        };

        if result.exit_code != 0 {
            return Err(DocumentError::ConversionFailed(format!(
                "pandoc exited with code {}: {}",
                result.exit_code, result.stderr
            )));
        }

        // 6. Verify output exists
        let output_path_relative = format!("output/{}", output_filename);
        let actual_relative = if self.shared_folder.file_exists(&output_path_relative) {
            output_path_relative
        } else if self.shared_folder.file_exists(&output_filename) {
            output_filename.clone()
        } else {
            return Err(DocumentError::OutputNotFound(output_filename));
        };

        let file_data = self.shared_folder.get_file(&actual_relative).map_err(|e| {
            DocumentError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        Ok(DocumentResult {
            output_path: self.shared_folder.host_path().join(&actual_relative),
            filename: output_filename,
            size_bytes: file_data.len() as u64,
            log: format!("{}{}", result.stdout, result.stderr),
            duration: start.elapsed(),
            format: request.output_format.clone(),
        })
    }

    /// Create multiple documents in batch.
    ///
    /// Each request is processed sequentially; partial failures are captured per-item.
    pub fn create_batch(
        &mut self,
        requests: &[DocumentRequest],
    ) -> Vec<Result<DocumentResult, DocumentError>> {
        requests.iter().map(|r| self.create(r)).collect()
    }

    /// Convert an existing file in the shared folder to another format.
    ///
    /// Detects the source format from the file extension, reads the file content,
    /// and converts it to the requested output format.
    pub fn convert(
        &mut self,
        input_filename: &str,
        output_format: OutputFormat,
        output_name: &str,
    ) -> Result<DocumentResult, DocumentError> {
        // Detect source format from extension
        let source_format = Self::detect_source_format(input_filename).ok_or_else(|| {
            DocumentError::UnsupportedConversion(format!(
                "Cannot detect format for: {}",
                input_filename
            ))
        })?;

        let content = self.shared_folder.get_file(input_filename).map_err(|e| {
            DocumentError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        let request = DocumentRequest {
            content: String::from_utf8_lossy(&content).into_owned(),
            source_format,
            output_format,
            output_name: output_name.into(),
            stylesheet: None,
            extra_args: Vec::new(),
            metadata: HashMap::new(),
        };

        self.create(&request)
    }

    /// Get the shared folder.
    pub fn shared_folder(&self) -> &SharedFolder {
        &self.shared_folder
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &DocumentPipelineConfig {
        &self.config
    }

    /// Supported conversions (source -> output format pairs).
    ///
    /// Returns all format pairs that pandoc supports via the document pipeline.
    pub fn supported_conversions() -> Vec<(SourceFormat, OutputFormat)> {
        vec![
            (SourceFormat::Markdown, OutputFormat::Pdf),
            (SourceFormat::Markdown, OutputFormat::Docx),
            (SourceFormat::Markdown, OutputFormat::Html),
            (SourceFormat::Markdown, OutputFormat::Latex),
            (SourceFormat::Markdown, OutputFormat::Epub),
            (SourceFormat::Markdown, OutputFormat::Odt),
            (SourceFormat::Markdown, OutputFormat::Pptx),
            (SourceFormat::Html, OutputFormat::Pdf),
            (SourceFormat::Html, OutputFormat::Docx),
            (SourceFormat::Html, OutputFormat::Latex),
            (SourceFormat::Html, OutputFormat::Epub),
            (SourceFormat::Latex, OutputFormat::Pdf),
            (SourceFormat::Latex, OutputFormat::Docx),
            (SourceFormat::Latex, OutputFormat::Html),
            (SourceFormat::Rst, OutputFormat::Pdf),
            (SourceFormat::Rst, OutputFormat::Docx),
            (SourceFormat::Rst, OutputFormat::Html),
            (SourceFormat::Csv, OutputFormat::Html),
        ]
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Ensure the pipeline container is running, creating it if needed.
    fn ensure_container(&mut self) -> Result<String, DocumentError> {
        if let Some(ref id) = self.container_id {
            return Ok(id.clone());
        }

        let name = format!(
            "ai_doc_pipeline_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let mut opts = CreateOptions::default();
        opts.bind_mounts.push((
            self.shared_folder
                .host_path()
                .to_string_lossy()
                .into_owned(),
            "/workspace".into(),
        ));
        opts.working_dir = Some("/workspace".into());
        opts.cmd = Some(vec!["sleep".into(), "7200".into()]); // 2 hour keep-alive

        let mut exec = self.executor.write().map_err(|_| {
            DocumentError::ConversionFailed("executor lock poisoned".into())
        })?;
        let id = exec
            .create(&self.config.image, &name, opts)
            .map_err(DocumentError::ContainerError)?;
        exec.start(&id).map_err(DocumentError::ContainerError)?;

        self.container_id = Some(id.clone());
        Ok(id)
    }

    /// Build the pandoc command line for a document request.
    fn build_pandoc_command(
        request: &DocumentRequest,
        input_filename: &str,
        output_filename: &str,
    ) -> Vec<String> {
        let mut cmd = vec![
            "pandoc".to_string(),
            format!("/workspace/{}", input_filename),
            "-o".to_string(),
            format!("/workspace/{}", output_filename),
            "-f".to_string(),
            request.source_format.pandoc_format().to_string(),
        ];

        // Add metadata
        for (key, value) in &request.metadata {
            cmd.push("-M".to_string());
            cmd.push(format!("{}={}", key, value));
        }

        // Add stylesheet for PDF/HTML
        if request.stylesheet.is_some() {
            match request.output_format {
                OutputFormat::Html | OutputFormat::Pdf => {
                    cmd.push("--css".to_string());
                    cmd.push("/workspace/style.css".to_string());
                }
                _ => {}
            }
        }

        // PDF-specific: use xelatex for Unicode support
        if request.output_format == OutputFormat::Pdf {
            cmd.push("--pdf-engine=xelatex".to_string());
        }

        // Add extra args
        cmd.extend(request.extra_args.iter().cloned());

        cmd
    }

    /// Detect the source format from a filename extension.
    fn detect_source_format(filename: &str) -> Option<SourceFormat> {
        let ext = filename.rsplit('.').next()?.to_lowercase();
        match ext.as_str() {
            "md" | "markdown" => Some(SourceFormat::Markdown),
            "html" | "htm" => Some(SourceFormat::Html),
            "tex" | "latex" => Some(SourceFormat::Latex),
            "csv" => Some(SourceFormat::Csv),
            "json" => Some(SourceFormat::Json),
            "txt" => Some(SourceFormat::PlainText),
            "rst" => Some(SourceFormat::Rst),
            _ => None,
        }
    }

    /// Cleanup the pipeline container (stop and remove).
    pub fn cleanup(&mut self) {
        if let Some(ref id) = self.container_id.take() {
            if let Ok(mut exec) = self.executor.write() {
                let _ = exec.stop(id, 5);
                let _ = exec.remove(id, true);
            }
        }
    }
}

impl Drop for DocumentPipeline {
    fn drop(&mut self) {
        if !self.config.reuse_container {
            self.cleanup();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OutputFormat tests
    // =========================================================================

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Pdf.extension(), "pdf");
        assert_eq!(OutputFormat::Docx.extension(), "docx");
        assert_eq!(OutputFormat::Pptx.extension(), "pptx");
        assert_eq!(OutputFormat::Xlsx.extension(), "xlsx");
        assert_eq!(OutputFormat::Odt.extension(), "odt");
        assert_eq!(OutputFormat::Html.extension(), "html");
        assert_eq!(OutputFormat::Latex.extension(), "tex");
        assert_eq!(OutputFormat::Epub.extension(), "epub");
        assert_eq!(OutputFormat::Png.extension(), "png");
        assert_eq!(OutputFormat::Svg.extension(), "svg");
    }

    #[test]
    fn test_output_format_from_str() {
        // Primary names
        assert_eq!(OutputFormat::from_str("pdf"), Some(OutputFormat::Pdf));
        assert_eq!(OutputFormat::from_str("docx"), Some(OutputFormat::Docx));
        assert_eq!(OutputFormat::from_str("pptx"), Some(OutputFormat::Pptx));
        assert_eq!(OutputFormat::from_str("xlsx"), Some(OutputFormat::Xlsx));
        assert_eq!(OutputFormat::from_str("odt"), Some(OutputFormat::Odt));
        assert_eq!(OutputFormat::from_str("html"), Some(OutputFormat::Html));
        assert_eq!(OutputFormat::from_str("latex"), Some(OutputFormat::Latex));
        assert_eq!(OutputFormat::from_str("epub"), Some(OutputFormat::Epub));
        assert_eq!(OutputFormat::from_str("png"), Some(OutputFormat::Png));
        assert_eq!(OutputFormat::from_str("svg"), Some(OutputFormat::Svg));

        // Aliases
        assert_eq!(OutputFormat::from_str("word"), Some(OutputFormat::Docx));
        assert_eq!(
            OutputFormat::from_str("powerpoint"),
            Some(OutputFormat::Pptx)
        );
        assert_eq!(OutputFormat::from_str("excel"), Some(OutputFormat::Xlsx));
        assert_eq!(OutputFormat::from_str("htm"), Some(OutputFormat::Html));
        assert_eq!(OutputFormat::from_str("tex"), Some(OutputFormat::Latex));

        // Case insensitive
        assert_eq!(OutputFormat::from_str("PDF"), Some(OutputFormat::Pdf));
        assert_eq!(OutputFormat::from_str("Docx"), Some(OutputFormat::Docx));
        assert_eq!(OutputFormat::from_str("WORD"), Some(OutputFormat::Docx));

        // Invalid
        assert_eq!(OutputFormat::from_str(""), None);
        assert_eq!(OutputFormat::from_str("bmp"), None);
        assert_eq!(OutputFormat::from_str("unknown"), None);
    }

    #[test]
    fn test_output_format_mime_type() {
        assert_eq!(OutputFormat::Pdf.mime_type(), "application/pdf");
        assert_eq!(
            OutputFormat::Docx.mime_type(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert_eq!(
            OutputFormat::Pptx.mime_type(),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        );
        assert_eq!(
            OutputFormat::Xlsx.mime_type(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
        assert_eq!(
            OutputFormat::Odt.mime_type(),
            "application/vnd.oasis.opendocument.text"
        );
        assert_eq!(OutputFormat::Html.mime_type(), "text/html");
        assert_eq!(OutputFormat::Latex.mime_type(), "application/x-latex");
        assert_eq!(OutputFormat::Epub.mime_type(), "application/epub+zip");
        assert_eq!(OutputFormat::Png.mime_type(), "image/png");
        assert_eq!(OutputFormat::Svg.mime_type(), "image/svg+xml");
    }

    // =========================================================================
    // SourceFormat tests
    // =========================================================================

    #[test]
    fn test_source_format_extension() {
        assert_eq!(SourceFormat::Markdown.extension(), "md");
        assert_eq!(SourceFormat::Html.extension(), "html");
        assert_eq!(SourceFormat::Latex.extension(), "tex");
        assert_eq!(SourceFormat::Csv.extension(), "csv");
        assert_eq!(SourceFormat::Json.extension(), "json");
        assert_eq!(SourceFormat::PlainText.extension(), "txt");
        assert_eq!(SourceFormat::Rst.extension(), "rst");
    }

    #[test]
    fn test_source_format_pandoc_format() {
        assert_eq!(SourceFormat::Markdown.pandoc_format(), "markdown");
        assert_eq!(SourceFormat::Html.pandoc_format(), "html");
        assert_eq!(SourceFormat::Latex.pandoc_format(), "latex");
        assert_eq!(SourceFormat::Csv.pandoc_format(), "csv");
        assert_eq!(SourceFormat::Json.pandoc_format(), "json");
        assert_eq!(SourceFormat::PlainText.pandoc_format(), "plain");
        assert_eq!(SourceFormat::Rst.pandoc_format(), "rst");
    }

    // =========================================================================
    // DocumentRequest tests
    // =========================================================================

    #[test]
    fn test_document_request_new() {
        let req = DocumentRequest::new("# Hello", OutputFormat::Pdf);
        assert_eq!(req.content, "# Hello");
        assert_eq!(req.output_format, OutputFormat::Pdf);
        assert_eq!(req.source_format, SourceFormat::Markdown);
        assert_eq!(req.output_name, "document");
        assert!(req.stylesheet.is_none());
        assert!(req.extra_args.is_empty());
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn test_document_request_with_name() {
        let req = DocumentRequest::new("content", OutputFormat::Docx).with_name("my_report");
        assert_eq!(req.output_name, "my_report");
        assert_eq!(req.content, "content");
    }

    #[test]
    fn test_document_request_with_metadata() {
        let req = DocumentRequest::new("content", OutputFormat::Pdf)
            .with_metadata("title", "Test Doc")
            .with_metadata("author", "AI")
            .with_metadata("date", "2026-02-21");
        assert_eq!(req.metadata.len(), 3);
        assert_eq!(req.metadata.get("title").unwrap(), "Test Doc");
        assert_eq!(req.metadata.get("author").unwrap(), "AI");
        assert_eq!(req.metadata.get("date").unwrap(), "2026-02-21");
    }

    #[test]
    fn test_document_request_with_stylesheet() {
        let req = DocumentRequest::new("content", OutputFormat::Html)
            .with_stylesheet("body { font-family: sans-serif; }");
        assert_eq!(
            req.stylesheet.as_deref(),
            Some("body { font-family: sans-serif; }")
        );
    }

    #[test]
    fn test_document_request_with_source_format() {
        let req =
            DocumentRequest::new("<h1>Hello</h1>", OutputFormat::Pdf)
                .with_source_format(SourceFormat::Html);
        assert_eq!(req.source_format, SourceFormat::Html);
        assert_eq!(req.content, "<h1>Hello</h1>");
    }

    // =========================================================================
    // DocumentPipelineConfig tests
    // =========================================================================

    #[test]
    fn test_document_pipeline_config_default() {
        let config = DocumentPipelineConfig::default();
        assert_eq!(config.image, "pandoc/extra:latest");
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.reuse_container);
    }

    // =========================================================================
    // Supported conversions test
    // =========================================================================

    #[test]
    fn test_supported_conversions_count() {
        let conversions = DocumentPipeline::supported_conversions();
        assert!(
            conversions.len() >= 18,
            "Expected at least 18 supported conversions, got {}",
            conversions.len()
        );

        // Verify some key conversions are present
        assert!(
            conversions.contains(&(SourceFormat::Markdown, OutputFormat::Pdf)),
            "Markdown -> PDF should be supported"
        );
        assert!(
            conversions.contains(&(SourceFormat::Markdown, OutputFormat::Docx)),
            "Markdown -> DOCX should be supported"
        );
        assert!(
            conversions.contains(&(SourceFormat::Html, OutputFormat::Pdf)),
            "HTML -> PDF should be supported"
        );
        assert!(
            conversions.contains(&(SourceFormat::Latex, OutputFormat::Pdf)),
            "LaTeX -> PDF should be supported"
        );
        assert!(
            conversions.contains(&(SourceFormat::Rst, OutputFormat::Html)),
            "RST -> HTML should be supported"
        );
        assert!(
            conversions.contains(&(SourceFormat::Csv, OutputFormat::Html)),
            "CSV -> HTML should be supported"
        );
    }

    // =========================================================================
    // detect_source_format tests
    // =========================================================================

    #[test]
    fn test_detect_source_format() {
        assert_eq!(
            DocumentPipeline::detect_source_format("readme.md"),
            Some(SourceFormat::Markdown)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("notes.markdown"),
            Some(SourceFormat::Markdown)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("page.html"),
            Some(SourceFormat::Html)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("index.htm"),
            Some(SourceFormat::Html)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("paper.tex"),
            Some(SourceFormat::Latex)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("paper.latex"),
            Some(SourceFormat::Latex)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("data.csv"),
            Some(SourceFormat::Csv)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("config.json"),
            Some(SourceFormat::Json)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("notes.txt"),
            Some(SourceFormat::PlainText)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("docs.rst"),
            Some(SourceFormat::Rst)
        );

        // With directory paths
        assert_eq!(
            DocumentPipeline::detect_source_format("dir/sub/file.md"),
            Some(SourceFormat::Markdown)
        );

        // Case insensitive
        assert_eq!(
            DocumentPipeline::detect_source_format("FILE.HTML"),
            Some(SourceFormat::Html)
        );
        assert_eq!(
            DocumentPipeline::detect_source_format("DATA.CSV"),
            Some(SourceFormat::Csv)
        );
    }

    #[test]
    fn test_detect_source_format_invalid() {
        assert_eq!(DocumentPipeline::detect_source_format("image.png"), None);
        assert_eq!(DocumentPipeline::detect_source_format("video.mp4"), None);
        assert_eq!(DocumentPipeline::detect_source_format("binary.exe"), None);
        assert_eq!(DocumentPipeline::detect_source_format("archive.zip"), None);
        assert_eq!(
            DocumentPipeline::detect_source_format("unknown.xyz"),
            None
        );
    }

    // =========================================================================
    // DocumentResult fields test
    // =========================================================================

    #[test]
    fn test_document_result_fields() {
        let result = DocumentResult {
            output_path: PathBuf::from("/tmp/docs/report.pdf"),
            filename: "report.pdf".into(),
            size_bytes: 42_000,
            log: "conversion complete".into(),
            duration: Duration::from_millis(1500),
            format: OutputFormat::Pdf,
        };

        assert_eq!(result.output_path, PathBuf::from("/tmp/docs/report.pdf"));
        assert_eq!(result.filename, "report.pdf");
        assert_eq!(result.size_bytes, 42_000);
        assert_eq!(result.log, "conversion complete");
        assert_eq!(result.duration, Duration::from_millis(1500));
        assert_eq!(result.format, OutputFormat::Pdf);
    }

    // =========================================================================
    // DocumentError Display test
    // =========================================================================

    #[test]
    fn test_document_error_display() {
        let err = DocumentError::ConversionFailed("pandoc failed".into());
        assert_eq!(format!("{}", err), "Conversion failed: pandoc failed");

        let err = DocumentError::UnsupportedConversion("bmp to pdf".into());
        assert_eq!(format!("{}", err), "Unsupported conversion: bmp to pdf");

        let err = DocumentError::OutputNotFound("report.pdf".into());
        assert_eq!(format!("{}", err), "Output file not found: report.pdf");

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = DocumentError::IoError(io_err);
        let display = format!("{}", err);
        assert!(
            display.contains("I/O error"),
            "Expected 'I/O error' in: {}",
            display
        );
        assert!(
            display.contains("file missing"),
            "Expected 'file missing' in: {}",
            display
        );

        // ContainerError variant
        let ce = ContainerError::OperationFailed("docker died".into());
        let err = DocumentError::ContainerError(ce);
        let display = format!("{}", err);
        assert!(
            display.contains("Container error"),
            "Expected 'Container error' in: {}",
            display
        );
    }

    // =========================================================================
    // build_pandoc_command test
    // =========================================================================

    #[test]
    fn test_build_pandoc_command() {
        // build_pandoc_command is an associated function that does not require
        // a DocumentPipeline instance (no Docker needed).

        // Basic Markdown -> PDF request
        let req = DocumentRequest::new("# Test", OutputFormat::Pdf)
            .with_name("report")
            .with_metadata("title", "My Report")
            .with_metadata("author", "AI");

        let cmd =
            DocumentPipeline::build_pandoc_command(&req, "input_report.md", "report.pdf");

        assert_eq!(cmd[0], "pandoc");
        assert_eq!(cmd[1], "/workspace/input_report.md");
        assert_eq!(cmd[2], "-o");
        assert_eq!(cmd[3], "/workspace/report.pdf");
        assert_eq!(cmd[4], "-f");
        assert_eq!(cmd[5], "markdown");

        // Should contain metadata flags
        let cmd_joined = cmd.join(" ");
        assert!(
            cmd_joined.contains("-M title=My Report")
                || cmd_joined.contains("-M author=AI"),
            "Command should contain metadata flags: {}",
            cmd_joined
        );

        // PDF should include xelatex engine
        assert!(
            cmd.contains(&"--pdf-engine=xelatex".to_string()),
            "PDF output should use xelatex: {:?}",
            cmd
        );

        // Test with stylesheet for HTML output
        let req_html = DocumentRequest::new("<h1>Hi</h1>", OutputFormat::Html)
            .with_source_format(SourceFormat::Html)
            .with_stylesheet("body { color: red; }");

        let cmd_html = DocumentPipeline::build_pandoc_command(
            &req_html,
            "input_page.html",
            "page.html",
        );
        assert!(
            cmd_html.contains(&"--css".to_string()),
            "HTML with stylesheet should include --css: {:?}",
            cmd_html
        );
        assert!(
            cmd_html.contains(&"/workspace/style.css".to_string()),
            "CSS path should be /workspace/style.css: {:?}",
            cmd_html
        );

        // Should NOT include --pdf-engine for HTML
        assert!(
            !cmd_html.contains(&"--pdf-engine=xelatex".to_string()),
            "HTML output should not use xelatex: {:?}",
            cmd_html
        );

        // Test extra args
        let req_extra = DocumentRequest {
            content: "test".into(),
            source_format: SourceFormat::Markdown,
            output_format: OutputFormat::Docx,
            output_name: "out".into(),
            stylesheet: None,
            extra_args: vec!["--toc".into(), "--highlight-style=tango".into()],
            metadata: HashMap::new(),
        };

        let cmd_extra = DocumentPipeline::build_pandoc_command(
            &req_extra,
            "input_out.md",
            "out.docx",
        );
        assert!(
            cmd_extra.contains(&"--toc".to_string()),
            "Extra args should include --toc: {:?}",
            cmd_extra
        );
        assert!(
            cmd_extra.contains(&"--highlight-style=tango".to_string()),
            "Extra args should include --highlight-style=tango: {:?}",
            cmd_extra
        );

        // DOCX should NOT include --pdf-engine
        assert!(
            !cmd_extra.contains(&"--pdf-engine=xelatex".to_string()),
            "DOCX output should not use xelatex: {:?}",
            cmd_extra
        );
    }

    // =========================================================================
    // From impls test
    // =========================================================================

    #[test]
    fn test_document_error_from_container_error() {
        let ce = ContainerError::DockerNotAvailable("no docker".into());
        let de: DocumentError = ce.into();
        match de {
            DocumentError::ContainerError(_) => {} // expected
            other => panic!("Expected ContainerError variant, got: {:?}", other),
        }
    }

    #[test]
    fn test_document_error_from_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let de: DocumentError = io.into();
        match de {
            DocumentError::IoError(ref e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
            }
            other => panic!("Expected IoError variant, got: {:?}", other),
        }
    }

    // =========================================================================
    // OutputFormat clone + equality tests
    // =========================================================================

    #[test]
    fn test_output_format_clone_eq() {
        let a = OutputFormat::Epub;
        let b = a.clone();
        assert_eq!(a, b);

        // Different variants should not be equal
        assert_ne!(OutputFormat::Pdf, OutputFormat::Docx);
        assert_ne!(OutputFormat::Html, OutputFormat::Latex);
        assert_ne!(OutputFormat::Png, OutputFormat::Svg);
    }
}
