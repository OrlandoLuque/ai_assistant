//! Multi-backend OCR pipeline.

use super::image_extraction::{ExtractedImage, ImageFormat};
use super::ocr_engine::{OcrEngine, OcrConfig, OcrLine, OcrResult};

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
#[non_exhaustive]
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
#[non_exhaustive]
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
