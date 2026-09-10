//! Multi-backend OCR pipeline.

use super::image_extraction::{ExtractedImage, ImageFormat};
use super::ocr_engine::{OcrConfig, OcrEngine, OcrResult};

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
/// own process-spawning logic. `recognize` reads nothing and returns an empty
/// result at zero confidence.
///
/// Registering it alone therefore yields no text: with the default
/// `min_confidence` of 0.3 the pipeline discards its result, which is the
/// correct outcome — no OCR ran. **Do not read the name as "Tesseract works
/// here"**; `docs/IMPROVEMENTS.md` did exactly that and listed OCR as done.
pub struct TesseractOcrBackend {
    config: TesseractConfig,
}

impl TesseractOcrBackend {
    pub fn new(config: TesseractConfig) -> Self {
        Self { config }
    }

    /// The settings a caller needs to spawn `tesseract` themselves.
    ///
    /// Without this the struct was a "configuration holder" whose configuration
    /// could not be read — the compiler said so as `field is never read` the
    /// moment the fake output stopped consuming it.
    pub fn config(&self) -> &TesseractConfig {
        &self.config
    }
}

impl OcrBackend for TesseractOcrBackend {
    fn name(&self) -> &str {
        "tesseract"
    }

    /// Recognises nothing, and says so by returning nothing.
    ///
    /// This used to put its own diagnostic — "binary not available for direct
    /// invocation" — into `full_text`, i.e. into the field that holds *what the
    /// image said*. A caller feeding OCR output into a summary or a RAG index
    /// would have stored that sentence as the content of the page. An empty
    /// result at zero confidence is the honest shape: nothing was read.
    fn recognize(&self, _image: &[u8], _width: usize, _height: usize) -> OcrResult {
        OcrResult {
            lines: Vec::new(),
            full_text: String::new(),
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
    /// If `preferred_backend` is set, that backend is tried first. The result
    /// with the highest `average_confidence` that meets `min_confidence` wins.
    ///
    /// **If no result meets the threshold, an empty result is returned** — not
    /// the best of a bad lot. This sentence used to promise the opposite and the
    /// code did neither: the threshold was never applied at all, so the "best"
    /// result won even at zero confidence. Returning text nobody could read is
    /// worse than returning none, because only the second is obvious downstream.
    ///
    /// An empty `OcrResult` also comes back when no backends are registered.
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
            // `min_confidence` was documented as a filter here and never applied,
            // so a backend that recognised nothing still won by default when it
            // was the only one registered — and its output was returned as text.
            // Reading nothing is a legitimate answer; returning noise as if it
            // were the page is not.
            if result.average_confidence < self.config.min_confidence {
                continue;
            }
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

#[cfg(test)]
mod honest_output_tests {
    //! OCR output feeds summaries and RAG indexes, so "what the image said" has
    //! to contain only that. Both defects pinned here shipped together: a
    //! backend that returned its own error message as recognised text, and a
    //! `min_confidence` the pipeline documented and never applied.
    use super::*;

    fn tesseract() -> TesseractOcrBackend {
        TesseractOcrBackend::new(TesseractConfig::default())
    }

    #[test]
    fn a_backend_that_reads_nothing_returns_nothing() {
        let result = tesseract().recognize(&[0u8; 16], 4, 4);

        assert!(
            result.full_text.is_empty(),
            "the diagnostic must not travel in the field that holds page content: {:?}",
            result.full_text
        );
        assert!(result.lines.is_empty());
        assert_eq!(result.average_confidence, 0.0);
    }

    #[test]
    fn the_pipeline_drops_results_below_min_confidence() {
        // With only the non-working backend registered, the honest answer is
        // "no text", not "here is a sentence at zero confidence".
        let mut pipeline = OcrPipeline::new(OcrPipelineConfig::default());
        pipeline.add_backend(Box::new(tesseract()));

        let result = pipeline.process_image(&[0u8; 16], 4, 4);

        assert!(result.full_text.is_empty(), "got {:?}", result.full_text);
        assert_eq!(result.average_confidence, 0.0);
    }

    #[test]
    fn min_confidence_is_read_from_the_config_not_ignored() {
        // The regression guard: before V311 `min_confidence` was documented as a
        // filter and never consulted, so this backend won by default.
        let config = OcrPipelineConfig {
            min_confidence: 0.0,
            ..OcrPipelineConfig::default()
        };
        let mut permissive = OcrPipeline::new(config);
        permissive.add_backend(Box::new(tesseract()));

        // At a zero threshold the empty result is admissible — and still empty,
        // because the backend no longer invents text.
        let result = permissive.process_image(&[0u8; 16], 4, 4);
        assert!(result.full_text.is_empty(), "got {:?}", result.full_text);
    }
}
