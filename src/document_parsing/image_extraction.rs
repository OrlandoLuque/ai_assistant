//! Image extraction from binary document streams.

use std::collections::HashMap;

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
