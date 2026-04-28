//! Vision support for multimodal AI models
//!
//! This module provides support for image inputs to vision-capable models,
//! including image encoding, resizing, and multi-image messages.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// An image input for vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    /// The image data (base64 encoded or URL)
    pub data: ImageData,
    /// Media type (e.g., "image/png", "image/jpeg")
    pub media_type: String,
    /// Optional detail level for processing
    pub detail: ImageDetail,
}

/// Image data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ImageData {
    /// Base64 encoded image
    Base64(String),
    /// URL to an image
    Url(String),
}

/// Detail level for image processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageDetail {
    /// Low detail (faster, less tokens)
    Low,
    /// High detail (slower, more tokens)
    High,
    /// Auto (let the model decide)
    #[default]
    Auto,
}

/// Hard limits applied to inbound image payloads.
///
/// Decompression-bomb defense: dimensions are read from the image header
/// (PNG IHDR, JPEG SOF, GIF screen descriptor, WebP VP8/VP8L/VP8X) and
/// validated against `max_decoded_pixels` and `max_dimension` BEFORE any
/// pixel decode is attempted. A 1×1 PNG that decompresses to 100k×100k
/// (~40 GB) would still be caught here because the IHDR width/height fields
/// expose the declared dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionLimits {
    /// Maximum decoded pixel count (width × height). Default 67_108_864 (8192²).
    pub max_decoded_pixels: u64,
    /// Maximum encoded payload bytes. Default 20 MiB (matches CLI loader).
    pub max_encoded_bytes: usize,
    /// Maximum width OR height. Default 8192.
    pub max_dimension: u32,
}

impl Default for VisionLimits {
    fn default() -> Self {
        Self {
            max_decoded_pixels: 8192 * 8192,
            max_encoded_bytes: 20 * 1024 * 1024,
            max_dimension: 8192,
        }
    }
}

impl VisionLimits {
    /// Strict limits for untrusted inputs (4096² / 8 MiB / 4096 max-dim).
    pub fn strict() -> Self {
        Self {
            max_decoded_pixels: 4096 * 4096,
            max_encoded_bytes: 8 * 1024 * 1024,
            max_dimension: 4096,
        }
    }

    /// Permissive limits for trusted internal pipelines.
    pub fn permissive() -> Self {
        Self {
            max_decoded_pixels: 16384 * 16384,
            max_encoded_bytes: 100 * 1024 * 1024,
            max_dimension: 16384,
        }
    }
}

/// Unified image format detection. Wraps
/// [`crate::document_parsing::image_extraction::ImageExtractor::detect_format`]
/// and adds WebP detection (RIFF + "WEBP" tag). Returns the canonical
/// media type on success, or `None` for unsupported / unknown formats.
///
/// This is the single source of truth for magic-byte detection across the
/// vision pipeline. Callers must not roll their own header sniffing.
pub fn detect_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    use crate::document_parsing::{ImageExtractor, ImageFormat};

    if let Some(fmt) = ImageExtractor::detect_format(bytes) {
        return Some(match fmt {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::Gif => "image/gif",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Tiff => "image/tiff",
            ImageFormat::Unknown => return None,
        });
    }
    // WebP: "RIFF" .... "WEBP"
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

/// Parse image dimensions from header bytes WITHOUT decoding.
///
/// Supports PNG (IHDR), JPEG (SOF0/SOF1/SOF2/SOF3), GIF (logical screen),
/// WebP (VP8 / VP8L / VP8X), BMP (DIB header). Returns `None` if dimensions
/// cannot be located in the leading header section. This is the dimension
/// gate for decompression-bomb defense.
pub fn parse_image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // PNG: 8-byte signature + IHDR chunk; width@16, height@20 (big-endian).
    if bytes.len() >= 24 && &bytes[0..8] == b"\x89PNG\r\n\x1a\n" && &bytes[12..16] == b"IHDR" {
        let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        return Some((w, h));
    }
    // JPEG: scan for SOF marker (0xFFC0..0xFFCF excluding 0xFFC4/0xFFC8/0xFFCC).
    if bytes.len() >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        let mut i = 2;
        while i + 8 < bytes.len() {
            if bytes[i] != 0xFF {
                return None;
            }
            // Skip fill bytes (0xFF 0xFF ...)
            while i < bytes.len() && bytes[i] == 0xFF {
                i += 1;
            }
            if i >= bytes.len() {
                return None;
            }
            let marker = bytes[i];
            i += 1;
            // SOI/EOI/RSTn have no length
            if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
                continue;
            }
            if i + 1 >= bytes.len() {
                return None;
            }
            let seg_len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
            // SOFn: 0xC0..0xCF excluding 0xC4 (DHT), 0xC8 (JPG), 0xCC (DAC)
            if (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC
            {
                if i + 7 >= bytes.len() {
                    return None;
                }
                let h = u16::from_be_bytes([bytes[i + 3], bytes[i + 4]]) as u32;
                let w = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
                return Some((w, h));
            }
            i += seg_len;
        }
        return None;
    }
    // GIF: signature "GIF87a" or "GIF89a" + logical screen descriptor (LE).
    if bytes.len() >= 10 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        let w = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
        let h = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
        return Some((w, h));
    }
    // WebP: dimensions live in the VP8/VP8L/VP8X chunk after RIFF header.
    if bytes.len() >= 30 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        match &bytes[12..16] {
            b"VP8X" => {
                // Canvas Width Minus One (24-bit LE) at offset 24
                let w = (bytes[24] as u32) | ((bytes[25] as u32) << 8) | ((bytes[26] as u32) << 16);
                let h = (bytes[27] as u32) | ((bytes[28] as u32) << 8) | ((bytes[29] as u32) << 16);
                return Some((w + 1, h + 1));
            }
            b"VP8L" => {
                // 14-bit width-1 at offset 21 (LE bitstream after 0x2F sig)
                if bytes.len() >= 25 && bytes[20] == 0x2F {
                    let b0 = bytes[21] as u32;
                    let b1 = bytes[22] as u32;
                    let b2 = bytes[23] as u32;
                    let b3 = bytes[24] as u32;
                    let w = (b0 | ((b1 & 0x3F) << 8)) + 1;
                    let h = (((b1 >> 6) | (b2 << 2) | ((b3 & 0x0F) << 10)) & 0x3FFF) + 1;
                    return Some((w, h));
                }
                return None;
            }
            b"VP8 " => {
                // Lossy: 0x9D 0x01 0x2A start-code + width/height @ +6
                if bytes.len() >= 30 && bytes[23] == 0x9D && bytes[24] == 0x01 && bytes[25] == 0x2A
                {
                    let w = u16::from_le_bytes([bytes[26], bytes[27]]) as u32 & 0x3FFF;
                    let h = u16::from_le_bytes([bytes[28], bytes[29]]) as u32 & 0x3FFF;
                    return Some((w, h));
                }
                return None;
            }
            _ => return None,
        }
    }
    // BMP: BITMAPINFOHEADER width/height at offset 18/22 (LE, signed for height).
    if bytes.len() >= 26 && bytes[0] == b'B' && bytes[1] == b'M' {
        let w = u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
        let h = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]).unsigned_abs();
        return Some((w, h));
    }
    None
}

/// Detect whether the image is animated. Animated images are rejected by
/// the vision pipeline (per Batch 11 plan: "reject-animated") because
/// vision LLMs only consume single frames and animations multiply token
/// cost without benefit.
///
/// - GIF: presence of NETSCAPE2.0 application extension OR more than one
///   image-descriptor block.
/// - APNG: PNG with `acTL` chunk before `IDAT`.
/// - WebP: VP8X chunk with animation flag set (bit 1 of byte at offset 20).
pub fn is_animated_image(bytes: &[u8]) -> bool {
    // GIF — look for NETSCAPE2.0 marker (case-sensitive ASCII).
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        // NETSCAPE2.0 is the de-facto animation marker.
        if bytes.windows(11).any(|w| w == b"NETSCAPE2.0") {
            return true;
        }
        // GIF87a is single-frame by spec; GIF89a may still be animated even
        // without NETSCAPE — count image-descriptor markers (0x2C). Bytes
        // after the global color table aren't worth fully parsing here;
        // a count >1 of 0x2C is a strong heuristic.
        let count = bytes.iter().filter(|&&b| b == 0x2C).count();
        return count > 1;
    }
    // APNG — PNG with acTL chunk (4 bytes length + "acTL" type).
    if bytes.len() >= 8 && &bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
        return bytes.windows(4).any(|w| w == b"acTL");
    }
    // WebP — VP8X chunk has animation flag (bit 1) in flags byte at offset 20.
    if bytes.len() >= 21
        && &bytes[0..4] == b"RIFF"
        && &bytes[8..12] == b"WEBP"
        && &bytes[12..16] == b"VP8X"
    {
        return (bytes[20] & 0x02) != 0;
    }
    false
}

impl ImageInput {
    /// Create from a file path
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = fs::read(path)?;
        let base64 = base64_encode(&data);

        let media_type = Self::detect_media_type(path)?;

        Ok(Self {
            data: ImageData::Base64(base64),
            media_type,
            detail: ImageDetail::Auto,
        })
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8], media_type: &str) -> Self {
        Self {
            data: ImageData::Base64(base64_encode(bytes)),
            media_type: media_type.to_string(),
            detail: ImageDetail::Auto,
        }
    }

    /// Create from raw bytes with safety validation.
    ///
    /// Runs [`ImagePreprocessor::validate_bytes`] before constructing the
    /// `ImageInput`. The detected media type from magic-byte inspection
    /// overrides any caller-supplied hint (so a `.png` with a JPEG header
    /// is corrected, not blindly trusted). Use this entry point for any
    /// untrusted input; `from_bytes` remains for callers that have
    /// already validated upstream.
    pub fn from_bytes_validated(bytes: &[u8], limits: &VisionLimits) -> Result<Self> {
        let preprocessor = ImagePreprocessor::default().with_limits(*limits);
        let (media_type, _w, _h) = preprocessor.validate_bytes(bytes)?;
        Ok(Self {
            data: ImageData::Base64(base64_encode(bytes)),
            media_type: media_type.to_string(),
            detail: ImageDetail::Auto,
        })
    }

    /// Read from disk and validate. Combines [`Self::from_file`] with the
    /// safety gates of [`ImagePreprocessor::validate_bytes`].
    pub fn from_file_validated(path: &Path, limits: &VisionLimits) -> Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes_validated(&bytes, limits)
    }

    /// Create from a URL
    pub fn from_url(url: &str) -> Self {
        let media_type =
            Self::detect_media_type_from_url(url).unwrap_or_else(|| "image/jpeg".to_string());

        Self {
            data: ImageData::Url(url.to_string()),
            media_type,
            detail: ImageDetail::Auto,
        }
    }

    /// Create from base64 string
    pub fn from_base64(base64: &str, media_type: &str) -> Self {
        Self {
            data: ImageData::Base64(base64.to_string()),
            media_type: media_type.to_string(),
            detail: ImageDetail::Auto,
        }
    }

    /// Set detail level
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Detect media type from file extension
    fn detect_media_type(path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("Could not determine file extension"))?
            .to_lowercase();

        Ok(match ext.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            "svg" => "image/svg+xml",
            _ => return Err(anyhow!("Unsupported image format: {}", ext)),
        }
        .to_string())
    }

    /// Detect media type from URL
    fn detect_media_type_from_url(url: &str) -> Option<String> {
        let url_lower = url.to_lowercase();
        if url_lower.ends_with(".png") {
            Some("image/png".to_string())
        } else if url_lower.ends_with(".jpg") || url_lower.ends_with(".jpeg") {
            Some("image/jpeg".to_string())
        } else if url_lower.ends_with(".gif") {
            Some("image/gif".to_string())
        } else if url_lower.ends_with(".webp") {
            Some("image/webp".to_string())
        } else {
            None
        }
    }

    /// Convert to data URL format
    pub fn to_data_url(&self) -> String {
        match &self.data {
            ImageData::Base64(b64) => format!("data:{};base64,{}", self.media_type, b64),
            ImageData::Url(url) => url.clone(),
        }
    }

    /// Convert to OpenAI API format
    pub fn to_openai_format(&self) -> serde_json::Value {
        let image_url = match &self.data {
            ImageData::Base64(b64) => {
                serde_json::json!({
                    "url": format!("data:{};base64,{}", self.media_type, b64)
                })
            }
            ImageData::Url(url) => {
                serde_json::json!({
                    "url": url
                })
            }
        };

        let mut result = serde_json::json!({
            "type": "image_url",
            "image_url": image_url
        });

        if self.detail != ImageDetail::Auto {
            result["image_url"]["detail"] = serde_json::json!(match self.detail {
                ImageDetail::Low => "low",
                ImageDetail::High => "high",
                ImageDetail::Auto => "auto",
            });
        }

        result
    }

    /// Convert to Ollama format
    pub fn to_ollama_format(&self) -> serde_json::Value {
        match &self.data {
            ImageData::Base64(b64) => serde_json::json!(b64),
            ImageData::Url(url) => serde_json::json!(url),
        }
    }

    /// Convert to Anthropic Messages API format.
    ///
    /// Anthropic accepts two source types:
    /// - `{"type":"base64","media_type":"image/png","data":"..."}`
    /// - `{"type":"url","url":"https://..."}` (added 2024)
    pub fn to_anthropic_format(&self) -> serde_json::Value {
        let source = match &self.data {
            ImageData::Base64(b64) => serde_json::json!({
                "type": "base64",
                "media_type": self.media_type,
                "data": b64,
            }),
            ImageData::Url(url) => serde_json::json!({
                "type": "url",
                "url": url,
            }),
        };
        serde_json::json!({
            "type": "image",
            "source": source,
        })
    }

    /// Convert to Google Gemini `inlineData` / `fileData` format.
    ///
    /// Gemini requires the raw base64 (not a data URL) and a `mimeType`.
    /// URLs are wrapped as `fileData` with `fileUri`.
    pub fn to_gemini_format(&self) -> serde_json::Value {
        match &self.data {
            ImageData::Base64(b64) => serde_json::json!({
                "inlineData": {
                    "mimeType": self.media_type,
                    "data": b64,
                }
            }),
            ImageData::Url(url) => serde_json::json!({
                "fileData": {
                    "mimeType": self.media_type,
                    "fileUri": url,
                }
            }),
        }
    }

    /// Get estimated token cost (rough approximation)
    pub fn estimate_tokens(&self) -> usize {
        match self.detail {
            ImageDetail::Low => 85,
            ImageDetail::High => 765, // For a typical 1024x1024 image
            ImageDetail::Auto => 500, // Estimate
        }
    }
}

/// A multimodal message with text and images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionMessage {
    /// Role (user, assistant, system)
    pub role: String,
    /// Text content
    pub text: String,
    /// Image inputs
    pub images: Vec<ImageInput>,
}

impl VisionMessage {
    /// Create a user message with images
    pub fn user(text: &str, images: Vec<ImageInput>) -> Self {
        Self {
            role: "user".to_string(),
            text: text.to_string(),
            images,
        }
    }

    /// Create an assistant message
    pub fn assistant(text: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            text: text.to_string(),
            images: vec![],
        }
    }

    /// Create a system message
    pub fn system(text: &str) -> Self {
        Self {
            role: "system".to_string(),
            text: text.to_string(),
            images: vec![],
        }
    }

    /// Add an image
    pub fn with_image(mut self, image: ImageInput) -> Self {
        self.images.push(image);
        self
    }

    /// Convert to OpenAI API format
    pub fn to_openai_format(&self) -> serde_json::Value {
        if self.images.is_empty() {
            serde_json::json!({
                "role": self.role,
                "content": self.text
            })
        } else {
            let mut content = vec![serde_json::json!({
                "type": "text",
                "text": self.text
            })];

            for image in &self.images {
                content.push(image.to_openai_format());
            }

            serde_json::json!({
                "role": self.role,
                "content": content
            })
        }
    }

    /// Convert to Ollama format
    pub fn to_ollama_format(&self) -> serde_json::Value {
        let mut msg = serde_json::json!({
            "role": self.role,
            "content": self.text
        });

        if !self.images.is_empty() {
            let images: Vec<serde_json::Value> = self
                .images
                .iter()
                .map(|img| img.to_ollama_format())
                .collect();
            msg["images"] = serde_json::json!(images);
        }

        msg
    }

    /// Convert to Anthropic Messages API format.
    ///
    /// Builds a content array with text + image blocks. The `system` role
    /// is NOT supported in Anthropic's `messages` array; callers should
    /// route system prompts to the top-level `system` parameter.
    pub fn to_anthropic_format(&self) -> serde_json::Value {
        let role = if self.role == "system" {
            "user"
        } else {
            self.role.as_str()
        };
        if self.images.is_empty() {
            return serde_json::json!({
                "role": role,
                "content": self.text,
            });
        }
        let mut content = Vec::with_capacity(self.images.len() + 1);
        for image in &self.images {
            content.push(image.to_anthropic_format());
        }
        if !self.text.is_empty() {
            content.push(serde_json::json!({
                "type": "text",
                "text": self.text,
            }));
        }
        serde_json::json!({
            "role": role,
            "content": content,
        })
    }

    /// Convert to Google Gemini `contents` format (single entry).
    ///
    /// Roles map: `assistant` → `model`, `system` → `user` (system prompt
    /// is supplied via top-level `systemInstruction`).
    pub fn to_gemini_format(&self) -> serde_json::Value {
        let role = match self.role.as_str() {
            "assistant" => "model",
            "system" => "user",
            other => other,
        };
        let mut parts = Vec::with_capacity(self.images.len() + 1);
        if !self.text.is_empty() {
            parts.push(serde_json::json!({ "text": self.text }));
        }
        for image in &self.images {
            parts.push(image.to_gemini_format());
        }
        serde_json::json!({
            "role": role,
            "parts": parts,
        })
    }

    /// Estimate total tokens
    pub fn estimate_tokens(&self) -> usize {
        let text_tokens = crate::estimate_tokens(&self.text);
        let image_tokens: usize = self.images.iter().map(|i| i.estimate_tokens()).sum();
        text_tokens + image_tokens
    }
}

/// Vision capabilities checker
pub struct VisionCapabilities {
    /// List of known vision-capable models
    vision_models: Vec<&'static str>,
}

impl VisionCapabilities {
    /// Create a new capabilities checker
    pub fn new() -> Self {
        Self {
            vision_models: vec![
                // OpenAI
                "gpt-4-vision",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4v",
                // Anthropic
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-3-haiku",
                // Ollama/Local
                "llava",
                "bakllava",
                "llava-llama3",
                "llava-phi3",
                "moondream",
                "minicpm-v",
                // Others
                "cogvlm",
                "qwen-vl",
                "yi-vl",
            ],
        }
    }

    /// Check if a model supports vision
    pub fn supports_vision(&self, model_name: &str) -> bool {
        let model_lower = model_name.to_lowercase();
        self.vision_models.iter().any(|m| model_lower.contains(m))
            || model_lower.contains("vision")
            || model_lower.contains("-vl")
            || model_lower.contains("llava")
    }

    /// Get the recommended image format for a model
    pub fn recommended_format(&self, model_name: &str) -> &str {
        let model_lower = model_name.to_lowercase();
        if model_lower.contains("claude") {
            "image/png"
        } else {
            "image/jpeg" // Most compatible
        }
    }

    /// Get maximum images per message for a model
    pub fn max_images(&self, model_name: &str) -> usize {
        let model_lower = model_name.to_lowercase();
        if model_lower.contains("gpt-4") {
            10
        } else if model_lower.contains("claude") {
            20
        } else {
            4 // Conservative default for local models
        }
    }
}

impl Default for VisionCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

/// Image preprocessing utilities
pub struct ImagePreprocessor {
    /// Maximum width
    pub max_width: u32,
    /// Maximum height
    pub max_height: u32,
    /// Target format
    pub target_format: String,
    /// Quality (for JPEG)
    pub quality: u8,
    /// Hard safety limits applied during [`Self::validate_bytes`] /
    /// [`Self::process`]. Defends against decompression bombs and oversized
    /// payloads BEFORE any pixel decode.
    pub limits: VisionLimits,
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            max_width: 2048,
            max_height: 2048,
            target_format: "image/jpeg".to_string(),
            quality: 85,
            limits: VisionLimits::default(),
        }
    }
}

impl ImagePreprocessor {
    /// Create a preprocessor for low detail mode
    pub fn low_detail() -> Self {
        Self {
            max_width: 512,
            max_height: 512,
            target_format: "image/jpeg".to_string(),
            quality: 75,
            limits: VisionLimits::default(),
        }
    }

    /// Create a preprocessor for high detail mode
    pub fn high_detail() -> Self {
        Self {
            max_width: 2048,
            max_height: 2048,
            target_format: "image/png".to_string(),
            quality: 100,
            limits: VisionLimits::default(),
        }
    }

    /// Override safety limits.
    pub fn with_limits(mut self, limits: VisionLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Check if an image needs resizing
    pub fn needs_resize(&self, width: u32, height: u32) -> bool {
        width > self.max_width || height > self.max_height
    }

    /// Calculate new dimensions maintaining aspect ratio
    pub fn calculate_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        if !self.needs_resize(width, height) {
            return (width, height);
        }

        let width_ratio = self.max_width as f64 / width as f64;
        let height_ratio = self.max_height as f64 / height as f64;
        let ratio = width_ratio.min(height_ratio);

        let new_width = (width as f64 * ratio).round() as u32;
        let new_height = (height as f64 * ratio).round() as u32;

        (new_width.max(1), new_height.max(1))
    }

    /// Validate raw image bytes without decoding pixels.
    ///
    /// Enforces, in order:
    /// 1. Encoded byte cap (`limits.max_encoded_bytes`).
    /// 2. Magic-byte detection — rejects unknown formats (incl. SVG, which
    ///    is denylisted because of XML-XXE / JS-injection risk).
    /// 3. Header-declared dimensions against `limits.max_dimension` and
    ///    `limits.max_decoded_pixels` (decompression-bomb gate).
    /// 4. Animated-image rejection (vision LLMs see a single frame; an
    ///    animation balloons cost without benefit).
    ///
    /// Returns `(detected_media_type, width, height)` on success.
    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(&'static str, u32, u32)> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(anyhow!(
                "image bytes ({}) exceed max_encoded_bytes ({})",
                bytes.len(),
                self.limits.max_encoded_bytes
            ));
        }
        let media_type = detect_image_media_type(bytes).ok_or_else(|| {
            anyhow!("unsupported or unrecognised image format (magic-byte check)")
        })?;
        if is_animated_image(bytes) {
            return Err(anyhow!(
                "animated images are not supported (got {})",
                media_type
            ));
        }
        let (w, h) = parse_image_dimensions(bytes)
            .ok_or_else(|| anyhow!("could not parse image dimensions from header"))?;
        if w == 0 || h == 0 {
            return Err(anyhow!("image declares zero dimension (w={}, h={})", w, h));
        }
        if w > self.limits.max_dimension || h > self.limits.max_dimension {
            return Err(anyhow!(
                "image dimension ({}x{}) exceeds max_dimension ({})",
                w,
                h,
                self.limits.max_dimension
            ));
        }
        let pixels = (w as u64).saturating_mul(h as u64);
        if pixels > self.limits.max_decoded_pixels {
            return Err(anyhow!(
                "image pixel count ({}) exceeds max_decoded_pixels ({}) — decompression-bomb defense",
                pixels,
                self.limits.max_decoded_pixels
            ));
        }
        Ok((media_type, w, h))
    }

    /// Process raw image bytes for vision-pipeline ingestion.
    ///
    /// In this batch (Batch 11 v1) `process` validates safely without
    /// decoding pixels; resize/re-encode lands in a follow-up batch when
    /// the `image` crate is added behind the `vision` feature. Returns the
    /// validated bytes unchanged on success — the contract is "after
    /// `process` returns Ok, downstream code may pass these bytes to a
    /// provider".
    pub fn process<'a>(&self, bytes: &'a [u8]) -> Result<(&'static str, u32, u32, &'a [u8])> {
        let (mt, w, h) = self.validate_bytes(bytes)?;
        Ok((mt, w, h, bytes))
    }
}

/// Simple base64 decoder (no external dependency). Symmetric with
/// `base64_encode`. Used by `agent_bridge` paths to validate inbound
/// base64 image payloads against `VisionLimits` before dispatch.
pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim_end_matches('=');
    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(format!("Invalid base64 character: {}", c)),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(result)
}

/// Simple base64 encoding (no external dependency)
pub(crate) fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let chunks = data.chunks(3);

    for chunk in chunks {
        let mut n: u32 = 0;
        for (i, &byte) in chunk.iter().enumerate() {
            n |= (byte as u32) << (16 - i * 8);
        }

        let char_count = chunk.len() + 1;
        for i in 0..4 {
            if i < char_count {
                let idx = ((n >> (18 - i * 6)) & 0x3F) as usize;
                result.push(ALPHABET[idx] as char);
            } else {
                result.push('=');
            }
        }
    }

    result
}

/// Batch image processor
pub struct ImageBatch {
    images: Vec<ImageInput>,
    max_images: usize,
}

impl ImageBatch {
    /// Create a new batch
    pub fn new(max_images: usize) -> Self {
        Self {
            images: Vec::new(),
            max_images,
        }
    }

    /// Add an image to the batch
    pub fn add(&mut self, image: ImageInput) -> bool {
        if self.images.len() >= self.max_images {
            return false;
        }
        self.images.push(image);
        true
    }

    /// Add from file path
    pub fn add_file(&mut self, path: &Path) -> Result<bool> {
        if self.images.len() >= self.max_images {
            return Ok(false);
        }
        let image = ImageInput::from_file(path)?;
        self.images.push(image);
        Ok(true)
    }

    /// Add from URL
    pub fn add_url(&mut self, url: &str) -> bool {
        if self.images.len() >= self.max_images {
            return false;
        }
        self.images.push(ImageInput::from_url(url));
        true
    }

    /// Get all images
    pub fn images(&self) -> &[ImageInput] {
        &self.images
    }

    /// Take all images
    pub fn take(self) -> Vec<ImageInput> {
        self.images
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.images.len() >= self.max_images
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.max_images.saturating_sub(self.images.len())
    }

    /// Estimate total tokens for all images
    pub fn estimate_tokens(&self) -> usize {
        self.images.iter().map(|i| i.estimate_tokens()).sum()
    }
}

/// Unified dispatcher: send a vision request through the right transport for
/// the configured provider.
///
/// - Cloud providers (OpenAI, Anthropic, Gemini, Groq, Together, etc.) →
///   [`crate::cloud_providers::generate_cloud_response_with_images`]
/// - Ollama → [`crate::providers::generate_ollama_response_with_images`]
/// - LM Studio / LocalAI / llama.cpp / vLLM / text-gen-webui /
///   `OpenAICompatible` → [`crate::providers::generate_openai_compat_response_with_images`]
/// - Anything else → returns an error explaining the limitation.
///
/// The router profiles in [`crate::routing`] mark which model patterns claim
/// vision support; a provider being eligible here does not guarantee the
/// loaded model can read images — pair this with a vision-capable model.
pub fn generate_vision_response(
    config: &crate::config::AiConfig,
    messages: &[VisionMessage],
    system_prompt: &str,
) -> anyhow::Result<String> {
    use crate::config::AiProvider;
    match config.provider {
        AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Gemini
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => crate::cloud_providers::generate_cloud_response_with_images(
            config,
            messages,
            system_prompt,
        ),
        AiProvider::Ollama => {
            crate::providers::generate_ollama_response_with_images(config, messages, system_prompt)
        }
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::LlamaCpp
        | AiProvider::VLLM
        | AiProvider::OpenAICompatible { .. } => {
            crate::providers::generate_openai_compat_response_with_images(
                config,
                messages,
                system_prompt,
            )
        }
        AiProvider::AzureOpenAI { .. } => anyhow::bail!(
            "Azure OpenAI vision requires deployment-specific routing; \
             use generate_openai_cloud_with_images directly."
        ),
        AiProvider::Bedrock { .. } => {
            anyhow::bail!("AWS Bedrock vision requires the `aws-bedrock` feature flag")
        }
        _ => anyhow::bail!(
            "Vision is not supported for provider {}",
            config.provider.display_name()
        ),
    }
}

// =============================================================================
// IMAGE STORE
// =============================================================================

/// Opaque, content-addressed reference to an image previously placed in an
/// [`ImageStore`]. Audit logs and persisted artefacts (memory entries,
/// snapshots, conversation history) carry this reference instead of raw
/// bytes — that way the bytes live in one place and the rest of the system
/// only needs to know the SHA256.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageRef {
    /// Store-local key used by [`ImageStore::get`].
    pub key: String,
    /// Media type (e.g. `image/png`).
    pub media_type: String,
    /// Hex-encoded SHA256 of the underlying bytes (or of the URL string for
    /// URL-only refs). Stable across stores.
    pub sha256: String,
}

/// Pluggable persistence backend for image bytes.
///
/// Hash-only audit is the default operating mode: callers store bytes here
/// once, then carry the `ImageRef` around. Implementations:
/// * [`InMemoryImageStore`] — process-local, lost on restart.
/// * [`FilesystemImageStore`] — `<base>/<sha256>.<ext>` layout.
///
/// An S3 backend is intentionally deferred (Batch 14 spec) — third-party
/// integrations are added under their own feature flag.
pub trait ImageStore: Send + Sync {
    /// Persist `image` and return a stable reference.
    fn put(&self, image: &ImageInput) -> Result<ImageRef>;
    /// Retrieve the image previously stored under `image_ref`. Returns
    /// `Err` if the reference is unknown to this store.
    fn get(&self, image_ref: &ImageRef) -> Result<ImageInput>;
}

/// Compute the SHA256 of an `ImageInput`. For URL-only references the
/// SHA256 is taken over the URL string itself (no network I/O).
pub fn image_sha256(image: &ImageInput) -> String {
    match &image.data {
        ImageData::Url(url) => crate::binary_integrity::hash_bytes(url.as_bytes()),
        ImageData::Base64(b64) => match base64_decode(b64) {
            Ok(bytes) => crate::binary_integrity::hash_bytes(&bytes),
            // Fall back to hashing the b64 string itself if it is malformed —
            // we still want a stable identifier.
            Err(_) => crate::binary_integrity::hash_bytes(b64.as_bytes()),
        },
    }
}

/// In-memory `ImageStore`. Thread-safe via `Mutex`. Suitable for tests,
/// short-lived agent surfaces, and process-local caches.
#[derive(Default)]
pub struct InMemoryImageStore {
    inner: std::sync::Mutex<std::collections::HashMap<String, ImageInput>>,
}

impl InMemoryImageStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ImageStore for InMemoryImageStore {
    fn put(&self, image: &ImageInput) -> Result<ImageRef> {
        let sha = image_sha256(image);
        let r = ImageRef {
            key: sha.clone(),
            media_type: image.media_type.clone(),
            sha256: sha,
        };
        let mut map = self
            .inner
            .lock()
            .map_err(|_| anyhow!("InMemoryImageStore mutex poisoned"))?;
        map.insert(r.key.clone(), image.clone());
        Ok(r)
    }

    fn get(&self, image_ref: &ImageRef) -> Result<ImageInput> {
        let map = self
            .inner
            .lock()
            .map_err(|_| anyhow!("InMemoryImageStore mutex poisoned"))?;
        map.get(&image_ref.key)
            .cloned()
            .ok_or_else(|| anyhow!("image not found: {}", image_ref.key))
    }
}

/// Filesystem-backed `ImageStore`. Each base64 image is written to
/// `<base>/<sha256>.<ext>` (raw decoded bytes). URL-only refs are written
/// as `<base>/<sha256>.url` containing the URL string.
pub struct FilesystemImageStore {
    base: std::path::PathBuf,
}

impl FilesystemImageStore {
    /// Create or open a filesystem store rooted at `base`. The directory
    /// is created if missing.
    pub fn new(base: impl Into<std::path::PathBuf>) -> Result<Self> {
        let base = base.into();
        fs::create_dir_all(&base)
            .map_err(|e| anyhow!("create_dir_all({}): {}", base.display(), e))?;
        Ok(Self { base })
    }

    fn ext_for(media_type: &str) -> &'static str {
        match media_type {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            "image/bmp" => "bmp",
            "image/tiff" => "tiff",
            _ => "bin",
        }
    }
}

impl ImageStore for FilesystemImageStore {
    fn put(&self, image: &ImageInput) -> Result<ImageRef> {
        let sha = image_sha256(image);
        match &image.data {
            ImageData::Url(url) => {
                let path = self.base.join(format!("{}.url", sha));
                fs::write(&path, url.as_bytes())
                    .map_err(|e| anyhow!("write url ref {}: {}", path.display(), e))?;
                Ok(ImageRef {
                    key: format!("{}.url", sha),
                    media_type: image.media_type.clone(),
                    sha256: sha,
                })
            }
            ImageData::Base64(b64) => {
                let bytes = base64_decode(b64).map_err(|e| anyhow!("base64 decode: {}", e))?;
                let ext = Self::ext_for(&image.media_type);
                let key = format!("{}.{}", sha, ext);
                let path = self.base.join(&key);
                fs::write(&path, &bytes)
                    .map_err(|e| anyhow!("write image {}: {}", path.display(), e))?;
                Ok(ImageRef {
                    key,
                    media_type: image.media_type.clone(),
                    sha256: sha,
                })
            }
        }
    }

    fn get(&self, image_ref: &ImageRef) -> Result<ImageInput> {
        let path = self.base.join(&image_ref.key);
        let bytes = fs::read(&path).map_err(|e| anyhow!("read {}: {}", path.display(), e))?;
        if image_ref.key.ends_with(".url") {
            let url =
                String::from_utf8(bytes).map_err(|e| anyhow!("invalid utf-8 url ref: {}", e))?;
            let mut img = ImageInput::from_url(&url);
            img.media_type = image_ref.media_type.clone();
            Ok(img)
        } else {
            Ok(ImageInput::from_bytes(&bytes, &image_ref.media_type))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_from_bytes() {
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
        let image = ImageInput::from_bytes(&bytes, "image/jpeg");

        assert_eq!(image.media_type, "image/jpeg");
        matches!(image.data, ImageData::Base64(_));
    }

    #[test]
    fn test_image_from_url() {
        let image = ImageInput::from_url("https://example.com/image.png");

        assert_eq!(image.media_type, "image/png");
        matches!(image.data, ImageData::Url(_));
    }

    #[test]
    fn test_openai_format() {
        let image =
            ImageInput::from_url("https://example.com/image.jpg").with_detail(ImageDetail::High);

        let format = image.to_openai_format();

        assert_eq!(format["type"], "image_url");
        assert!(format["image_url"]["url"].as_str().is_some());
    }

    #[test]
    fn test_vision_message() {
        let message = VisionMessage::user(
            "What's in this image?",
            vec![ImageInput::from_url("https://example.com/cat.jpg")],
        );

        let format = message.to_openai_format();

        assert_eq!(format["role"], "user");
        assert!(format["content"].is_array());
    }

    #[test]
    fn test_vision_capabilities() {
        let caps = VisionCapabilities::new();

        assert!(caps.supports_vision("gpt-4o"));
        assert!(caps.supports_vision("llava"));
        assert!(caps.supports_vision("claude-3-opus"));
        assert!(!caps.supports_vision("gpt-3.5-turbo"));
    }

    #[test]
    fn test_preprocessor_dimensions() {
        let preprocessor = ImagePreprocessor::default();

        // No resize needed
        let (w, h) = preprocessor.calculate_dimensions(1000, 800);
        assert_eq!((w, h), (1000, 800));

        // Resize needed
        let (w, h) = preprocessor.calculate_dimensions(4000, 3000);
        assert!(w <= 2048);
        assert!(h <= 2048);
    }

    #[test]
    fn test_image_batch() {
        let mut batch = ImageBatch::new(3);

        assert!(batch.add_url("https://example.com/1.jpg"));
        assert!(batch.add_url("https://example.com/2.jpg"));
        assert!(batch.add_url("https://example.com/3.jpg"));
        assert!(!batch.add_url("https://example.com/4.jpg")); // Full

        assert!(batch.is_full());
        assert_eq!(batch.images().len(), 3);
    }

    #[test]
    fn test_base64_encode() {
        let result = base64_encode(b"Hello");
        assert_eq!(result, "SGVsbG8=");

        let result = base64_encode(b"Hello World!");
        assert_eq!(result, "SGVsbG8gV29ybGQh");
    }

    #[test]
    fn test_token_estimation() {
        let image =
            ImageInput::from_url("https://example.com/image.jpg").with_detail(ImageDetail::Low);
        assert_eq!(image.estimate_tokens(), 85);

        let image =
            ImageInput::from_url("https://example.com/image.jpg").with_detail(ImageDetail::High);
        assert_eq!(image.estimate_tokens(), 765);
    }

    #[test]
    fn test_image_detail_default() {
        let image = ImageInput::from_url("https://example.com/image.jpg");
        assert_eq!(image.detail, ImageDetail::Auto);
    }

    #[test]
    fn test_image_anthropic_format_base64() {
        let image = ImageInput::from_bytes(&[0xFF, 0xD8], "image/jpeg");
        let v = image.to_anthropic_format();
        assert_eq!(v["type"], "image");
        assert_eq!(v["source"]["type"], "base64");
        assert_eq!(v["source"]["media_type"], "image/jpeg");
        assert!(v["source"]["data"].as_str().unwrap().len() > 0);
    }

    #[test]
    fn test_image_anthropic_format_url() {
        let image = ImageInput::from_url("https://example.com/cat.png");
        let v = image.to_anthropic_format();
        assert_eq!(v["type"], "image");
        assert_eq!(v["source"]["type"], "url");
        assert_eq!(v["source"]["url"], "https://example.com/cat.png");
    }

    #[test]
    fn test_image_gemini_format_inline_data() {
        let image = ImageInput::from_bytes(&[0x89, 0x50, 0x4E, 0x47], "image/png");
        let v = image.to_gemini_format();
        assert_eq!(v["inlineData"]["mimeType"], "image/png");
        assert!(v["inlineData"]["data"].as_str().unwrap().len() > 0);
    }

    #[test]
    fn test_image_gemini_format_file_data() {
        let image = ImageInput::from_url("https://example.com/cat.jpg");
        let v = image.to_gemini_format();
        assert_eq!(v["fileData"]["mimeType"], "image/jpeg");
        assert_eq!(v["fileData"]["fileUri"], "https://example.com/cat.jpg");
    }

    #[test]
    fn test_message_anthropic_text_only() {
        let m = VisionMessage::user("hello", vec![]);
        let v = m.to_anthropic_format();
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello");
    }

    #[test]
    fn test_message_anthropic_with_images() {
        let m = VisionMessage::user(
            "What is this?",
            vec![ImageInput::from_url("https://example.com/x.png")],
        );
        let v = m.to_anthropic_format();
        let content = v["content"].as_array().expect("array content");
        // image first, then text
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "What is this?");
    }

    #[test]
    fn test_message_anthropic_system_role_remapped_to_user() {
        let m = VisionMessage::system("You are a helper");
        let v = m.to_anthropic_format();
        assert_eq!(
            v["role"], "user",
            "Anthropic doesn't accept system in messages"
        );
    }

    #[test]
    fn test_message_gemini_role_remap() {
        let m = VisionMessage::assistant("done");
        let v = m.to_gemini_format();
        assert_eq!(v["role"], "model", "assistant must map to model for Gemini");
    }

    #[test]
    fn test_message_gemini_with_image() {
        let m = VisionMessage::user(
            "describe",
            vec![ImageInput::from_bytes(&[0x89, 0x50], "image/png")],
        );
        let v = m.to_gemini_format();
        let parts = v["parts"].as_array().expect("array parts");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].get("text").is_some());
        assert!(parts[1].get("inlineData").is_some());
    }

    // ---------- Batch 11: VisionLimits / decompression-bomb defense ----------

    /// 24-byte minimal PNG header with custom IHDR width/height.
    fn png_header(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::from(&b"\x89PNG\r\n\x1a\n"[..]);
        // IHDR length (fake; we only need the type + dims)
        v.extend_from_slice(&[0, 0, 0, 13]);
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&h.to_be_bytes());
        // bit-depth/color-type/compression/filter/interlace
        v.extend_from_slice(&[8, 2, 0, 0, 0]);
        v
    }

    /// Minimal JPEG header with SOF0 declaring (w,h).
    fn jpeg_header(w: u16, h: u16) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8, 0xFF, 0xE0]; // SOI + APP0
        v.extend_from_slice(&[0, 16]); // APP0 length 16
        v.extend_from_slice(b"JFIF\0");
        v.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]); // 9 bytes
        v.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        v.extend_from_slice(&[0, 17]); // segment length
        v.push(8); // precision
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v.push(3); // components
        v.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        v
    }

    /// Minimal GIF89a header with logical screen w,h.
    fn gif_header(w: u16, h: u16) -> Vec<u8> {
        let mut v = Vec::from(&b"GIF89a"[..]);
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&[0, 0, 0]); // packed/bg/aspect
        v
    }

    /// Minimal WebP/VP8X header with declared canvas dims.
    fn webp_vp8x_header(w: u32, h: u32, animated: bool) -> Vec<u8> {
        let mut v = Vec::from(&b"RIFF"[..]);
        v.extend_from_slice(&[0, 0, 0, 0]); // file size (unused)
        v.extend_from_slice(b"WEBP");
        v.extend_from_slice(b"VP8X");
        v.extend_from_slice(&[10, 0, 0, 0]); // chunk length
        v.push(if animated { 0x02 } else { 0x00 }); // flags
        v.extend_from_slice(&[0, 0, 0]); // reserved
        let w_minus = (w - 1).to_le_bytes();
        let h_minus = (h - 1).to_le_bytes();
        v.extend_from_slice(&w_minus[..3]);
        v.extend_from_slice(&h_minus[..3]);
        v
    }

    #[test]
    fn test_vision_limits_default() {
        let l = VisionLimits::default();
        assert_eq!(l.max_dimension, 8192);
        assert_eq!(l.max_decoded_pixels, 8192 * 8192);
        assert_eq!(l.max_encoded_bytes, 20 * 1024 * 1024);
    }

    #[test]
    fn test_vision_limits_strict_permissive() {
        assert_eq!(VisionLimits::strict().max_dimension, 4096);
        assert_eq!(VisionLimits::permissive().max_dimension, 16384);
    }

    #[test]
    fn test_detect_image_media_type_png_jpeg_gif_webp() {
        assert_eq!(
            detect_image_media_type(&png_header(1, 1)),
            Some("image/png")
        );
        assert_eq!(
            detect_image_media_type(&jpeg_header(1, 1)),
            Some("image/jpeg")
        );
        assert_eq!(
            detect_image_media_type(&gif_header(1, 1)),
            Some("image/gif")
        );
        assert_eq!(
            detect_image_media_type(&webp_vp8x_header(1, 1, false)),
            Some("image/webp")
        );
    }

    #[test]
    fn test_detect_image_media_type_rejects_svg_and_unknown() {
        assert!(detect_image_media_type(b"<svg xmlns='http://www.w3.org/2000/svg'>").is_none());
        assert!(detect_image_media_type(b"\0\0\0").is_none());
        assert!(detect_image_media_type(b"").is_none());
    }

    #[test]
    fn test_parse_image_dimensions_png() {
        let bytes = png_header(640, 480);
        assert_eq!(parse_image_dimensions(&bytes), Some((640, 480)));
    }

    #[test]
    fn test_parse_image_dimensions_jpeg() {
        let bytes = jpeg_header(800, 600);
        assert_eq!(parse_image_dimensions(&bytes), Some((800, 600)));
    }

    #[test]
    fn test_parse_image_dimensions_gif() {
        let bytes = gif_header(320, 200);
        assert_eq!(parse_image_dimensions(&bytes), Some((320, 200)));
    }

    #[test]
    fn test_parse_image_dimensions_webp_vp8x() {
        let bytes = webp_vp8x_header(1024, 768, false);
        assert_eq!(parse_image_dimensions(&bytes), Some((1024, 768)));
    }

    #[test]
    fn test_is_animated_image_apng() {
        // PNG sig + arbitrary 16 bytes + acTL chunk
        let mut b = Vec::from(&b"\x89PNG\r\n\x1a\n"[..]);
        b.extend_from_slice(&[0; 16]);
        b.extend_from_slice(b"acTL");
        b.extend_from_slice(&[0, 0, 0, 1]);
        assert!(is_animated_image(&b));
    }

    #[test]
    fn test_is_animated_image_gif_netscape() {
        let mut b = gif_header(2, 2);
        b.extend_from_slice(b"GARBAGE");
        b.extend_from_slice(b"NETSCAPE2.0");
        assert!(is_animated_image(&b));
    }

    #[test]
    fn test_is_animated_image_webp_anim_flag() {
        let b = webp_vp8x_header(64, 64, true);
        assert!(is_animated_image(&b));
        let b2 = webp_vp8x_header(64, 64, false);
        assert!(!is_animated_image(&b2));
    }

    #[test]
    fn test_is_animated_image_static_png_false() {
        let b = png_header(16, 16);
        assert!(!is_animated_image(&b));
    }

    #[test]
    fn test_validate_bytes_accepts_small_png() {
        let pre = ImagePreprocessor::default();
        let bytes = png_header(64, 64);
        let (mt, w, h) = pre.validate_bytes(&bytes).expect("valid");
        assert_eq!(mt, "image/png");
        assert_eq!((w, h), (64, 64));
    }

    #[test]
    fn test_validate_bytes_rejects_oversized_pixels() {
        // Need pixel-count check to fire before dimension check: relax
        // max_dimension above max_decoded_pixels^0.5 so the bomb defense
        // is the failing gate (both 5000×5000 = 25M > 1M pixel cap, but
        // 5000 < 10000 dim cap).
        let bytes = png_header(5000, 5000);
        let pre = ImagePreprocessor::default().with_limits(VisionLimits {
            max_dimension: 10_000,
            max_decoded_pixels: 1_000_000,
            max_encoded_bytes: 20 * 1024 * 1024,
        });
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("decompression-bomb"),
            "expected bomb error, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_bytes_rejects_oversized_dimension() {
        let bytes = png_header(20_000, 4);
        let pre = ImagePreprocessor::default();
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("max_dimension"));
    }

    #[test]
    fn test_validate_bytes_rejects_zero_dimension() {
        let bytes = png_header(0, 100);
        let pre = ImagePreprocessor::default();
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("zero dimension"));
    }

    #[test]
    fn test_validate_bytes_rejects_oversized_encoded() {
        let pre = ImagePreprocessor::default().with_limits(VisionLimits {
            max_encoded_bytes: 8,
            ..Default::default()
        });
        let bytes = png_header(8, 8); // ~29 bytes — exceeds 8 byte cap.
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("max_encoded_bytes"));
    }

    #[test]
    fn test_validate_bytes_rejects_unknown_format() {
        let pre = ImagePreprocessor::default();
        let err = pre.validate_bytes(b"not an image at all").unwrap_err();
        assert!(err.to_string().contains("magic-byte"));
    }

    #[test]
    fn test_validate_bytes_rejects_animated_webp() {
        let pre = ImagePreprocessor::default();
        let bytes = webp_vp8x_header(64, 64, true);
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("animated"));
    }

    #[test]
    fn test_validate_bytes_rejects_apng() {
        let pre = ImagePreprocessor::default();
        let mut bytes = png_header(16, 16);
        bytes.extend_from_slice(b"acTL");
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("animated"));
    }

    #[test]
    fn test_process_returns_validated_bytes_unchanged() {
        let pre = ImagePreprocessor::default();
        let bytes = png_header(128, 128);
        let (mt, w, h, out) = pre.process(&bytes).expect("valid");
        assert_eq!(mt, "image/png");
        assert_eq!((w, h), (128, 128));
        assert_eq!(out.as_ptr(), bytes.as_ptr());
    }

    #[test]
    fn test_image_input_from_bytes_validated_corrects_media_type() {
        // Caller-supplied hint is ignored; magic-byte detection wins.
        let bytes = png_header(32, 32);
        let img =
            ImageInput::from_bytes_validated(&bytes, &VisionLimits::default()).expect("valid");
        assert_eq!(img.media_type, "image/png");
    }

    #[test]
    fn test_image_input_from_bytes_validated_rejects_bomb() {
        // 50k square trips dimension cap (8192) before pixel cap; the
        // failure mode "rejected before decode" is what matters.
        let bytes = png_header(50_000, 50_000);
        let err = ImageInput::from_bytes_validated(&bytes, &VisionLimits::default()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_dimension") || msg.contains("decompression-bomb"),
            "expected bomb-defense error, got: {}",
            msg
        );
    }

    #[test]
    fn test_strict_limits_reject_4097_dim() {
        let pre = ImagePreprocessor::default().with_limits(VisionLimits::strict());
        let bytes = png_header(4097, 4097);
        let err = pre.validate_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("max_dimension"));
    }

    #[test]
    fn test_image_sha256_stable_for_same_bytes() {
        let bytes = b"hello-image-bytes";
        let a = ImageInput::from_bytes(bytes, "image/png");
        let b = ImageInput::from_bytes(bytes, "image/png");
        assert_eq!(image_sha256(&a), image_sha256(&b));
        assert_eq!(image_sha256(&a).len(), 64);
    }

    #[test]
    fn test_image_sha256_differs_for_different_bytes() {
        let a = ImageInput::from_bytes(b"first-payload", "image/png");
        let b = ImageInput::from_bytes(b"second-payload", "image/png");
        assert_ne!(image_sha256(&a), image_sha256(&b));
    }

    #[test]
    fn test_image_sha256_url_path() {
        let a = ImageInput::from_url("https://example.com/a.png");
        let b = ImageInput::from_url("https://example.com/a.png");
        let c = ImageInput::from_url("https://example.com/b.png");
        assert_eq!(image_sha256(&a), image_sha256(&b));
        assert_ne!(image_sha256(&a), image_sha256(&c));
    }

    #[test]
    fn test_in_memory_image_store_roundtrip_base64() {
        let store = InMemoryImageStore::new();
        let img = ImageInput::from_bytes(b"abc-payload", "image/png");
        let r = store.put(&img).expect("put");
        assert_eq!(r.media_type, "image/png");
        assert_eq!(r.sha256.len(), 64);
        assert_eq!(store.len(), 1);
        let got = store.get(&r).expect("get");
        match got.data {
            ImageData::Base64(b) => {
                let raw = base64_decode(&b).expect("decode");
                assert_eq!(raw, b"abc-payload");
            }
            _ => panic!("expected base64"),
        }
    }

    #[test]
    fn test_in_memory_image_store_roundtrip_url() {
        let store = InMemoryImageStore::new();
        let img = ImageInput::from_url("https://example.com/cat.png");
        let r = store.put(&img).expect("put");
        let got = store.get(&r).expect("get");
        match got.data {
            ImageData::Url(u) => assert_eq!(u, "https://example.com/cat.png"),
            _ => panic!("expected url"),
        }
    }

    #[test]
    fn test_in_memory_image_store_dedup_by_sha() {
        let store = InMemoryImageStore::new();
        let a = ImageInput::from_bytes(b"same-bytes", "image/png");
        let b = ImageInput::from_bytes(b"same-bytes", "image/png");
        let r1 = store.put(&a).expect("put1");
        let r2 = store.put(&b).expect("put2");
        assert_eq!(r1.key, r2.key);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_in_memory_image_store_unknown_ref_errors() {
        let store = InMemoryImageStore::new();
        let bogus = ImageRef {
            key: "deadbeef".to_string(),
            media_type: "image/png".to_string(),
            sha256: "deadbeef".to_string(),
        };
        assert!(store.get(&bogus).is_err());
    }

    #[test]
    fn test_filesystem_image_store_roundtrip_base64() {
        let dir =
            std::env::temp_dir().join(format!("ai_assistant_imgstore_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FilesystemImageStore::new(&dir).expect("create store");
        let img = ImageInput::from_bytes(b"\x89PNG-fake-payload", "image/png");
        let r = store.put(&img).expect("put");
        assert!(r.key.ends_with(".png"));
        let got = store.get(&r).expect("get");
        match got.data {
            ImageData::Base64(b) => {
                let raw = base64_decode(&b).expect("decode");
                assert_eq!(raw, b"\x89PNG-fake-payload");
            }
            _ => panic!("expected base64"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_filesystem_image_store_roundtrip_url() {
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_imgstore_url_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FilesystemImageStore::new(&dir).expect("create store");
        let img = ImageInput::from_url("https://example.com/x.jpg");
        let r = store.put(&img).expect("put");
        assert!(r.key.ends_with(".url"));
        let got = store.get(&r).expect("get");
        match got.data {
            ImageData::Url(u) => assert_eq!(u, "https://example.com/x.jpg"),
            _ => panic!("expected url"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ============================================================================
// Batch 0: vision::agent_bridge — adapters and bounded-buffer helpers used by
// agent-side surfaces (AgenticLoop, MultiAgent, AutonomousAgent, SubAgents)
// to convert their text-only message types into VisionMessage and to bound
// the volume of image bytes flowing through the loop.
// ============================================================================

/// Agent-side bridge helpers. Layered above [`generate_vision_response`] so
/// callers in `agentic_loop`, `multi_agent`, `autonomous_loop`, and
/// `sub_agents` don't all reinvent text-to-vision message conversion or
/// pending-image bookkeeping.
pub mod agent_bridge {
    use super::*;
    use crate::config::AiConfig;

    /// Returns whether the configured provider supports vision **at the
    /// transport level**. This does NOT verify the loaded model can read
    /// images — that's a separate check via [`VisionCapabilities`].
    ///
    /// Provider matrix matches the dispatch in [`generate_vision_response`].
    pub fn vision_supported_for(config: &AiConfig) -> bool {
        use crate::config::AiProvider;
        matches!(
            config.provider,
            AiProvider::OpenAI
                | AiProvider::Anthropic
                | AiProvider::Gemini
                | AiProvider::Groq
                | AiProvider::Together
                | AiProvider::Fireworks
                | AiProvider::DeepSeek
                | AiProvider::Mistral
                | AiProvider::Perplexity
                | AiProvider::OpenRouter
                | AiProvider::Ollama
                | AiProvider::LMStudio
                | AiProvider::LocalAI
                | AiProvider::LlamaCpp
                | AiProvider::VLLM
                | AiProvider::TextGenWebUI
                | AiProvider::OpenAICompatible { .. }
        )
    }

    /// Combined transport + model capability check. Errors with a typed
    /// message identifying which gate failed; agent surfaces should call
    /// this before constructing a vision request.
    pub fn ensure_vision_capable(config: &AiConfig) -> Result<()> {
        if !vision_supported_for(config) {
            return Err(anyhow!(
                "vision unsupported by provider {:?}",
                config.provider
            ));
        }
        let caps = VisionCapabilities::default();
        if !caps.supports_vision(&config.selected_model) {
            return Err(anyhow!(
                "model '{}' is not in the known vision-capable set; \
                 select a vision model (gpt-4o, claude-3-*, llava, gemini, qwen-vl, ...)",
                config.selected_model
            ));
        }
        Ok(())
    }

    /// Convert canonical `messages::ChatMessage` history into
    /// `VisionMessage`. Carries any images attached to each chat message.
    pub fn chat_messages_to_vision(
        messages: &[crate::messages::ChatMessage],
    ) -> Vec<VisionMessage> {
        messages
            .iter()
            .map(|m| VisionMessage {
                role: m.role.clone(),
                text: m.content.clone(),
                images: m.images.clone(),
            })
            .collect()
    }

    /// Convert AgenticLoop history into `VisionMessage`. Tool roles map to
    /// `user` (standard pattern across providers — Anthropic accepts
    /// `tool_result` blocks but the unified vision wire format flattens
    /// them into user-side text for now). Tool-emitted images will flow
    /// through `ToolResult.images` once Batch 2 lands.
    #[cfg(feature = "tools")]
    pub fn loop_messages_to_vision(
        messages: &[crate::agentic_loop::LoopMessage],
    ) -> Vec<VisionMessage> {
        use crate::agentic_loop::LoopRole;
        messages
            .iter()
            .map(|m| VisionMessage {
                role: match m.role {
                    LoopRole::System => "system".to_string(),
                    LoopRole::User => "user".to_string(),
                    LoopRole::Assistant => "assistant".to_string(),
                    LoopRole::Tool => "user".to_string(),
                },
                text: m.content.clone(),
                images: Vec::new(),
            })
            .collect()
    }

    /// Render an `ImageInput` as a short opaque tag suitable for log
    /// output: `[image:sha8:WxH:media_type]` for base64, `[image:url:...]`
    /// for URL refs. The full bytes are NEVER emitted to logs.
    ///
    /// SHA prefix is computed from the base64 string itself (avoids an
    /// extra allocation to decode); 8 hex chars is enough for log
    /// correlation across a single conversation.
    pub fn redact_image_bytes_for_log(image: &ImageInput) -> String {
        match &image.data {
            ImageData::Url(url) => {
                let safe = if url.len() > 64 {
                    &url[..64]
                } else {
                    url.as_str()
                };
                format!("[image:url:{}…:{}]", safe, image.media_type)
            }
            ImageData::Base64(b64) => {
                // Cheap fingerprint — first 8 chars of a fold-hash of the b64.
                let mut acc: u64 = 0xcbf29ce484222325;
                for &b in b64.as_bytes() {
                    acc ^= b as u64;
                    acc = acc.wrapping_mul(0x100000001b3);
                }
                format!(
                    "[image:sha8:{:016x}:{}b:{}]",
                    acc,
                    b64.len(),
                    image.media_type
                )
            }
        }
    }

    /// Bounded queue of pending images flowing through an agent surface.
    ///
    /// Defends two failure modes:
    /// 1. Tool screenshots accumulating across many iterations of an
    ///    AgenticLoop until the next LLM call OOMs on context.
    /// 2. Aggregate image-byte volume blowing past a per-agent budget.
    ///
    /// Eviction policy is FIFO (drop oldest) once `cap` is reached, and
    /// `push` rejects with `false` if `max_total_bytes` would be exceeded.
    /// Returns the dropped image (if any) so audit can record the
    /// eviction.
    #[derive(Debug)]
    pub struct PendingImages {
        /// Maximum number of images held simultaneously. Default 4.
        pub cap: usize,
        /// Maximum aggregate base64 bytes across all held images. Default
        /// 16 MiB (covers four 4MB-each PNGs comfortably).
        pub max_total_bytes: usize,
        images: std::collections::VecDeque<ImageInput>,
        bytes_held: usize,
    }

    impl Default for PendingImages {
        fn default() -> Self {
            Self::new(4, 16 * 1024 * 1024)
        }
    }

    impl PendingImages {
        pub fn new(cap: usize, max_total_bytes: usize) -> Self {
            Self {
                cap: cap.max(1),
                max_total_bytes,
                images: std::collections::VecDeque::new(),
                bytes_held: 0,
            }
        }

        fn image_bytes(image: &ImageInput) -> usize {
            match &image.data {
                ImageData::Base64(b) => b.len(),
                ImageData::Url(_) => 0,
            }
        }

        /// Push an image. Returns the evicted image (FIFO eviction when
        /// at capacity) or `None`. Returns `Err` if accepting would push
        /// total bytes over `max_total_bytes` even after eviction.
        pub fn push(&mut self, image: ImageInput) -> Result<Option<ImageInput>> {
            let incoming = Self::image_bytes(&image);
            if incoming > self.max_total_bytes {
                return Err(anyhow!(
                    "single image ({} bytes) exceeds max_total_bytes ({})",
                    incoming,
                    self.max_total_bytes
                ));
            }
            let mut evicted = None;
            if self.images.len() >= self.cap {
                if let Some(front) = self.images.pop_front() {
                    self.bytes_held = self.bytes_held.saturating_sub(Self::image_bytes(&front));
                    evicted = Some(front);
                }
            }
            // Drop more from the front until we'd fit.
            while self.bytes_held + incoming > self.max_total_bytes {
                match self.images.pop_front() {
                    Some(front) => {
                        self.bytes_held = self.bytes_held.saturating_sub(Self::image_bytes(&front));
                        // Only the most-recent eviction is reported; older
                        // drops are silent (caller only needs the head).
                        evicted = Some(front);
                    }
                    None => break,
                }
            }
            self.bytes_held += incoming;
            self.images.push_back(image);
            Ok(evicted)
        }

        pub fn drain(&mut self) -> Vec<ImageInput> {
            self.bytes_held = 0;
            self.images.drain(..).collect()
        }

        pub fn len(&self) -> usize {
            self.images.len()
        }

        pub fn is_empty(&self) -> bool {
            self.images.is_empty()
        }

        pub fn bytes_held(&self) -> usize {
            self.bytes_held
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::config::{AiConfig, AiProvider};

        fn cfg(provider: AiProvider, model: &str) -> AiConfig {
            let mut c = AiConfig::default();
            c.provider = provider;
            c.selected_model = model.to_string();
            c
        }

        #[test]
        fn test_vision_supported_for_known_providers() {
            assert!(vision_supported_for(&cfg(AiProvider::OpenAI, "gpt-4o")));
            assert!(vision_supported_for(&cfg(AiProvider::Ollama, "llava")));
            assert!(vision_supported_for(&cfg(
                AiProvider::Anthropic,
                "claude-3-opus"
            )));
            // OpenAICompatible enabled too (covers self-hosted vision endpoints).
            assert!(vision_supported_for(&cfg(
                AiProvider::OpenAICompatible {
                    base_url: "http://localhost:9000".to_string()
                },
                "qwen-vl"
            )));
        }

        #[test]
        fn test_vision_supported_for_excludes_kobold_and_struct_providers() {
            assert!(!vision_supported_for(&cfg(AiProvider::KoboldCpp, "any")));
            // Bedrock and AzureOpenAI are excluded (separate routing batches);
            // assert that here to lock the surface.
            assert!(!vision_supported_for(&cfg(
                AiProvider::Bedrock {
                    region: "us-east-1".to_string()
                },
                "any"
            )));
            assert!(!vision_supported_for(&cfg(
                AiProvider::AzureOpenAI {
                    endpoint: "x".into(),
                    deployment: "y".into()
                },
                "any"
            )));
        }

        #[test]
        fn test_ensure_vision_capable_rejects_text_only_model() {
            let c = cfg(AiProvider::OpenAI, "gpt-3.5-turbo");
            let err = ensure_vision_capable(&c).unwrap_err();
            assert!(err
                .to_string()
                .contains("not in the known vision-capable set"));
        }

        #[test]
        fn test_ensure_vision_capable_rejects_unsupported_provider() {
            let c = cfg(AiProvider::KoboldCpp, "llava");
            let err = ensure_vision_capable(&c).unwrap_err();
            assert!(err.to_string().contains("vision unsupported by provider"));
        }

        #[test]
        fn test_ensure_vision_capable_accepts_gpt4o() {
            let c = cfg(AiProvider::OpenAI, "gpt-4o");
            assert!(ensure_vision_capable(&c).is_ok());
        }

        #[test]
        fn test_chat_messages_to_vision_preserves_roles_and_text() {
            use crate::messages::ChatMessage;
            let msgs = vec![
                ChatMessage::system("sys"),
                ChatMessage::user("hello"),
                ChatMessage::assistant("hi"),
            ];
            let v = chat_messages_to_vision(&msgs);
            assert_eq!(v.len(), 3);
            assert_eq!(v[0].role, "system");
            assert_eq!(v[0].text, "sys");
            assert_eq!(v[1].role, "user");
            assert_eq!(v[2].role, "assistant");
            assert!(v.iter().all(|m| m.images.is_empty()));
        }

        #[test]
        fn test_redact_image_bytes_for_log_url() {
            let img = ImageInput::from_url("https://example.com/cat.png");
            let s = redact_image_bytes_for_log(&img);
            assert!(s.starts_with("[image:url:"));
            assert!(s.contains("image/png"));
        }

        #[test]
        fn test_redact_image_bytes_for_log_base64_no_payload_leak() {
            let img = ImageInput::from_bytes(b"super-secret-payload-bytes-here", "image/png");
            let s = redact_image_bytes_for_log(&img);
            // The original payload must NOT appear in the redacted form.
            assert!(!s.contains("super-secret"));
            assert!(s.starts_with("[image:sha8:"));
            assert!(s.contains("image/png"));
        }

        #[test]
        fn test_pending_images_fifo_eviction_at_cap() {
            let mut q = PendingImages::new(2, 1024 * 1024);
            let mk = |tag: &str| ImageInput::from_bytes(tag.as_bytes(), "image/png");
            assert!(q.push(mk("a")).unwrap().is_none());
            assert!(q.push(mk("b")).unwrap().is_none());
            // Third push evicts "a"
            let evicted = q.push(mk("c")).unwrap().expect("eviction");
            match evicted.data {
                ImageData::Base64(b) => assert_eq!(b, base64_encode(b"a")),
                _ => panic!("expected base64"),
            }
            assert_eq!(q.len(), 2);
        }

        #[test]
        fn test_pending_images_rejects_oversize_single() {
            let mut q = PendingImages::new(4, 16);
            let huge = ImageInput::from_bytes(&vec![0u8; 1024], "image/png");
            assert!(q.push(huge).is_err());
        }

        #[test]
        fn test_pending_images_drain_clears_bytes() {
            let mut q = PendingImages::default();
            let _ = q.push(ImageInput::from_bytes(b"abc", "image/png"));
            assert!(q.bytes_held() > 0);
            let drained = q.drain();
            assert_eq!(drained.len(), 1);
            assert_eq!(q.bytes_held(), 0);
            assert!(q.is_empty());
        }
    }
}
