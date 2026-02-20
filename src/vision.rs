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
pub enum ImageData {
    /// Base64 encoded image
    Base64(String),
    /// URL to an image
    Url(String),
}

/// Detail level for image processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Low detail (faster, less tokens)
    Low,
    /// High detail (slower, more tokens)
    High,
    /// Auto (let the model decide)
    #[default]
    Auto,
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
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            max_width: 2048,
            max_height: 2048,
            target_format: "image/jpeg".to_string(),
            quality: 85,
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
        }
    }

    /// Create a preprocessor for high detail mode
    pub fn high_detail() -> Self {
        Self {
            max_width: 2048,
            max_height: 2048,
            target_format: "image/png".to_string(),
            quality: 100,
        }
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
}

/// Simple base64 encoding (no external dependency)
fn base64_encode(data: &[u8]) -> String {
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
}
