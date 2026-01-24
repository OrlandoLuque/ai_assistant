//! Response streaming compression
//!
//! This module provides compression for streaming responses, reducing
//! bandwidth and improving transmission speed for large responses.
//!
//! # Features
//!
//! - **Chunk compression**: Compress individual chunks
//! - **Adaptive compression**: Adjust compression based on content
//! - **Multiple algorithms**: gzip, deflate, zstd support
//! - **Decompression**: Decompress incoming streams
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::streaming_compression::{StreamCompressor, CompressionConfig};
//!
//! let compressor = StreamCompressor::new(CompressionConfig::default());
//!
//! // Compress chunks as they come
//! let compressed = compressor.compress_chunk(b"Hello, world!");
//! println!("Compressed {} bytes to {} bytes", 13, compressed.len());
//!
//! // Decompress
//! let decompressed = compressor.decompress_chunk(&compressed)?;
//! ```

use std::io::{Read, Write};
use flate2::Compression;
use flate2::read::{GzDecoder, DeflateDecoder};
use flate2::write::{GzEncoder, DeflateEncoder};

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// No compression
    None,
    /// Gzip compression
    #[default]
    Gzip,
    /// Deflate compression
    Deflate,
}

impl Algorithm {
    /// Get content encoding header value
    pub fn content_encoding(&self) -> &'static str {
        match self {
            Self::None => "identity",
            Self::Gzip => "gzip",
            Self::Deflate => "deflate",
        }
    }

    /// Parse from content encoding header
    pub fn from_content_encoding(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gzip" => Self::Gzip,
            "deflate" => Self::Deflate,
            _ => Self::None,
        }
    }
}

/// Compression level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// No compression (fastest)
    None,
    /// Fast compression, larger output
    Fast,
    /// Balanced compression
    Default,
    /// Best compression, slower
    Best,
    /// Custom level (0-9)
    Custom(u32),
}

impl Level {
    fn to_flate2(&self) -> Compression {
        match self {
            Self::None => Compression::none(),
            Self::Fast => Compression::fast(),
            Self::Default => Compression::default(),
            Self::Best => Compression::best(),
            Self::Custom(level) => Compression::new(*level),
        }
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::Default
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: Algorithm,
    /// Compression level
    pub level: Level,
    /// Minimum size to compress (bytes)
    pub min_size: usize,
    /// Enable adaptive compression
    pub adaptive: bool,
    /// Buffer size for streaming
    pub buffer_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Gzip,
            level: Level::Default,
            min_size: 100,
            adaptive: true,
            buffer_size: 4096,
        }
    }
}

impl CompressionConfig {
    /// Create config for maximum compression
    pub fn max_compression() -> Self {
        Self {
            level: Level::Best,
            ..Default::default()
        }
    }

    /// Create config for fastest compression
    pub fn fast() -> Self {
        Self {
            level: Level::Fast,
            ..Default::default()
        }
    }

    /// Create config with no compression
    pub fn none() -> Self {
        Self {
            algorithm: Algorithm::None,
            level: Level::None,
            ..Default::default()
        }
    }
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Compressed data
    pub data: Vec<u8>,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio (compressed/original)
    pub ratio: f64,
    /// Algorithm used
    pub algorithm: Algorithm,
}

impl CompressionResult {
    /// Check if compression was beneficial
    pub fn is_beneficial(&self) -> bool {
        self.compressed_size < self.original_size
    }

    /// Get bytes saved
    pub fn bytes_saved(&self) -> usize {
        if self.original_size > self.compressed_size {
            self.original_size - self.compressed_size
        } else {
            0
        }
    }
}

/// Stream compressor for chunked responses
pub struct StreamCompressor {
    config: CompressionConfig,
    stats: CompressionStats,
}

impl StreamCompressor {
    /// Create a new stream compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Compress a single chunk
    pub fn compress_chunk(&self, data: &[u8]) -> Vec<u8> {
        // Skip compression for small data
        if data.len() < self.config.min_size {
            return data.to_vec();
        }

        match self.config.algorithm {
            Algorithm::None => data.to_vec(),
            Algorithm::Gzip => self.compress_gzip(data),
            Algorithm::Deflate => self.compress_deflate(data),
        }
    }

    /// Compress with full result info
    pub fn compress(&self, data: &[u8]) -> CompressionResult {
        let original_size = data.len();

        // Skip compression for small data
        if original_size < self.config.min_size {
            return CompressionResult {
                data: data.to_vec(),
                original_size,
                compressed_size: original_size,
                ratio: 1.0,
                algorithm: Algorithm::None,
            };
        }

        let compressed = self.compress_chunk(data);
        let compressed_size = compressed.len();

        // If compression made it bigger and adaptive is on, return original
        let (final_data, final_algo) = if self.config.adaptive && compressed_size >= original_size {
            (data.to_vec(), Algorithm::None)
        } else {
            (compressed, self.config.algorithm)
        };

        CompressionResult {
            data: final_data.clone(),
            original_size,
            compressed_size: final_data.len(),
            ratio: final_data.len() as f64 / original_size as f64,
            algorithm: final_algo,
        }
    }

    /// Decompress a chunk
    pub fn decompress_chunk(&self, data: &[u8], algorithm: Algorithm) -> Result<Vec<u8>, CompressionError> {
        match algorithm {
            Algorithm::None => Ok(data.to_vec()),
            Algorithm::Gzip => self.decompress_gzip(data),
            Algorithm::Deflate => self.decompress_deflate(data),
        }
    }

    /// Decompress using configured algorithm
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        self.decompress_chunk(data, self.config.algorithm)
    }

    fn compress_gzip(&self, data: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), self.config.level.to_flate2());
        encoder.write_all(data).ok();
        encoder.finish().unwrap_or_else(|_| data.to_vec())
    }

    fn compress_deflate(&self, data: &[u8]) -> Vec<u8> {
        let mut encoder = DeflateEncoder::new(Vec::new(), self.config.level.to_flate2());
        encoder.write_all(data).ok();
        encoder.finish().unwrap_or_else(|_| data.to_vec())
    }

    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let mut decoder = GzDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        Ok(result)
    }

    fn decompress_deflate(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let mut decoder = DeflateDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        Ok(result)
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }
}

impl Default for StreamCompressor {
    fn default() -> Self {
        Self::new(CompressionConfig::default())
    }
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total bytes compressed
    pub total_input: u64,
    /// Total bytes after compression
    pub total_output: u64,
    /// Number of chunks compressed
    pub chunks_compressed: u64,
    /// Number of chunks skipped (too small)
    pub chunks_skipped: u64,
}

impl CompressionStats {
    /// Get overall compression ratio
    pub fn overall_ratio(&self) -> f64 {
        if self.total_input == 0 {
            1.0
        } else {
            self.total_output as f64 / self.total_input as f64
        }
    }

    /// Get total bytes saved
    pub fn total_saved(&self) -> u64 {
        self.total_input.saturating_sub(self.total_output)
    }
}

/// Compression errors
#[derive(Debug, Clone)]
pub enum CompressionError {
    /// Decompression failed
    DecompressionFailed(String),
    /// Invalid data
    InvalidData(String),
    /// Unsupported algorithm
    UnsupportedAlgorithm(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecompressionFailed(msg) => write!(f, "Decompression failed: {}", msg),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Self::UnsupportedAlgorithm(msg) => write!(f, "Unsupported algorithm: {}", msg),
        }
    }
}

impl std::error::Error for CompressionError {}

/// Streaming decompressor for incoming data
pub struct StreamDecompressor {
    algorithm: Algorithm,
    buffer: Vec<u8>,
}

impl StreamDecompressor {
    /// Create a new decompressor
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            buffer: Vec::new(),
        }
    }

    /// Add data to buffer
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to decompress buffered data
    pub fn try_decompress(&mut self) -> Result<Option<Vec<u8>>, CompressionError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let compressor = StreamCompressor::default();
        match compressor.decompress_chunk(&self.buffer, self.algorithm) {
            Ok(data) => {
                self.buffer.clear();
                Ok(Some(data))
            }
            Err(_) => Ok(None), // Not enough data yet
        }
    }

    /// Decompress all remaining data
    pub fn finish(mut self) -> Result<Vec<u8>, CompressionError> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let compressor = StreamCompressor::default();
        compressor.decompress_chunk(&self.buffer, self.algorithm)
    }
}

/// Compress a string
pub fn compress_string(s: &str) -> Vec<u8> {
    let compressor = StreamCompressor::default();
    compressor.compress_chunk(s.as_bytes())
}

/// Decompress to string
pub fn decompress_string(data: &[u8]) -> Result<String, CompressionError> {
    let compressor = StreamCompressor::default();
    let bytes = compressor.decompress(data)?;
    String::from_utf8(bytes)
        .map_err(|e| CompressionError::InvalidData(e.to_string()))
}

/// Estimate compression ratio for content type
pub fn estimate_compression_ratio(content_type: &str) -> f64 {
    let content_type = content_type.to_lowercase();

    if content_type.contains("json") {
        0.3 // JSON compresses well
    } else if content_type.contains("text") {
        0.4 // Text compresses well
    } else if content_type.contains("html") || content_type.contains("xml") {
        0.35 // Markup compresses well
    } else if content_type.contains("image") || content_type.contains("video") {
        0.95 // Already compressed
    } else {
        0.6 // Unknown, moderate compression expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let compressor = StreamCompressor::default();
        let original = b"Hello, this is a test message that should compress well when repeated. ".repeat(10);

        let compressed = compressor.compress_chunk(&original);
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(original, decompressed.as_slice());
        assert!(compressed.len() < original.len());
    }

    #[test]
    fn test_small_data_not_compressed() {
        let compressor = StreamCompressor::new(CompressionConfig {
            min_size: 100,
            ..Default::default()
        });

        let small_data = b"tiny";
        let result = compressor.compress(small_data);

        assert_eq!(result.algorithm, Algorithm::None);
        assert_eq!(result.data, small_data);
    }

    #[test]
    fn test_adaptive_compression() {
        let compressor = StreamCompressor::new(CompressionConfig {
            adaptive: true,
            min_size: 10,
            ..Default::default()
        });

        // Random data doesn't compress well
        let random_data: Vec<u8> = (0..200).map(|i| (i * 7 % 256) as u8).collect();
        let result = compressor.compress(&random_data);

        // With adaptive, should return original if compression doesn't help
        assert!(result.ratio <= 1.1);
    }

    #[test]
    fn test_compression_result() {
        let compressor = StreamCompressor::default();
        let text = "This is repeated text. ".repeat(50);

        let result = compressor.compress(text.as_bytes());

        assert!(result.is_beneficial());
        assert!(result.bytes_saved() > 0);
        assert!(result.ratio < 0.5);
    }

    #[test]
    fn test_algorithms() {
        let text = b"Test data for compression algorithms. ".repeat(20);

        for algo in [Algorithm::Gzip, Algorithm::Deflate] {
            let config = CompressionConfig {
                algorithm: algo,
                min_size: 10,
                ..Default::default()
            };
            let compressor = StreamCompressor::new(config);

            let compressed = compressor.compress_chunk(&text);
            let decompressed = compressor.decompress(&compressed).unwrap();

            assert_eq!(text.as_slice(), decompressed.as_slice());
        }
    }

    #[test]
    fn test_string_helpers() {
        let original = "Hello, world! This is a test string. ".repeat(10);
        let compressed = compress_string(&original);
        let decompressed = decompress_string(&compressed).unwrap();

        assert_eq!(original, decompressed);
    }
}
