//! Cache compression for reduced memory and disk usage
//!
//! This module provides transparent compression for cached data,
//! significantly reducing storage requirements.
//!
//! # Features
//!
//! - **Multiple algorithms**: gzip, deflate, LZ4-style fast compression
//! - **Automatic selection**: Choose best algorithm based on data
//! - **Transparent API**: Compress/decompress automatically
//! - **Streaming support**: Process large data efficiently
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::cache_compression::{CompressedCache, CompressionAlgorithm};
//!
//! let mut cache = CompressedCache::new(CompressionAlgorithm::Gzip);
//!
//! cache.insert("key1", "Some large text data...".to_string());
//! let value = cache.get("key1");
//! ```

use flate2::read::{DeflateDecoder, GzDecoder};
use flate2::write::{DeflateEncoder, GzEncoder};
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

/// Compression algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression (good balance)
    Gzip,
    /// Deflate compression (slightly faster)
    Deflate,
    /// Fast LZ-style compression (fastest)
    Fast,
    /// Best compression ratio
    Best,
    /// Automatic selection based on data
    Auto,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Gzip
    }
}

/// Compression level
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum CompressionLevel {
    /// Fastest compression
    Fast,
    /// Default balance
    Default,
    /// Best compression ratio
    Best,
    /// Custom level (0-9)
    Custom(u32),
}

impl CompressionLevel {
    fn to_flate2(&self) -> Compression {
        match self {
            CompressionLevel::Fast => Compression::fast(),
            CompressionLevel::Default => Compression::default(),
            CompressionLevel::Best => Compression::best(),
            CompressionLevel::Custom(level) => Compression::new(*level),
        }
    }
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::Default
    }
}

/// Compressed data wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    /// The compressed bytes
    pub data: Vec<u8>,
    /// Original size before compression
    pub original_size: usize,
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
}

impl CompressedData {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.data.len() as f64 / self.original_size as f64
    }

    /// Get space saved in bytes
    pub fn space_saved(&self) -> isize {
        self.original_size as isize - self.data.len() as isize
    }

    /// Get space saved percentage
    pub fn space_saved_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - self.compression_ratio()) * 100.0
    }
}

/// Compress data using the specified algorithm
pub fn compress(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: CompressionLevel,
) -> CompressedData {
    let algorithm = if algorithm == CompressionAlgorithm::Auto {
        select_algorithm(data)
    } else {
        algorithm
    };

    let compressed = match algorithm {
        CompressionAlgorithm::None => data.to_vec(),
        CompressionAlgorithm::Gzip => compress_gzip(data, level),
        CompressionAlgorithm::Deflate => compress_deflate(data, level),
        CompressionAlgorithm::Fast => compress_fast(data),
        CompressionAlgorithm::Best => compress_gzip(data, CompressionLevel::Best),
        CompressionAlgorithm::Auto => unreachable!(),
    };

    CompressedData {
        data: compressed,
        original_size: data.len(),
        algorithm,
    }
}

/// Decompress data
pub fn decompress(compressed: &CompressedData) -> Result<Vec<u8>, CompressionError> {
    match compressed.algorithm {
        CompressionAlgorithm::None => Ok(compressed.data.clone()),
        CompressionAlgorithm::Gzip | CompressionAlgorithm::Best => {
            decompress_gzip(&compressed.data)
        }
        CompressionAlgorithm::Deflate => decompress_deflate(&compressed.data),
        CompressionAlgorithm::Fast => decompress_fast(&compressed.data),
        CompressionAlgorithm::Auto => {
            // Try each algorithm
            decompress_gzip(&compressed.data)
                .or_else(|_| decompress_deflate(&compressed.data))
                .or_else(|_| decompress_fast(&compressed.data))
        }
    }
}

fn compress_gzip(data: &[u8], level: CompressionLevel) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), level.to_flate2());
    encoder
        .write_all(data)
        .expect("compression should not fail");
    encoder.finish().expect("compression should not fail")
}

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decoder = GzDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
    Ok(result)
}

fn compress_deflate(data: &[u8], level: CompressionLevel) -> Vec<u8> {
    let mut encoder = DeflateEncoder::new(Vec::new(), level.to_flate2());
    encoder
        .write_all(data)
        .expect("compression should not fail");
    encoder.finish().expect("compression should not fail")
}

fn decompress_deflate(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decoder = DeflateDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
    Ok(result)
}

/// Fast LZ-style compression (simple RLE + dictionary)
fn compress_fast(data: &[u8]) -> Vec<u8> {
    // Simple but fast compression using deflate with fast level
    compress_deflate(data, CompressionLevel::Fast)
}

fn decompress_fast(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    decompress_deflate(data)
}

/// Select best algorithm based on data characteristics
fn select_algorithm(data: &[u8]) -> CompressionAlgorithm {
    if data.len() < 100 {
        // Very small data, compression overhead not worth it
        return CompressionAlgorithm::None;
    }

    // Check entropy/compressibility
    let unique_bytes = {
        let mut seen = [false; 256];
        for &b in data.iter().take(1000) {
            seen[b as usize] = true;
        }
        seen.iter().filter(|&&x| x).count()
    };

    if unique_bytes > 200 {
        // High entropy, likely already compressed
        CompressionAlgorithm::None
    } else if data.len() > 100_000 {
        // Large data, use best compression
        CompressionAlgorithm::Best
    } else {
        // Default to gzip
        CompressionAlgorithm::Gzip
    }
}

/// Compression error types
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CompressionError {
    /// Compression failed
    CompressionFailed(String),
    /// Decompression failed
    DecompressionFailed(String),
    /// Invalid data
    InvalidData(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            CompressionError::DecompressionFailed(msg) => {
                write!(f, "Decompression failed: {}", msg)
            }
            CompressionError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for CompressionError {}

/// A cache that automatically compresses stored values
pub struct CompressedCache<V> {
    algorithm: CompressionAlgorithm,
    level: CompressionLevel,
    data: HashMap<String, CompressedData>,
    stats: CacheCompressionStats,
    _marker: std::marker::PhantomData<V>,
}

impl<V: Serialize + for<'de> Deserialize<'de>> CompressedCache<V> {
    /// Create a new compressed cache
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            level: CompressionLevel::Default,
            data: HashMap::new(),
            stats: CacheCompressionStats::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create with custom compression level
    pub fn with_level(algorithm: CompressionAlgorithm, level: CompressionLevel) -> Self {
        Self {
            algorithm,
            level,
            data: HashMap::new(),
            stats: CacheCompressionStats::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Insert a value
    pub fn insert(&mut self, key: impl Into<String>, value: V) {
        let key = key.into();
        let serialized = serde_json::to_vec(&value).unwrap_or_default();
        let compressed = compress(&serialized, self.algorithm, self.level);

        self.stats.original_bytes += compressed.original_size;
        self.stats.compressed_bytes += compressed.data.len();
        self.stats.items += 1;

        self.data.insert(key, compressed);
    }

    /// Get a value
    pub fn get(&self, key: &str) -> Option<V> {
        let compressed = self.data.get(key)?;
        let decompressed = decompress(compressed).ok()?;
        serde_json::from_slice(&decompressed).ok()
    }

    /// Remove a value
    pub fn remove(&mut self, key: &str) -> Option<V> {
        let compressed = self.data.remove(key)?;

        self.stats.original_bytes = self
            .stats
            .original_bytes
            .saturating_sub(compressed.original_size);
        self.stats.compressed_bytes = self
            .stats
            .compressed_bytes
            .saturating_sub(compressed.data.len());
        self.stats.items = self.stats.items.saturating_sub(1);

        let decompressed = decompress(&compressed).ok()?;
        serde_json::from_slice(&decompressed).ok()
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get number of items
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.data.clear();
        self.stats = CacheCompressionStats::default();
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CacheCompressionStats {
        &self.stats
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
}

/// Statistics for compressed cache
#[derive(Debug, Clone, Default)]
pub struct CacheCompressionStats {
    /// Total original bytes (uncompressed)
    pub original_bytes: usize,
    /// Total compressed bytes
    pub compressed_bytes: usize,
    /// Number of items
    pub items: usize,
}

impl CacheCompressionStats {
    /// Get overall compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.compressed_bytes as f64 / self.original_bytes as f64
    }

    /// Get space saved in bytes
    pub fn space_saved(&self) -> usize {
        self.original_bytes.saturating_sub(self.compressed_bytes)
    }

    /// Get space saved percentage
    pub fn space_saved_percent(&self) -> f64 {
        if self.original_bytes == 0 {
            return 0.0;
        }
        (1.0 - self.compression_ratio()) * 100.0
    }

    /// Get average item size (compressed)
    pub fn avg_compressed_size(&self) -> usize {
        if self.items == 0 {
            return 0;
        }
        self.compressed_bytes / self.items
    }

    /// Get average item size (original)
    pub fn avg_original_size(&self) -> usize {
        if self.items == 0 {
            return 0;
        }
        self.original_bytes / self.items
    }
}

/// Compress a string
pub fn compress_string(s: &str, algorithm: CompressionAlgorithm) -> CompressedData {
    compress(s.as_bytes(), algorithm, CompressionLevel::Default)
}

/// Decompress to string
pub fn decompress_string(compressed: &CompressedData) -> Result<String, CompressionError> {
    let bytes = decompress(compressed)?;
    String::from_utf8(bytes).map_err(|e| CompressionError::InvalidData(e.to_string()))
}

/// Streaming compressor for large data
pub struct StreamingCompressor {
    encoder: GzEncoder<Vec<u8>>,
    bytes_written: usize,
}

impl StreamingCompressor {
    /// Create a new streaming compressor
    pub fn new() -> Self {
        Self {
            encoder: GzEncoder::new(Vec::new(), Compression::default()),
            bytes_written: 0,
        }
    }

    /// Write data to compressor
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let written = self.encoder.write(data)?;
        self.bytes_written += written;
        Ok(written)
    }

    /// Finish compression and get result
    pub fn finish(self) -> std::io::Result<CompressedData> {
        let data = self.encoder.finish()?;
        Ok(CompressedData {
            data,
            original_size: self.bytes_written,
            algorithm: CompressionAlgorithm::Gzip,
        })
    }
}

impl Default for StreamingCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let original =
            "Hello, World! This is a test string that should compress well. ".repeat(100);
        let compressed = compress(
            original.as_bytes(),
            CompressionAlgorithm::Gzip,
            CompressionLevel::Default,
        );

        assert!(compressed.data.len() < original.len());

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(String::from_utf8(decompressed).unwrap(), original);
    }

    #[test]
    fn test_compression_ratio() {
        let original = "aaaaaaaaaa".repeat(1000); // Highly compressible
        let compressed = compress(
            original.as_bytes(),
            CompressionAlgorithm::Best,
            CompressionLevel::Best,
        );

        assert!(compressed.compression_ratio() < 0.1); // Should be very well compressed
        assert!(compressed.space_saved_percent() > 90.0);
    }

    #[test]
    fn test_compressed_cache() {
        let mut cache: CompressedCache<String> = CompressedCache::new(CompressionAlgorithm::Gzip);

        cache.insert("key1", "Hello World!".to_string());
        cache.insert("key2", "Another value".to_string());

        assert_eq!(cache.get("key1"), Some("Hello World!".to_string()));
        assert_eq!(cache.get("key2"), Some("Another value".to_string()));
        assert_eq!(cache.get("key3"), None);
    }

    #[test]
    fn test_auto_algorithm_selection() {
        // Small data
        let small = "Hi";
        assert_eq!(
            select_algorithm(small.as_bytes()),
            CompressionAlgorithm::None
        );

        // Larger text data
        let text = "Hello World! ".repeat(100);
        let algo = select_algorithm(text.as_bytes());
        assert!(algo == CompressionAlgorithm::Gzip || algo == CompressionAlgorithm::Best);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: CompressedCache<String> = CompressedCache::new(CompressionAlgorithm::Gzip);

        let large_value = "x".repeat(10000);
        cache.insert("large", large_value);

        let stats = cache.stats();
        assert!(stats.original_bytes > 0);
        assert!(stats.compressed_bytes > 0);
        assert!(stats.compressed_bytes < stats.original_bytes);
        assert!(stats.space_saved_percent() > 0.0);
    }

    #[test]
    fn test_different_algorithms() {
        let data = "Test data for compression".repeat(100);

        for algo in [
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Deflate,
            CompressionAlgorithm::Fast,
        ] {
            let compressed = compress(data.as_bytes(), algo, CompressionLevel::Default);
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(String::from_utf8(decompressed).unwrap(), data);
        }
    }

    #[test]
    fn test_streaming_compressor() {
        let mut compressor = StreamingCompressor::new();

        compressor.write(b"Hello ").unwrap();
        compressor.write(b"World!").unwrap();

        let compressed = compressor.finish().unwrap();
        assert!(compressed.original_size == 12);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(String::from_utf8(decompressed).unwrap(), "Hello World!");
    }

    #[test]
    fn test_compress_decompress_string_helpers() {
        let original = "Repetitive text for compression testing. ".repeat(50);
        let compressed = compress_string(&original, CompressionAlgorithm::Deflate);
        assert_eq!(compressed.algorithm, CompressionAlgorithm::Deflate);
        assert!(compressed.data.len() < original.len());

        let decompressed = decompress_string(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_cache_remove_and_clear() {
        let mut cache: CompressedCache<String> = CompressedCache::new(CompressionAlgorithm::Gzip);

        cache.insert("a", "value_a".to_string());
        cache.insert("b", "value_b".to_string());
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
        assert!(cache.contains_key("a"));

        // Remove one key and verify stats update
        let removed = cache.remove("a");
        assert_eq!(removed, Some("value_a".to_string()));
        assert!(!cache.contains_key("a"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().items, 1);

        // Clear all and verify everything resets
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().items, 0);
        assert_eq!(cache.stats().original_bytes, 0);
        assert_eq!(cache.stats().compressed_bytes, 0);
    }

    #[test]
    fn test_no_compression_roundtrip_and_stats_averages() {
        let mut cache: CompressedCache<String> =
            CompressedCache::with_level(CompressionAlgorithm::None, CompressionLevel::Fast);

        let val1 = "hello".to_string();
        let val2 = "world".to_string();
        cache.insert("k1", val1.clone());
        cache.insert("k2", val2.clone());

        // With None compression, data passes through unchanged
        assert_eq!(cache.get("k1"), Some(val1));
        assert_eq!(cache.get("k2"), Some(val2));

        let stats = cache.stats();
        assert_eq!(stats.items, 2);
        assert!(stats.avg_compressed_size() > 0);
        assert!(stats.avg_original_size() > 0);
        // None compression means ratio is 1.0 (compressed == original)
        assert!((stats.compression_ratio() - 1.0).abs() < 0.01);
        assert_eq!(stats.space_saved(), 0);

        // Verify keys iterator
        let keys: Vec<&String> = cache.keys().collect();
        assert_eq!(keys.len(), 2);
    }
}
