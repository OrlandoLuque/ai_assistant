//! Internal storage format abstraction.
//!
//! Provides a unified API for serializing and persisting internal data structures.
//! When the `binary-storage` feature is enabled, uses bincode + gzip compression
//! for significantly better performance and smaller files. Falls back to JSON
//! when the feature is not enabled.
//!
//! # Auto-detection
//!
//! `load_internal` and `deserialize_internal` automatically detect the format
//! of existing data (gzip magic bytes for binary, `{`/`[` for JSON), so legacy
//! JSON files continue to work after migration.
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::internal_storage::{save_internal, load_internal};
//! use serde::{Serialize, Deserialize};
//! use std::path::Path;
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct MyData { value: i32 }
//!
//! let data = MyData { value: 42 };
//! save_internal(&data, Path::new("data.bin")).unwrap();
//! let loaded: MyData = load_internal(Path::new("data.bin")).unwrap();
//! assert_eq!(data, loaded);
//! ```

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Detected storage format of a file or byte buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageFormat {
    /// bincode + gzip compressed binary format.
    Binary,
    /// JSON text format (human-readable).
    Json,
}

impl std::fmt::Display for StorageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageFormat::Binary => write!(f, "Binary (bincode+gzip)"),
            StorageFormat::Json => write!(f, "JSON"),
        }
    }
}

/// Metadata about an internal storage file.
#[derive(Debug, Clone)]
pub struct InternalFileInfo {
    /// Path to the file.
    pub path: PathBuf,
    /// Detected storage format.
    pub format: StorageFormat,
    /// Size on disk in bytes.
    pub size_bytes: u64,
    /// Size after decompression (only for Binary format).
    pub uncompressed_bytes: Option<u64>,
    /// Compression ratio (compressed / uncompressed). Lower is better.
    pub compression_ratio: Option<f64>,
}

// ============================================================================
// Format detection
// ============================================================================

/// Gzip magic bytes: 0x1F 0x8B
const GZIP_MAGIC: [u8; 2] = [0x1F, 0x8B];

/// Detect the storage format of a byte slice by inspecting the first bytes.
///
/// - Starts with gzip magic bytes (0x1F 0x8B) → Binary (bincode+gzip)
/// - Starts with `{` or `[` (after trimming whitespace) → Json
/// - Otherwise → Binary (assumes raw bincode without gzip)
pub fn detect_format(bytes: &[u8]) -> StorageFormat {
    if bytes.len() >= 2 && bytes[0] == GZIP_MAGIC[0] && bytes[1] == GZIP_MAGIC[1] {
        return StorageFormat::Binary;
    }
    // Check for JSON start characters (trim leading whitespace/BOM)
    for &b in bytes.iter().take(64) {
        match b {
            b' ' | b'\t' | b'\n' | b'\r' | 0xEF | 0xBB | 0xBF => continue, // whitespace + UTF-8 BOM
            b'{' | b'[' => return StorageFormat::Json,
            _ => break,
        }
    }
    // Default to binary if we can't determine
    StorageFormat::Binary
}

// ============================================================================
// Serialization (in-memory)
// ============================================================================

/// Serialize data to bytes using the optimal internal format.
///
/// With `binary-storage` feature: bincode + gzip compression.
/// Without: JSON (useful for debugging).
pub fn serialize_internal<T: Serialize>(data: &T) -> Result<Vec<u8>> {
    serialize_impl(data)
}

/// Deserialize data from bytes, auto-detecting the format.
///
/// Handles both bincode+gzip and JSON transparently, so data serialized
/// in either format can be read regardless of the current feature flags.
pub fn deserialize_internal<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.is_empty() {
        anyhow::bail!("Cannot deserialize empty data");
    }
    match detect_format(bytes) {
        StorageFormat::Binary => deserialize_binary(bytes),
        StorageFormat::Json => deserialize_json(bytes),
    }
}

// ============================================================================
// File I/O
// ============================================================================

/// Save data to a file using the optimal internal format.
///
/// With `binary-storage` feature: bincode + gzip compression.
/// Without: JSON pretty-printed.
///
/// Creates parent directories if they don't exist.
pub fn save_internal<T: Serialize>(data: &T, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
    }
    let bytes = serialize_internal(data)?;
    std::fs::write(path, &bytes)
        .with_context(|| format!("Failed to write file: {}", path.display()))?;

    debug_dump_impl(data, path);

    Ok(())
}

/// Load data from a file, auto-detecting the format.
///
/// Transparently reads both bincode+gzip (binary) and JSON (legacy) files.
/// This ensures backward compatibility when migrating from JSON to binary storage.
pub fn load_internal<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    deserialize_internal(&bytes)
        .with_context(|| format!("Failed to deserialize file: {}", path.display()))
}

// ============================================================================
// Debug / inspection tools
// ============================================================================

/// Load an internal file and return its contents as pretty-printed JSON.
///
/// Useful for inspecting binary files: `dump_as_json::<ChatSessionStore>(path)`
pub fn dump_as_json<T: Serialize + DeserializeOwned>(path: &Path) -> Result<String> {
    let data: T = load_internal(path)?;
    serde_json::to_string_pretty(&data).context("Failed to serialize to JSON")
}

/// Convert an internal file (binary or JSON) to a `.debug.json` file alongside it.
///
/// Returns the path of the created debug file.
pub fn convert_to_json<T: Serialize + DeserializeOwned>(path: &Path) -> Result<PathBuf> {
    let json = dump_as_json::<T>(path)?;
    let debug_path = append_extension(path, "debug.json");
    std::fs::write(&debug_path, &json)
        .with_context(|| format!("Failed to write debug file: {}", debug_path.display()))?;
    Ok(debug_path)
}

/// Convert a JSON file to the binary internal format.
///
/// Reads JSON from `json_path`, deserializes, then saves as bincode+gzip to `binary_path`.
pub fn convert_json_to_binary<T: Serialize + DeserializeOwned>(
    json_path: &Path,
    binary_path: &Path,
) -> Result<()> {
    let json_str = std::fs::read_to_string(json_path)
        .with_context(|| format!("Failed to read JSON file: {}", json_path.display()))?;
    let data: T = serde_json::from_str(&json_str)
        .with_context(|| format!("Failed to parse JSON: {}", json_path.display()))?;
    save_internal(&data, binary_path)
}

/// Get metadata about an internal storage file.
///
/// Reports format, size on disk, and (for binary files) uncompressed size and
/// compression ratio.
pub fn file_info(path: &Path) -> Result<InternalFileInfo> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let size_bytes = bytes.len() as u64;
    let format = detect_format(&bytes);

    let (uncompressed_bytes, compression_ratio) = match format {
        StorageFormat::Binary => {
            // Try to decompress to get uncompressed size
            match decompress_gzip(&bytes) {
                Ok(decompressed) => {
                    let uncompressed = decompressed.len() as u64;
                    let ratio = if uncompressed > 0 {
                        size_bytes as f64 / uncompressed as f64
                    } else {
                        1.0
                    };
                    (Some(uncompressed), Some(ratio))
                }
                Err(_) => (None, None),
            }
        }
        StorageFormat::Json => (None, None),
    };

    Ok(InternalFileInfo {
        path: path.to_path_buf(),
        format,
        size_bytes,
        uncompressed_bytes,
        compression_ratio,
    })
}

// ============================================================================
// Macros
// ============================================================================

/// In debug builds, save a `.debug.json` copy alongside binary files.
/// In release builds, this is a no-op (zero cost).
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! debug_dump {
    ($data:expr, $path:expr) => {
        if let Ok(json) = serde_json::to_string_pretty($data) {
            let debug_path = $path.with_extension("debug.json");
            let _ = std::fs::write(&debug_path, &json);
        }
    };
}

/// In release builds, this is a no-op.
#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! debug_dump {
    ($data:expr, $path:expr) => {};
}

// ============================================================================
// Private implementation — binary-storage feature
// ============================================================================

#[cfg(feature = "binary-storage")]
fn serialize_impl<T: Serialize>(data: &T) -> Result<Vec<u8>> {
    let bincode_bytes = bincode::serialize(data).context("bincode serialization failed")?;
    compress_gzip(&bincode_bytes)
}

#[cfg(not(feature = "binary-storage"))]
fn serialize_impl<T: Serialize>(data: &T) -> Result<Vec<u8>> {
    let json = serde_json::to_string_pretty(data).context("JSON serialization failed")?;
    Ok(json.into_bytes())
}

fn deserialize_binary<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    // Try gzip decompression first
    let decompressed = match decompress_gzip(bytes) {
        Ok(d) => d,
        Err(_) => {
            // Might be raw bincode without gzip
            bytes.to_vec()
        }
    };

    // Try bincode first
    #[cfg(feature = "binary-storage")]
    {
        match bincode::deserialize::<T>(&decompressed) {
            Ok(data) => return Ok(data),
            Err(_) => {
                // Fall through to JSON attempt
            }
        }
    }

    // Last resort: try JSON even though it didn't look like JSON
    serde_json::from_slice(&decompressed)
        .context("Failed to deserialize binary data (tried bincode and JSON)")
}

fn deserialize_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).context("Failed to deserialize JSON data")
}

fn compress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(data)
        .context("gzip compression write failed")?;
    encoder.finish().context("gzip compression finish failed")
}

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .context("gzip decompression failed")?;
    Ok(decompressed)
}

/// In debug builds, write a .debug.json alongside saved files.
#[cfg(debug_assertions)]
fn debug_dump_impl<T: Serialize>(data: &T, path: &Path) {
    if let Ok(json) = serde_json::to_string_pretty(data) {
        let debug_path = append_extension(path, "debug.json");
        let _ = std::fs::write(&debug_path, &json);
    }
}

#[cfg(not(debug_assertions))]
fn debug_dump_impl<T: Serialize>(_data: &T, _path: &Path) {
    // No-op in release
}

/// Append an extension to a path without replacing the existing one.
/// E.g., "data.bin" → "data.bin.debug.json"
fn append_extension(path: &Path, ext: &str) -> PathBuf {
    let mut new_path = path.as_os_str().to_owned();
    new_path.push(".");
    new_path.push(ext);
    PathBuf::from(new_path)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestData {
        name: String,
        values: Vec<i32>,
        nested: Option<TestNested>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestNested {
        flag: bool,
        score: f64,
    }

    fn sample_data() -> TestData {
        TestData {
            name: "test_data".to_string(),
            values: vec![1, 2, 3, 42, 100],
            nested: Some(TestNested {
                flag: true,
                score: 0.95,
            }),
        }
    }

    #[test]
    fn test_serialize_deserialize_round_trip() {
        let data = sample_data();
        let bytes = serialize_internal(&data).unwrap();
        let loaded: TestData = deserialize_internal(&bytes).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_save_load_round_trip() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        save_internal(&data, &path).unwrap();
        assert!(path.exists());

        let loaded: TestData = load_internal(&path).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_load_json_legacy() {
        // Simulate a legacy JSON file
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.json");

        let json = serde_json::to_string_pretty(&data).unwrap();
        std::fs::write(&path, &json).unwrap();

        // load_internal should auto-detect JSON
        let loaded: TestData = load_internal(&path).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_detect_format_json_object() {
        assert_eq!(detect_format(b"{\"key\": 1}"), StorageFormat::Json);
    }

    #[test]
    fn test_detect_format_json_array() {
        assert_eq!(detect_format(b"[1, 2, 3]"), StorageFormat::Json);
    }

    #[test]
    fn test_detect_format_json_with_whitespace() {
        assert_eq!(detect_format(b"  \n  {\"key\": 1}"), StorageFormat::Json);
    }

    #[test]
    fn test_detect_format_gzip() {
        assert_eq!(
            detect_format(&[0x1F, 0x8B, 0x08, 0x00]),
            StorageFormat::Binary
        );
    }

    #[test]
    fn test_detect_format_raw_binary() {
        assert_eq!(detect_format(&[0x00, 0x01, 0x02]), StorageFormat::Binary);
    }

    #[test]
    fn test_detect_format_empty() {
        assert_eq!(detect_format(&[]), StorageFormat::Binary);
    }

    #[test]
    fn test_deserialize_empty_returns_error() {
        let result = deserialize_internal::<TestData>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_creates_parent_directories() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("deep").join("test.bin");

        save_internal(&data, &path).unwrap();
        assert!(path.exists());

        let loaded: TestData = load_internal(&path).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_load_nonexistent_file_returns_error() {
        let result = load_internal::<TestData>(Path::new("/nonexistent/path/file.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_dump_as_json() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        save_internal(&data, &path).unwrap();
        let json_str = dump_as_json::<TestData>(&path).unwrap();

        // Verify it's valid JSON
        let from_json: TestData = serde_json::from_str(&json_str).unwrap();
        assert_eq!(data, from_json);
    }

    #[test]
    fn test_convert_to_json_creates_debug_file() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        save_internal(&data, &path).unwrap();
        let debug_path = convert_to_json::<TestData>(&path).unwrap();

        assert!(debug_path.exists());
        let json_str = std::fs::read_to_string(&debug_path).unwrap();
        let from_json: TestData = serde_json::from_str(&json_str).unwrap();
        assert_eq!(data, from_json);
    }

    #[test]
    fn test_convert_json_to_binary_round_trip() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("source.json");
        let binary_path = dir.path().join("target.bin");

        // Write JSON source
        let json = serde_json::to_string_pretty(&data).unwrap();
        std::fs::write(&json_path, &json).unwrap();

        // Convert to binary
        convert_json_to_binary::<TestData>(&json_path, &binary_path).unwrap();

        // Load from binary
        let loaded: TestData = load_internal(&binary_path).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_file_info_json() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");

        let json = serde_json::to_string_pretty(&data).unwrap();
        std::fs::write(&path, &json).unwrap();

        let info = file_info(&path).unwrap();
        assert_eq!(info.format, StorageFormat::Json);
        assert_eq!(info.size_bytes, json.len() as u64);
        assert!(info.uncompressed_bytes.is_none());
        assert!(info.compression_ratio.is_none());
    }

    #[test]
    fn test_file_info_binary() {
        let data = sample_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        save_internal(&data, &path).unwrap();
        let info = file_info(&path).unwrap();

        // Format depends on feature flag
        #[cfg(feature = "binary-storage")]
        {
            assert_eq!(info.format, StorageFormat::Binary);
            assert!(info.uncompressed_bytes.is_some());
            assert!(info.compression_ratio.is_some());
        }

        assert!(info.size_bytes > 0);
    }

    #[test]
    fn test_storage_format_display() {
        assert_eq!(StorageFormat::Binary.to_string(), "Binary (bincode+gzip)");
        assert_eq!(StorageFormat::Json.to_string(), "JSON");
    }

    #[test]
    fn test_large_data_round_trip() {
        // Test with larger data to exercise compression
        let data = TestData {
            name: "x".repeat(10000),
            values: (0..1000).collect(),
            nested: Some(TestNested {
                flag: false,
                score: 3.14159,
            }),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");

        save_internal(&data, &path).unwrap();
        let loaded: TestData = load_internal(&path).unwrap();
        assert_eq!(data, loaded);

        // With binary-storage, compressed file should be smaller than JSON
        #[cfg(feature = "binary-storage")]
        {
            let json_size = serde_json::to_string(&data).unwrap().len();
            let file_size = std::fs::metadata(&path).unwrap().len() as usize;
            assert!(file_size < json_size, "Binary should be smaller than JSON");
        }
    }

    #[test]
    fn test_serialize_deserialize_simple_types() {
        // String
        let s = "hello world".to_string();
        let bytes = serialize_internal(&s).unwrap();
        let loaded: String = deserialize_internal(&bytes).unwrap();
        assert_eq!(s, loaded);

        // Vec
        let v: Vec<u32> = vec![1, 2, 3, 4, 5];
        let bytes = serialize_internal(&v).unwrap();
        let loaded: Vec<u32> = deserialize_internal(&bytes).unwrap();
        assert_eq!(v, loaded);
    }

    #[test]
    fn test_append_extension() {
        let path = Path::new("data.bin");
        let result = append_extension(path, "debug.json");
        assert!(result.to_str().unwrap().ends_with("data.bin.debug.json"));
    }
}
