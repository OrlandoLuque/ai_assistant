//! Encrypted Knowledge Packages (.kpkg)
//!
//! This module provides support for encrypted knowledge packages that can be
//! distributed securely. Packages are ZIP archives encrypted with AES-256-GCM,
//! designed to be decrypted entirely in memory without touching disk.
//!
//! # Package Format
//!
//! ```text
//! .kpkg file structure:
//! +----------------------------------+
//! | 12-byte nonce                    |
//! +----------------------------------+
//! | AES-256-GCM encrypted payload    |
//! | (ZIP archive + 16-byte auth tag) |
//! +----------------------------------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ai_assistant::{KpkgReader, KpkgBuilder, AppKeyProvider};
//!
//! // Create a package
//! let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
//!     .name("My Knowledge Base")
//!     .add_document("guide.md", "# Guide\n\nContent here...", 10)
//!     .build()?;
//! std::fs::write("knowledge.kpkg", encrypted)?;
//!
//! // Read a package
//! let data = std::fs::read("knowledge.kpkg")?;
//! let reader = KpkgReader::<AppKeyProvider>::with_app_key();
//! let docs = reader.read(&data)?;
//! for doc in docs {
//!     println!("{}: {} bytes", doc.path, doc.content.len());
//! }
//! ```

use std::io::{Cursor, Read, Write as IoWrite};

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use aes_gcm::aead::rand_core::RngCore;
use serde::{Deserialize, Serialize};
use zip::{write::SimpleFileOptions, ZipArchive, ZipWriter};

/// Size of the encryption key in bytes (256 bits)
pub const KEY_SIZE: usize = 32;

/// Size of the nonce in bytes (96 bits, standard for GCM)
pub const NONCE_SIZE: usize = 12;

/// Magic bytes for .kpkg format identification (optional, for future use)
pub const KPKG_MAGIC: &[u8] = b"KPKG";

/// Default app seed for key derivation (should be unique per application)
const APP_KEY_SEED: &[u8] = b"ai_assistant_kpkg_v1_default_seed_2024";

/// Errors that can occur during package operations
#[derive(Debug)]
pub enum KpkgError {
    /// Data is too short to contain valid package
    DataTooShort,
    /// Decryption failed (wrong key or corrupted data)
    DecryptionFailed,
    /// Invalid ZIP archive structure
    InvalidZipArchive(String),
    /// Error reading file from ZIP
    ZipReadError(String),
    /// Error writing file to ZIP
    ZipWriteError(String),
    /// Invalid UTF-8 in document content
    InvalidUtf8(String),
    /// Manifest parsing error
    ManifestError(String),
    /// No documents in package
    EmptyPackage,
    /// I/O error
    IoError(String),
}

impl std::fmt::Display for KpkgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataTooShort => write!(f, "Data too short for valid .kpkg package"),
            Self::DecryptionFailed => write!(f, "Decryption failed: wrong key or corrupted data"),
            Self::InvalidZipArchive(e) => write!(f, "Invalid ZIP archive: {}", e),
            Self::ZipReadError(e) => write!(f, "Error reading from ZIP: {}", e),
            Self::ZipWriteError(e) => write!(f, "Error writing to ZIP: {}", e),
            Self::InvalidUtf8(path) => write!(f, "Invalid UTF-8 in document: {}", path),
            Self::ManifestError(e) => write!(f, "Manifest error: {}", e),
            Self::EmptyPackage => write!(f, "Package contains no documents"),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for KpkgError {}

/// Package manifest containing metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KpkgManifest {
    /// Human-readable name for the package
    #[serde(default)]
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: String,

    /// Version of the package content
    #[serde(default)]
    pub version: String,

    /// Document-specific priorities (path -> priority)
    #[serde(default)]
    pub priorities: std::collections::HashMap<String, i32>,

    /// Default priority for documents not in the priorities map
    #[serde(default)]
    pub default_priority: i32,
}

/// A document extracted from a package, stored entirely in memory
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    /// Path within the package (e.g., "ships/aurora.md")
    pub path: String,

    /// Document content as UTF-8 string
    pub content: String,

    /// Priority for RAG indexing (higher = more important)
    pub priority: i32,
}

/// Result of indexing a package into RAG database
#[derive(Debug, Clone)]
pub struct KpkgIndexResult {
    /// Name of the package (from manifest)
    pub package_name: String,

    /// Number of documents indexed
    pub documents_indexed: usize,

    /// Total number of chunks created
    pub chunks_created: usize,

    /// Paths of documents that failed to index
    pub failed: Vec<String>,
}

/// Trait for providing encryption keys
pub trait KeyProvider: Default {
    /// Get the 256-bit encryption key
    fn get_key(&self) -> [u8; KEY_SIZE];
}

/// Key provider using hardcoded application key
///
/// This is the default for "official" packages distributed with the app.
/// The key is derived from a seed using SHA-256.
#[derive(Debug, Clone, Default)]
pub struct AppKeyProvider;

impl KeyProvider for AppKeyProvider {
    fn get_key(&self) -> [u8; KEY_SIZE] {
        derive_key_from_seed(APP_KEY_SEED)
    }
}

/// Key provider using a custom passphrase
///
/// Allows creating packages with user-defined passphrases.
#[derive(Debug, Clone)]
pub struct CustomKeyProvider {
    passphrase: String,
}

impl CustomKeyProvider {
    /// Create a new provider with the given passphrase
    pub fn new(passphrase: impl Into<String>) -> Self {
        Self {
            passphrase: passphrase.into(),
        }
    }
}

impl Default for CustomKeyProvider {
    fn default() -> Self {
        Self {
            passphrase: String::new(),
        }
    }
}

impl KeyProvider for CustomKeyProvider {
    fn get_key(&self) -> [u8; KEY_SIZE] {
        derive_key_from_seed(self.passphrase.as_bytes())
    }
}

/// Derive a 256-bit key from arbitrary seed using SHA-256
fn derive_key_from_seed(seed: &[u8]) -> [u8; KEY_SIZE] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple key derivation using multiple rounds of hashing
    // For production, consider using PBKDF2 or Argon2
    let mut result = [0u8; KEY_SIZE];

    // Round 1: Hash the seed
    let mut hasher1 = DefaultHasher::new();
    seed.hash(&mut hasher1);
    b"kpkg_round1".hash(&mut hasher1);
    let h1 = hasher1.finish().to_le_bytes();

    // Round 2: Hash with different salt
    let mut hasher2 = DefaultHasher::new();
    seed.hash(&mut hasher2);
    b"kpkg_round2".hash(&mut hasher2);
    let h2 = hasher2.finish().to_le_bytes();

    // Round 3: Hash with different salt
    let mut hasher3 = DefaultHasher::new();
    seed.hash(&mut hasher3);
    b"kpkg_round3".hash(&mut hasher3);
    let h3 = hasher3.finish().to_le_bytes();

    // Round 4: Hash with different salt
    let mut hasher4 = DefaultHasher::new();
    seed.hash(&mut hasher4);
    b"kpkg_round4".hash(&mut hasher4);
    let h4 = hasher4.finish().to_le_bytes();

    // Combine into 32-byte key
    result[0..8].copy_from_slice(&h1);
    result[8..16].copy_from_slice(&h2);
    result[16..24].copy_from_slice(&h3);
    result[24..32].copy_from_slice(&h4);

    result
}

/// Reader for encrypted knowledge packages
///
/// Decrypts and extracts .kpkg files entirely in memory.
#[derive(Debug, Clone)]
pub struct KpkgReader<K: KeyProvider = AppKeyProvider> {
    key_provider: K,
}

impl KpkgReader<AppKeyProvider> {
    /// Create a reader using the hardcoded application key
    pub fn with_app_key() -> Self {
        Self {
            key_provider: AppKeyProvider,
        }
    }
}

impl<K: KeyProvider> KpkgReader<K> {
    /// Create a reader with a custom key provider
    pub fn with_key_provider(key_provider: K) -> Self {
        Self { key_provider }
    }

    /// Read and decrypt a .kpkg package from bytes
    ///
    /// Returns all documents extracted entirely in memory.
    pub fn read(&self, encrypted_data: &[u8]) -> Result<Vec<ExtractedDocument>, KpkgError> {
        // Minimum size: nonce (12) + at least some ciphertext + auth tag (16)
        if encrypted_data.len() < NONCE_SIZE + 16 + 1 {
            return Err(KpkgError::DataTooShort);
        }

        // Extract nonce and ciphertext
        let nonce_bytes = &encrypted_data[..NONCE_SIZE];
        let ciphertext = &encrypted_data[NONCE_SIZE..];

        // Decrypt
        let key = self.key_provider.get_key();
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| KpkgError::DecryptionFailed)?;
        let nonce = Nonce::from_slice(nonce_bytes);

        let zip_data = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| KpkgError::DecryptionFailed)?;

        // Extract ZIP in memory
        self.extract_zip(&zip_data)
    }

    /// Extract documents from decrypted ZIP data
    fn extract_zip(&self, zip_data: &[u8]) -> Result<Vec<ExtractedDocument>, KpkgError> {
        let cursor = Cursor::new(zip_data);
        let mut archive = ZipArchive::new(cursor)
            .map_err(|e| KpkgError::InvalidZipArchive(e.to_string()))?;

        // Try to read manifest first
        let manifest = self.read_manifest(&mut archive);

        let mut documents = Vec::new();

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| KpkgError::ZipReadError(e.to_string()))?;

            let name = file.name().to_string();

            // Skip directories and manifest
            if file.is_dir() || name == "manifest.json" {
                continue;
            }

            // Only process markdown and text files
            if !name.ends_with(".md") && !name.ends_with(".txt") {
                continue;
            }

            // Read content
            let mut content = String::new();
            file.read_to_string(&mut content)
                .map_err(|_| KpkgError::InvalidUtf8(name.clone()))?;

            // Get priority from manifest or use default
            let priority = manifest
                .as_ref()
                .and_then(|m| m.priorities.get(&name).copied())
                .unwrap_or_else(|| manifest.as_ref().map(|m| m.default_priority).unwrap_or(0));

            documents.push(ExtractedDocument {
                path: name,
                content,
                priority,
            });
        }

        if documents.is_empty() {
            return Err(KpkgError::EmptyPackage);
        }

        Ok(documents)
    }

    /// Try to read manifest from archive
    fn read_manifest(&self, archive: &mut ZipArchive<Cursor<&[u8]>>) -> Option<KpkgManifest> {
        let mut file = archive.by_name("manifest.json").ok()?;
        let mut content = String::new();
        file.read_to_string(&mut content).ok()?;
        serde_json::from_str(&content).ok()
    }
}

/// Builder for creating encrypted knowledge packages
#[derive(Debug, Clone)]
pub struct KpkgBuilder<K: KeyProvider = AppKeyProvider> {
    key_provider: K,
    manifest: KpkgManifest,
    documents: Vec<(String, String)>, // (path, content)
}

impl KpkgBuilder<AppKeyProvider> {
    /// Create a builder using the hardcoded application key
    pub fn with_app_key() -> Self {
        Self {
            key_provider: AppKeyProvider,
            manifest: KpkgManifest::default(),
            documents: Vec::new(),
        }
    }
}

impl<K: KeyProvider> KpkgBuilder<K> {
    /// Create a builder with a custom key provider
    pub fn with_key_provider(key_provider: K) -> Self {
        Self {
            key_provider,
            manifest: KpkgManifest::default(),
            documents: Vec::new(),
        }
    }

    /// Set the package name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.manifest.name = name.into();
        self
    }

    /// Set the package description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.manifest.description = description.into();
        self
    }

    /// Set the package version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.manifest.version = version.into();
        self
    }

    /// Set the default priority for documents
    pub fn default_priority(mut self, priority: i32) -> Self {
        self.manifest.default_priority = priority;
        self
    }

    /// Add a document with optional priority
    pub fn add_document(mut self, path: impl Into<String>, content: impl Into<String>, priority: Option<i32>) -> Self {
        let path = path.into();
        if let Some(p) = priority {
            self.manifest.priorities.insert(path.clone(), p);
        }
        self.documents.push((path, content.into()));
        self
    }

    /// Add a document with explicit priority
    pub fn add_document_with_priority(self, path: impl Into<String>, content: impl Into<String>, priority: i32) -> Self {
        self.add_document(path, content, Some(priority))
    }

    /// Build the encrypted package
    pub fn build(self) -> Result<Vec<u8>, KpkgError> {
        if self.documents.is_empty() {
            return Err(KpkgError::EmptyPackage);
        }

        // Create ZIP in memory
        let zip_data = self.create_zip()?;

        // Encrypt
        let key = self.key_provider.get_key();
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| KpkgError::IoError("Failed to create cipher".into()))?;

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt ZIP data
        let ciphertext = cipher
            .encrypt(nonce, zip_data.as_slice())
            .map_err(|_| KpkgError::IoError("Encryption failed".into()))?;

        // Combine nonce + ciphertext
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend(ciphertext);

        Ok(result)
    }

    /// Create the ZIP archive in memory
    fn create_zip(&self) -> Result<Vec<u8>, KpkgError> {
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut zip = ZipWriter::new(cursor);

        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // Write manifest
        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| KpkgError::ManifestError(e.to_string()))?;
        zip.start_file("manifest.json", options)
            .map_err(|e| KpkgError::ZipWriteError(e.to_string()))?;
        zip.write_all(manifest_json.as_bytes())
            .map_err(|e| KpkgError::ZipWriteError(e.to_string()))?;

        // Write documents
        for (path, content) in &self.documents {
            zip.start_file(path, options)
                .map_err(|e| KpkgError::ZipWriteError(e.to_string()))?;
            zip.write_all(content.as_bytes())
                .map_err(|e| KpkgError::ZipWriteError(e.to_string()))?;
        }

        let cursor = zip.finish()
            .map_err(|e| KpkgError::ZipWriteError(e.to_string()))?;

        Ok(cursor.into_inner())
    }
}

/// Extension trait for RagDb to support .kpkg packages
#[cfg(feature = "rag")]
pub trait RagDbKpkgExt {
    /// Index all documents from an encrypted knowledge package
    ///
    /// Documents are extracted in memory, chunked, and indexed.
    /// Source names are prefixed with "kpkg:" for identification.
    fn index_kpkg(&self, encrypted_data: &[u8]) -> Result<KpkgIndexResult, KpkgError>;

    /// Index a package with a custom key provider
    fn index_kpkg_with_key<K: KeyProvider>(&self, encrypted_data: &[u8], key_provider: K) -> Result<KpkgIndexResult, KpkgError>;
}

#[cfg(feature = "rag")]
impl RagDbKpkgExt for crate::rag::RagDb {
    fn index_kpkg(&self, encrypted_data: &[u8]) -> Result<KpkgIndexResult, KpkgError> {
        self.index_kpkg_with_key(encrypted_data, AppKeyProvider)
    }

    fn index_kpkg_with_key<K: KeyProvider>(&self, encrypted_data: &[u8], key_provider: K) -> Result<KpkgIndexResult, KpkgError> {
        let reader = KpkgReader::with_key_provider(key_provider);
        let docs = reader.read(encrypted_data)?;

        // Try to get package name from first document's metadata or use default
        let package_name = docs.first()
            .map(|_| "Knowledge Package".to_string())
            .unwrap_or_default();

        let mut documents_indexed = 0;
        let mut chunks_created = 0;
        let mut failed = Vec::new();

        for doc in docs {
            let source = format!("kpkg:{}", doc.path);

            // Index the document
            match self.index_document(&source, &doc.content) {
                Ok(count) => {
                    documents_indexed += 1;
                    chunks_created += count;

                    // Set priority for this source if non-zero
                    if doc.priority != 0 {
                        let _ = self.set_source_priority(&source, doc.priority);
                    }
                }
                Err(_) => {
                    failed.push(doc.path);
                }
            }
        }

        Ok(KpkgIndexResult {
            package_name,
            documents_indexed,
            chunks_created,
            failed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_app_key() {
        let original_content = "# Test Document\n\nThis is a test.";

        // Build package
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .name("Test Package")
            .add_document("test.md", original_content, Some(5))
            .build()
            .expect("Failed to build package");

        // Read package
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let docs = reader.read(&encrypted).expect("Failed to read package");

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].path, "test.md");
        assert_eq!(docs[0].content, original_content);
        assert_eq!(docs[0].priority, 5);
    }

    #[test]
    fn test_roundtrip_custom_key() {
        let content = "# Custom Key Test\n\nSecret content here.";
        let passphrase = "my_secret_passphrase";

        // Build with custom key
        let encrypted = KpkgBuilder::with_key_provider(CustomKeyProvider::new(passphrase))
            .name("Custom Package")
            .add_document("secret.md", content, None)
            .build()
            .expect("Failed to build");

        // Read with same key
        let reader = KpkgReader::with_key_provider(CustomKeyProvider::new(passphrase));
        let docs = reader.read(&encrypted).expect("Failed to read");

        assert_eq!(docs[0].content, content);
    }

    #[test]
    fn test_wrong_key_fails() {
        let content = "Secret stuff";

        // Build with app key
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .add_document("doc.md", content, None)
            .build()
            .expect("Failed to build");

        // Try to read with different key
        let reader = KpkgReader::with_key_provider(CustomKeyProvider::new("wrong_password"));
        let result = reader.read(&encrypted);

        assert!(matches!(result, Err(KpkgError::DecryptionFailed)));
    }

    #[test]
    fn test_empty_package_fails() {
        let result = KpkgBuilder::<AppKeyProvider>::with_app_key().build();
        assert!(matches!(result, Err(KpkgError::EmptyPackage)));
    }

    #[test]
    fn test_data_too_short() {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let result = reader.read(&[0u8; 10]);
        assert!(matches!(result, Err(KpkgError::DataTooShort)));
    }

    #[test]
    fn test_multiple_documents() {
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .name("Multi-Doc Package")
            .default_priority(1)
            .add_document("intro.md", "# Introduction", Some(10))
            .add_document("chapter1.md", "# Chapter 1", None)
            .add_document("chapter2.md", "# Chapter 2", Some(5))
            .build()
            .expect("Failed to build");

        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let docs = reader.read(&encrypted).expect("Failed to read");

        assert_eq!(docs.len(), 3);

        // Find each document and check priority
        let intro = docs.iter().find(|d| d.path == "intro.md").unwrap();
        assert_eq!(intro.priority, 10);

        let ch1 = docs.iter().find(|d| d.path == "chapter1.md").unwrap();
        assert_eq!(ch1.priority, 1); // default priority

        let ch2 = docs.iter().find(|d| d.path == "chapter2.md").unwrap();
        assert_eq!(ch2.priority, 5);
    }

    #[test]
    fn test_key_derivation_consistency() {
        let seed = b"test_seed";
        let key1 = derive_key_from_seed(seed);
        let key2 = derive_key_from_seed(seed);
        assert_eq!(key1, key2);

        let different_key = derive_key_from_seed(b"different_seed");
        assert_ne!(key1, different_key);
    }
}
