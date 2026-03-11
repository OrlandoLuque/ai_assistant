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

use aes_gcm::aead::rand_core::RngCore;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use serde::{Deserialize, Serialize};
use zip::{write::SimpleFileOptions, ZipArchive, ZipWriter};

/// Size of the encryption key in bytes (256 bits)
pub const KEY_SIZE: usize = 32;

/// Size of the nonce in bytes (96 bits, standard for GCM)
pub const NONCE_SIZE: usize = 12;

/// Magic bytes for .kpkg format identification (optional, for future use)
pub const KPKG_MAGIC: &[u8] = b"KPKG";

/// Default app seed for key derivation.
///
/// **SECURITY WARNING**: This seed is embedded in the source code. Packages
/// encrypted with `AppKeyProvider` are NOT confidential — any build of this
/// binary can decrypt them. Use `CustomKeyProvider` with a user-supplied
/// passphrase for real confidentiality.
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

/// Example pair for few-shot learning
///
/// Used to provide example input/output pairs that help the AI understand
/// the expected format and style of responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExamplePair {
    /// The example input/question
    pub input: String,

    /// The expected output/response
    pub output: String,

    /// Optional category for organizing examples
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

impl ExamplePair {
    /// Create a new example pair
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            category: None,
        }
    }

    /// Create a new example pair with category
    pub fn with_category(
        input: impl Into<String>,
        output: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            category: Some(category.into()),
        }
    }

    /// Format the example for prompt injection
    pub fn format_for_prompt(&self) -> String {
        format!("User: {}\nAssistant: {}", self.input, self.output)
    }
}

/// RAG configuration specific to a knowledge package
///
/// Allows packages to specify their preferred RAG settings that override
/// or merge with the global configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RagPackageConfig {
    /// Preferred chunk size in tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,

    /// Overlap between chunks in tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_overlap: Option<usize>,

    /// Number of top results to retrieve
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Maximum context tokens to include
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<usize>,

    /// Minimum relevance score (0.0 to 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_relevance: Option<f32>,

    /// Chunking strategy: "sentence", "paragraph", "fixed", "semantic"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunking_strategy: Option<String>,

    /// Whether to use hybrid search (BM25 + semantic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_hybrid_search: Option<bool>,

    /// Priority boost for all documents in this package
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_boost: Option<i32>,
}

impl RagPackageConfig {
    /// Check if this config has any non-default values
    pub fn is_empty(&self) -> bool {
        self.chunk_size.is_none()
            && self.chunk_overlap.is_none()
            && self.top_k.is_none()
            && self.max_context_tokens.is_none()
            && self.min_relevance.is_none()
            && self.chunking_strategy.is_none()
            && self.use_hybrid_search.is_none()
            && self.priority_boost.is_none()
    }
}

/// Metadata about the knowledge package
///
/// Contains authorship, licensing, and organizational information.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct KpkgMetadata {
    /// Author or creator of the package
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,

    /// ISO 8601 timestamp of creation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// ISO 8601 timestamp of last update
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,

    /// Tags for categorization
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Language code (e.g., "en", "es", "zh")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// License identifier (e.g., "MIT", "CC-BY-4.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// URL for more information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Custom metadata fields
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub custom: std::collections::HashMap<String, String>,
}

impl KpkgMetadata {
    /// Check if this metadata has any non-default values
    pub fn is_empty(&self) -> bool {
        self.author.is_none()
            && self.created_at.is_none()
            && self.updated_at.is_none()
            && self.tags.is_empty()
            && self.language.is_none()
            && self.license.is_none()
            && self.url.is_none()
            && self.custom.is_empty()
    }
}

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

    // === New fields for professional KPKG ===
    /// System prompt to use with this knowledge base
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Persona description for the AI assistant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,

    /// Few-shot examples for response formatting
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<ExamplePair>,

    /// RAG configuration for this package
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rag_config: Option<RagPackageConfig>,

    /// Package metadata (author, license, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<KpkgMetadata>,
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

/// Extended result that includes the manifest
///
/// Use this when you need access to package metadata like system_prompt,
/// persona, examples, and rag_config after indexing.
#[derive(Debug, Clone)]
pub struct KpkgIndexResultExt {
    /// Basic index result
    pub result: KpkgIndexResult,

    /// Full manifest with all metadata
    pub manifest: KpkgManifest,
}

impl KpkgIndexResultExt {
    /// Get the number of documents indexed
    pub fn documents_indexed(&self) -> usize {
        self.result.documents_indexed
    }

    /// Get the number of chunks created
    pub fn chunks_created(&self) -> usize {
        self.result.chunks_created
    }

    /// Check if there were any failures
    pub fn has_failures(&self) -> bool {
        !self.result.failed.is_empty()
    }

    /// Get the system prompt if available
    pub fn system_prompt(&self) -> Option<&str> {
        self.manifest.system_prompt.as_deref()
    }

    /// Get the persona if available
    pub fn persona(&self) -> Option<&str> {
        self.manifest.persona.as_deref()
    }

    /// Get the examples for few-shot learning
    pub fn examples(&self) -> &[ExamplePair] {
        &self.manifest.examples
    }

    /// Get the RAG config if specified
    pub fn rag_config(&self) -> Option<&RagPackageConfig> {
        self.manifest.rag_config.as_ref()
    }

    /// Get the metadata if available
    pub fn metadata(&self) -> Option<&KpkgMetadata> {
        self.manifest.metadata.as_ref()
    }

    /// Format all examples for prompt injection
    pub fn format_examples_for_prompt(&self) -> String {
        if self.manifest.examples.is_empty() {
            return String::new();
        }

        self.manifest
            .examples
            .iter()
            .map(|ex| ex.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Build a combined system prompt from manifest fields
    ///
    /// Combines system_prompt and persona into a single prompt.
    pub fn build_effective_system_prompt(&self) -> Option<String> {
        match (&self.manifest.system_prompt, &self.manifest.persona) {
            (Some(sys), Some(persona)) => Some(format!("{}\n\nPersona: {}", sys, persona)),
            (Some(sys), None) => Some(sys.clone()),
            (None, Some(persona)) => Some(format!("Persona: {}", persona)),
            (None, None) => None,
        }
    }
}

/// Trait for providing encryption keys
pub trait KeyProvider: Default {
    /// Get the 256-bit encryption key
    fn get_key(&self) -> [u8; KEY_SIZE];
}

/// Key provider using a built-in application key.
///
/// Intended for "official" packages distributed alongside the application binary.
/// The key is derived from a seed embedded in the source code.
///
/// **SECURITY WARNING**: This provides **obfuscation, not confidentiality**.
/// Anyone with access to this binary or source can decrypt packages created
/// with `AppKeyProvider`. For real encryption, use [`CustomKeyProvider`] with
/// a user-supplied passphrase.
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

/// Derive a 256-bit key from arbitrary seed using SHA-256 HKDF-like construction
fn derive_key_from_seed(seed: &[u8]) -> [u8; KEY_SIZE] {
    use crate::request_signing::sha256::sha256;

    // HKDF-like: extract then expand
    let mut salted = b"kpkg_key_derivation_v2".to_vec();
    salted.extend_from_slice(seed);
    let prk = sha256(&salted);

    // Expand
    let mut expand_input = prk.to_vec();
    expand_input.extend_from_slice(b"kpkg_expand");
    expand_input.push(0x01);
    sha256(&expand_input)
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
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| KpkgError::DecryptionFailed)?;
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
        let mut archive =
            ZipArchive::new(cursor).map_err(|e| KpkgError::InvalidZipArchive(e.to_string()))?;

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

    /// Decrypt and read only the manifest from a package
    ///
    /// Useful for inspecting package metadata without loading all documents.
    pub fn read_manifest_only(&self, encrypted_data: &[u8]) -> Result<KpkgManifest, KpkgError> {
        if encrypted_data.len() < NONCE_SIZE + 16 + 1 {
            return Err(KpkgError::DataTooShort);
        }

        let nonce_bytes = &encrypted_data[..NONCE_SIZE];
        let ciphertext = &encrypted_data[NONCE_SIZE..];

        let key = self.key_provider.get_key();
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| KpkgError::DecryptionFailed)?;
        let nonce = Nonce::from_slice(nonce_bytes);

        let zip_data = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| KpkgError::DecryptionFailed)?;

        let cursor = Cursor::new(&zip_data[..]);
        let mut archive =
            ZipArchive::new(cursor).map_err(|e| KpkgError::InvalidZipArchive(e.to_string()))?;

        self.read_manifest(&mut archive)
            .ok_or_else(|| KpkgError::ManifestError("Manifest not found or invalid".into()))
    }

    /// Read and decrypt a package, returning both documents and manifest
    ///
    /// Returns a tuple of (documents, manifest) for cases where both are needed.
    pub fn read_with_manifest(
        &self,
        encrypted_data: &[u8],
    ) -> Result<(Vec<ExtractedDocument>, KpkgManifest), KpkgError> {
        if encrypted_data.len() < NONCE_SIZE + 16 + 1 {
            return Err(KpkgError::DataTooShort);
        }

        let nonce_bytes = &encrypted_data[..NONCE_SIZE];
        let ciphertext = &encrypted_data[NONCE_SIZE..];

        let key = self.key_provider.get_key();
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| KpkgError::DecryptionFailed)?;
        let nonce = Nonce::from_slice(nonce_bytes);

        let zip_data = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| KpkgError::DecryptionFailed)?;

        let cursor = Cursor::new(&zip_data[..]);
        let mut archive =
            ZipArchive::new(cursor).map_err(|e| KpkgError::InvalidZipArchive(e.to_string()))?;

        // Read manifest first
        let manifest = self
            .read_manifest(&mut archive)
            .ok_or_else(|| KpkgError::ManifestError("Manifest not found or invalid".into()))?;

        // Extract documents
        let mut documents = Vec::new();

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| KpkgError::ZipReadError(e.to_string()))?;

            let name = file.name().to_string();

            if file.is_dir() || name == "manifest.json" {
                continue;
            }

            if !name.ends_with(".md") && !name.ends_with(".txt") {
                continue;
            }

            let mut content = String::new();
            file.read_to_string(&mut content)
                .map_err(|_| KpkgError::InvalidUtf8(name.clone()))?;

            let priority = manifest
                .priorities
                .get(&name)
                .copied()
                .unwrap_or(manifest.default_priority);

            documents.push(ExtractedDocument {
                path: name,
                content,
                priority,
            });
        }

        if documents.is_empty() {
            return Err(KpkgError::EmptyPackage);
        }

        Ok((documents, manifest))
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

    /// Set the system prompt for this knowledge base
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.manifest.system_prompt = Some(prompt.into());
        self
    }

    /// Set the persona description
    pub fn persona(mut self, persona: impl Into<String>) -> Self {
        self.manifest.persona = Some(persona.into());
        self
    }

    /// Add a few-shot example
    pub fn add_example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.manifest.examples.push(ExamplePair::new(input, output));
        self
    }

    /// Add a few-shot example with category
    pub fn add_example_with_category(
        mut self,
        input: impl Into<String>,
        output: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        self.manifest
            .examples
            .push(ExamplePair::with_category(input, output, category));
        self
    }

    /// Set the chunk size for RAG
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .chunk_size = Some(size);
        self
    }

    /// Set the chunk overlap for RAG
    pub fn chunk_overlap(mut self, overlap: usize) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .chunk_overlap = Some(overlap);
        self
    }

    /// Set the top_k for RAG retrieval
    pub fn top_k(mut self, k: usize) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .top_k = Some(k);
        self
    }

    /// Set the max context tokens for RAG
    pub fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .max_context_tokens = Some(tokens);
        self
    }

    /// Set the minimum relevance score for RAG
    pub fn min_relevance(mut self, score: f32) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .min_relevance = Some(score);
        self
    }

    /// Set the chunking strategy
    pub fn chunking_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .chunking_strategy = Some(strategy.into());
        self
    }

    /// Enable or disable hybrid search
    pub fn use_hybrid_search(mut self, enabled: bool) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .use_hybrid_search = Some(enabled);
        self
    }

    /// Set priority boost for all documents in package
    pub fn priority_boost(mut self, boost: i32) -> Self {
        self.manifest
            .rag_config
            .get_or_insert_with(RagPackageConfig::default)
            .priority_boost = Some(boost);
        self
    }

    /// Set the author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .author = Some(author.into());
        self
    }

    /// Set the language code
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .language = Some(lang.into());
        self
    }

    /// Set the license
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .license = Some(license.into());
        self
    }

    /// Set the URL
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .url = Some(url.into());
        self
    }

    /// Add a tag
    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .tags
            .push(tag.into());
        self
    }

    /// Add a custom metadata field
    pub fn add_custom_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .custom
            .insert(key.into(), value.into());
        self
    }

    /// Set created_at timestamp (ISO 8601)
    pub fn created_at(mut self, timestamp: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .created_at = Some(timestamp.into());
        self
    }

    /// Set updated_at timestamp (ISO 8601)
    pub fn updated_at(mut self, timestamp: impl Into<String>) -> Self {
        self.manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default)
            .updated_at = Some(timestamp.into());
        self
    }

    /// Set timestamps to current time
    pub fn with_current_timestamps(mut self) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let metadata = self
            .manifest
            .metadata
            .get_or_insert_with(KpkgMetadata::default);
        if metadata.created_at.is_none() {
            metadata.created_at = Some(now.clone());
        }
        metadata.updated_at = Some(now);
        self
    }

    /// Add a document with optional priority
    pub fn add_document(
        mut self,
        path: impl Into<String>,
        content: impl Into<String>,
        priority: Option<i32>,
    ) -> Self {
        let path = path.into();
        if let Some(p) = priority {
            self.manifest.priorities.insert(path.clone(), p);
        }
        self.documents.push((path, content.into()));
        self
    }

    /// Add a document with explicit priority
    pub fn add_document_with_priority(
        self,
        path: impl Into<String>,
        content: impl Into<String>,
        priority: i32,
    ) -> Self {
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

        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

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

        let cursor = zip
            .finish()
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
    fn index_kpkg_with_key<K: KeyProvider>(
        &self,
        encrypted_data: &[u8],
        key_provider: K,
    ) -> Result<KpkgIndexResult, KpkgError>;

    /// Index a package and return extended result with manifest
    ///
    /// Use this when you need access to manifest fields like system_prompt,
    /// persona, examples, and rag_config after indexing.
    fn index_kpkg_ext(&self, encrypted_data: &[u8]) -> Result<KpkgIndexResultExt, KpkgError>;

    /// Index a package with custom key and return extended result
    fn index_kpkg_ext_with_key<K: KeyProvider>(
        &self,
        encrypted_data: &[u8],
        key_provider: K,
    ) -> Result<KpkgIndexResultExt, KpkgError>;
}

#[cfg(feature = "rag")]
impl RagDbKpkgExt for crate::rag::RagDb {
    fn index_kpkg(&self, encrypted_data: &[u8]) -> Result<KpkgIndexResult, KpkgError> {
        self.index_kpkg_with_key(encrypted_data, AppKeyProvider)
    }

    fn index_kpkg_with_key<K: KeyProvider>(
        &self,
        encrypted_data: &[u8],
        key_provider: K,
    ) -> Result<KpkgIndexResult, KpkgError> {
        let reader = KpkgReader::with_key_provider(key_provider);
        let docs = reader.read(encrypted_data)?;

        // Try to get package name from first document's metadata or use default
        let package_name = docs
            .first()
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

    fn index_kpkg_ext(&self, encrypted_data: &[u8]) -> Result<KpkgIndexResultExt, KpkgError> {
        self.index_kpkg_ext_with_key(encrypted_data, AppKeyProvider)
    }

    fn index_kpkg_ext_with_key<K: KeyProvider>(
        &self,
        encrypted_data: &[u8],
        key_provider: K,
    ) -> Result<KpkgIndexResultExt, KpkgError> {
        let reader = KpkgReader::with_key_provider(key_provider);
        let (docs, manifest) = reader.read_with_manifest(encrypted_data)?;

        let package_name = manifest.name.clone();

        let mut documents_indexed = 0;
        let mut chunks_created = 0;
        let mut failed = Vec::new();

        // Get priority boost from rag_config
        let priority_boost = manifest
            .rag_config
            .as_ref()
            .and_then(|c| c.priority_boost)
            .unwrap_or(0);

        for doc in docs {
            let source = format!("kpkg:{}", doc.path);

            match self.index_document(&source, &doc.content) {
                Ok(count) => {
                    documents_indexed += 1;
                    chunks_created += count;

                    // Apply priority from manifest + boost
                    let effective_priority = doc.priority + priority_boost;
                    if effective_priority != 0 {
                        let _ = self.set_source_priority(&source, effective_priority);
                    }
                }
                Err(_) => {
                    failed.push(doc.path);
                }
            }
        }

        Ok(KpkgIndexResultExt {
            result: KpkgIndexResult {
                package_name,
                documents_indexed,
                chunks_created,
                failed,
            },
            manifest,
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
        // Same seed must produce the same key
        let seed = b"test_seed";
        let key1 = derive_key_from_seed(seed);
        let key2 = derive_key_from_seed(seed);
        assert_eq!(key1, key2);

        // Different seed must produce a different key
        let different_key = derive_key_from_seed(b"different_seed");
        assert_ne!(key1, different_key);
    }

    #[test]
    fn test_key_derivation_avalanche() {
        // Changing a single byte in the seed should change the key substantially
        let seed_a = b"avalanche_test_seed_0";
        let seed_b = b"avalanche_test_seed_1"; // differs by 1 byte (last char)

        let key_a = derive_key_from_seed(seed_a);
        let key_b = derive_key_from_seed(seed_b);

        // Count differing bits between the two keys
        let differing_bits: u32 = key_a
            .iter()
            .zip(key_b.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        // With 256 bits total, a good hash should flip ~50% of bits.
        // We assert > 25% (64 bits) differ to allow margin.
        assert!(
            differing_bits > 64,
            "Avalanche effect too weak: only {} of 256 bits differ",
            differing_bits
        );
    }

    // === New tests for professional KPKG features ===

    #[test]
    fn test_example_pair_creation() {
        let ex1 = ExamplePair::new("What is rust?", "Rust is a systems programming language.");
        assert_eq!(ex1.input, "What is rust?");
        assert_eq!(ex1.output, "Rust is a systems programming language.");
        assert!(ex1.category.is_none());

        let ex2 =
            ExamplePair::with_category("How do I compile?", "Use cargo build.", "compilation");
        assert_eq!(ex2.category, Some("compilation".to_string()));
    }

    #[test]
    fn test_example_pair_format() {
        let ex = ExamplePair::new("Hello", "Hi there!");
        let formatted = ex.format_for_prompt();
        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_rag_config_defaults() {
        let config = RagPackageConfig::default();
        assert!(config.is_empty());
        assert!(config.chunk_size.is_none());
        assert!(config.top_k.is_none());
        assert!(config.min_relevance.is_none());
    }

    #[test]
    fn test_rag_config_not_empty() {
        let mut config = RagPackageConfig::default();
        config.chunk_size = Some(512);
        assert!(!config.is_empty());
    }

    #[test]
    fn test_metadata_defaults() {
        let meta = KpkgMetadata::default();
        assert!(meta.is_empty());
        assert!(meta.author.is_none());
        assert!(meta.tags.is_empty());
    }

    #[test]
    fn test_metadata_not_empty() {
        let mut meta = KpkgMetadata::default();
        meta.author = Some("Test Author".to_string());
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_manifest_with_new_fields() {
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .name("Professional Package")
            .description("A package with all features")
            .system_prompt("You are a helpful assistant.")
            .persona("Friendly and professional")
            .add_example("Hello", "Hi! How can I help?")
            .add_example_with_category("What's 2+2?", "4", "math")
            .author("Test Author")
            .language("en")
            .license("MIT")
            .add_tag("test")
            .add_tag("example")
            .chunk_size(512)
            .top_k(5)
            .min_relevance(0.5)
            .priority_boost(10)
            .add_document("doc.md", "# Test Content", None)
            .build()
            .expect("Failed to build");

        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let manifest = reader
            .read_manifest_only(&encrypted)
            .expect("Failed to read manifest");

        assert_eq!(manifest.name, "Professional Package");
        assert_eq!(
            manifest.system_prompt,
            Some("You are a helpful assistant.".to_string())
        );
        assert_eq!(
            manifest.persona,
            Some("Friendly and professional".to_string())
        );
        assert_eq!(manifest.examples.len(), 2);
        assert_eq!(manifest.examples[0].input, "Hello");
        assert_eq!(manifest.examples[1].category, Some("math".to_string()));

        // Check metadata
        let meta = manifest.metadata.expect("Metadata should exist");
        assert_eq!(meta.author, Some("Test Author".to_string()));
        assert_eq!(meta.language, Some("en".to_string()));
        assert_eq!(meta.license, Some("MIT".to_string()));
        assert_eq!(meta.tags, vec!["test", "example"]);

        // Check RAG config
        let rag = manifest.rag_config.expect("RAG config should exist");
        assert_eq!(rag.chunk_size, Some(512));
        assert_eq!(rag.top_k, Some(5));
        assert_eq!(rag.min_relevance, Some(0.5));
        assert_eq!(rag.priority_boost, Some(10));
    }

    #[test]
    fn test_read_with_manifest() {
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .name("Test")
            .system_prompt("System prompt here")
            .add_document("doc.md", "Content", None)
            .build()
            .expect("Failed to build");

        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let (docs, manifest) = reader
            .read_with_manifest(&encrypted)
            .expect("Failed to read");

        assert_eq!(docs.len(), 1);
        assert_eq!(manifest.name, "Test");
        assert_eq!(
            manifest.system_prompt,
            Some("System prompt here".to_string())
        );
    }

    #[test]
    fn test_backward_compatible_manifest() {
        // Old-style manifest without new fields should still deserialize
        let old_manifest_json = r#"{
            "name": "Old Package",
            "description": "Legacy format",
            "version": "1.0.0",
            "priorities": {},
            "default_priority": 0
        }"#;

        let manifest: KpkgManifest =
            serde_json::from_str(old_manifest_json).expect("Failed to deserialize old manifest");

        assert_eq!(manifest.name, "Old Package");
        assert!(manifest.system_prompt.is_none());
        assert!(manifest.persona.is_none());
        assert!(manifest.examples.is_empty());
        assert!(manifest.rag_config.is_none());
        assert!(manifest.metadata.is_none());
    }

    #[test]
    fn test_kpkg_index_result_ext_helpers() {
        let result = KpkgIndexResultExt {
            result: KpkgIndexResult {
                package_name: "Test".to_string(),
                documents_indexed: 5,
                chunks_created: 20,
                failed: vec![],
            },
            manifest: KpkgManifest {
                name: "Test".to_string(),
                system_prompt: Some("You are helpful.".to_string()),
                persona: Some("Expert".to_string()),
                examples: vec![ExamplePair::new("Q1", "A1"), ExamplePair::new("Q2", "A2")],
                ..Default::default()
            },
        };

        assert_eq!(result.documents_indexed(), 5);
        assert_eq!(result.chunks_created(), 20);
        assert!(!result.has_failures());
        assert_eq!(result.system_prompt(), Some("You are helpful."));
        assert_eq!(result.persona(), Some("Expert"));
        assert_eq!(result.examples().len(), 2);

        let effective = result.build_effective_system_prompt().unwrap();
        assert!(effective.contains("You are helpful."));
        assert!(effective.contains("Expert"));

        let examples_str = result.format_examples_for_prompt();
        assert!(examples_str.contains("Q1"));
        assert!(examples_str.contains("A1"));
    }

    #[test]
    fn test_timestamps() {
        let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
            .name("Timestamped")
            .with_current_timestamps()
            .add_document("doc.md", "Content", None)
            .build()
            .expect("Failed to build");

        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let manifest = reader
            .read_manifest_only(&encrypted)
            .expect("Failed to read");

        let meta = manifest.metadata.expect("Metadata should exist");
        assert!(meta.created_at.is_some());
        assert!(meta.updated_at.is_some());
    }
}
