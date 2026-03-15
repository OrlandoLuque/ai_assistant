//! Secure credential management
//!
//! Provides `SecureString` — a memory-safe string wrapper that zeroes its content
//! on drop and never leaks secrets through `Debug`/`Display`/`Serialize` — and
//! `CredentialResolver`, a pluggable chain-of-responsibility resolver for API keys.
//!
//! # Security guarantees
//!
//! - **Zeroization**: `SecureString` overwrites all bytes with `0u8` via
//!   `ptr::write_volatile` on drop, preventing compiler elision.
//! - **Constant-time comparison**: `PartialEq` uses XOR-accumulation to avoid
//!   timing side-channels.
//! - **No accidental leaks**: `Debug` prints `SecureString(***)`, `Display` prints
//!   `***`, and `Serialize` emits `"***"`. The only way to read the value is the
//!   explicit `expose()` method.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::ptr;

use serde::{Deserialize, Serialize};

// =============================================================================
// SecureString
// =============================================================================

/// A string wrapper that zeroes its memory on drop and never reveals its content
/// through `Debug`, `Display`, or `Serialize`.
pub struct SecureString {
    inner: Vec<u8>,
}

impl SecureString {
    /// Create a new `SecureString` from anything convertible to `String`.
    pub fn new(s: impl Into<String>) -> Self {
        let s: String = s.into();
        Self {
            inner: s.into_bytes(),
        }
    }

    /// The **only** way to access the underlying value.
    ///
    /// Returns the content as `&str`. If the bytes are not valid UTF-8
    /// (which should not happen when created via the public API), returns
    /// a replacement string.
    pub fn expose(&self) -> &str {
        std::str::from_utf8(&self.inner).unwrap_or("\u{FFFD}")
    }

    /// Returns `true` if the secure string is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the byte length of the secure string.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Explicitly zero all bytes in-place. Called automatically by `Drop`.
    pub fn zeroize(&mut self) {
        for byte in self.inner.iter_mut() {
            // write_volatile prevents the compiler from optimizing away the store
            unsafe {
                ptr::write_volatile(byte as *mut u8, 0u8);
            }
        }
    }
}

// -- Drop: zero memory before deallocation -----------------------------------

impl Drop for SecureString {
    fn drop(&mut self) {
        self.zeroize();
    }
}

// -- Debug: never reveal content ---------------------------------------------

impl fmt::Debug for SecureString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecureString(***)")
    }
}

// -- Display: never reveal content -------------------------------------------

impl fmt::Display for SecureString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

// -- Clone: deep copy --------------------------------------------------------

impl Clone for SecureString {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// -- PartialEq: constant-time comparison -------------------------------------

impl PartialEq for SecureString {
    fn eq(&self, other: &Self) -> bool {
        if self.inner.len() != other.inner.len() {
            return false;
        }
        let mut result = 0u8;
        for (a, b) in self.inner.iter().zip(other.inner.iter()) {
            result |= a ^ b;
        }
        result == 0
    }
}

impl Eq for SecureString {}

// -- From conversions --------------------------------------------------------

impl From<String> for SecureString {
    fn from(s: String) -> Self {
        Self {
            inner: s.into_bytes(),
        }
    }
}

impl From<&str> for SecureString {
    fn from(s: &str) -> Self {
        Self {
            inner: s.as_bytes().to_vec(),
        }
    }
}

// -- Serde: Serialize always emits "***", Deserialize wraps input ------------

impl Serialize for SecureString {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("***")
    }
}

impl<'de> Deserialize<'de> for SecureString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(SecureString::new(s))
    }
}

// =============================================================================
// CredentialError
// =============================================================================

/// Error type for credential resolution failures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CredentialError {
    /// The requested key was not found in any source.
    NotFound {
        key_name: String,
        sources_tried: Vec<String>,
    },
    /// A source contained data but in an invalid format.
    InvalidFormat {
        source: String,
        detail: String,
    },
}

impl fmt::Display for CredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CredentialError::NotFound {
                key_name,
                sources_tried,
            } => {
                write!(
                    f,
                    "Credential '{}' not found (sources tried: {})",
                    key_name,
                    sources_tried.join(", ")
                )
            }
            CredentialError::InvalidFormat { source, detail } => {
                write!(
                    f,
                    "Invalid credential format in source '{}': {}",
                    source, detail
                )
            }
        }
    }
}

impl std::error::Error for CredentialError {}

// =============================================================================
// CredentialSource trait
// =============================================================================

/// A source that can resolve a credential key to its value.
///
/// Implementations must be `Send + Sync` so they can be shared across threads.
pub trait CredentialSource: Send + Sync {
    /// Human-readable name of this source (e.g. `"env"`, `"file:/path"`).
    fn name(&self) -> &str;

    /// Attempt to resolve `key_name`. Returns `None` if the key is not available.
    fn resolve(&self, key_name: &str) -> Option<SecureString>;
}

// =============================================================================
// Built-in sources
// =============================================================================

/// Resolves credentials from environment variables.
///
/// The `key_name` is looked up directly as an environment variable name.
#[derive(Debug)]
pub struct EnvVarSource;

impl CredentialSource for EnvVarSource {
    fn name(&self) -> &str {
        "env"
    }

    fn resolve(&self, key_name: &str) -> Option<SecureString> {
        std::env::var(key_name).ok().map(SecureString::new)
    }
}

/// Resolves credentials from a file with `KEY_NAME=value` lines.
///
/// Blank lines and lines starting with `#` are ignored.
#[derive(Debug)]
pub struct FileSource {
    path: PathBuf,
}

impl FileSource {
    /// Create a new `FileSource` pointing to the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl CredentialSource for FileSource {
    fn name(&self) -> &str {
        "file"
    }

    fn resolve(&self, key_name: &str) -> Option<SecureString> {
        use std::io::BufRead;

        let file = std::fs::File::open(&self.path).ok()?;
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = trimmed.split_once('=') {
                if key.trim() == key_name {
                    return Some(SecureString::new(value.trim()));
                }
            }
        }
        None
    }
}

/// Resolves credentials via a user-provided callback.
pub struct CallbackSource {
    name: String,
    callback: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
}

impl CallbackSource {
    /// Create a new `CallbackSource` with a name and resolver function.
    pub fn new(
        name: impl Into<String>,
        callback: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            callback: Box::new(callback),
        }
    }
}

impl fmt::Debug for CallbackSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackSource")
            .field("name", &self.name)
            .finish()
    }
}

impl CredentialSource for CallbackSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self, key_name: &str) -> Option<SecureString> {
        (self.callback)(key_name).map(SecureString::new)
    }
}

/// Resolves credentials from a static in-memory map. Primarily for testing.
#[derive(Debug, Clone)]
pub struct StaticSource {
    entries: HashMap<String, String>,
}

impl StaticSource {
    /// Create a new empty `StaticSource`.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a key-value pair.
    pub fn insert(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.entries.insert(key.into(), value.into());
        self
    }
}

impl Default for StaticSource {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialSource for StaticSource {
    fn name(&self) -> &str {
        "static"
    }

    fn resolve(&self, key_name: &str) -> Option<SecureString> {
        self.entries.get(key_name).map(|v| SecureString::new(v.clone()))
    }
}

// =============================================================================
// CredentialResolver
// =============================================================================

/// A chain-of-responsibility resolver that tries multiple `CredentialSource`s
/// in order and returns the first hit.
pub struct CredentialResolver {
    chain: Vec<Box<dyn CredentialSource>>,
}

impl fmt::Debug for CredentialResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.chain.iter().map(|s| s.name()).collect();
        f.debug_struct("CredentialResolver")
            .field("chain", &names)
            .finish()
    }
}

impl CredentialResolver {
    /// Create a resolver with an empty chain.
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    /// Create a resolver pre-loaded with `EnvVarSource`.
    pub fn with_env() -> Self {
        let mut resolver = Self::new();
        resolver.add_source(EnvVarSource);
        resolver
    }

    /// Append a source to the chain. Sources are tried in insertion order.
    pub fn add_source(&mut self, source: impl CredentialSource + 'static) -> &mut Self {
        self.chain.push(Box::new(source));
        self
    }

    /// Try each source in order and return the first successful resolution.
    pub fn resolve(&self, key_name: &str) -> Option<SecureString> {
        for source in &self.chain {
            if let Some(value) = source.resolve(key_name) {
                return Some(value);
            }
        }
        None
    }

    /// Like `resolve`, but returns a descriptive `CredentialError` on failure.
    pub fn resolve_or_error(&self, key_name: &str) -> Result<SecureString, CredentialError> {
        let mut sources_tried = Vec::new();
        for source in &self.chain {
            sources_tried.push(source.name().to_string());
            if let Some(value) = source.resolve(key_name) {
                return Ok(value);
            }
        }
        Err(CredentialError::NotFound {
            key_name: key_name.to_string(),
            sources_tried,
        })
    }
}

impl Default for CredentialResolver {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // -- SecureString tests ---------------------------------------------------

    #[test]
    fn test_secure_string_new() {
        let s = SecureString::new("hello");
        assert_eq!(s.expose(), "hello");
    }

    #[test]
    fn test_secure_string_expose() {
        let s = SecureString::new("my_secret_key_12345");
        assert_eq!(s.expose(), "my_secret_key_12345");
    }

    #[test]
    fn test_secure_string_is_empty_true() {
        let s = SecureString::new("");
        assert!(s.is_empty());
    }

    #[test]
    fn test_secure_string_is_empty_false() {
        let s = SecureString::new("x");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_secure_string_len() {
        let s = SecureString::new("abcdef");
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn test_secure_string_len_empty() {
        let s = SecureString::new("");
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_secure_string_zeroize_impl() {
        let mut s = SecureString::new("hello");
        s.zeroize();
        assert!(s.inner.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_string_drop_zeroes_bytes() {
        // Verify the zeroize logic works by calling it explicitly,
        // since we cannot safely inspect memory after drop.
        let mut s = SecureString::new("secret_key_12345");
        assert!(!s.inner.iter().all(|&b| b == 0));
        s.zeroize();
        assert_eq!(s.inner.len(), 16);
        assert!(s.inner.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_string_debug_format() {
        let s = SecureString::new("super_secret");
        let debug = format!("{:?}", s);
        assert_eq!(debug, "SecureString(***)");
        assert!(!debug.contains("super_secret"));
    }

    #[test]
    fn test_secure_string_display_format() {
        let s = SecureString::new("super_secret");
        let display = format!("{}", s);
        assert_eq!(display, "***");
        assert!(!display.contains("super_secret"));
    }

    #[test]
    fn test_secure_string_clone() {
        let original = SecureString::new("cloneable");
        let cloned = original.clone();
        assert_eq!(original.expose(), cloned.expose());
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_secure_string_clone_independence() {
        let original = SecureString::new("independent");
        let mut cloned = original.clone();
        cloned.zeroize();
        // Original must be unaffected
        assert_eq!(original.expose(), "independent");
        assert!(cloned.inner.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_string_eq_same() {
        let a = SecureString::new("same_value");
        let b = SecureString::new("same_value");
        assert_eq!(a, b);
    }

    #[test]
    fn test_secure_string_eq_different() {
        let a = SecureString::new("value_a");
        let b = SecureString::new("value_b");
        assert_ne!(a, b);
    }

    #[test]
    fn test_secure_string_eq_different_length() {
        let a = SecureString::new("short");
        let b = SecureString::new("much_longer_string");
        assert_ne!(a, b);
    }

    #[test]
    fn test_secure_string_from_string() {
        let s: SecureString = String::from("from_string").into();
        assert_eq!(s.expose(), "from_string");
    }

    #[test]
    fn test_secure_string_from_str() {
        let s: SecureString = "from_str".into();
        assert_eq!(s.expose(), "from_str");
    }

    #[test]
    fn test_secure_string_serialize() {
        let s = SecureString::new("never_leaked");
        let json = serde_json::to_string(&s).expect("serialize should succeed");
        assert_eq!(json, r#""***""#);
        assert!(!json.contains("never_leaked"));
    }

    #[test]
    fn test_secure_string_deserialize() {
        let json = r#""deserialized_value""#;
        let s: SecureString = serde_json::from_str(json).expect("deserialize should succeed");
        assert_eq!(s.expose(), "deserialized_value");
    }

    // -- CredentialResolver tests ---------------------------------------------

    #[test]
    fn test_resolver_empty_chain_returns_none() {
        let resolver = CredentialResolver::new();
        assert!(resolver.resolve("ANY_KEY").is_none());
    }

    #[test]
    fn test_env_var_source() {
        // Set a temporary env var for this test
        let key = "AI_ASSISTANT_TEST_CRED_ENV_42";
        std::env::set_var(key, "env_secret");
        let source = EnvVarSource;
        let result = source.resolve(key);
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|s| s.expose()), Some("env_secret"));
        std::env::remove_var(key);
    }

    #[test]
    fn test_env_var_source_missing() {
        let source = EnvVarSource;
        assert!(source.resolve("DEFINITELY_NOT_SET_XYZ_999").is_none());
    }

    #[test]
    fn test_file_source() {
        let dir = std::env::temp_dir().join("ai_assistant_test_cred_file");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("creds.env");
        {
            let mut f = std::fs::File::create(&file_path).expect("create temp file");
            writeln!(f, "# comment line").expect("write");
            writeln!(f, "").expect("write");
            writeln!(f, "MY_API_KEY=file_secret_123").expect("write");
            writeln!(f, "OTHER_KEY=other_value").expect("write");
        }

        let source = FileSource::new(&file_path);
        let result = source.resolve("MY_API_KEY");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|s| s.expose()), Some("file_secret_123"));

        // Key not in file
        assert!(source.resolve("MISSING_KEY").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_file_source_missing_file() {
        let source = FileSource::new("/nonexistent/path/creds.env");
        assert!(source.resolve("ANY").is_none());
    }

    #[test]
    fn test_callback_source() {
        let source = CallbackSource::new("my_callback", |key| {
            if key == "CB_KEY" {
                Some("callback_secret".to_string())
            } else {
                None
            }
        });

        assert_eq!(source.name(), "my_callback");
        let result = source.resolve("CB_KEY");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|s| s.expose()), Some("callback_secret"));
        assert!(source.resolve("OTHER").is_none());
    }

    #[test]
    fn test_static_source() {
        let source = StaticSource::new()
            .insert("KEY_A", "value_a")
            .insert("KEY_B", "value_b");

        let a = source.resolve("KEY_A");
        assert!(a.is_some());
        assert_eq!(a.as_ref().map(|s| s.expose()), Some("value_a"));

        let b = source.resolve("KEY_B");
        assert!(b.is_some());
        assert_eq!(b.as_ref().map(|s| s.expose()), Some("value_b"));

        assert!(source.resolve("KEY_C").is_none());
    }

    #[test]
    fn test_chain_priority_first_wins() {
        let mut resolver = CredentialResolver::new();
        resolver.add_source(
            StaticSource::new()
                .insert("SHARED_KEY", "first_source"),
        );
        resolver.add_source(
            StaticSource::new()
                .insert("SHARED_KEY", "second_source"),
        );

        let result = resolver.resolve("SHARED_KEY");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|s| s.expose()), Some("first_source"));
    }

    #[test]
    fn test_chain_fallback_to_second() {
        let mut resolver = CredentialResolver::new();
        resolver.add_source(StaticSource::new()); // empty — will miss
        resolver.add_source(
            StaticSource::new()
                .insert("ONLY_IN_SECOND", "second_value"),
        );

        let result = resolver.resolve("ONLY_IN_SECOND");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|s| s.expose()), Some("second_value"));
    }

    #[test]
    fn test_resolve_or_error_success() {
        let mut resolver = CredentialResolver::new();
        resolver.add_source(
            StaticSource::new()
                .insert("FOUND_KEY", "found_value"),
        );

        let result = resolver.resolve_or_error("FOUND_KEY");
        assert!(result.is_ok());
        assert_eq!(result.as_ref().map(|s| s.expose()), Ok("found_value"));
    }

    #[test]
    fn test_resolve_or_error_not_found() {
        let mut resolver = CredentialResolver::new();
        resolver.add_source(StaticSource::new());

        let result = resolver.resolve_or_error("MISSING_KEY");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            CredentialError::NotFound {
                key_name,
                sources_tried,
            } => {
                assert_eq!(key_name, "MISSING_KEY");
                assert_eq!(sources_tried, &["static"]);
            }
            _ => panic!("Expected NotFound variant"),
        }
    }

    #[test]
    fn test_resolve_or_error_multiple_sources_tried() {
        let mut resolver = CredentialResolver::new();
        resolver.add_source(StaticSource::new());
        resolver.add_source(CallbackSource::new("my_cb", |_| None));

        let result = resolver.resolve_or_error("NOPE");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            CredentialError::NotFound { sources_tried, .. } => {
                assert_eq!(sources_tried, &["static", "my_cb"]);
            }
            _ => panic!("Expected NotFound variant"),
        }
    }

    // -- CredentialError display tests ----------------------------------------

    #[test]
    fn test_credential_error_not_found_display() {
        let err = CredentialError::NotFound {
            key_name: "API_KEY".to_string(),
            sources_tried: vec!["env".to_string(), "file".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("API_KEY"));
        assert!(msg.contains("env"));
        assert!(msg.contains("file"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_credential_error_invalid_format_display() {
        let err = CredentialError::InvalidFormat {
            source: "file".to_string(),
            detail: "missing equals sign".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("file"));
        assert!(msg.contains("missing equals sign"));
        assert!(msg.contains("Invalid credential format"));
    }

    #[test]
    fn test_with_env_constructor() {
        let resolver = CredentialResolver::with_env();
        let debug = format!("{:?}", resolver);
        assert!(debug.contains("env"));
    }

    #[test]
    fn test_resolver_default() {
        let resolver = CredentialResolver::default();
        assert!(resolver.resolve("ANYTHING").is_none());
    }
}
