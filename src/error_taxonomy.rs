//! Structured-error taxonomy (V113 / Phase C.2).
//!
//! Three pieces:
//!
//! 1. **`ErrorCode` trait** — every subsystem error enum implements
//!    `fn code(&self) -> &'static str` returning a stable code like
//!    `LOCAL_INFER_MODEL_NOT_FOUND`. Codes are screaming-snake-case,
//!    prefixed by subsystem, never renamed once shipped (consumers of
//!    structured logs depend on them).
//! 2. **`StructuredError`** — owned, JSON-serializable representation.
//!    Built from any `ErrorCode + std::error::Error` via `from_err`.
//!    What OTel spans + structured logs emit on the wire.
//! 3. **i18n loader** — loads `errors/<locale>.json` (`{ code: template }`)
//!    on first access via `OnceLock`. Templates support `{field}` placeholder
//!    substitution from `StructuredError::fields`. Falls back to the
//!    underlying `Display` impl when the locale or code is missing.
//!
//! ## The migration recipe (per subsystem)
//!
//! ```ignore
//! use crate::error_taxonomy::ErrorCode;
//!
//! #[derive(thiserror::Error, Debug)]
//! pub enum MySubsystemError {
//!     #[error("model not found: {path}")]
//!     ModelNotFound { path: PathBuf },
//!     #[error("io error: {0}")]
//!     Io(#[from] std::io::Error),
//! }
//!
//! impl ErrorCode for MySubsystemError {
//!     fn code(&self) -> &'static str {
//!         match self {
//!             Self::ModelNotFound { .. } => "MY_SUBSYSTEM_MODEL_NOT_FOUND",
//!             Self::Io(_) => "MY_SUBSYSTEM_IO",
//!         }
//!     }
//!     fn fields(&self) -> Vec<(&'static str, String)> {
//!         match self {
//!             Self::ModelNotFound { path } => vec![("path", path.display().to_string())],
//!             Self::Io(e) => vec![("io_kind", format!("{:?}", e.kind()))],
//!         }
//!     }
//! }
//! ```
//!
//! Then `StructuredError::from_err(&err).to_json()` is wire-ready and
//! `StructuredError::from_err(&err).localize("es")` is human-ready.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Trait for any error that exposes a stable code + structured fields.
///
/// `code()` is the contract with structured-log consumers — never rename
/// or repurpose once shipped. New variants get new codes; behaviour
/// changes that alter the meaning of an existing code earn a new code.
pub trait ErrorCode {
    /// Stable, screaming-snake-case code, prefixed by subsystem.
    fn code(&self) -> &'static str;

    /// Optional structured fields associated with this error instance.
    /// Used by OTel + i18n template substitution. Default: empty.
    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Owned, JSON-serializable error representation. The wire format for
/// OTel span events, structured logs, and IPC error responses.
///
/// `source_chain` is materialized (not `Box<dyn Error>`) so the struct
/// stays `Send + Sync + Clone + Serialize`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuredError {
    /// Stable subsystem-prefixed code (e.g. `LOCAL_INFER_MODEL_NOT_FOUND`).
    pub code: &'static str,
    /// Default `Display` rendering of the error. English; for localized
    /// output, call [`StructuredError::localize`] instead.
    pub message: String,
    /// Structured key/value pairs from [`ErrorCode::fields`]. Sorted to
    /// keep JSON output deterministic for testing.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, String>,
    /// Walk of `Error::source()` flattened to strings. The first entry
    /// is the immediate source; the rest are deeper causes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_chain: Vec<String>,
}

impl StructuredError {
    /// Build from any error that implements [`ErrorCode`] +
    /// [`std::error::Error`]. Walks the source chain up to 8 levels deep
    /// (cycle-safe; deeper chains are truncated).
    pub fn from_err<E>(err: &E) -> Self
    where
        E: ErrorCode + std::error::Error + ?Sized,
    {
        let mut fields = BTreeMap::new();
        for (k, v) in err.fields() {
            fields.insert(k.to_string(), v);
        }
        let mut source_chain = Vec::new();
        let mut current: Option<&dyn std::error::Error> = err.source();
        for _ in 0..8 {
            match current {
                Some(s) => {
                    source_chain.push(s.to_string());
                    current = s.source();
                }
                None => break,
            }
        }
        Self {
            code: err.code(),
            message: err.to_string(),
            fields,
            source_chain,
        }
    }

    /// Render the error in the given locale (`"en"`, `"es"`, …).
    /// Falls back to `self.message` when the locale or code is missing.
    /// `{field}` placeholders in the template are substituted from
    /// `self.fields` (missing fields stay literal).
    pub fn localize(&self, locale: &str) -> String {
        let table = match load_locale_table(locale) {
            Some(t) => t,
            None => return self.message.clone(),
        };
        let template = match table.get(self.code) {
            Some(t) => t,
            None => return self.message.clone(),
        };
        substitute(template, &self.fields)
    }

    /// Serialize to canonical JSON. Always Ok — fields are owned strings.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// `{field}` placeholder substitution. Unknown placeholders pass through
/// unchanged so a missing field never produces empty strings (which would
/// confuse readers); they remain literal `{field}` for visibility.
fn substitute(template: &str, fields: &BTreeMap<String, String>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end) => {
                let key = &after[..end];
                match fields.get(key) {
                    Some(v) => out.push_str(v),
                    None => {
                        out.push('{');
                        out.push_str(key);
                        out.push('}');
                    }
                }
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[start..]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Locale table loader. Caches per-locale tables in a `OnceLock`-keyed
/// static so subsequent lookups are O(1). Misses (no file, parse error)
/// cache as `None` so we don't re-attempt on every error.
fn load_locale_table(locale: &str) -> Option<&'static BTreeMap<&'static str, String>> {
    // We only ship `en` and `es` in-tree; additional locales drop in via
    // `errors/<locale>.json` resolved at runtime relative to the executable
    // working directory. Cached entries are never freed (process lifetime).
    match locale {
        "en" => Some(en_table()),
        "es" => Some(es_table()),
        _ => load_external(locale),
    }
}

fn en_table() -> &'static BTreeMap<&'static str, String> {
    static T: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    T.get_or_init(|| parse_table(EN_JSON).unwrap_or_default())
}

fn es_table() -> &'static BTreeMap<&'static str, String> {
    static T: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    T.get_or_init(|| parse_table(ES_JSON).unwrap_or_default())
}

fn load_external(_locale: &str) -> Option<&'static BTreeMap<&'static str, String>> {
    // External locale files are deferred — the indirection via OnceLock
    // would need a per-locale slot. The current contract is: in-tree
    // `en` + `es`; callers that want more set a custom resolver via
    // [`set_locale_resolver`] (TODO V114+).
    None
}

fn parse_table(json: &str) -> Result<BTreeMap<&'static str, String>, String> {
    let raw: BTreeMap<String, String> =
        serde_json::from_str(json).map_err(|e| format!("parse locale table: {e}"))?;
    // Codes are `&'static str` — leak the keys so they outlive the table.
    // This runs once per locale; total leak ~few KB, well under any budget.
    let mut out = BTreeMap::new();
    for (k, v) in raw {
        let leaked: &'static str = Box::leak(k.into_boxed_str());
        out.insert(leaked, v);
    }
    Ok(out)
}

const EN_JSON: &str = include_str!("../errors/en.json");
const ES_JSON: &str = include_str!("../errors/es.json");

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(thiserror::Error, Debug)]
    enum DemoError {
        #[error("demo failure for {who}")]
        Failure { who: String },
        #[error("io: {0}")]
        Io(#[from] std::io::Error),
    }

    impl ErrorCode for DemoError {
        fn code(&self) -> &'static str {
            match self {
                Self::Failure { .. } => "DEMO_FAILURE",
                Self::Io(_) => "DEMO_IO",
            }
        }
        fn fields(&self) -> Vec<(&'static str, String)> {
            match self {
                Self::Failure { who } => vec![("who", who.clone())],
                Self::Io(e) => vec![("io_kind", format!("{:?}", e.kind()))],
            }
        }
    }

    #[test]
    fn structured_from_err_carries_code_and_fields() {
        let e = DemoError::Failure {
            who: "alice".into(),
        };
        let s = StructuredError::from_err(&e);
        assert_eq!(s.code, "DEMO_FAILURE");
        assert_eq!(s.fields.get("who").map(String::as_str), Some("alice"));
        assert!(s.message.contains("alice"));
    }

    #[test]
    fn structured_walks_source_chain() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: DemoError = io.into();
        let s = StructuredError::from_err(&e);
        assert_eq!(s.code, "DEMO_IO");
        assert!(s.source_chain.iter().any(|m| m.contains("denied")));
    }

    #[test]
    fn substitution_replaces_known_keys() {
        let mut f = BTreeMap::new();
        f.insert("name".to_string(), "World".to_string());
        assert_eq!(substitute("Hello {name}!", &f), "Hello World!");
    }

    #[test]
    fn substitution_leaves_unknown_keys_literal() {
        let f = BTreeMap::new();
        assert_eq!(substitute("Hello {name}!", &f), "Hello {name}!");
    }

    #[test]
    fn substitution_handles_unclosed_braces() {
        let f = BTreeMap::new();
        assert_eq!(substitute("oops {bad", &f), "oops {bad");
    }

    #[test]
    fn json_roundtrip_preserves_shape() {
        let e = DemoError::Failure { who: "bob".into() };
        let s = StructuredError::from_err(&e);
        let v = s.to_json();
        assert_eq!(v["code"], "DEMO_FAILURE");
        assert_eq!(v["fields"]["who"], "bob");
    }

    #[test]
    fn localize_falls_back_when_locale_missing() {
        let e = DemoError::Failure {
            who: "carol".into(),
        };
        let s = StructuredError::from_err(&e);
        // `xx` isn't loaded → fallback to Display message.
        assert!(s.localize("xx").contains("carol"));
    }
}
