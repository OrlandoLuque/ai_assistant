//! On-disk cache layout for downloaded benchmark datasets.
//!
//! Each benchmark lives under:
//!
//! ```text
//!   <root>/<loader_name>/<file>
//! ```
//!
//! `<root>` defaults to `$CARGO_TARGET_DIR/eval_benchmarks` during tests and
//! `<data_dir>/ai_assistant/eval_benchmarks` in production, but callers can
//! override via `BenchmarkCache::with_root`.

use std::path::{Path, PathBuf};

use super::types::BenchmarkError;

/// Location on disk where benchmark datasets are stored.
#[derive(Debug, Clone)]
pub struct BenchmarkCache {
    root: PathBuf,
}

impl BenchmarkCache {
    /// Build a cache rooted at `root`. The directory is created lazily.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Default cache root: `<CARGO_TARGET_DIR or ./target>/eval_benchmarks`.
    ///
    /// We deliberately avoid pulling in the `dirs` crate — zero new deps is
    /// one of V90's goals — and stick to paths we know are writable.
    pub fn default_root() -> Self {
        // CARGO_TARGET_DIR is set by cargo; fall back to ./target.
        let base = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target"));
        Self::with_root(base.join("eval_benchmarks"))
    }

    /// Root path (may not yet exist on disk).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory for a single loader (e.g. `<root>/truthfulqa`). Creates it
    /// if missing.
    pub fn dir_for(&self, loader_name: &str) -> Result<PathBuf, BenchmarkError> {
        let sanitized = sanitize(loader_name);
        if sanitized.is_empty() {
            return Err(BenchmarkError::Io(format!(
                "invalid loader name: {loader_name:?}"
            )));
        }
        let dir = self.root.join(&sanitized);
        std::fs::create_dir_all(&dir)
            .map_err(|e| BenchmarkError::Io(format!("mkdir {}: {e}", dir.display())))?;
        Ok(dir)
    }

    /// Full path for a named file inside a loader's dir.
    pub fn file_for(&self, loader_name: &str, file: &str) -> Result<PathBuf, BenchmarkError> {
        let dir = self.dir_for(loader_name)?;
        let sanitized = sanitize(file);
        if sanitized.is_empty() {
            return Err(BenchmarkError::Io(format!("invalid file name: {file:?}")));
        }
        Ok(dir.join(sanitized))
    }

    /// Returns `true` if `file` exists and its size matches `expected` (when
    /// `expected` is `Some`). Used to short-circuit downloads.
    pub fn is_cached(&self, path: &Path, expected: Option<u64>) -> bool {
        match std::fs::metadata(path) {
            Ok(m) if m.is_file() => match expected {
                Some(n) if n > 0 => m.len() == n,
                _ => m.len() > 0,
            },
            _ => false,
        }
    }
}

/// Strip path separators and other risky characters so a loader name can't
/// escape its cache directory.
fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_removes_path_separators() {
        assert_eq!(sanitize("../../etc/passwd"), "....etcpasswd");
        assert_eq!(sanitize("truthful_qa.v1"), "truthful_qa.v1");
    }

    #[test]
    fn dir_for_creates_subdir() {
        let tmp = std::env::temp_dir().join(format!("aic_eb_{}", std::process::id()));
        let cache = BenchmarkCache::with_root(&tmp);
        let dir = cache.dir_for("truthfulqa").unwrap();
        assert!(dir.exists());
        assert!(dir.ends_with("truthfulqa"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn is_cached_checks_size() {
        let tmp = std::env::temp_dir().join(format!("aic_eb2_{}", std::process::id()));
        let cache = BenchmarkCache::with_root(&tmp);
        let f = cache.file_for("x", "y.bin").unwrap();
        std::fs::write(&f, b"hello").unwrap();
        assert!(cache.is_cached(&f, None));
        assert!(cache.is_cached(&f, Some(5)));
        assert!(!cache.is_cached(&f, Some(99)));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn rejects_empty_loader_name() {
        let cache = BenchmarkCache::with_root(std::env::temp_dir());
        assert!(cache.dir_for("").is_err());
        assert!(cache.dir_for("/////").is_err());
    }
}
