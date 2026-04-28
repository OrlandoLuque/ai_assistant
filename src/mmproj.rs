// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Multimodal projector (mmproj) discovery and validation.
//!
//! `mmproj` files are standalone GGUF projector adapters that pair with a
//! base LLM to enable vision in `llama.cpp` / `llama-server` / `koboldcpp`.
//! The projector is loaded at server startup (`--mmproj <path>`), not per
//! request — so this module's job is to **validate the path** and surface
//! a typed handle that callers can store in `AiConfig` or pass to a future
//! embedded launcher.
//!
//! ## Validation
//!
//! [`MultimodalProjector::from_path`] performs:
//!
//! 1. Reject `..` components defensively (before canonicalize, so a symlink
//!    chain ending in `..` cannot bypass the check on race).
//! 2. Canonicalize to an absolute path (resolves symlinks once).
//! 3. Confirm the target exists and is a regular file (not a directory or
//!    device).
//! 4. Open with a 4-byte read of the GGUF magic (`0x47 0x47 0x55 0x46`).
//!    The read is bounded so a device file or pipe cannot hang the call.
//! 5. Sanity-check the size: real projectors are at least a few hundred
//!    MB; we treat anything below [`MIN_PROJECTOR_BYTES`] (1 MiB) as a
//!    user error and reject it.
//!
//! ## What this module does *not* do
//!
//! * It never reads the full file. The handle stores only the path, size,
//!   and a flag confirming the GGUF magic.
//! * It does not parse the GGUF tensor table — too brittle across formats,
//!   and the runtime (llama.cpp) does the real check anyway.
//! * It does not download from HuggingFace; the user must pre-fetch.
//! * It does not start a server. Spawning `llama-server --mmproj ...` is a
//!   future scope item.
//!
//! Logging deliberately emits only the file *name*, never the absolute
//! path, to avoid leaking machine layout when logs are shared.

#![cfg(feature = "vision")]

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Minimum size, in bytes, for a candidate mmproj file. Real projectors
/// are typically 100 MB – 2 GB; anything well under a megabyte is almost
/// certainly the wrong file.
pub const MIN_PROJECTOR_BYTES: u64 = 1_048_576; // 1 MiB

/// GGUF format magic bytes (ASCII "GGUF").
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Validated handle to a multimodal projector file on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalProjector {
    /// Absolute, canonicalized path. Stored absolute so callers don't
    /// re-resolve relative to a different cwd.
    path: PathBuf,
    /// File size in bytes at validation time.
    size_bytes: u64,
    /// True after [`Self::from_path`] confirmed the GGUF magic. Always
    /// true for handles produced by `from_path`; the field exists so
    /// future relaxed constructors can opt out.
    gguf_validated: bool,
}

impl MultimodalProjector {
    /// Validate `path` and return a handle. Performs the full check
    /// pipeline described at the module level.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, MmprojValidationError> {
        let raw = path.as_ref();

        // Defense-in-depth: reject `..` *before* canonicalize so a
        // symlink race can't substitute a path that walks out of the
        // intended directory between checks.
        for component in raw.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(MmprojValidationError::PathTraversal);
            }
        }

        let canonical = std::fs::canonicalize(raw).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                MmprojValidationError::NotFound
            } else {
                MmprojValidationError::IoError(e.to_string())
            }
        })?;

        let metadata = std::fs::metadata(&canonical)
            .map_err(|e| MmprojValidationError::IoError(e.to_string()))?;

        if !metadata.is_file() {
            return Err(MmprojValidationError::NotAFile);
        }

        let size_bytes = metadata.len();
        if size_bytes < MIN_PROJECTOR_BYTES {
            return Err(MmprojValidationError::TooSmall { actual: size_bytes });
        }

        let mut file = std::fs::File::open(&canonical)
            .map_err(|e| MmprojValidationError::IoError(e.to_string()))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| MmprojValidationError::IoError(e.to_string()))?;
        if magic != GGUF_MAGIC {
            return Err(MmprojValidationError::InvalidMagic { found: magic });
        }

        Ok(Self {
            path: canonical,
            size_bytes,
            gguf_validated: true,
        })
    }

    /// Absolute, canonicalized path of the validated projector.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Filename only — safe for log output (does not leak directory layout).
    pub fn filename(&self) -> std::borrow::Cow<'_, str> {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_else(|| std::borrow::Cow::Borrowed("<unnamed>"))
    }

    /// File size at validation time.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Whether the GGUF magic header was confirmed. Always `true` for
    /// handles produced by [`Self::from_path`].
    pub fn is_gguf_validated(&self) -> bool {
        self.gguf_validated
    }
}

/// Typed validation errors. Each variant carries enough information for
/// the caller to render an actionable message without disclosing the
/// full filesystem path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MmprojValidationError {
    /// Path did not exist at validation time.
    NotFound,
    /// Path resolved to a directory, symlink, device, or pipe.
    NotAFile,
    /// Path contained a `..` component. Rejected before canonicalize as
    /// defense-in-depth against symlink-race substitution.
    PathTraversal,
    /// File is below [`MIN_PROJECTOR_BYTES`] — almost certainly the wrong file.
    TooSmall { actual: u64 },
    /// First 4 bytes did not match the GGUF magic. The actual bytes seen
    /// are returned so the caller can report what was there instead.
    InvalidMagic { found: [u8; 4] },
    /// Underlying I/O error during canonicalize / open / read.
    IoError(String),
}

impl fmt::Display for MmprojValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "mmproj path does not exist"),
            Self::NotAFile => write!(f, "mmproj path is not a regular file"),
            Self::PathTraversal => {
                write!(f, "mmproj path contains a `..` component (rejected)")
            }
            Self::TooSmall { actual } => write!(
                f,
                "mmproj file is only {} bytes — real projectors are at least {} bytes",
                actual, MIN_PROJECTOR_BYTES
            ),
            Self::InvalidMagic { found } => write!(
                f,
                "mmproj file does not start with GGUF magic (got {:02X?})",
                found
            ),
            Self::IoError(msg) => write!(f, "mmproj I/O error: {}", msg),
        }
    }
}

impl std::error::Error for MmprojValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("ai_assistant_mmproj_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    fn write_file(name: &str, bytes: &[u8]) -> PathBuf {
        let p = tmpdir().join(name);
        std::fs::write(&p, bytes).expect("write");
        p
    }

    fn synthetic_gguf(size: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(size.max(4));
        v.extend_from_slice(&GGUF_MAGIC);
        v.resize(size.max(4), 0xAB);
        v
    }

    #[test]
    fn from_path_accepts_valid_gguf() {
        let p = write_file(
            "valid.gguf",
            &synthetic_gguf(MIN_PROJECTOR_BYTES as usize + 16),
        );
        let proj = MultimodalProjector::from_path(&p).expect("valid file should pass");
        assert!(proj.is_gguf_validated());
        assert!(proj.size_bytes() >= MIN_PROJECTOR_BYTES);
        assert!(proj.path().is_absolute());
    }

    #[test]
    fn from_path_rejects_wrong_magic() {
        let mut bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        bytes.resize(MIN_PROJECTOR_BYTES as usize + 16, 0);
        let p = write_file("bad_magic.gguf", &bytes);
        let err = MultimodalProjector::from_path(&p).expect_err("bad magic must fail");
        assert!(matches!(
            err,
            MmprojValidationError::InvalidMagic {
                found: [0xDE, 0xAD, 0xBE, 0xEF]
            }
        ));
    }

    #[test]
    fn from_path_rejects_too_small() {
        let p = write_file("tiny.gguf", &synthetic_gguf(64));
        let err = MultimodalProjector::from_path(&p).expect_err("tiny file must fail");
        assert!(matches!(
            err,
            MmprojValidationError::TooSmall { actual: 64 }
        ));
    }

    #[test]
    fn from_path_rejects_missing() {
        let p = tmpdir().join("does_not_exist.gguf");
        let err = MultimodalProjector::from_path(&p).expect_err("missing file must fail");
        assert!(matches!(err, MmprojValidationError::NotFound));
    }

    #[test]
    fn from_path_rejects_directory() {
        let d = tmpdir().join("a_dir");
        let _ = std::fs::create_dir_all(&d);
        let err = MultimodalProjector::from_path(&d).expect_err("directory must fail");
        // canonicalize succeeds for a directory; we should fall through
        // to NotAFile.
        assert!(matches!(err, MmprojValidationError::NotAFile));
    }

    #[test]
    fn from_path_rejects_traversal() {
        let p = tmpdir().join("..").join("evil.gguf");
        let err = MultimodalProjector::from_path(&p).expect_err("traversal must fail");
        assert!(matches!(err, MmprojValidationError::PathTraversal));
    }

    #[test]
    fn filename_is_short_no_directory() {
        let p = write_file(
            "logs_safe.gguf",
            &synthetic_gguf(MIN_PROJECTOR_BYTES as usize + 8),
        );
        let proj = MultimodalProjector::from_path(&p).expect("valid");
        let fname = proj.filename();
        assert_eq!(&*fname, "logs_safe.gguf");
        assert!(!fname.contains(std::path::MAIN_SEPARATOR));
    }

    #[test]
    fn from_path_returns_absolute() {
        let p = write_file(
            "abs.gguf",
            &synthetic_gguf(MIN_PROJECTOR_BYTES as usize + 8),
        );
        let proj = MultimodalProjector::from_path(&p).expect("valid");
        assert!(
            proj.path().is_absolute(),
            "canonicalize must yield absolute"
        );
    }
}
