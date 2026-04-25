//! File-reference expansion (V104.7)
//!
//! Parses `@path` and `@path#L37-42` tokens out of free-form prompt text,
//! reads the corresponding file (or line range) from disk, and substitutes
//! the contents inline.
//!
//! # Security baseline
//!
//! - Reference paths are resolved relative to a `project_root` and the
//!   resolved path must stay inside that root (no `..` escape).
//! - Symlinks are rejected by default.
//! - Each file is capped at `max_file_size` bytes; total expansion over a
//!   single message is capped at `max_total_expanded` bytes.
//! - References inside fenced code blocks (` ``` ` or `~~~`) are NOT
//!   expanded — code samples must be quotable verbatim.
//! - References preceded by an alphanumeric/`@` character are skipped, to
//!   avoid mis-parsing email addresses or `@@scoped/package` names.
//!
//! # Format
//!
//! Successful expansions are inserted in place of the original token as:
//!
//! ```text
//! <file path="src/foo.rs" lines="10-20">
//! ...content...
//! </file>
//! ```
//!
//! Skipped references stay verbatim in the output and are also reported in
//! [`FileRefExpansion::skipped`] so the caller can warn the user.
//!
//! # Quick start
//!
//! ```no_run
//! use ai_assistant::file_references::{expand_file_refs, FileRefConfig};
//! use std::path::PathBuf;
//!
//! let cfg = FileRefConfig {
//!     project_root: PathBuf::from("./my-project"),
//!     ..FileRefConfig::default()
//! };
//! let res = expand_file_refs("see @src/main.rs#L1-20", &cfg);
//! println!("{}", res.expanded_text);
//! ```

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for file-reference expansion.
#[derive(Debug, Clone)]
pub struct FileRefConfig {
    /// Root directory references are resolved against. References whose
    /// resolved path falls outside this root are rejected.
    pub project_root: PathBuf,
    /// Maximum size of any single file (bytes). Default: 256 KiB.
    pub max_file_size: u64,
    /// Maximum total expanded size over a single call (bytes). Once this
    /// is hit further references are skipped and `truncated = true`.
    /// Default: 1 MiB.
    pub max_total_expanded: u64,
    /// Maximum number of references expanded per call. Default: 32.
    pub max_refs_per_message: usize,
    /// Whether symlinks may be expanded. Default false.
    pub follow_symlinks: bool,
}

impl Default for FileRefConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::new(),
            max_file_size: 256 * 1024,
            max_total_expanded: 1024 * 1024,
            max_refs_per_message: 32,
            follow_symlinks: false,
        }
    }
}

// ============================================================================
// Parsed reference + expansion result
// ============================================================================

/// A parsed reference token, before any I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRef {
    /// Original token as it appeared in the input, e.g. `@src/foo.rs#L10-20`.
    pub raw: String,
    /// Path component (always relative; resolved against `project_root`).
    pub path: PathBuf,
    /// Inclusive line range start (1-based), if specified.
    pub line_start: Option<u64>,
    /// Inclusive line range end (1-based), if specified.
    pub line_end: Option<u64>,
    /// Byte offset where the token starts in the input (UTF-8 char-aligned).
    pub byte_offset: usize,
    /// Byte length of the token.
    pub byte_length: usize,
}

/// A successfully expanded reference.
#[derive(Debug, Clone)]
pub struct ExpandedRef {
    pub raw: String,
    pub path: PathBuf,
    pub line_start: Option<u64>,
    pub line_end: Option<u64>,
    /// The block that replaced the token in the output text.
    pub block: String,
}

/// Result of `expand_file_refs`.
#[derive(Debug, Clone)]
pub struct FileRefExpansion {
    /// Input text with references substituted in place.
    pub expanded_text: String,
    /// Successfully expanded references (in order of appearance).
    pub refs: Vec<ExpandedRef>,
    /// References that were parsed but not expanded, with the reason.
    pub skipped: Vec<(FileRef, FileRefError)>,
    /// True if expansion stopped early due to total-size or count caps.
    pub truncated: bool,
}

/// Reasons a reference may fail to expand.
#[derive(Debug, Clone)]
pub enum FileRefError {
    NotFound(PathBuf),
    OutsideRoot(PathBuf),
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    NotRegularFile(PathBuf),
    SymlinkRejected(PathBuf),
    InvalidUtf8(PathBuf),
    LineRangeOutOfBounds {
        path: PathBuf,
        start: u64,
        end: u64,
        max_lines: u64,
    },
    InvalidLineRange {
        path: PathBuf,
        start: u64,
        end: u64,
    },
    Io(String),
    QuotaExceeded,
}

impl std::fmt::Display for FileRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "file not found: {}", p.display()),
            Self::OutsideRoot(p) => {
                write!(f, "path escapes project root: {}", p.display())
            }
            Self::TooLarge { path, size, limit } => write!(
                f,
                "file too large: {} ({} > {} bytes)",
                path.display(),
                size,
                limit
            ),
            Self::NotRegularFile(p) => {
                write!(f, "not a regular file: {}", p.display())
            }
            Self::SymlinkRejected(p) => {
                write!(f, "symlink rejected: {}", p.display())
            }
            Self::InvalidUtf8(p) => {
                write!(f, "invalid UTF-8: {}", p.display())
            }
            Self::LineRangeOutOfBounds {
                path,
                start,
                end,
                max_lines,
            } => write!(
                f,
                "line range L{}-{} out of bounds for {} ({} lines)",
                start,
                end,
                path.display(),
                max_lines
            ),
            Self::InvalidLineRange { path, start, end } => write!(
                f,
                "invalid line range L{}-{} for {}",
                start,
                end,
                path.display()
            ),
            Self::Io(s) => write!(f, "I/O error: {}", s),
            Self::QuotaExceeded => write!(f, "expansion quota exceeded"),
        }
    }
}

impl std::error::Error for FileRefError {}

// ============================================================================
// Parsing
// ============================================================================

/// Extract all `@path[#L<start>[-<end>]]` references from `text`.
///
/// Skips:
/// - tokens preceded by an alphanumeric, `_`, `.` or `@` (avoids emails,
///   `@@scoped/foo`, etc.),
/// - tokens inside fenced code blocks.
pub fn parse_file_refs(text: &str) -> Vec<FileRef> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    let mut in_fence = false;
    let mut fence_marker: Option<&[u8]> = None;
    let mut at_line_start = true;

    while i < bytes.len() {
        // Track newlines / fence boundaries on a line basis.
        if at_line_start {
            // Skip leading whitespace for fence detection only.
            let mut j = i;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            let rest = &bytes[j..];
            if let Some(marker) = fence_marker {
                if rest.starts_with(marker) {
                    in_fence = false;
                    fence_marker = None;
                }
            } else if rest.starts_with(b"```") {
                in_fence = true;
                fence_marker = Some(b"```");
            } else if rest.starts_with(b"~~~") {
                in_fence = true;
                fence_marker = Some(b"~~~");
            }
        }

        let b = bytes[i];
        if b == b'\n' {
            at_line_start = true;
            i += 1;
            continue;
        }
        if b != b' ' && b != b'\t' {
            at_line_start = false;
        }

        if in_fence {
            i += 1;
            continue;
        }

        if b == b'@' {
            // Check char before: skip if alphanumeric/_/./@.
            let before_ok = if i == 0 {
                true
            } else {
                let p = bytes[i - 1];
                !(p.is_ascii_alphanumeric() || p == b'_' || p == b'.' || p == b'@')
            };
            if before_ok {
                if let Some(parsed) = try_parse_ref(text, i) {
                    let len = parsed.byte_length;
                    out.push(parsed);
                    i += len;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn try_parse_ref(text: &str, start: usize) -> Option<FileRef> {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[start], b'@');
    let mut i = start + 1;
    let path_start = i;
    while i < bytes.len() {
        let c = bytes[i];
        if is_path_char(c) {
            i += 1;
        } else {
            break;
        }
    }
    if i == path_start {
        return None;
    }
    let path_end = i;
    let path_str = &text[path_start..path_end];
    // Trim trailing punctuation that's likely not part of the path.
    let trimmed_path = trim_trailing_punct(path_str);
    if trimmed_path.is_empty() {
        return None;
    }
    let trimmed_len = trimmed_path.len();
    let path_end = path_start + trimmed_len;
    let mut i = path_end;

    let mut line_start = None;
    let mut line_end = None;
    if i + 2 <= bytes.len() && &bytes[i..i + 2] == b"#L" {
        let after_l = i + 2;
        let mut j = after_l;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > after_l {
            let n: u64 = text[after_l..j].parse().ok()?;
            line_start = Some(n);
            line_end = Some(n);
            i = j;
            if i < bytes.len() && bytes[i] == b'-' {
                let after_dash = i + 1;
                let mut k = after_dash;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                if k > after_dash {
                    let n2: u64 = text[after_dash..k].parse().ok()?;
                    line_end = Some(n2);
                    i = k;
                }
            }
        }
    }

    let raw = text[start..i].to_string();
    let path = PathBuf::from(trimmed_path.replace('\\', "/"));
    Some(FileRef {
        raw,
        path,
        line_start,
        line_end,
        byte_offset: start,
        byte_length: i - start,
    })
}

fn is_path_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' || c == b'/' || c == b'\\'
}

/// Trim trailing characters that are likely sentence punctuation rather
/// than part of the path: `. , ; : ! ? ) ] }`.
/// Note: a single trailing `.` is kept if it follows alphanumerics
/// (e.g. `foo.rs`); we only strip if it's the FINAL char and there's
/// already a `.` somewhere earlier in the segment, which is a heuristic
/// for `Cargo.toml.` ending a sentence — but we keep it simple and only
/// strip a small set of clearly-sentence punctuation.
fn trim_trailing_punct(s: &str) -> &str {
    let mut end = s.len();
    let bytes = s.as_bytes();
    while end > 0 {
        let c = bytes[end - 1];
        let punct = matches!(c, b',' | b';' | b':' | b'!' | b'?' | b')' | b']' | b'}');
        if punct {
            end -= 1;
        } else {
            break;
        }
    }
    &s[..end]
}

// ============================================================================
// Expansion
// ============================================================================

/// Parse references from `text`, expand each one against `config.project_root`,
/// and return the substituted text plus per-reference outcomes.
pub fn expand_file_refs(text: &str, config: &FileRefConfig) -> FileRefExpansion {
    let refs = parse_file_refs(text);
    let mut expanded: Vec<ExpandedRef> = Vec::new();
    let mut skipped: Vec<(FileRef, FileRefError)> = Vec::new();
    let mut truncated = false;
    let mut total_bytes: u64 = 0;

    // Resolve project_root once.
    let root_canonical = match dunce_canonicalize(&config.project_root) {
        Ok(p) => p,
        Err(_) => {
            // No usable root — every ref will fail OutsideRoot.
            for r in refs {
                skipped.push((r.clone(), FileRefError::OutsideRoot(r.path.clone())));
            }
            return FileRefExpansion {
                expanded_text: text.to_string(),
                refs: expanded,
                skipped,
                truncated,
            };
        }
    };

    // Decide outcome for each reference.
    let mut outcomes: Vec<(usize, usize, Option<String>)> = Vec::new();
    // (byte_offset, byte_length, replacement) — replacement None means leave verbatim.

    for r in &refs {
        if expanded.len() >= config.max_refs_per_message {
            truncated = true;
            skipped.push((r.clone(), FileRefError::QuotaExceeded));
            outcomes.push((r.byte_offset, r.byte_length, None));
            continue;
        }
        match try_expand_one(r, config, &root_canonical) {
            Ok((block, used)) => {
                if total_bytes.saturating_add(used) > config.max_total_expanded {
                    truncated = true;
                    skipped.push((r.clone(), FileRefError::QuotaExceeded));
                    outcomes.push((r.byte_offset, r.byte_length, None));
                } else {
                    total_bytes += used;
                    outcomes.push((r.byte_offset, r.byte_length, Some(block.clone())));
                    expanded.push(ExpandedRef {
                        raw: r.raw.clone(),
                        path: r.path.clone(),
                        line_start: r.line_start,
                        line_end: r.line_end,
                        block,
                    });
                }
            }
            Err(e) => {
                skipped.push((r.clone(), e));
                outcomes.push((r.byte_offset, r.byte_length, None));
            }
        }
    }

    // Build expanded_text.
    let mut out = String::with_capacity(text.len() + total_bytes as usize);
    let mut cursor = 0usize;
    for (offset, length, replacement) in outcomes {
        if cursor < offset {
            out.push_str(&text[cursor..offset]);
        }
        match replacement {
            Some(r) => out.push_str(&r),
            None => out.push_str(&text[offset..offset + length]),
        }
        cursor = offset + length;
    }
    if cursor < text.len() {
        out.push_str(&text[cursor..]);
    }

    FileRefExpansion {
        expanded_text: out,
        refs: expanded,
        skipped,
        truncated,
    }
}

fn try_expand_one(
    r: &FileRef,
    config: &FileRefConfig,
    root_canonical: &Path,
) -> Result<(String, u64), FileRefError> {
    // Reject absolute paths up front. Note: on Windows `Path::is_absolute()`
    // returns false for `/etc/passwd`-style paths (no drive letter), but we
    // still want to reject those for security: a leading `/` or `\` is a
    // strong signal of root-anchored intent.
    let path_str = r.path.to_string_lossy();
    let starts_with_root = path_str.starts_with('/') || path_str.starts_with('\\');
    if r.path.is_absolute() || starts_with_root {
        return Err(FileRefError::OutsideRoot(r.path.clone()));
    }
    // Reject any '..' segment up front (cheap pre-check).
    for comp in r.path.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err(FileRefError::OutsideRoot(r.path.clone()));
        }
    }

    let joined = root_canonical.join(&r.path);
    let meta = match if config.follow_symlinks {
        fs::metadata(&joined)
    } else {
        fs::symlink_metadata(&joined)
    } {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(FileRefError::NotFound(r.path.clone()));
        }
        Err(e) => return Err(FileRefError::Io(e.to_string())),
    };

    if meta.file_type().is_symlink() && !config.follow_symlinks {
        return Err(FileRefError::SymlinkRejected(r.path.clone()));
    }
    if !meta.is_file() {
        return Err(FileRefError::NotRegularFile(r.path.clone()));
    }

    // Realpath escape check.
    let real = match dunce_canonicalize(&joined) {
        Ok(p) => p,
        Err(e) => return Err(FileRefError::Io(e.to_string())),
    };
    if !real.starts_with(root_canonical) {
        return Err(FileRefError::OutsideRoot(r.path.clone()));
    }

    let size = meta.len();
    if size > config.max_file_size {
        return Err(FileRefError::TooLarge {
            path: r.path.clone(),
            size,
            limit: config.max_file_size,
        });
    }

    let bytes = fs::read(&real).map_err(|e| FileRefError::Io(e.to_string()))?;
    let content = match std::str::from_utf8(&bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return Err(FileRefError::InvalidUtf8(r.path.clone())),
    };

    let (selected, lines_attr) = if let (Some(s), Some(e)) = (r.line_start, r.line_end) {
        if s == 0 || e == 0 || s > e {
            return Err(FileRefError::InvalidLineRange {
                path: r.path.clone(),
                start: s,
                end: e,
            });
        }
        let total: u64 = content.lines().count() as u64;
        if s > total {
            return Err(FileRefError::LineRangeOutOfBounds {
                path: r.path.clone(),
                start: s,
                end: e,
                max_lines: total,
            });
        }
        let clamped_end = e.min(total);
        let mut buf = String::new();
        for (idx, line) in content.lines().enumerate() {
            let n = (idx as u64) + 1;
            if n >= s && n <= clamped_end {
                buf.push_str(line);
                buf.push('\n');
            } else if n > clamped_end {
                break;
            }
        }
        (buf, format!("{}-{}", s, clamped_end))
    } else {
        (content, String::new())
    };

    let path_str = r.path.to_string_lossy().replace('\\', "/");
    let header = if lines_attr.is_empty() {
        format!("<file path=\"{}\">", path_str)
    } else {
        format!("<file path=\"{}\" lines=\"{}\">", path_str, lines_attr)
    };
    let mut block = String::with_capacity(header.len() + selected.len() + 16);
    block.push_str(&header);
    block.push('\n');
    block.push_str(&selected);
    if !block.ends_with('\n') {
        block.push('\n');
    }
    block.push_str("</file>");

    let used = block.len() as u64;
    Ok((block, used))
}

/// `std::fs::canonicalize` returns UNC paths on Windows (`\\?\C:\foo`),
/// which trip up downstream `starts_with` and joins. This is a tiny
/// shim equivalent to the `dunce` crate: strip the UNC prefix on Windows.
fn dunce_canonicalize(p: &Path) -> io::Result<PathBuf> {
    let canon = fs::canonicalize(p)?;
    #[cfg(windows)]
    {
        let s = canon.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(rest));
        }
    }
    Ok(canon)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn tmpdir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ai_assistant_filerefs_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    fn cfg_for(root: &Path) -> FileRefConfig {
        FileRefConfig {
            project_root: root.to_path_buf(),
            ..FileRefConfig::default()
        }
    }

    // ---------- parser ----------

    #[test]
    fn parse_simple_ref() {
        let refs = parse_file_refs("see @src/foo.rs for details");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].path, PathBuf::from("src/foo.rs"));
        assert!(refs[0].line_start.is_none());
    }

    #[test]
    fn parse_ref_with_line_range() {
        let refs = parse_file_refs("look at @src/lib.rs#L37-42 plz");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].line_start, Some(37));
        assert_eq!(refs[0].line_end, Some(42));
    }

    #[test]
    fn parse_ref_with_single_line() {
        let refs = parse_file_refs("@src/foo.rs#L10 only");
        assert_eq!(refs[0].line_start, Some(10));
        assert_eq!(refs[0].line_end, Some(10));
    }

    #[test]
    fn parse_skips_email() {
        let refs = parse_file_refs("contact me at user@example.com today");
        assert!(refs.is_empty());
    }

    #[test]
    fn parse_skips_inside_fence() {
        let text = "Outside @a.rs\n```\ninside @b.rs\n```\nafter @c.rs";
        let refs = parse_file_refs(text);
        let paths: Vec<_> = refs
            .iter()
            .map(|r| r.path.to_string_lossy().to_string())
            .collect();
        assert_eq!(paths, vec!["a.rs", "c.rs"]);
    }

    #[test]
    fn parse_handles_trailing_punctuation() {
        let refs = parse_file_refs("(see @src/foo.rs), then @bar.rs.");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].path, PathBuf::from("src/foo.rs"));
        // Trailing '.' is kept (could be a real path); we don't strip it.
        assert!(refs[1].path.to_string_lossy().starts_with("bar.rs"));
    }

    #[test]
    fn parse_multiple_refs_per_line() {
        let refs = parse_file_refs("@a.rs and @b.rs#L1-5 and @c/d.rs");
        assert_eq!(refs.len(), 3);
    }

    // ---------- expansion ----------

    #[test]
    fn expand_simple_file() {
        let dir = tmpdir("simple");
        write(&dir.join("foo.txt"), "alpha\nbeta\ngamma\n");
        let res = expand_file_refs("see @foo.txt now", &cfg_for(&dir));
        assert_eq!(res.refs.len(), 1);
        assert!(res.expanded_text.contains("<file path=\"foo.txt\">"));
        assert!(res.expanded_text.contains("alpha"));
        assert!(res.skipped.is_empty());
    }

    #[test]
    fn expand_with_line_range() {
        let dir = tmpdir("lines");
        write(&dir.join("a.txt"), "1\n2\n3\n4\n5\n6\n7\n8\n");
        let res = expand_file_refs("@a.txt#L3-5", &cfg_for(&dir));
        assert_eq!(res.refs.len(), 1);
        let block = &res.refs[0].block;
        assert!(block.contains("lines=\"3-5\""));
        assert!(block.contains("3\n4\n5"));
        assert!(!block.contains("\n2\n"));
        assert!(!block.contains("\n6\n"));
    }

    #[test]
    fn expand_clamps_end_to_eof() {
        let dir = tmpdir("clamp");
        write(&dir.join("a.txt"), "1\n2\n3\n");
        let res = expand_file_refs("@a.txt#L2-99", &cfg_for(&dir));
        assert_eq!(res.refs.len(), 1);
        assert!(res.refs[0].block.contains("lines=\"2-3\""));
    }

    #[test]
    fn expand_rejects_outside_root() {
        let dir = tmpdir("escape");
        let _outside = std::env::temp_dir().join("outside_target.txt");
        let _ = fs::write(&_outside, b"secret");
        // Path with .. should be rejected at parse-precheck level.
        let res = expand_file_refs("@../outside_target.txt", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        assert_eq!(res.skipped.len(), 1);
        match &res.skipped[0].1 {
            FileRefError::OutsideRoot(_) => {}
            other => panic!("expected OutsideRoot, got {:?}", other),
        }
    }

    #[test]
    fn expand_rejects_absolute_path() {
        let dir = tmpdir("abs");
        let res = expand_file_refs("@/etc/passwd", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        assert_eq!(res.skipped.len(), 1);
        match &res.skipped[0].1 {
            FileRefError::OutsideRoot(_) => {}
            other => panic!("expected OutsideRoot, got {:?}", other),
        }
    }

    #[test]
    fn expand_reports_not_found() {
        let dir = tmpdir("notfound");
        let res = expand_file_refs("@missing.txt", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        match &res.skipped[0].1 {
            FileRefError::NotFound(_) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
        assert!(res.expanded_text.contains("@missing.txt"));
    }

    #[test]
    fn expand_size_cap_per_file() {
        let dir = tmpdir("size");
        let big = "x".repeat(2000);
        write(&dir.join("big.txt"), &big);
        let mut cfg = cfg_for(&dir);
        cfg.max_file_size = 1000;
        let res = expand_file_refs("@big.txt", &cfg);
        assert!(res.refs.is_empty());
        match &res.skipped[0].1 {
            FileRefError::TooLarge { .. } => {}
            other => panic!("expected TooLarge, got {:?}", other),
        }
    }

    #[test]
    fn expand_respects_max_refs_per_message() {
        let dir = tmpdir("count");
        write(&dir.join("a.txt"), "a\n");
        let mut cfg = cfg_for(&dir);
        cfg.max_refs_per_message = 2;
        let res = expand_file_refs("@a.txt @a.txt @a.txt @a.txt", &cfg);
        assert_eq!(res.refs.len(), 2);
        assert!(res.truncated);
        assert_eq!(res.skipped.len(), 2);
    }

    #[test]
    fn expand_respects_total_size_cap() {
        let dir = tmpdir("total");
        let body = "y".repeat(500);
        write(&dir.join("a.txt"), &body);
        let mut cfg = cfg_for(&dir);
        cfg.max_total_expanded = 800;
        let res = expand_file_refs("@a.txt @a.txt @a.txt", &cfg);
        // First fits, second causes truncation.
        assert_eq!(res.refs.len(), 1);
        assert!(res.truncated);
    }

    #[test]
    fn expand_invalid_line_range() {
        let dir = tmpdir("invalid_range");
        write(&dir.join("a.txt"), "x\ny\nz\n");
        let res = expand_file_refs("@a.txt#L5-2", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        match &res.skipped[0].1 {
            FileRefError::InvalidLineRange { .. } => {}
            other => panic!("expected InvalidLineRange, got {:?}", other),
        }
    }

    #[test]
    fn expand_line_range_out_of_bounds() {
        let dir = tmpdir("oob");
        write(&dir.join("a.txt"), "x\ny\n");
        let res = expand_file_refs("@a.txt#L10-20", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        match &res.skipped[0].1 {
            FileRefError::LineRangeOutOfBounds { .. } => {}
            other => panic!("expected LineRangeOutOfBounds, got {:?}", other),
        }
    }

    #[test]
    fn expand_ignores_invalid_utf8() {
        let dir = tmpdir("utf8");
        let path = dir.join("bad.bin");
        fs::write(&path, &[0xFFu8, 0xFE, 0xFD]).unwrap();
        let res = expand_file_refs("@bad.bin", &cfg_for(&dir));
        assert!(res.refs.is_empty());
        match &res.skipped[0].1 {
            FileRefError::InvalidUtf8(_) => {}
            other => panic!("expected InvalidUtf8, got {:?}", other),
        }
    }

    #[test]
    fn expand_preserves_text_around_refs() {
        let dir = tmpdir("around");
        write(&dir.join("a.txt"), "AAA\n");
        let res = expand_file_refs("before @a.txt after", &cfg_for(&dir));
        assert!(res.expanded_text.starts_with("before "));
        assert!(res.expanded_text.ends_with(" after"));
        assert!(res.expanded_text.contains("AAA"));
    }

    #[test]
    fn expand_unresolvable_root_skips_all() {
        let cfg = FileRefConfig {
            project_root: PathBuf::from("/this/path/should/never/exist/zzz"),
            ..FileRefConfig::default()
        };
        let res = expand_file_refs("@a.txt", &cfg);
        assert!(res.refs.is_empty());
        assert_eq!(res.skipped.len(), 1);
    }
}
