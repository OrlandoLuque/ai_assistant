//! Slash commands (V104.2) — discovered, project-local prompt templates.
//!
//! A slash command is a `.md` file with optional YAML frontmatter that
//! lives under a commands directory (project-local or user-global). The
//! user invokes it as `/<name> [args...]`; the body is rendered into a
//! prompt with `{{argN}}` and `{{name}}` placeholders substituted.
//!
//! ## File format
//!
//! ```text
//! ---
//! name: review
//! description: Review code for bugs and clarity
//! args:
//!   - file_path
//!   - focus
//! model: claude-opus-4-7
//! tags: [code, review]
//! ---
//! Please review the file at {{file_path}}.
//! Focus on: {{focus}}.
//! ```
//!
//! Positional args also bind to `{{1}}`, `{{2}}`, etc. so commands work
//! even if the caller doesn't know the schema.
//!
//! ## Discovery
//!
//! Roots are scanned in order; later roots **override** earlier roots on
//! same `name`. Typical setup:
//!
//! 1. user-global: `<config-dir>/ai_assistant/commands/`
//! 2. project root: `<project>/.ai_assistant/commands/`
//!
//! ## Trust model
//!
//! Project-local command files are *user-supplied content* — they could
//! be hostile. Trust is delegated to the same `ApprovalHandler` used by
//! [`crate::project_conventions`]; the caller decides whether to require
//! it. Commands above `max_file_size` are rejected; symlinks are rejected
//! by default; UTF-8 is enforced; only `.md` files are loaded.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for slash-command discovery.
#[derive(Debug, Clone)]
pub struct SlashCommandConfig {
    /// Maximum bytes per command file. Default 64 KiB.
    pub max_file_size: u64,
    /// Whether to follow symlinks when reading commands. Default false.
    pub follow_symlinks: bool,
    /// Maximum number of commands loaded per root. Default 256.
    pub max_per_root: usize,
}

impl Default for SlashCommandConfig {
    fn default() -> Self {
        Self {
            max_file_size: 64 * 1024,
            follow_symlinks: false,
            max_per_root: 256,
        }
    }
}

// ============================================================================
// Types
// ============================================================================

/// A loaded slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommand {
    /// Canonical name (lowercase, no spaces). Defaults to filename stem
    /// if no `name:` in frontmatter.
    pub name: String,
    /// Optional description from frontmatter.
    pub description: Option<String>,
    /// Optional positional arg names from frontmatter.
    pub args: Vec<String>,
    /// Optional `model:` hint from frontmatter.
    pub model: Option<String>,
    /// Optional tags from frontmatter.
    pub tags: Vec<String>,
    /// Raw template body (everything after the frontmatter block).
    pub body: String,
    /// Path the command was loaded from.
    pub source_path: PathBuf,
}

/// Outcome of rendering a slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedCommand {
    /// The substituted body, ready to send as a prompt.
    pub prompt: String,
    /// Names of placeholders that were referenced but unbound (left
    /// verbatim in the output).
    pub unbound_placeholders: Vec<String>,
    /// Args supplied beyond the declared schema (still bound positionally).
    pub extra_positional: usize,
}

/// Errors during command discovery or load.
#[derive(Debug)]
pub enum SlashCommandError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    NotRegularFile(PathBuf),
    SymlinkRejected(PathBuf),
    InvalidUtf8(PathBuf),
    InvalidFrontmatter {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for SlashCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "I/O error reading {}: {}", path.display(), source)
            }
            Self::TooLarge { path, size, limit } => write!(
                f,
                "command file too large: {} ({} > {} bytes)",
                path.display(),
                size,
                limit
            ),
            Self::NotRegularFile(p) => write!(f, "not a regular file: {}", p.display()),
            Self::SymlinkRejected(p) => write!(f, "symlink rejected: {}", p.display()),
            Self::InvalidUtf8(p) => write!(f, "invalid UTF-8: {}", p.display()),
            Self::InvalidFrontmatter { path, message } => {
                write!(f, "invalid frontmatter in {}: {}", path.display(), message)
            }
        }
    }
}

impl std::error::Error for SlashCommandError {}

/// In-memory registry. Later roots override earlier roots on duplicate
/// command names.
#[derive(Debug, Default, Clone)]
pub struct SlashCommandRegistry {
    by_name: BTreeMap<String, SlashCommand>,
    /// Commands that failed to load, with reason — surface to user as a warning.
    pub load_errors: Vec<(PathBuf, String)>,
}

impl SlashCommandRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, cmd: SlashCommand) {
        self.by_name.insert(cmd.name.clone(), cmd);
    }

    pub fn get(&self, name: &str) -> Option<&SlashCommand> {
        self.by_name.get(&name.to_lowercase())
    }

    pub fn names(&self) -> Vec<String> {
        self.by_name.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Scan each root in order and build a [`SlashCommandRegistry`].
///
/// Roots are typically `[user_dir, project_dir]`; later roots override
/// earlier on duplicate names.
pub fn discover_slash_commands(
    roots: &[PathBuf],
    cfg: &SlashCommandConfig,
) -> SlashCommandRegistry {
    let mut reg = SlashCommandRegistry::new();
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut count = 0usize;
        for entry in entries.flatten() {
            if count >= cfg.max_per_root {
                break;
            }
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            match load_one(&path, cfg) {
                Ok(cmd) => {
                    reg.insert(cmd);
                    count += 1;
                }
                Err(e) => {
                    reg.load_errors.push((path, e.to_string()));
                }
            }
        }
    }
    reg
}

/// Render a command body by substituting `{{key}}` placeholders.
///
/// Binding rules:
/// - `{{<arg_name>}}` binds to the positional arg at that schema index.
/// - `{{1}}`, `{{2}}`, ... bind to positional args (1-based).
/// - Unbound placeholders are left verbatim AND reported.
pub fn render_slash_command(cmd: &SlashCommand, args: &[String]) -> RenderedCommand {
    let mut bindings: BTreeMap<String, String> = BTreeMap::new();

    // Bind positional 1-based.
    for (i, v) in args.iter().enumerate() {
        bindings.insert(format!("{}", i + 1), v.clone());
    }
    // Bind named.
    for (i, name) in cmd.args.iter().enumerate() {
        if let Some(v) = args.get(i) {
            bindings.insert(name.to_lowercase(), v.clone());
        }
    }

    let mut out = String::with_capacity(cmd.body.len());
    let mut unbound: Vec<String> = Vec::new();
    let bytes = cmd.body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 <= bytes.len() && &bytes[i..i + 2] == b"{{" {
            // Find `}}`
            if let Some(end_rel) = find_subseq(&bytes[i + 2..], b"}}") {
                let key_raw = &cmd.body[i + 2..i + 2 + end_rel];
                let key = key_raw.trim().to_lowercase();
                if let Some(v) = bindings.get(&key) {
                    out.push_str(v);
                } else {
                    out.push_str("{{");
                    out.push_str(key_raw);
                    out.push_str("}}");
                    if !unbound.contains(&key) && !key.is_empty() {
                        unbound.push(key);
                    }
                }
                i += 2 + end_rel + 2;
                continue;
            }
        }
        // copy one byte (safe — we never split a UTF-8 codepoint at a `{` boundary)
        out.push(bytes[i] as char);
        i += 1;
    }

    let extra = args.len().saturating_sub(cmd.args.len());
    RenderedCommand {
        prompt: out,
        unbound_placeholders: unbound,
        extra_positional: extra,
    }
}

// ============================================================================
// Internals
// ============================================================================

fn load_one(path: &Path, cfg: &SlashCommandConfig) -> Result<SlashCommand, SlashCommandError> {
    let meta = match if cfg.follow_symlinks {
        fs::metadata(path)
    } else {
        fs::symlink_metadata(path)
    } {
        Ok(m) => m,
        Err(e) => {
            return Err(SlashCommandError::Io {
                path: path.to_path_buf(),
                source: e,
            });
        }
    };
    if meta.file_type().is_symlink() && !cfg.follow_symlinks {
        return Err(SlashCommandError::SymlinkRejected(path.to_path_buf()));
    }
    if !meta.is_file() {
        return Err(SlashCommandError::NotRegularFile(path.to_path_buf()));
    }
    let size = meta.len();
    if size > cfg.max_file_size {
        return Err(SlashCommandError::TooLarge {
            path: path.to_path_buf(),
            size,
            limit: cfg.max_file_size,
        });
    }

    let bytes = fs::read(path).map_err(|e| SlashCommandError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| SlashCommandError::InvalidUtf8(path.to_path_buf()))?;

    let (frontmatter, body) = split_frontmatter(text);

    // Default name: filename stem, lowercased.
    let default_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unnamed")
        .to_lowercase();

    let mut name = default_name;
    let mut description: Option<String> = None;
    let mut args: Vec<String> = Vec::new();
    let mut model: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();

    if let Some(fm) = frontmatter {
        for (k, v) in parse_frontmatter(fm).map_err(|e| SlashCommandError::InvalidFrontmatter {
            path: path.to_path_buf(),
            message: e,
        })? {
            match k.as_str() {
                "name" => name = v.trim().to_lowercase(),
                "description" => description = Some(v.trim().to_string()),
                "model" => model = Some(v.trim().to_string()),
                "args" => args = parse_inline_list(&v),
                "tags" => tags = parse_inline_list(&v),
                _ => {}
            }
        }
    }

    Ok(SlashCommand {
        name,
        description,
        args,
        model,
        tags,
        body: body.to_string(),
        source_path: path.to_path_buf(),
    })
}

/// Returns `(frontmatter_text, body)`. Frontmatter is `---\n...\n---\n`
/// at the very start.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    // Strip BOM
    let stripped = text.strip_prefix('\u{feff}').unwrap_or(text);
    if !stripped.starts_with("---\n") && !stripped.starts_with("---\r\n") {
        return (None, stripped);
    }
    let after_first = if stripped.starts_with("---\r\n") {
        &stripped[5..]
    } else {
        &stripped[4..]
    };
    // Find next `---` on its own line.
    let mut idx = 0usize;
    let mut at_line_start = true;
    let bytes = after_first.as_bytes();
    while idx < bytes.len() {
        if at_line_start && idx + 3 <= bytes.len() && &bytes[idx..idx + 3] == b"---" {
            // Must be followed by newline or EOF.
            let after = idx + 3;
            let ok = after >= bytes.len()
                || bytes[after] == b'\n'
                || (after + 1 <= bytes.len() && bytes[after] == b'\r');
            if ok {
                let fm = &after_first[..idx];
                let mut body_start = after;
                while body_start < bytes.len()
                    && (bytes[body_start] == b'\n' || bytes[body_start] == b'\r')
                {
                    body_start += 1;
                }
                return (Some(fm), &after_first[body_start..]);
            }
        }
        at_line_start = bytes[idx] == b'\n';
        idx += 1;
    }
    // Unterminated frontmatter — treat as if there were none.
    (None, stripped)
}

/// Parse a tiny key:value frontmatter (one entry per line). Multi-line
/// arrays use `[a, b, c]` inline syntax. Quoted values supported.
/// Returns Err only on truly malformed lines (key without `:`).
fn parse_frontmatter(fm: &str) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (lineno, raw) in fm.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // YAML-style block array: `args:` then `- foo` lines.
        if line.starts_with("- ") {
            // Append to last key as a list item (only if last value is `[...]`-extensible).
            if let Some(last) = out.last_mut() {
                let item = line[2..].trim();
                if last.1.is_empty() {
                    last.1 = format!("[{}]", item);
                } else if last.1.starts_with('[') && last.1.ends_with(']') {
                    let inner = &last.1[1..last.1.len() - 1];
                    last.1 = format!("[{}, {}]", inner, item);
                }
                continue;
            } else {
                return Err(format!("dangling list item at line {}", lineno + 1));
            }
        }
        if let Some(colon) = line.find(':') {
            let key = line[..colon].trim().to_lowercase();
            let val = line[colon + 1..].trim().to_string();
            out.push((key, val));
        } else {
            return Err(format!("missing ':' on line {}", lineno + 1));
        }
    }
    Ok(out)
}

/// Parse `[a, b, "quoted item", c]` or whitespace-separated bare values.
fn parse_inline_list(raw: &str) -> Vec<String> {
    let s = raw.trim();
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };
    inner
        .split(',')
        .map(|p| {
            let t = p.trim();
            t.trim_matches(|c: char| c == '"' || c == '\'').to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn find_subseq(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    for i in 0..=hay.len() - needle.len() {
        if &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ai_assistant_slashcmds_{}_{}",
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

    // ---------- frontmatter parsing ----------

    #[test]
    fn split_no_frontmatter() {
        let (fm, body) = split_frontmatter("hello\nworld\n");
        assert!(fm.is_none());
        assert_eq!(body, "hello\nworld\n");
    }

    #[test]
    fn split_with_frontmatter() {
        let (fm, body) = split_frontmatter("---\nname: foo\n---\nbody here\n");
        assert_eq!(fm, Some("name: foo\n"));
        assert_eq!(body, "body here\n");
    }

    #[test]
    fn split_strips_bom() {
        let (fm, body) = split_frontmatter("\u{feff}---\nname: foo\n---\nbody\n");
        assert_eq!(fm, Some("name: foo\n"));
        assert_eq!(body, "body\n");
    }

    #[test]
    fn split_unterminated_frontmatter_returns_none() {
        let (fm, body) = split_frontmatter("---\nname: foo\nstill no closer\n");
        assert!(fm.is_none());
        assert!(body.starts_with("---"));
    }

    #[test]
    fn parse_frontmatter_basic() {
        let pairs = parse_frontmatter("name: foo\ndescription: bar\n").unwrap();
        assert_eq!(
            pairs,
            vec![
                ("name".into(), "foo".into()),
                ("description".into(), "bar".into()),
            ]
        );
    }

    #[test]
    fn parse_frontmatter_inline_list() {
        let pairs = parse_frontmatter("args: [a, b, c]\n").unwrap();
        assert_eq!(pairs[0].1, "[a, b, c]");
    }

    #[test]
    fn parse_frontmatter_block_list() {
        let pairs = parse_frontmatter("args:\n- a\n- b\n").unwrap();
        assert_eq!(pairs[0].1, "[a, b]");
    }

    #[test]
    fn parse_frontmatter_rejects_missing_colon() {
        let err = parse_frontmatter("nameonly\n").unwrap_err();
        assert!(err.contains("missing ':'"));
    }

    #[test]
    fn parse_frontmatter_skips_comments_and_blanks() {
        let pairs = parse_frontmatter("# comment\n\nname: foo\n").unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "name");
    }

    // ---------- inline list ----------

    #[test]
    fn inline_list_parses_brackets() {
        let v = parse_inline_list("[a, \"b c\", d]");
        assert_eq!(v, vec!["a", "b c", "d"]);
    }

    #[test]
    fn inline_list_parses_bare() {
        let v = parse_inline_list("a, b, c");
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    // ---------- discovery ----------

    #[test]
    fn discover_loads_md_files() {
        let dir = tmpdir("disc");
        write(&dir.join("a.md"), "---\nname: alpha\n---\nbody A\n");
        write(&dir.join("b.md"), "body B (no frontmatter)\n");
        write(&dir.join("ignored.txt"), "not a command");
        let reg = discover_slash_commands(&[dir], &SlashCommandConfig::default());
        assert_eq!(reg.len(), 2);
        assert!(reg.get("alpha").is_some());
        assert!(reg.get("b").is_some()); // filename stem fallback
    }

    #[test]
    fn discover_later_root_overrides_earlier() {
        let a = tmpdir("ovr_a");
        let b = tmpdir("ovr_b");
        write(
            &a.join("x.md"),
            "---\nname: x\ndescription: from-a\n---\nA\n",
        );
        write(
            &b.join("x.md"),
            "---\nname: x\ndescription: from-b\n---\nB\n",
        );
        let reg = discover_slash_commands(&[a, b], &SlashCommandConfig::default());
        assert_eq!(reg.get("x").unwrap().description.as_deref(), Some("from-b"));
    }

    #[test]
    fn discover_skips_oversized() {
        let dir = tmpdir("size");
        let big_body = "x".repeat(200);
        write(&dir.join("big.md"), &big_body);
        let mut cfg = SlashCommandConfig::default();
        cfg.max_file_size = 50;
        let reg = discover_slash_commands(&[dir], &cfg);
        assert_eq!(reg.len(), 0);
        assert_eq!(reg.load_errors.len(), 1);
    }

    #[test]
    fn discover_caps_per_root() {
        let dir = tmpdir("cap");
        for i in 0..10 {
            write(&dir.join(&format!("c{}.md", i)), "body\n");
        }
        let mut cfg = SlashCommandConfig::default();
        cfg.max_per_root = 3;
        let reg = discover_slash_commands(&[dir], &cfg);
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn discover_handles_missing_root_gracefully() {
        let reg = discover_slash_commands(
            &[PathBuf::from("/no/such/dir/zzz")],
            &SlashCommandConfig::default(),
        );
        assert!(reg.is_empty());
    }

    // ---------- rendering ----------

    fn make_cmd(args: &[&str], body: &str) -> SlashCommand {
        SlashCommand {
            name: "t".into(),
            description: None,
            args: args.iter().map(|s| s.to_string()).collect(),
            model: None,
            tags: vec![],
            body: body.into(),
            source_path: PathBuf::from("/tmp/t.md"),
        }
    }

    #[test]
    fn render_named_args() {
        let cmd = make_cmd(&["file", "focus"], "Review {{file}} for {{focus}}.");
        let r = render_slash_command(&cmd, &["a.rs".into(), "perf".into()]);
        assert_eq!(r.prompt, "Review a.rs for perf.");
        assert!(r.unbound_placeholders.is_empty());
    }

    #[test]
    fn render_positional_args() {
        let cmd = make_cmd(&[], "Pos1 = {{1}}, pos2 = {{2}}.");
        let r = render_slash_command(&cmd, &["alpha".into(), "beta".into()]);
        assert_eq!(r.prompt, "Pos1 = alpha, pos2 = beta.");
    }

    #[test]
    fn render_reports_unbound_placeholders() {
        let cmd = make_cmd(&["a"], "{{a}} {{b}} {{c}}");
        let r = render_slash_command(&cmd, &["X".into()]);
        assert_eq!(r.prompt, "X {{b}} {{c}}");
        assert_eq!(r.unbound_placeholders.len(), 2);
    }

    #[test]
    fn render_reports_extra_positional() {
        let cmd = make_cmd(&["a"], "{{a}}");
        let r = render_slash_command(&cmd, &["X".into(), "Y".into(), "Z".into()]);
        assert_eq!(r.extra_positional, 2);
    }

    #[test]
    fn render_handles_no_placeholders() {
        let cmd = make_cmd(&[], "static body");
        let r = render_slash_command(&cmd, &["ignored".into()]);
        assert_eq!(r.prompt, "static body");
    }

    #[test]
    fn render_is_case_insensitive_for_placeholders() {
        let cmd = make_cmd(&["File"], "{{FILE}} {{file}}");
        let r = render_slash_command(&cmd, &["a.rs".into()]);
        assert_eq!(r.prompt, "a.rs a.rs");
    }

    // ---------- end-to-end ----------

    #[test]
    fn end_to_end_load_and_render() {
        let dir = tmpdir("e2e");
        write(
            &dir.join("review.md"),
            "---\nname: review\ndescription: Review code\nargs:\n- file\n- focus\n---\nReview {{file}}, focus on {{focus}}.\n",
        );
        let reg = discover_slash_commands(&[dir], &SlashCommandConfig::default());
        let cmd = reg.get("review").unwrap();
        assert_eq!(cmd.description.as_deref(), Some("Review code"));
        assert_eq!(cmd.args, vec!["file", "focus"]);
        let r = render_slash_command(cmd, &["src/lib.rs".into(), "memory leaks".into()]);
        assert!(r.prompt.contains("Review src/lib.rs"));
        assert!(r.prompt.contains("memory leaks"));
    }

    #[test]
    fn registry_get_is_case_insensitive() {
        let mut reg = SlashCommandRegistry::new();
        reg.insert(make_cmd(&[], "x"));
        // The default name in make_cmd is "t".
        assert!(reg.get("T").is_some());
    }
}
