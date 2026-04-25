//! Project conventions loader (V104.1)
//!
//! Loads `CLAUDE.md` and/or `AGENTS.md` from a project root, merging them
//! into a single conventions document with heading-aware deduplication
//! (CLAUDE.md takes precedence on duplicate headings).
//!
//! Implements V104.1 of the OpenCode-parity plan: deterministic merge
//! (no LLM call), trust prompt on first load, size cap, symlink rejection,
//! BOM/CRLF normalization, case-insensitive filesystem probe.
//!
//! # Quick start
//!
//! ```no_run
//! use ai_assistant::project_conventions::{
//!     load_project_conventions, ConventionsConfig,
//! };
//! use std::path::Path;
//!
//! let cfg = ConventionsConfig::default();
//! let loaded = load_project_conventions(
//!     Path::new("./my-project"),
//!     None, // no trust cache
//!     None, // no approval handler => non-interactive: untrusted projects denied
//!     &cfg,
//! ).expect("load failed");
//!
//! if let Some(conv) = loaded {
//!     println!("{}", conv.content);
//! }
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent_policy::{ApprovalHandler, RiskLevel};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the project-conventions loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConventionsConfig {
    /// Maximum size in bytes per convention file. Default 256 KiB.
    pub max_file_size: u64,
    /// Whether to follow symlinks. Default false (security baseline).
    pub follow_symlinks: bool,
    /// Whether to require a trust prompt on first load of an unknown
    /// project root. Default true.
    pub require_trust_prompt: bool,
    /// When to emit the merge notice to stderr / log.
    pub merge_notice: MergeNotice,
}

impl Default for ConventionsConfig {
    fn default() -> Self {
        Self {
            max_file_size: 256 * 1024,
            follow_symlinks: false,
            require_trust_prompt: true,
            merge_notice: MergeNotice::First,
        }
    }
}

/// Controls when the merge notice is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeNotice {
    /// Emit only the first time a given project root is loaded in
    /// this process (default).
    First,
    /// Emit on every load.
    Always,
    /// Never emit.
    Never,
}

// ============================================================================
// Result types
// ============================================================================

/// Identifies which convention file contributed to a load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConventionSource {
    ClaudeMd(PathBuf),
    AgentsMd(PathBuf),
}

/// The result of loading project conventions.
#[derive(Debug, Clone)]
pub struct LoadedConventions {
    /// Final merged convention text, ready to inject into an agent prompt.
    pub content: String,
    /// Files that contributed.
    pub sources: Vec<ConventionSource>,
    /// Merge statistics, present when both files were merged.
    pub merge_summary: Option<MergeSummary>,
    /// Trust check outcome.
    pub trust_status: TrustStatus,
}

/// Summary of a heading-aware merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeSummary {
    /// Number of CLAUDE.md sections that overrode an AGENTS.md section
    /// with the same (normalized) heading.
    pub claude_overrides: usize,
    /// Number of AGENTS.md sections that were unique (no CLAUDE.md
    /// counterpart) and were therefore appended.
    pub agents_unique: usize,
}

/// Trust check outcome for the project root being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustStatus {
    /// Project root was already in the trust cache.
    Trusted,
    /// Project root was newly approved by the user via the handler.
    NewlyTrusted,
    /// Trust prompt was required but no handler was provided —
    /// load is denied and returns `Ok(None)`.
    UntrustedNoHandler,
    /// User denied the trust prompt — load is denied.
    UntrustedDenied,
    /// Trust prompts were disabled by config.
    PromptDisabled,
}

// ============================================================================
// Errors
// ============================================================================

/// Errors that can occur loading project conventions.
#[derive(Debug)]
pub enum ConventionsError {
    /// I/O error during read.
    Io(io::Error),
    /// File exceeded the configured maximum size.
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    /// File was not a regular file (e.g., device, FIFO, socket).
    NotRegularFile(PathBuf),
    /// File was a symlink and `follow_symlinks` was false, OR the
    /// resolved target escaped the project root.
    SymlinkRejected(PathBuf),
    /// File contents were not valid UTF-8.
    InvalidUtf8(PathBuf),
}

impl std::fmt::Display for ConventionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::TooLarge { path, size, limit } => write!(
                f,
                "convention file too large: {} ({} bytes > {} bytes limit)",
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
                write!(f, "invalid UTF-8 in: {}", p.display())
            }
        }
    }
}

impl std::error::Error for ConventionsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ConventionsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// Trust cache
// ============================================================================

/// Persistent record of project roots the user has approved as trusted.
pub trait TrustCache: Send + Sync {
    fn is_trusted(&self, project_root: &Path) -> bool;
    fn mark_trusted(&self, project_root: &Path) -> Result<(), io::Error>;
}

/// JSON-file-backed trust cache, default location is the user's
/// platform config dir under `ai_assistant/trust.json`.
pub struct FileTrustCache {
    path: PathBuf,
}

impl FileTrustCache {
    /// Create a cache rooted at the default platform config location.
    pub fn default_location() -> Self {
        Self {
            path: default_trust_cache_path(),
        }
    }

    /// Create a cache rooted at an explicit path (used by tests).
    pub fn at(path: PathBuf) -> Self {
        Self { path }
    }

    fn load(&self) -> BTreeSet<PathBuf> {
        let bytes = match fs::read(&self.path) {
            Ok(b) => b,
            Err(_) => return BTreeSet::new(),
        };
        serde_json::from_slice(&bytes).unwrap_or_default()
    }

    fn save(&self, set: &BTreeSet<PathBuf>) -> Result<(), io::Error> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(set).map_err(io::Error::other)?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, json)?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

impl TrustCache for FileTrustCache {
    fn is_trusted(&self, project_root: &Path) -> bool {
        let canonical = match fs::canonicalize(project_root) {
            Ok(p) => p,
            Err(_) => project_root.to_path_buf(),
        };
        self.load().contains(&canonical)
    }

    fn mark_trusted(&self, project_root: &Path) -> Result<(), io::Error> {
        let canonical =
            fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
        let mut set = self.load();
        set.insert(canonical);
        self.save(&set)
    }
}

fn default_trust_cache_path() -> PathBuf {
    if let Some(dir) = config_dir() {
        dir.join("ai_assistant").join("trust.json")
    } else {
        PathBuf::from("ai_assistant_trust.json")
    }
}

fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA").ok().map(PathBuf::from)
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Application Support"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".config"))
            })
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

// ============================================================================
// Public entry point
// ============================================================================

/// Load project conventions from `project_root`, merging `CLAUDE.md` and
/// `AGENTS.md` if both exist (heading-aware, CLAUDE.md wins on duplicates).
///
/// Returns `Ok(None)` if neither file exists OR the trust check denied
/// the load. Returns `Err(_)` only on I/O / size / encoding failures.
pub fn load_project_conventions(
    project_root: &Path,
    trust_cache: Option<&dyn TrustCache>,
    trust_handler: Option<&dyn ApprovalHandler>,
    config: &ConventionsConfig,
) -> Result<Option<LoadedConventions>, ConventionsError> {
    let canonical_root = match fs::canonicalize(project_root) {
        Ok(p) => p,
        Err(_) => project_root.to_path_buf(),
    };

    let candidates = probe_convention_files(&canonical_root);
    if candidates.is_empty() {
        return Ok(None);
    }

    // Trust check — done before reading anything off disk beyond the
    // probe, so a hostile project's contents never reach us until the
    // user OKs it.
    let trust_status = check_trust(
        &canonical_root,
        trust_cache,
        trust_handler,
        config.require_trust_prompt,
    );
    match trust_status {
        TrustStatus::Trusted | TrustStatus::NewlyTrusted | TrustStatus::PromptDisabled => {}
        TrustStatus::UntrustedNoHandler | TrustStatus::UntrustedDenied => {
            return Ok(None);
        }
    }

    // Read everything we found.
    let mut claude_text: Option<String> = None;
    let mut agents_text: Option<String> = None;
    let mut sources: Vec<ConventionSource> = Vec::new();

    for (source, path) in &candidates {
        let text = read_convention_file(
            path,
            &canonical_root,
            config.max_file_size,
            config.follow_symlinks,
        )?;
        match source {
            ConventionSource::ClaudeMd(_) => claude_text = Some(text),
            ConventionSource::AgentsMd(_) => agents_text = Some(text),
        }
        sources.push(source.clone());
    }

    let (content, merge_summary) = match (claude_text, agents_text) {
        (Some(c), Some(a)) => {
            let (merged, summary) = merge_conventions(&c, &a);
            (merged, Some(summary))
        }
        (Some(c), None) => (c, None),
        (None, Some(a)) => (a, None),
        (None, None) => return Ok(None),
    };

    Ok(Some(LoadedConventions {
        content,
        sources,
        merge_summary,
        trust_status,
    }))
}

// ============================================================================
// Trust check
// ============================================================================

fn check_trust(
    project_root: &Path,
    cache: Option<&dyn TrustCache>,
    handler: Option<&dyn ApprovalHandler>,
    require_prompt: bool,
) -> TrustStatus {
    if !require_prompt {
        return TrustStatus::PromptDisabled;
    }
    if let Some(c) = cache {
        if c.is_trusted(project_root) {
            return TrustStatus::Trusted;
        }
    }
    let handler = match handler {
        Some(h) => h,
        None => return TrustStatus::UntrustedNoHandler,
    };
    let prompt = format!(
        "Load project conventions (CLAUDE.md / AGENTS.md) from {}? \
         These files will be injected into agent prompts and may contain \
         instructions the agent should follow.",
        project_root.display()
    );
    let approved = handler.request_approval(&prompt, RiskLevel::Medium);
    if !approved {
        return TrustStatus::UntrustedDenied;
    }
    if let Some(c) = cache {
        let _ = c.mark_trusted(project_root);
    }
    TrustStatus::NewlyTrusted
}

// ============================================================================
// File probe (case-insensitive on Windows/macOS de facto via OS, but we
// also try common case variants explicitly so it works the same on case-
// sensitive Linux filesystems where the user may name a file `claude.md`).
// ============================================================================

fn probe_convention_files(root: &Path) -> Vec<(ConventionSource, PathBuf)> {
    let mut out = Vec::new();
    for name in &["CLAUDE.md", "claude.md", "Claude.md"] {
        let p = root.join(name);
        if p.exists() {
            out.push((ConventionSource::ClaudeMd(p.clone()), p));
            break;
        }
    }
    for name in &["AGENTS.md", "agents.md", "Agents.md"] {
        let p = root.join(name);
        if p.exists() {
            out.push((ConventionSource::AgentsMd(p.clone()), p));
            break;
        }
    }
    out
}

// ============================================================================
// File read with security checks
// ============================================================================

fn read_convention_file(
    path: &Path,
    project_root: &Path,
    max_size: u64,
    follow_symlinks: bool,
) -> Result<String, ConventionsError> {
    let meta = fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() && !follow_symlinks {
        return Err(ConventionsError::SymlinkRejected(path.to_path_buf()));
    }
    if !meta.file_type().is_file() && !meta.file_type().is_symlink() {
        return Err(ConventionsError::NotRegularFile(path.to_path_buf()));
    }

    // Resolve and check escape from root.
    let resolved = fs::canonicalize(path)?;
    if !resolved.starts_with(project_root) {
        return Err(ConventionsError::SymlinkRejected(resolved));
    }

    let resolved_meta = fs::metadata(&resolved)?;
    if !resolved_meta.is_file() {
        return Err(ConventionsError::NotRegularFile(resolved));
    }
    let size = resolved_meta.len();
    if size > max_size {
        return Err(ConventionsError::TooLarge {
            path: resolved,
            size,
            limit: max_size,
        });
    }

    let bytes = fs::read(&resolved)?;
    let text =
        String::from_utf8(bytes).map_err(|_| ConventionsError::InvalidUtf8(resolved.clone()))?;
    Ok(normalize_text(text))
}

/// Strip BOM if present and normalize CRLF / lone CR to LF.
fn normalize_text(s: String) -> String {
    let s = s.strip_prefix('\u{feff}').unwrap_or(&s).to_string();
    // Replace CRLF first, then any remaining bare CR.
    let s = s.replace("\r\n", "\n");
    s.replace('\r', "\n")
}

// ============================================================================
// Heading-aware merge
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    /// Original heading line including the `#` prefix, e.g. `## Reglas`.
    /// `None` means the preamble (text before the first heading).
    heading: Option<String>,
    /// Normalized heading (lowercased, trimmed, whitespace collapsed)
    /// for matching across files. `None` for preamble.
    heading_normalized: Option<String>,
    /// Body of the section, NOT including the heading line itself.
    body: String,
}

/// Merge `claude` (winner) and `agents` (base) into a single document.
fn merge_conventions(claude: &str, agents: &str) -> (String, MergeSummary) {
    let claude_sections = parse_sections(claude);
    let agents_sections = parse_sections(agents);

    let claude_norm: BTreeSet<String> = claude_sections
        .iter()
        .filter_map(|s| s.heading_normalized.clone())
        .collect();

    let mut out: Vec<Section> = Vec::new();
    let mut overrides = 0usize;
    let mut unique_agents = 0usize;

    // Pre-heading preamble:
    // CLAUDE.md preamble wins; if absent, AGENTS.md preamble carries.
    let claude_preamble = claude_sections
        .first()
        .filter(|s| s.heading.is_none())
        .cloned();
    let agents_preamble = agents_sections
        .first()
        .filter(|s| s.heading.is_none())
        .cloned();

    if let Some(c) = claude_preamble.clone() {
        if !c.body.trim().is_empty() {
            out.push(c);
        }
    } else if let Some(a) = agents_preamble {
        if !a.body.trim().is_empty() {
            out.push(a);
        }
    }

    // Walk AGENTS.md headings; include unless CLAUDE.md will override.
    for sec in agents_sections.iter().filter(|s| s.heading.is_some()) {
        let norm = sec.heading_normalized.as_ref().unwrap();
        if !claude_norm.contains(norm) {
            out.push(sec.clone());
            unique_agents += 1;
        }
        // else: skipped here, will be replaced by CLAUDE's version below.
    }

    // Walk CLAUDE.md headings; for each, either override an existing
    // entry (already-omitted from agents) by appending in place, or
    // append at end if AGENTS didn't have it.
    let agents_norm: BTreeSet<String> = agents_sections
        .iter()
        .filter_map(|s| s.heading_normalized.clone())
        .collect();

    for sec in claude_sections.iter().filter(|s| s.heading.is_some()) {
        let norm = sec.heading_normalized.as_ref().unwrap();
        if agents_norm.contains(norm) {
            overrides += 1;
        }
        out.push(sec.clone());
    }

    let mut text = String::new();
    for (i, sec) in out.iter().enumerate() {
        if let Some(h) = &sec.heading {
            if i > 0 && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(h);
            text.push('\n');
        }
        text.push_str(&sec.body);
        if !sec.body.ends_with('\n') {
            text.push('\n');
        }
    }

    let summary = MergeSummary {
        claude_overrides: overrides,
        agents_unique: unique_agents,
    };

    (text.trim_end().to_string() + "\n", summary)
}

/// Parse a markdown document into ordered sections, respecting fenced
/// code blocks (so a `#` line inside ``` ``` is NOT treated as a heading).
fn parse_sections(text: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();
    let mut in_fence = false;
    let mut fence_marker: Option<String> = None;

    for line in text.split('\n') {
        // Fence detection: ``` or ~~~, optionally with language tag.
        let trimmed = line.trim_start();
        let is_fence_open_close = if let Some(marker) = &fence_marker {
            trimmed.starts_with(marker.as_str())
        } else {
            trimmed.starts_with("```") || trimmed.starts_with("~~~")
        };

        if is_fence_open_close {
            if in_fence {
                in_fence = false;
                fence_marker = None;
            } else {
                in_fence = true;
                fence_marker = Some(if trimmed.starts_with("~~~") {
                    "~~~".to_string()
                } else {
                    "```".to_string()
                });
            }
            current_body.push_str(line);
            current_body.push('\n');
            continue;
        }

        let is_heading = !in_fence && is_atx_heading(line);
        if is_heading {
            // Push the previous section.
            sections.push(Section {
                heading: current_heading.take(),
                heading_normalized: None,
                body: std::mem::take(&mut current_body),
            });
            current_heading = Some(line.trim_end().to_string());
            continue;
        }

        current_body.push_str(line);
        current_body.push('\n');
    }
    sections.push(Section {
        heading: current_heading,
        heading_normalized: None,
        body: current_body,
    });

    // Compute normalized headings.
    for sec in &mut sections {
        sec.heading_normalized = sec.heading.as_deref().map(normalize_heading);
    }

    // Drop a leading empty preamble that has no content at all (some files
    // start straight with `# Title`).
    if let Some(first) = sections.first() {
        if first.heading.is_none() && first.body.trim().is_empty() {
            sections.remove(0);
        }
    }

    sections
}

fn is_atx_heading(line: &str) -> bool {
    // ATX: 1-6 leading `#` followed by space (or end of line).
    let mut count = 0;
    for c in line.chars() {
        if c == '#' {
            count += 1;
            if count > 6 {
                return false;
            }
        } else {
            return count >= 1 && (c == ' ' || c == '\t');
        }
    }
    false
}

/// Normalize a heading line for cross-file matching:
/// - prefix with heading level (so `## A` and `### A` don't collide),
/// - drop leading `#` chars and surrounding whitespace,
/// - lowercase,
/// - collapse internal whitespace to single space.
fn normalize_heading(line: &str) -> String {
    let s = line.trim_start();
    let mut level = 0usize;
    let mut rest = s;
    while let Some(r) = rest.strip_prefix('#') {
        level += 1;
        rest = r;
    }
    let s = rest.trim();
    let mut out = String::with_capacity(s.len() + 3);
    out.push_str(&format!("{}:", level));
    let mut last_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.extend(c.to_lowercase());
            last_space = false;
        }
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::Mutex;

    // --- helpers -----------------------------------------------------------

    struct DenyAll;
    impl ApprovalHandler for DenyAll {
        fn request_approval(&self, _action: &str, _risk: RiskLevel) -> bool {
            false
        }
    }

    struct InMemTrustCache {
        inner: Mutex<BTreeSet<PathBuf>>,
    }
    impl InMemTrustCache {
        fn new() -> Self {
            Self {
                inner: Mutex::new(BTreeSet::new()),
            }
        }
    }
    impl TrustCache for InMemTrustCache {
        fn is_trusted(&self, root: &Path) -> bool {
            let canon = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
            self.inner.lock().unwrap().contains(&canon)
        }
        fn mark_trusted(&self, root: &Path) -> Result<(), io::Error> {
            let canon = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
            self.inner.lock().unwrap().insert(canon);
            Ok(())
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("ai_assistant_conventions_tests")
            .join(format!(
                "{}_{}_{}",
                name,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(p: &Path, contents: &str) {
        fs::write(p, contents).unwrap();
    }

    fn permissive_cfg() -> ConventionsConfig {
        ConventionsConfig {
            require_trust_prompt: false,
            ..Default::default()
        }
    }

    // --- normalization & parsing -----------------------------------------

    #[test]
    fn normalize_strips_bom_and_crlf() {
        let s = "\u{feff}line1\r\nline2\rline3\n".to_string();
        assert_eq!(normalize_text(s), "line1\nline2\nline3\n");
    }

    #[test]
    fn normalize_heading_lowercases_and_collapses() {
        assert_eq!(
            normalize_heading("##  Reglas   Importantes"),
            "2:reglas importantes"
        );
        assert_eq!(normalize_heading("# A"), "1:a");
        assert_eq!(normalize_heading("### CamelCase"), "3:camelcase");
    }

    #[test]
    fn is_atx_heading_recognises_levels() {
        assert!(is_atx_heading("# Title"));
        assert!(is_atx_heading("###### Deep"));
        assert!(!is_atx_heading("####### Too deep"));
        assert!(!is_atx_heading("Plain text"));
        assert!(!is_atx_heading("##"));
        assert!(is_atx_heading("##\tTab"));
    }

    #[test]
    fn parse_sections_skips_headings_in_code_blocks() {
        let md = "# Real\nbody\n```\n# Not a heading\n```\n## Another\nx\n";
        let secs = parse_sections(md);
        let titles: Vec<_> = secs.iter().filter_map(|s| s.heading.clone()).collect();
        assert_eq!(titles, vec!["# Real".to_string(), "## Another".to_string()]);
    }

    #[test]
    fn parse_sections_handles_tilde_fences() {
        let md = "# A\n~~~\n## not\n~~~\n## B\n";
        let secs = parse_sections(md);
        let titles: Vec<_> = secs.iter().filter_map(|s| s.heading.clone()).collect();
        assert_eq!(titles, vec!["# A".to_string(), "## B".to_string()]);
    }

    #[test]
    fn parse_sections_no_headings_yields_single_preamble() {
        let secs = parse_sections("just text\nmore\n");
        assert_eq!(secs.len(), 1);
        assert!(secs[0].heading.is_none());
    }

    // --- merge logic ------------------------------------------------------

    #[test]
    fn merge_only_claude_when_only_claude() {
        let (out, _) = merge_conventions("# A\n1\n", "");
        assert!(out.contains("# A"));
        assert!(out.contains("1"));
    }

    #[test]
    fn merge_appends_unique_agents_sections() {
        let claude = "# A\nclaude-a\n";
        let agents = "# A\nagents-a\n# B\nagents-b\n";
        let (out, summary) = merge_conventions(claude, agents);
        assert_eq!(summary.claude_overrides, 1);
        assert_eq!(summary.agents_unique, 1);
        assert!(out.contains("# B"));
        assert!(out.contains("agents-b"));
        assert!(out.contains("claude-a"));
        assert!(
            !out.contains("agents-a"),
            "CLAUDE must override AGENTS for #A"
        );
    }

    #[test]
    fn merge_normalizes_headings_for_matching() {
        let claude = "##  Reglas\nclaude-rules\n";
        let agents = "## reglas\nagents-rules\n";
        let (out, summary) = merge_conventions(claude, agents);
        assert_eq!(summary.claude_overrides, 1);
        assert_eq!(summary.agents_unique, 0);
        assert!(out.contains("claude-rules"));
        assert!(!out.contains("agents-rules"));
    }

    #[test]
    fn merge_preserves_distinct_levels() {
        let claude = "## A\nc\n";
        let agents = "### A\na\n";
        let (out, summary) = merge_conventions(claude, agents);
        // Different heading levels → different sections.
        assert_eq!(summary.claude_overrides, 0);
        assert_eq!(summary.agents_unique, 1);
        assert!(out.contains("c"));
        assert!(out.contains("a"));
    }

    #[test]
    fn merge_claude_preamble_wins() {
        let claude = "claude-preamble\n# A\nx\n";
        let agents = "agents-preamble\n# A\ny\n";
        let (out, _) = merge_conventions(claude, agents);
        assert!(out.starts_with("claude-preamble"));
        assert!(!out.contains("agents-preamble"));
    }

    #[test]
    fn merge_falls_back_to_agents_preamble_when_claude_lacks_one() {
        let claude = "# A\nx\n";
        let agents = "agents-preamble\n# B\ny\n";
        let (out, _) = merge_conventions(claude, agents);
        assert!(out.contains("agents-preamble"));
    }

    // --- end-to-end loader ------------------------------------------------

    #[test]
    fn load_returns_none_when_no_files() {
        let root = temp_root("none");
        let cfg = permissive_cfg();
        let res = load_project_conventions(&root, None, None, &cfg).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn load_only_claude() {
        let root = temp_root("only_claude");
        write(&root.join("CLAUDE.md"), "# T\nbody\n");
        let cfg = permissive_cfg();
        let res = load_project_conventions(&root, None, None, &cfg)
            .unwrap()
            .unwrap();
        assert!(res.content.contains("# T"));
        assert_eq!(res.sources.len(), 1);
        assert!(matches!(res.sources[0], ConventionSource::ClaudeMd(_)));
        assert!(res.merge_summary.is_none());
    }

    #[test]
    fn load_only_agents() {
        let root = temp_root("only_agents");
        write(&root.join("AGENTS.md"), "# T\nbody\n");
        let cfg = permissive_cfg();
        let res = load_project_conventions(&root, None, None, &cfg)
            .unwrap()
            .unwrap();
        assert!(res.content.contains("# T"));
        assert_eq!(res.sources.len(), 1);
        assert!(matches!(res.sources[0], ConventionSource::AgentsMd(_)));
        assert!(res.merge_summary.is_none());
    }

    #[test]
    fn load_merges_both_with_summary() {
        let root = temp_root("both");
        write(&root.join("CLAUDE.md"), "# Reglas\nclaude\n");
        write(
            &root.join("AGENTS.md"),
            "# Reglas\nagents\n# Extra\nagents-extra\n",
        );
        let cfg = permissive_cfg();
        let res = load_project_conventions(&root, None, None, &cfg)
            .unwrap()
            .unwrap();
        let summary = res.merge_summary.unwrap();
        assert_eq!(summary.claude_overrides, 1);
        assert_eq!(summary.agents_unique, 1);
        assert!(res.content.contains("claude"));
        assert!(!res.content.contains("agents\n"));
        assert!(res.content.contains("agents-extra"));
    }

    #[test]
    fn load_size_cap_rejects_oversized_file() {
        let root = temp_root("oversize");
        let big = "x".repeat(2048);
        write(&root.join("CLAUDE.md"), &big);
        let cfg = ConventionsConfig {
            max_file_size: 1024,
            require_trust_prompt: false,
            ..Default::default()
        };
        let err = load_project_conventions(&root, None, None, &cfg).unwrap_err();
        assert!(matches!(err, ConventionsError::TooLarge { .. }));
    }

    #[test]
    fn load_trust_denies_when_no_handler_and_prompt_required() {
        let root = temp_root("trust_no_handler");
        write(&root.join("CLAUDE.md"), "# T\nx\n");
        let cfg = ConventionsConfig {
            require_trust_prompt: true,
            ..Default::default()
        };
        let res = load_project_conventions(&root, None, None, &cfg).unwrap();
        assert!(res.is_none(), "no handler => denied");
    }

    #[test]
    fn load_trust_denies_when_handler_says_no() {
        let root = temp_root("trust_deny");
        write(&root.join("CLAUDE.md"), "# T\nx\n");
        let cfg = ConventionsConfig {
            require_trust_prompt: true,
            ..Default::default()
        };
        let cache = InMemTrustCache::new();
        let res = load_project_conventions(&root, Some(&cache), Some(&DenyAll), &cfg).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn load_trust_caches_after_first_approval() {
        let root = temp_root("trust_cache");
        write(&root.join("CLAUDE.md"), "# T\nx\n");
        let cfg = ConventionsConfig {
            require_trust_prompt: true,
            ..Default::default()
        };
        let cache = InMemTrustCache::new();

        // Track approval invocations.
        struct CountingHandler {
            count: Mutex<usize>,
        }
        impl ApprovalHandler for CountingHandler {
            fn request_approval(&self, _: &str, _: RiskLevel) -> bool {
                *self.count.lock().unwrap() += 1;
                true
            }
        }
        let h = CountingHandler {
            count: Mutex::new(0),
        };

        let r1 = load_project_conventions(&root, Some(&cache), Some(&h), &cfg)
            .unwrap()
            .unwrap();
        assert_eq!(r1.trust_status, TrustStatus::NewlyTrusted);

        let r2 = load_project_conventions(&root, Some(&cache), Some(&h), &cfg)
            .unwrap()
            .unwrap();
        assert_eq!(r2.trust_status, TrustStatus::Trusted);

        assert_eq!(
            *h.count.lock().unwrap(),
            1,
            "handler should be invoked once only"
        );
    }

    #[test]
    fn load_prompt_disabled_short_circuits_handler() {
        let root = temp_root("prompt_off");
        write(&root.join("CLAUDE.md"), "# T\nx\n");
        let cfg = ConventionsConfig {
            require_trust_prompt: false,
            ..Default::default()
        };
        let res = load_project_conventions(&root, None, None, &cfg)
            .unwrap()
            .unwrap();
        assert_eq!(res.trust_status, TrustStatus::PromptDisabled);
    }

    #[test]
    fn load_normalizes_bom_and_crlf_in_files() {
        let root = temp_root("bom_crlf");
        let body = "\u{feff}# Title\r\nbody1\r\nbody2\r\n";
        write(&root.join("CLAUDE.md"), body);
        let cfg = permissive_cfg();
        let res = load_project_conventions(&root, None, None, &cfg)
            .unwrap()
            .unwrap();
        assert!(!res.content.contains('\r'));
        assert!(!res.content.contains('\u{feff}'));
        assert!(res.content.contains("# Title\nbody1"));
    }

    #[test]
    fn file_trust_cache_round_trips() {
        let dir = temp_root("trust_cache_file");
        let cache = FileTrustCache::at(dir.join("trust.json"));
        let project = temp_root("project_for_cache");
        assert!(!cache.is_trusted(&project));
        cache.mark_trusted(&project).unwrap();
        let cache2 = FileTrustCache::at(dir.join("trust.json"));
        assert!(cache2.is_trusted(&project));
    }

    // Suppress unused-warning for RefCell import on platforms that
    // don't end up using it. (Kept in case future tests need single-
    // threaded interior mutability.)
    #[allow(dead_code)]
    fn _unused_marker(_: RefCell<()>) {}
}
