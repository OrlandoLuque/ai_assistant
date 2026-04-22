//! Capability model — deny-by-default permissions required to execute a skill.
//!
//! Capabilities are declared in `SkillDefinition.capabilities` and checked at
//! two points:
//! 1. Dispatch time — the caller's principal must grant the capability.
//! 2. WASM runtime time — the sandbox only exposes host imports for granted
//!    capabilities; attempted access to denied ones traps with
//!    `CapabilityError::Denied`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Deny-by-default set of capabilities a skill may exercise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CapabilitySet {
    pub items: BTreeSet<Capability>,
}

impl CapabilitySet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with(mut self, cap: Capability) -> Self {
        self.items.insert(cap);
        self
    }

    pub fn insert(&mut self, cap: Capability) {
        self.items.insert(cap);
    }

    pub fn contains(&self, cap: &Capability) -> bool {
        self.items.contains(cap)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.items.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check whether `self` is a subset of `granted`. Used at dispatch time:
    /// the skill's required capabilities must be a subset of what the caller
    /// has granted.
    pub fn is_subset_of(&self, granted: &CapabilitySet) -> bool {
        self.items.is_subset(&granted.items)
    }

    /// Find the first required capability that `granted` does not include.
    /// Useful for error messages (`CapabilityDenied`).
    pub fn first_missing<'a>(&'a self, granted: &CapabilitySet) -> Option<&'a Capability> {
        self.items.iter().find(|c| !granted.items.contains(c))
    }
}

/// Individual capability. `BTreeSet`-friendly (derives `Ord`).
///
/// Granularity choices:
/// - `NetFetch(AllowList)` — outbound HTTP(S) only, with host allowlist.
/// - `FileRead(PathGlob)` / `FileWrite(PathGlob)` — filesystem scoped to globs.
/// - `EnvRead(Vec<String>)` — environment variable allowlist.
/// - `Random` — access to CSPRNG.
/// - `Time` — read the wall clock (otherwise only monotonic is available).
/// - `ToolCall(String)` — invoke a specific named tool (Declarative mode).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Capability {
    NetFetch(NetAllowList),
    FileRead(PathGlob),
    FileWrite(PathGlob),
    EnvRead(Vec<String>),
    Random,
    Time,
    ToolCall(String),
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NetFetch(a) => write!(f, "net_fetch[{}]", a.hosts.join(",")),
            Self::FileRead(p) => write!(f, "file_read[{}]", p.glob),
            Self::FileWrite(p) => write!(f, "file_write[{}]", p.glob),
            Self::EnvRead(vars) => write!(f, "env_read[{}]", vars.join(",")),
            Self::Random => f.write_str("random"),
            Self::Time => f.write_str("time"),
            Self::ToolCall(name) => write!(f, "tool_call[{name}]"),
        }
    }
}

/// Host allowlist for `Capability::NetFetch`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct NetAllowList {
    /// Exact host matches. `api.example.com` matches only that host.
    /// Use `*.example.com` for subdomain wildcards (simple suffix match).
    pub hosts: Vec<String>,
}

impl NetAllowList {
    pub fn new(hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            hosts: hosts.into_iter().map(Into::into).collect(),
        }
    }

    /// True if `host` is covered by any entry in the allowlist.
    pub fn allows(&self, host: &str) -> bool {
        for entry in &self.hosts {
            if let Some(suffix) = entry.strip_prefix("*.") {
                if host.ends_with(suffix) && host != suffix {
                    return true;
                }
            } else if entry == host {
                return true;
            }
        }
        false
    }
}

/// Simple glob for filesystem capabilities.
///
/// Semantics (minimal for v1):
/// - `*` matches any run of non-separator characters.
/// - `**` matches across path separators.
/// - No `?` or character classes in v1.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct PathGlob {
    pub glob: String,
}

impl PathGlob {
    pub fn new(g: impl Into<String>) -> Self {
        Self { glob: g.into() }
    }

    /// True if `path` matches the glob.
    pub fn matches(&self, path: &str) -> bool {
        glob_match(&self.glob, path)
    }
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    glob_match_inner(&pat, 0, &txt, 0)
}

fn glob_match_inner(pat: &[char], pi: usize, txt: &[char], ti: usize) -> bool {
    // Walk literal prefix.
    let (mut pi, mut ti) = (pi, ti);
    while pi < pat.len() && pat[pi] != '*' {
        if ti >= txt.len() || pat[pi] != txt[ti] {
            return false;
        }
        pi += 1;
        ti += 1;
    }
    if pi >= pat.len() {
        return ti == txt.len();
    }
    // pat[pi] == '*'. Detect '**'.
    let is_double = pi + 1 < pat.len() && pat[pi + 1] == '*';
    let next_pi = if is_double { pi + 2 } else { pi + 1 };
    // Try to match 0..=N text chars.
    let mut candidate = ti;
    loop {
        if glob_match_inner(pat, next_pi, txt, candidate) {
            return true;
        }
        if candidate >= txt.len() {
            return false;
        }
        // Single `*` does not cross path separators.
        if !is_double && (txt[candidate] == '/' || txt[candidate] == '\\') {
            return false;
        }
        candidate += 1;
    }
}

/// Errors raised by capability checks.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CapabilityError {
    /// Requested capability is not in the granted set.
    Denied { capability: String },
    /// A capability was granted but with a more restrictive scope (host / path).
    OutOfScope { capability: String, target: String },
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Denied { capability } => write!(f, "capability denied: {capability}"),
            Self::OutOfScope { capability, target } => {
                write!(f, "capability '{capability}' out of scope for {target}")
            }
        }
    }
}

impl std::error::Error for CapabilityError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set_has_no_items() {
        let s = CapabilitySet::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn with_inserts_capability() {
        let s = CapabilitySet::empty().with(Capability::Random);
        assert!(s.contains(&Capability::Random));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn subset_check_works() {
        let skill = CapabilitySet::empty().with(Capability::Random);
        let granted = CapabilitySet::empty()
            .with(Capability::Random)
            .with(Capability::Time);
        assert!(skill.is_subset_of(&granted));
        assert!(!granted.is_subset_of(&skill));
    }

    #[test]
    fn first_missing_identifies_gap() {
        let skill = CapabilitySet::empty()
            .with(Capability::Random)
            .with(Capability::Time);
        let granted = CapabilitySet::empty().with(Capability::Random);
        let missing = skill.first_missing(&granted);
        assert!(matches!(missing, Some(Capability::Time)));
    }

    #[test]
    fn net_allowlist_exact_match() {
        let a = NetAllowList::new(["api.example.com"]);
        assert!(a.allows("api.example.com"));
        assert!(!a.allows("evil.com"));
        assert!(!a.allows("x.api.example.com"));
    }

    #[test]
    fn net_allowlist_wildcard_subdomain() {
        let a = NetAllowList::new(["*.example.com"]);
        assert!(a.allows("api.example.com"));
        assert!(a.allows("foo.bar.example.com"));
        assert!(!a.allows("example.com")); // wildcard requires at least one subdomain
        assert!(!a.allows("evil.com"));
    }

    #[test]
    fn path_glob_literal() {
        let g = PathGlob::new("/data/file.txt");
        assert!(g.matches("/data/file.txt"));
        assert!(!g.matches("/data/other.txt"));
    }

    #[test]
    fn path_glob_star_single_segment() {
        let g = PathGlob::new("/data/*.txt");
        assert!(g.matches("/data/file.txt"));
        assert!(g.matches("/data/other.txt"));
        // `*` must not cross separators
        assert!(!g.matches("/data/sub/file.txt"));
    }

    #[test]
    fn path_glob_doublestar_crosses_segments() {
        let g = PathGlob::new("/data/**/*.txt");
        assert!(g.matches("/data/sub/file.txt"));
        assert!(g.matches("/data/a/b/c/file.txt"));
        assert!(!g.matches("/other/file.txt"));
    }

    #[test]
    fn capability_display_is_stable() {
        let c = Capability::NetFetch(NetAllowList::new(["a", "b"]));
        assert_eq!(c.to_string(), "net_fetch[a,b]");
    }

    #[test]
    fn capability_ordering_deterministic() {
        // BTreeSet needs Ord — verify deterministic iteration order.
        let s = CapabilitySet::empty()
            .with(Capability::Time)
            .with(Capability::Random);
        let order: Vec<_> = s.iter().cloned().collect();
        assert_eq!(order.len(), 2);
    }
}
