//! Browser Security Policy — granular permissions for web browsing automation.
//!
//! Controls what the browser agent can do: navigate, read, interact, execute JS,
//! access cookies, download files, take screenshots.
//!
//! Also provides URL validation (scheme whitelist, SSRF protection) and
//! dangerous JS pattern detection.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::IpAddr;

// ============================================================================
// Browser Policy
// ============================================================================

/// Granular permissions for browser automation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BrowserPolicy {
    /// Allowed URL schemes. Default: ["https"]. Add "http" for non-TLS.
    pub allowed_schemes: Vec<String>,
    /// Domain allowlist (None = allow all non-blocked).
    pub domain_allowlist: Option<HashSet<String>>,
    /// Domain blocklist (always enforced).
    pub domain_blocklist: HashSet<String>,
    /// Block private/reserved IP ranges (SSRF protection).
    pub block_private_ips: bool,
    /// Block cloud metadata endpoints (169.254.169.254, etc.).
    pub block_metadata_endpoints: bool,
    /// JavaScript execution permission level.
    pub js_permission: JsPermission,
    /// Allow reading cookies/localStorage/sessionStorage.
    pub allow_cookie_access: bool,
    /// Allow form submission (click submit buttons, POST forms).
    pub allow_form_submission: bool,
    /// Allow file downloads.
    pub allow_downloads: bool,
    /// Allow screenshots.
    pub allow_screenshots: bool,
    /// Maximum redirect chain depth before blocking.
    pub max_redirect_depth: usize,
    /// Maximum page size in bytes before truncation.
    pub max_page_size_bytes: usize,
    /// Require user approval for interaction (click, type).
    pub require_approval_for_interaction: bool,
}

/// JavaScript execution permission level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum JsPermission {
    /// No JavaScript execution allowed.
    Disabled,
    /// Only DOM queries (querySelector, title, textContent).
    ReadOnly,
    /// DOM queries + modification (innerHTML, etc.).
    Mutating,
    /// Full access including network (DANGEROUS — opt-in only).
    Full,
}

impl Default for JsPermission {
    fn default() -> Self {
        Self::ReadOnly
    }
}

impl BrowserPolicy {
    /// Restrictive policy (default) — HTTPS only, block SSRF, read-only JS.
    pub fn restrictive() -> Self {
        let mut blocked = HashSet::new();
        blocked.insert("metadata.google.internal".to_string());
        blocked.insert("metadata.google.com".to_string());

        Self {
            allowed_schemes: vec!["https".to_string()],
            domain_allowlist: None,
            domain_blocklist: blocked,
            block_private_ips: true,
            block_metadata_endpoints: true,
            js_permission: JsPermission::ReadOnly,
            allow_cookie_access: false,
            allow_form_submission: false,
            allow_downloads: false,
            allow_screenshots: true,
            max_redirect_depth: 5,
            max_page_size_bytes: 10 * 1024 * 1024, // 10 MB
            require_approval_for_interaction: true,
        }
    }

    /// Permissive policy (development) — HTTP+HTTPS, no SSRF blocking.
    pub fn permissive() -> Self {
        Self {
            allowed_schemes: vec!["https".to_string(), "http".to_string()],
            domain_allowlist: None,
            domain_blocklist: HashSet::new(),
            block_private_ips: false,
            block_metadata_endpoints: false,
            js_permission: JsPermission::Mutating,
            allow_cookie_access: false,
            allow_form_submission: true,
            allow_downloads: false,
            allow_screenshots: true,
            max_redirect_depth: 10,
            max_page_size_bytes: 50 * 1024 * 1024,
            require_approval_for_interaction: false,
        }
    }

    /// Validate a URL against this policy.
    pub fn validate_url(&self, url: &str) -> UrlValidation {
        // 1. Scheme check
        let scheme = url.split("://").next().unwrap_or("").to_lowercase();
        if !self.allowed_schemes.iter().any(|s| s == &scheme) {
            return UrlValidation::Blocked {
                reason: format!(
                    "Scheme '{}' not allowed (allowed: {:?})",
                    scheme, self.allowed_schemes
                ),
            };
        }

        // 2. Extract host
        let host = extract_host(url);
        if host.is_empty() {
            return UrlValidation::Blocked {
                reason: "Could not extract host from URL".to_string(),
            };
        }

        // 3. Domain blocklist
        let host_lower = host.to_lowercase();
        if self.domain_blocklist.contains(&host_lower) {
            return UrlValidation::Blocked {
                reason: format!("Host '{}' is in blocklist", host),
            };
        }

        // 4. Domain allowlist (if configured)
        if let Some(ref allowlist) = self.domain_allowlist {
            if !allowlist.contains(&host_lower) {
                return UrlValidation::Blocked {
                    reason: format!("Host '{}' not in allowlist", host),
                };
            }
        }

        // 5. Private IP check
        if self.block_private_ips {
            if let Ok(ip) = host.parse::<IpAddr>() {
                if is_private_ip(&ip) {
                    return UrlValidation::Blocked {
                        reason: format!("IP {} is private/reserved (SSRF protection)", ip),
                    };
                }
            }
            // Check common localhost aliases
            if host_lower == "localhost"
                || host_lower == "127.0.0.1"
                || host_lower == "::1"
                || host_lower == "0.0.0.0"
            {
                return UrlValidation::Blocked {
                    reason: "Localhost access blocked (SSRF protection)".to_string(),
                };
            }
        }

        // 6. Metadata endpoint check
        if self.block_metadata_endpoints {
            if host_lower == "169.254.169.254"
                || host_lower == "metadata.google.internal"
                || host_lower == "metadata.google.com"
            {
                return UrlValidation::Blocked {
                    reason: "Cloud metadata endpoint blocked".to_string(),
                };
            }
        }

        UrlValidation::Allowed
    }

    /// Validate JavaScript code against the permission level.
    ///
    /// SECURITY MODEL: the pattern checks below are **defense-in-depth, not a
    /// hard boundary**. Substring matching on JS source is bypassable by any
    /// determined adversary (`window['fe'+'tch']`, `Function(...)`, unicode
    /// escapes, `atob` decoding). They catch the obvious/accidental cases.
    /// For untrusted input the real boundary must be the browser itself —
    /// run with `JsPermission::Disabled`, an isolated/sandboxed context, or a
    /// restrictive CSP. Do not treat a `ReadOnly`/`Mutating` pass as proof the
    /// script is safe.
    pub fn validate_js(&self, js: &str) -> JsValidation {
        match self.js_permission {
            JsPermission::Disabled => {
                return JsValidation::Blocked {
                    reason: "JavaScript execution is disabled".to_string(),
                };
            }
            JsPermission::ReadOnly => {
                // Block anything that mutates or accesses network
                if contains_dangerous_pattern(js) || contains_mutating_pattern(js) {
                    return JsValidation::Blocked {
                        reason: "Pattern blocked in ReadOnly JS mode".to_string(),
                    };
                }
            }
            JsPermission::Mutating => {
                // Allow DOM mutation but block network access
                if contains_dangerous_pattern(js) {
                    return JsValidation::Blocked {
                        reason: "Network/exfiltration pattern blocked".to_string(),
                    };
                }
            }
            JsPermission::Full => {
                // Only block the most critical exfiltration patterns
                if contains_critical_pattern(js) {
                    return JsValidation::Blocked {
                        reason: "Critical exfiltration pattern blocked".to_string(),
                    };
                }
            }
        }
        JsValidation::Allowed
    }
}

impl Default for BrowserPolicy {
    fn default() -> Self {
        Self::restrictive()
    }
}

/// Result of URL validation.
#[derive(Debug, Clone)]
pub enum UrlValidation {
    Allowed,
    Blocked { reason: String },
}

impl UrlValidation {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Result of JS validation.
#[derive(Debug, Clone)]
pub enum JsValidation {
    Allowed,
    Blocked { reason: String },
}

impl JsValidation {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

// ============================================================================
// Dangerous patterns
// ============================================================================

/// Patterns that indicate network access or data exfiltration.
fn contains_dangerous_pattern(js: &str) -> bool {
    let lower = js.to_lowercase();
    let patterns = [
        "fetch(",
        "xmlhttprequest",
        "navigator.sendbeacon",
        "websocket",
        "eventsource",
        "window.open",
        "document.cookie",
        "localstorage",
        "sessionstorage",
        "indexeddb",
        "navigator.clipboard",
        "navigator.credentials",
        "rtcpeerconnection",
        "webkitrtcpeerconnection",
        "importscripts",
        "serviceworker",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Patterns that indicate DOM mutation.
fn contains_mutating_pattern(js: &str) -> bool {
    let lower = js.to_lowercase();
    let patterns = [
        "innerhtml",
        "outerhtml",
        "insertadjacenthtml",
        "document.write",
        ".remove(",
        ".appendchild",
        ".replacechild",
        ".setattribute",
        "createelement",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Only the most critical exfiltration patterns (for Full mode).
fn contains_critical_pattern(js: &str) -> bool {
    let lower = js.to_lowercase();
    let patterns = [
        "navigator.sendbeacon",
        "importscripts",
        "serviceworker.register",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Extract host from a URL string.
fn extract_host(url: &str) -> String {
    let without_scheme = url.split("://").nth(1).unwrap_or(url);
    let authority = without_scheme.split('/').next().unwrap_or("");
    // Strip userinfo: in `userinfo@host:port` the real host is after the
    // LAST '@'. Without this, `https://attacker.com@192.168.1.1/` yields
    // host `attacker.com@192.168.1.1`, which fails IP parsing, so the
    // private-IP / metadata-endpoint checks are bypassed and the browser
    // navigates to the real host (192.168.1.1) after the '@' — an SSRF hole.
    let host_port = match authority.rsplit_once('@') {
        Some((_userinfo, host)) => host,
        None => authority,
    };
    // Drop the port. Bracketed IPv6 literals (`[::1]:443`) keep their
    // colons inside the brackets, so split on the closing bracket first.
    let host = if let Some(stripped) = host_port.strip_prefix('[') {
        stripped.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    host.to_string()
}

/// Check if an IP address is private/reserved.
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == 10
                || (o[0] == 172 && (16..=31).contains(&o[1]))
                || (o[0] == 192 && o[1] == 168)
                || (o[0] == 169 && o[1] == 254)
                || o[0] == 127
                || o[0] == 0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_private_ip(&IpAddr::V4(v4)))
        }
    }
}

// ============================================================================
// Tool Permission Categories
// ============================================================================

/// Fine-grained permission categories for MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolPermission {
    /// Read data (files, DB, API).
    Read,
    /// Write/modify data.
    Write,
    /// Execute code or commands.
    Execute,
    /// Delete data irreversibly.
    Delete,
    /// Create new persistent resources.
    Create,
    /// System administration operations.
    Admin,
    /// Outbound network access.
    Network,
    /// Local filesystem access.
    Filesystem,
    /// Access credentials/secrets.
    CredentialAccess,
    /// Act as another user.
    UserImpersonation,
    /// Operations that cost money.
    CostIncurring,
    /// Send data to external endpoints.
    DataExfiltration,
    /// Modify state that survives the session.
    PersistentStateModification,
    /// Change own permission boundary.
    PrivilegeEscalation,
}

impl std::fmt::Display for ToolPermission {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Execute => write!(f, "execute"),
            Self::Delete => write!(f, "delete"),
            Self::Create => write!(f, "create"),
            Self::Admin => write!(f, "admin"),
            Self::Network => write!(f, "network"),
            Self::Filesystem => write!(f, "filesystem"),
            Self::CredentialAccess => write!(f, "credential_access"),
            Self::UserImpersonation => write!(f, "user_impersonation"),
            Self::CostIncurring => write!(f, "cost_incurring"),
            Self::DataExfiltration => write!(f, "data_exfiltration"),
            Self::PersistentStateModification => write!(f, "persistent_state_modification"),
            Self::PrivilegeEscalation => write!(f, "privilege_escalation"),
            _ => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restrictive_blocks_http() {
        let policy = BrowserPolicy::restrictive();
        assert!(policy.validate_url("http://example.com").is_allowed() == false);
        assert!(policy.validate_url("https://example.com").is_allowed());
    }

    #[test]
    fn test_blocks_file_urls() {
        let policy = BrowserPolicy::restrictive();
        let result = policy.validate_url("file:///etc/passwd");
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_blocks_data_urls() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy
            .validate_url("data:text/html,<script>alert(1)</script>")
            .is_allowed());
    }

    #[test]
    fn test_blocks_javascript_urls() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy.validate_url("javascript:alert(1)").is_allowed());
    }

    #[test]
    fn test_blocks_private_ips() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy
            .validate_url("https://192.168.1.1/admin")
            .is_allowed());
        assert!(!policy.validate_url("https://10.0.0.1/").is_allowed());
        assert!(!policy.validate_url("https://127.0.0.1/").is_allowed());
        assert!(!policy.validate_url("https://localhost/").is_allowed());
    }

    #[test]
    fn test_blocks_metadata_endpoints() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy
            .validate_url("https://169.254.169.254/latest/meta-data/")
            .is_allowed());
        assert!(!policy
            .validate_url("https://metadata.google.internal/")
            .is_allowed());
    }

    #[test]
    fn test_userinfo_cannot_bypass_private_ip_check() {
        // SSRF regression: the real host is after the last '@'. Before the
        // extract_host fix, `attacker.com@192.168.1.1` failed IP parsing
        // and slipped past the private-IP gate while the browser would
        // navigate to 192.168.1.1.
        let policy = BrowserPolicy::restrictive();
        assert!(!policy
            .validate_url("https://attacker.com@192.168.1.1/")
            .is_allowed());
        assert!(!policy
            .validate_url("https://user:pass@127.0.0.1/admin")
            .is_allowed());
        assert!(!policy
            .validate_url("https://foo@169.254.169.254/latest/meta-data/")
            .is_allowed());
        // IPv6 loopback literal in brackets must still be caught.
        assert!(!policy.validate_url("https://[::1]/").is_allowed());
        // A legitimate userinfo on a public host stays allowed.
        assert!(policy
            .validate_url("https://user@example.com/")
            .is_allowed());
    }

    #[test]
    fn test_allows_public_https() {
        let policy = BrowserPolicy::restrictive();
        assert!(policy.validate_url("https://example.com").is_allowed());
        assert!(policy
            .validate_url("https://docs.rust-lang.org/book/")
            .is_allowed());
    }

    #[test]
    fn test_permissive_allows_http() {
        let policy = BrowserPolicy::permissive();
        assert!(policy.validate_url("http://example.com").is_allowed());
        assert!(policy.validate_url("https://example.com").is_allowed());
    }

    #[test]
    fn test_js_readonly_blocks_fetch() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy.validate_js("fetch('https://evil.com')").is_allowed());
        assert!(!policy
            .validate_js("new WebSocket('ws://evil.com')")
            .is_allowed());
        assert!(!policy.validate_js("document.cookie").is_allowed());
    }

    #[test]
    fn test_js_readonly_blocks_mutation() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy.validate_js("el.innerHTML = 'test'").is_allowed());
        assert!(!policy.validate_js("document.write('hi')").is_allowed());
    }

    #[test]
    fn test_js_readonly_allows_queries() {
        let policy = BrowserPolicy::restrictive();
        assert!(policy
            .validate_js("document.querySelector('h1').textContent")
            .is_allowed());
        assert!(policy.validate_js("document.title").is_allowed());
    }

    #[test]
    fn test_js_disabled() {
        let mut policy = BrowserPolicy::restrictive();
        policy.js_permission = JsPermission::Disabled;
        assert!(!policy.validate_js("document.title").is_allowed());
    }

    #[test]
    fn test_blocks_rtc_and_clipboard() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy.validate_js("new RTCPeerConnection()").is_allowed());
        assert!(!policy
            .validate_js("navigator.clipboard.readText()")
            .is_allowed());
    }

    #[test]
    fn test_blocks_service_worker() {
        let policy = BrowserPolicy::restrictive();
        assert!(!policy
            .validate_js("navigator.serviceWorker.register('/sw.js')")
            .is_allowed());
    }

    #[test]
    fn test_tool_permission_display() {
        assert_eq!(ToolPermission::Execute.to_string(), "execute");
        assert_eq!(
            ToolPermission::DataExfiltration.to_string(),
            "data_exfiltration"
        );
    }

    #[test]
    fn test_domain_blocklist() {
        let mut policy = BrowserPolicy::restrictive();
        policy.domain_blocklist.insert("evil.com".to_string());
        assert!(!policy.validate_url("https://evil.com/page").is_allowed());
        assert!(policy.validate_url("https://good.com/page").is_allowed());
    }

    #[test]
    fn test_domain_allowlist() {
        let mut policy = BrowserPolicy::restrictive();
        policy.domain_allowlist = Some(HashSet::from(["docs.rust-lang.org".to_string()]));
        assert!(policy
            .validate_url("https://docs.rust-lang.org/book/")
            .is_allowed());
        assert!(!policy.validate_url("https://other.com/").is_allowed());
    }
}
