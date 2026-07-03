//! Shared SSRF host-classification helpers (audit V188).
//!
//! Several outbound-fetch guards (`os_tools` http_get, `media_generation`
//! image download, `cloud_connectors` S3 endpoints, `event_source` pollers)
//! independently re-implemented "is this URL host private/internal?" and each
//! missed cases the audit flagged:
//!
//! * integer-encoded IPv4 — `http://2130706433/`, `http://0x7f000001/`,
//!   `http://017700000001/`, `http://127.1/`, `http://0x7f.0.0.1/` all
//!   resolve to loopback/internal via `getaddrinfo` but fail
//!   `IpAddr::parse`, so a parse-only guard skips the private-range check;
//! * bracketed / IPv4-mapped / ULA / link-local IPv6 literals; and
//! * `userinfo@host` authorities (`http://trusted.com@127.0.0.1/`).
//!
//! This module centralises a single correct implementation. It is a
//! best-effort *pre-flight* check: it does NOT resolve DNS, so a hostname
//! whose A record points at a private address (DNS rebinding) still needs a
//! connect-time guard, which the HTTP clients used here do not expose.

use std::net::{IpAddr, Ipv4Addr};

/// Extract the host from a URL, stripping the scheme, path, `userinfo@`
/// prefix and `:port` suffix, and unwrapping a bracketed IPv6 literal.
///
/// Returns `None` when there is no `://` authority or the host is empty.
/// The returned slice borrows from `url`.
pub fn extract_host(url: &str) -> Option<&str> {
    let without_scheme = url.split("://").nth(1)?;
    let authority = without_scheme.split('/').next().unwrap_or("");
    if authority.is_empty() {
        return None;
    }
    // The real host is after the LAST '@' (userinfo may itself contain '@').
    let host_port = match authority.rsplit_once('@') {
        Some((_userinfo, host)) => host,
        None => authority,
    };
    // Bracketed IPv6 (`[::1]:443`) keeps its colons inside the brackets, so
    // unwrap the bracket before splitting on the port colon.
    let host = if let Some(stripped) = host_port.strip_prefix('[') {
        stripped.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

/// Parse a single dotted segment as an `inet_aton`-style unsigned integer:
/// `0x`/`0X` prefix → hex, a leading `0` → octal, otherwise decimal.
fn parse_int_segment(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        if hex.is_empty() {
            return None;
        }
        u32::from_str_radix(hex, 16).ok()
    } else if s.len() > 1 && s.starts_with('0') {
        u32::from_str_radix(&s[1..], 8).ok()
    } else {
        s.parse::<u32>().ok()
    }
}

/// Decode the legacy `inet_aton` IPv4 forms that `getaddrinfo` still accepts
/// but `Ipv4Addr::parse` rejects: 1–4 dotted parts, each in decimal, hex
/// (`0x..`) or octal (`0..`); fewer than 4 parts pack the trailing value into
/// the low bytes (`127.1` == `127.0.0.1`, `2130706433` == `127.0.0.1`).
fn parse_inet_aton_ipv4(host: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.is_empty() || parts.len() > 4 {
        return None;
    }
    let nums: Vec<u32> = parts
        .iter()
        .map(|p| parse_int_segment(p))
        .collect::<Option<_>>()?;
    let addr: u32 = match nums.as_slice() {
        [a] => *a,
        [a, b] => {
            if *a > 0xff || *b > 0x00ff_ffff {
                return None;
            }
            (a << 24) | b
        }
        [a, b, c] => {
            if *a > 0xff || *b > 0xff || *c > 0xffff {
                return None;
            }
            (a << 24) | (b << 16) | c
        }
        [a, b, c, d] => {
            if *a > 0xff || *b > 0xff || *c > 0xff || *d > 0xff {
                return None;
            }
            (a << 24) | (b << 16) | (c << 8) | d
        }
        _ => return None,
    };
    Some(Ipv4Addr::from(addr))
}

/// Parse a host string into an [`IpAddr`], accepting the standard textual
/// forms PLUS the integer/legacy IPv4 encodings that resolvers still map to
/// the same address. Returns `None` for genuine hostnames.
pub fn parse_host_ip(host: &str) -> Option<IpAddr> {
    // Standard dotted-decimal IPv4 or textual IPv6.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(ip);
    }
    // Defensive: a bracketed IPv6 literal that reached us un-unwrapped.
    if let Some(inner) = host.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        if let Ok(ip) = inner.parse::<IpAddr>() {
            return Some(ip);
        }
    }
    // Legacy inet_aton IPv4 encodings.
    parse_inet_aton_ipv4(host).map(IpAddr::V4)
}

/// True if an IPv4 address is loopback / private / link-local / unspecified /
/// CGNAT — i.e. must be blocked for SSRF.
fn is_blocked_v4(v4: &Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_loopback()            // 127.0.0.0/8
        || v4.is_private()      // 10/8, 172.16/12, 192.168/16
        || v4.is_link_local()   // 169.254.0.0/16 (incl. cloud metadata)
        || v4.is_unspecified()  // 0.0.0.0
        || o[0] == 0            // 0.0.0.0/8
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // CGNAT 100.64.0.0/10
}

/// True if the IP is in a range that must be blocked for SSRF: loopback,
/// private, link-local, ULA, unspecified, CGNAT, or an IPv4-mapped IPv6 of
/// any of those.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // ULA fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
                || v6.to_ipv4_mapped().is_some_and(|v4| is_blocked_v4(&v4))
        }
    }
}

/// True if the host string is a well-known internal/loopback name or a cloud
/// metadata hostname. Case-insensitive.
pub fn is_internal_hostname(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    lower == "localhost"
        || lower == "metadata"
        || lower == "metadata.google.internal"
        || lower.ends_with(".local")
        || lower.ends_with(".localhost")
        || lower.ends_with(".internal")
}

/// One-shot SSRF pre-flight for a full URL: returns `true` when the URL's
/// host is an internal hostname or resolves (via literal / encoded IP) to a
/// blocked range. Does **not** resolve DNS. Returns `false` when no host can
/// be extracted (callers that want fail-closed should check `extract_host`
/// themselves).
pub fn is_ssrf_blocked_url(url: &str) -> bool {
    let Some(host) = extract_host(url) else {
        return false;
    };
    if is_internal_hostname(host) {
        return true;
    }
    parse_host_ip(host).is_some_and(|ip| is_blocked_ip(&ip))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_host_stripping_userinfo_port_brackets() {
        assert_eq!(extract_host("http://example.com/path"), Some("example.com"));
        assert_eq!(
            extract_host("https://example.com:8443/x"),
            Some("example.com")
        );
        // userinfo must not become the host
        assert_eq!(
            extract_host("http://trusted.com@127.0.0.1/"),
            Some("127.0.0.1")
        );
        assert_eq!(extract_host("http://a@b@10.0.0.1/"), Some("10.0.0.1"));
        // bracketed IPv6
        assert_eq!(extract_host("http://[::1]:8080/"), Some("::1"));
        assert_eq!(extract_host("http://[fd00::1]/"), Some("fd00::1"));
        assert_eq!(extract_host("not-a-url"), None);
    }

    #[test]
    fn parses_encoded_ipv4_forms() {
        let loop_v4: IpAddr = "127.0.0.1".parse().unwrap();
        assert_eq!(parse_host_ip("2130706433"), Some(loop_v4)); // decimal
        assert_eq!(parse_host_ip("0x7f000001"), Some(loop_v4)); // hex
        assert_eq!(parse_host_ip("017700000001"), Some(loop_v4)); // octal
        assert_eq!(parse_host_ip("127.1"), Some(loop_v4)); // short form
        assert_eq!(parse_host_ip("0x7f.0.0.1"), Some(loop_v4)); // dotted hex
        assert_eq!(parse_host_ip("0177.0.0.1"), Some(loop_v4)); // dotted octal
                                                                // decimal form of the cloud metadata address 169.254.169.254
        assert_eq!(
            parse_host_ip("2852039166"),
            Some("169.254.169.254".parse().unwrap())
        );
        // Genuine hostnames must NOT parse as IPs.
        assert_eq!(parse_host_ip("api.openai.com"), None);
        assert_eq!(parse_host_ip("example.com"), None);
        // Standard public IP still parses.
        assert_eq!(parse_host_ip("8.8.8.8"), Some("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn blocks_private_and_reserved_ips() {
        for s in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.5.5",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "100.64.0.1", // CGNAT
            "::1",
            "fd12:3456::1",     // ULA (was missed by the old 0xfd00 check)
            "fe80::1",          // link-local
            "::",               // unspecified
            "::ffff:127.0.0.1", // IPv4-mapped loopback
        ] {
            let ip: IpAddr = s.parse().unwrap();
            assert!(is_blocked_ip(&ip), "expected {s} to be blocked");
        }
        for s in ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888"] {
            let ip: IpAddr = s.parse().unwrap();
            assert!(!is_blocked_ip(&ip), "expected {s} to be allowed");
        }
    }

    #[test]
    fn full_url_guard_catches_bypasses() {
        for url in [
            "http://127.0.0.1/",
            "http://2130706433/",                  // decimal loopback
            "http://0x7f000001/latest/meta-data/", // hex loopback
            "http://trusted.com@169.254.169.254/", // userinfo + metadata
            "http://[::1]:9000/",                  // bracketed IPv6 loopback
            "http://[::ffff:10.0.0.1]/",           // mapped private
            "http://localhost/",
            "https://foo.internal/",
        ] {
            assert!(is_ssrf_blocked_url(url), "expected {url} to be blocked");
        }
        for url in ["https://api.openai.com/v1", "http://8.8.8.8/"] {
            assert!(!is_ssrf_blocked_url(url), "expected {url} to be allowed");
        }
    }
}
