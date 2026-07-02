//! Network Policy — egress control for agent HTTP requests.
//!
//! Inspired by NemoClaw (NVIDIA): default-deny network policy that only
//! allows explicitly whitelisted endpoints. Prevents SSRF, data exfiltration,
//! and unauthorized external access.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::IpAddr;

/// Network policy for controlling outbound HTTP requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NetworkPolicy {
    /// If true, all outbound requests are denied unless explicitly allowed.
    pub default_deny: bool,
    /// Allowed hostnames/domains (exact match or wildcard *.example.com).
    pub allowed_hosts: HashSet<String>,
    /// Blocked hostnames (checked before allowed).
    pub blocked_hosts: HashSet<String>,
    /// Blocked IP ranges (private networks, cloud metadata).
    pub blocked_ip_ranges: Vec<String>,
    /// Whether to log all blocked requests.
    pub log_blocked: bool,
    /// Whether to log all allowed requests (for audit).
    pub log_allowed: bool,
}

impl NetworkPolicy {
    /// Permissive policy — allows everything (development mode).
    pub fn permissive() -> Self {
        Self {
            default_deny: false,
            allowed_hosts: HashSet::new(),
            blocked_hosts: HashSet::new(),
            blocked_ip_ranges: Vec::new(),
            log_blocked: true,
            log_allowed: false,
        }
    }

    /// Restrictive policy — blocks private IPs + cloud metadata.
    pub fn restrictive() -> Self {
        let mut policy = Self {
            default_deny: false,
            allowed_hosts: HashSet::new(),
            blocked_hosts: HashSet::new(),
            blocked_ip_ranges: vec![
                "10.0.0.0/8".into(),
                "172.16.0.0/12".into(),
                "192.168.0.0/16".into(),
                "169.254.0.0/16".into(), // AWS/Azure/GCP metadata
                "127.0.0.0/8".into(),
                "0.0.0.0/8".into(),
            ],
            log_blocked: true,
            log_allowed: true,
        };
        // Block common cloud metadata endpoints
        policy
            .blocked_hosts
            .insert("metadata.google.internal".into());
        policy.blocked_hosts.insert("metadata.google.com".into());
        policy
    }

    /// Paranoid policy — default deny, only whitelisted hosts allowed.
    pub fn paranoid() -> Self {
        Self {
            default_deny: true,
            allowed_hosts: HashSet::new(),
            blocked_hosts: HashSet::new(),
            blocked_ip_ranges: vec![
                "10.0.0.0/8".into(),
                "172.16.0.0/12".into(),
                "192.168.0.0/16".into(),
                "169.254.0.0/16".into(),
                "127.0.0.0/8".into(),
                "0.0.0.0/8".into(),
            ],
            log_blocked: true,
            log_allowed: true,
        }
    }

    /// Add an allowed host.
    pub fn allow_host(&mut self, host: &str) {
        self.allowed_hosts.insert(host.to_lowercase());
    }

    /// Add a blocked host.
    pub fn block_host(&mut self, host: &str) {
        self.blocked_hosts.insert(host.to_lowercase());
    }

    /// Check if a hostname is allowed by this policy.
    pub fn check_host(&self, host: &str) -> PolicyDecision {
        let host_lower = host.to_lowercase();

        // 1. Check explicit block list first (highest priority)
        if self.blocked_hosts.contains(&host_lower) {
            return PolicyDecision::Denied {
                reason: format!("Host '{}' is in blocked list", host),
            };
        }

        // 2. Check wildcard blocks
        for blocked in &self.blocked_hosts {
            if blocked.starts_with("*.") {
                let domain = &blocked[2..];
                // Require a dot boundary: `*.openai.com` must match
                // `api.openai.com` and `openai.com`, but NOT `evilopenai.com`.
                if host_lower == domain
                    || host_lower
                        .strip_suffix(domain)
                        .is_some_and(|p| p.ends_with('.'))
                {
                    return PolicyDecision::Denied {
                        reason: format!("Host '{}' matches blocked pattern '{}'", host, blocked),
                    };
                }
            }
        }

        // 3. Check explicit allow list
        if self.allowed_hosts.contains(&host_lower) {
            return PolicyDecision::Allowed;
        }

        // 4. Check wildcard allows
        for allowed in &self.allowed_hosts {
            if allowed.starts_with("*.") {
                let domain = &allowed[2..];
                // Same dot-boundary requirement as the block path.
                if host_lower == domain
                    || host_lower
                        .strip_suffix(domain)
                        .is_some_and(|p| p.ends_with('.'))
                {
                    return PolicyDecision::Allowed;
                }
            }
        }

        // 5. Default policy
        if self.default_deny {
            PolicyDecision::Denied {
                reason: format!("Host '{}' not in allow list (default-deny policy)", host),
            }
        } else {
            PolicyDecision::Allowed
        }
    }

    /// Check if an IP address is in a blocked range.
    pub fn check_ip(&self, ip: &IpAddr) -> PolicyDecision {
        if self.is_private_ip(ip) && !self.blocked_ip_ranges.is_empty() {
            return PolicyDecision::Denied {
                reason: format!("IP {} is in a private/reserved range", ip),
            };
        }
        PolicyDecision::Allowed
    }

    /// Check if an IP is in a private/reserved range.
    fn is_private_ip(&self, ip: &IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                // 10.0.0.0/8
                octets[0] == 10
                // 172.16.0.0/12
                || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                // 192.168.0.0/16
                || (octets[0] == 192 && octets[1] == 168)
                // 169.254.0.0/16 (link-local / cloud metadata)
                || (octets[0] == 169 && octets[1] == 254)
                // 127.0.0.0/8 (loopback)
                || octets[0] == 127
                // 0.0.0.0/8
                || octets[0] == 0
            }
            IpAddr::V6(v6) => {
                v6.is_loopback() || v6.segments()[0] == 0xfd00 // ULA
            }
        }
    }
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self::restrictive()
    }
}

/// Result of a network policy check.
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    Allowed,
    Denied { reason: String },
}

impl PolicyDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied { .. })
    }
}

// ============================================================================
// Presets for common integrations (inspired by NemoClaw)
// ============================================================================

impl NetworkPolicy {
    /// Preset for an agent that uses web search APIs.
    pub fn preset_web_search() -> Self {
        let mut policy = Self::paranoid();
        policy.allow_host("html.duckduckgo.com");
        policy.allow_host("api.search.brave.com");
        policy.allow_host("*.googleapis.com");
        policy.allow_host("api.bing.microsoft.com");
        policy.allow_host("serpapi.com");
        policy.allow_host("api.tavily.com");
        policy
    }

    /// Preset for an agent that uses LLM cloud APIs.
    pub fn preset_llm_apis() -> Self {
        let mut policy = Self::paranoid();
        policy.allow_host("api.openai.com");
        policy.allow_host("api.anthropic.com");
        policy.allow_host("generativelanguage.googleapis.com");
        policy.allow_host("*.api.amazone.com"); // Bedrock
        policy.allow_host("api.groq.com");
        policy.allow_host("api.together.xyz");
        policy.allow_host("localhost"); // For local Ollama/LM Studio
        policy
    }

    /// Preset combining web search + LLM APIs.
    pub fn preset_research_agent() -> Self {
        let mut policy = Self::preset_llm_apis();
        let search = Self::preset_web_search();
        for host in search.allowed_hosts {
            policy.allow_host(&host);
        }
        policy
    }

    /// Preset for home automation: allows common local services.
    /// Whitelists Home Assistant, OpenHAB, MQTT brokers on local network.
    pub fn preset_home_automation() -> Self {
        let mut policy = Self::restrictive();
        policy.allow_host("homeassistant.local");
        policy.allow_host("*.homeassistant.local");
        policy.allow_host("openhab.local");
        policy.allow_host("*.openhab.local");
        policy.allow_host("mqtt.local");
        policy.allow_host("mosquitto.local");
        // Allow local network access for home devices
        // (restrictive() blocks some private ranges via blocked_ip_ranges,
        //  but home automation typically needs local network access)
        policy.blocked_ip_ranges.clear();
        policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permissive_allows_all() {
        let policy = NetworkPolicy::permissive();
        assert!(policy.check_host("anything.com").is_allowed());
        assert!(policy.check_host("evil.com").is_allowed());
    }

    #[test]
    fn test_paranoid_blocks_by_default() {
        let policy = NetworkPolicy::paranoid();
        assert!(policy.check_host("unknown.com").is_denied());
    }

    #[test]
    fn test_paranoid_allows_whitelisted() {
        let mut policy = NetworkPolicy::paranoid();
        policy.allow_host("api.openai.com");
        assert!(policy.check_host("api.openai.com").is_allowed());
        assert!(policy.check_host("evil.com").is_denied());
    }

    #[test]
    fn test_blocked_overrides_allowed() {
        let mut policy = NetworkPolicy::permissive();
        policy.block_host("evil.com");
        assert!(policy.check_host("evil.com").is_denied());
        assert!(policy.check_host("good.com").is_allowed());
    }

    #[test]
    fn test_wildcard_allow() {
        let mut policy = NetworkPolicy::paranoid();
        policy.allow_host("*.openai.com");
        assert!(policy.check_host("api.openai.com").is_allowed());
        assert!(policy.check_host("cdn.openai.com").is_allowed());
        assert!(policy.check_host("evil.com").is_denied());
    }

    #[test]
    fn test_private_ip_blocked() {
        let policy = NetworkPolicy::restrictive();
        let private_ip: IpAddr = "10.0.0.1".parse().unwrap();
        let public_ip: IpAddr = "8.8.8.8".parse().unwrap();
        let metadata_ip: IpAddr = "169.254.169.254".parse().unwrap();

        assert!(policy.check_ip(&private_ip).is_denied());
        assert!(policy.check_ip(&metadata_ip).is_denied());
        assert!(policy.check_ip(&public_ip).is_allowed());
    }

    #[test]
    fn test_localhost_blocked() {
        let policy = NetworkPolicy::restrictive();
        let localhost: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(policy.check_ip(&localhost).is_denied());
    }

    #[test]
    fn test_preset_web_search() {
        let policy = NetworkPolicy::preset_web_search();
        assert!(policy.check_host("html.duckduckgo.com").is_allowed());
        assert!(policy.check_host("api.tavily.com").is_allowed());
        assert!(policy.check_host("evil.com").is_denied());
    }

    #[test]
    fn test_preset_llm_apis() {
        let policy = NetworkPolicy::preset_llm_apis();
        assert!(policy.check_host("api.openai.com").is_allowed());
        assert!(policy.check_host("api.anthropic.com").is_allowed());
        assert!(policy.check_host("evil.com").is_denied());
    }

    #[test]
    fn test_case_insensitive() {
        let mut policy = NetworkPolicy::paranoid();
        policy.allow_host("API.OpenAI.COM");
        assert!(policy.check_host("api.openai.com").is_allowed());
    }
}
