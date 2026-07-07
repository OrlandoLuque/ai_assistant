//! HomeBackend trait and shared types for home automation backends.

use serde::{Deserialize, Serialize};

/// State of a home automation device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    pub entity_id: String,
    pub name: String,
    pub state: String,
    pub attributes: serde_json::Value,
    pub last_changed: String,
}

/// Configuration for a home automation backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HomeConfig {
    pub backend_type: String,
    pub base_url: String,
    #[serde(skip_serializing)]
    pub token: String,
    pub verify_ssl: bool,
    pub timeout_secs: u64,
}

impl Default for HomeConfig {
    fn default() -> Self {
        Self {
            backend_type: "home_assistant".into(),
            base_url: "http://homeassistant.local:8123".into(),
            token: String::new(),
            verify_ssl: true,
            timeout_secs: 10,
        }
    }
}

/// Abstraction over home automation platforms.
pub trait HomeBackend: Send + Sync {
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String>;
    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String>;
    fn call_service(
        &self,
        domain: &str,
        service: &str,
        entity_id: &str,
        data: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String>;
    fn list_scenes(&self) -> Result<Vec<DeviceState>, String>;
    fn list_automations(&self) -> Result<Vec<DeviceState>, String>;
}

// ============================================================================
// Validation (shared security helpers)
// ============================================================================

pub fn validate_entity_id(entity_id: &str) -> Result<(), String> {
    if !entity_id.contains('.') {
        return Err(format!(
            "Invalid entity_id '{}': expected 'domain.name'",
            entity_id
        ));
    }
    if entity_id.contains("..") || entity_id.contains('/') || entity_id.contains('\\') {
        return Err("Invalid entity_id: path traversal characters".into());
    }
    if entity_id.len() > 256 {
        return Err("Entity ID too long".into());
    }
    Ok(())
}

pub fn validate_domain(domain: &str) -> Result<(), String> {
    if domain.is_empty() || domain.len() > 64 {
        return Err("Invalid domain length".into());
    }
    if !domain
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(format!(
            "Invalid domain '{}': alphanumeric and underscore only",
            domain
        ));
    }
    Ok(())
}

pub fn validate_service_name(service: &str) -> Result<(), String> {
    if service.is_empty() || service.len() > 64 {
        return Err("Invalid service name length".into());
    }
    if !service
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(format!(
            "Invalid service '{}': alphanumeric and underscore only",
            service
        ));
    }
    Ok(())
}

pub fn extract_domain(entity_id: &str) -> Result<String, String> {
    validate_entity_id(entity_id)?;
    entity_id
        .split('.')
        .next()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Cannot extract domain from '{}'", entity_id))
}

/// Validate a broker/service URL for SSRF.
///
/// Home hubs legitimately live on the LAN, so — unlike the general
/// [`crate::ssrf`] guard — private/loopback ranges are intentionally NOT
/// blocked here. Only cloud-metadata endpoints are denied, including the
/// non-AWS/GCP ones (Alibaba, Oracle) and integer-encoded IP forms that the
/// old substring check missed (e.g. `http://2852039166/` == 169.254.169.254).
pub fn validate_backend_url(url: &str) -> Result<(), String> {
    let lower = url.to_lowercase();
    // Metadata hostnames.
    if lower.contains("metadata.google") {
        return Err("Blocked: SSRF target (metadata endpoint)".into());
    }
    // Resolve the host — decoding encoded-IP forms — and block metadata IPs.
    if let Some(host) = crate::ssrf::extract_host(url) {
        if crate::ssrf::parse_host_ip(host).is_some_and(|ip| is_cloud_metadata_ip(&ip)) {
            return Err("Blocked: SSRF target (metadata endpoint)".into());
        }
    }
    // Literal fallback for hosts we cannot parse.
    if lower.contains("169.254.")
        || lower.contains("100.100.100.200")
        || lower.contains("192.0.0.192")
    {
        return Err("Blocked: SSRF target (metadata endpoint)".into());
    }
    Ok(())
}

/// True if `ip` is a well-known cloud instance-metadata address. Covers the
/// link-local range used by AWS/Azure/GCP/DigitalOcean plus the Alibaba and
/// Oracle OCI classic endpoints. Does **not** match general private ranges.
fn is_cloud_metadata_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let o = v4.octets();
            (o[0] == 169 && o[1] == 254) // 169.254.0.0/16 (169.254.169.254, ...)
                || o == [100, 100, 100, 200] // Alibaba Cloud
                || o == [192, 0, 0, 192] // Oracle OCI classic
        }
        std::net::IpAddr::V6(_) => false,
    }
}

#[cfg(test)]
mod backend_url_tests {
    use super::validate_backend_url;

    #[test]
    fn blocks_cloud_metadata_including_encoded_and_non_aws() {
        // AWS/Azure/GCP link-local metadata, literal and integer-encoded.
        assert!(validate_backend_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_backend_url("http://2852039166/latest/meta-data/").is_err());
        assert!(validate_backend_url("http://metadata.google.internal/").is_err());
        // Non-AWS/GCP metadata endpoints the old substring check missed.
        assert!(validate_backend_url("http://100.100.100.200/").is_err()); // Alibaba
        assert!(validate_backend_url("http://192.0.0.192/").is_err()); // Oracle
    }

    #[test]
    fn allows_lan_hosts() {
        // Home hubs live on the LAN — private ranges must NOT be blocked here.
        assert!(validate_backend_url("http://192.168.1.50:8123/").is_ok());
        assert!(validate_backend_url("http://10.0.0.5/").is_ok());
        assert!(validate_backend_url("http://homeassistant.local:8123/").is_ok());
        assert!(validate_backend_url("https://api.example.com/").is_ok());
    }
}
