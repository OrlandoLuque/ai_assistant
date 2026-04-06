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
pub fn validate_backend_url(url: &str) -> Result<(), String> {
    let lower = url.to_lowercase();
    if lower.contains("169.254.") || lower.contains("metadata.google") {
        return Err("Blocked: SSRF target (metadata endpoint)".into());
    }
    Ok(())
}
