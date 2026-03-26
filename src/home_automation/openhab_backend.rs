//! OpenHAB Home Backend — device control via OpenHAB REST API.
//!
//! API reference: https://www.openhab.org/docs/configuration/restdocs.html

use super::backend::{DeviceState, HomeBackend, validate_entity_id, validate_backend_url};
use serde::{Deserialize, Serialize};

/// OpenHAB backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenHabConfig {
    /// Base URL (e.g., "http://openhab.local:8080").
    pub base_url: String,
    /// API token for authentication (optional, depends on OpenHAB config).
    #[serde(skip_serializing)]
    pub api_token: Option<String>,
    /// HTTP timeout in seconds.
    pub timeout_secs: u64,
    /// Verify SSL certificates.
    pub verify_ssl: bool,
}

impl Default for OpenHabConfig {
    fn default() -> Self {
        Self {
            base_url: "http://openhab.local:8080".into(),
            api_token: None,
            timeout_secs: 10,
            verify_ssl: true,
        }
    }
}

/// OpenHAB REST API backend.
pub struct OpenHabBackend {
    config: OpenHabConfig,
}

impl OpenHabBackend {
    pub fn new(config: OpenHabConfig) -> Result<Self, String> {
        validate_backend_url(&config.base_url)?;
        Ok(Self { config })
    }

    fn get(&self, path: &str) -> Result<serde_json::Value, String> {
        let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), path);
        validate_backend_url(&url)?;
        let mut req = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .set("Accept", "application/json");
        if let Some(ref token) = self.config.api_token {
            req = req.set("Authorization", &format!("Bearer {}", token));
        }
        let response = req.call().map_err(|e| format!("OpenHAB GET error: {}", e))?;
        response
            .into_json::<serde_json::Value>()
            .map_err(|e| format!("JSON parse error: {}", e))
    }

    fn post_command(&self, item_name: &str, command: &str) -> Result<serde_json::Value, String> {
        let url = format!(
            "{}/rest/items/{}",
            self.config.base_url.trim_end_matches('/'),
            item_name
        );
        validate_backend_url(&url)?;
        let mut req = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .set("Content-Type", "text/plain")
            .set("Accept", "application/json");
        if let Some(ref token) = self.config.api_token {
            req = req.set("Authorization", &format!("Bearer {}", token));
        }
        req.send_string(command)
            .map_err(|e| format!("OpenHAB POST error: {}", e))?;
        Ok(serde_json::json!({"command_sent": command, "item": item_name}))
    }

    /// Parse an OpenHAB item JSON into DeviceState.
    fn parse_item(item: &serde_json::Value) -> DeviceState {
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let label = item.get("label").and_then(|v| v.as_str()).unwrap_or(name);
        let state = item.get("state").and_then(|v| v.as_str()).unwrap_or("NULL");
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

        // Map OpenHAB type to domain
        let domain = match item_type {
            t if t.starts_with("Switch") => "switch",
            t if t.starts_with("Dimmer") => "light",
            t if t.starts_with("Color") => "light",
            t if t.starts_with("Number") => "sensor",
            t if t.starts_with("String") => "sensor",
            t if t.starts_with("Contact") => "binary_sensor",
            t if t.starts_with("Rollershutter") => "cover",
            t if t.starts_with("Player") => "media_player",
            _ => "sensor",
        };

        let entity_id = format!("{}.{}", domain, name.to_lowercase());

        // Build attributes from metadata and state description
        let mut attrs = serde_json::json!({
            "openhab_type": item_type,
        });
        if let Some(metadata) = item.get("metadata") {
            attrs["metadata"] = metadata.clone();
        }
        if let Some(state_desc) = item.get("stateDescription") {
            attrs["state_description"] = state_desc.clone();
        }
        if let Some(tags) = item.get("tags") {
            attrs["tags"] = tags.clone();
        }

        DeviceState {
            entity_id,
            name: label.to_string(),
            state: state.to_string(),
            attributes: attrs,
            last_changed: String::new(), // OpenHAB doesn't return this in items list
        }
    }
}

impl HomeBackend for OpenHabBackend {
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String> {
        let items = self.get("/rest/items")?;
        let arr = items.as_array().ok_or("Expected JSON array from /rest/items")?;
        let devices: Vec<DeviceState> = arr
            .iter()
            .map(Self::parse_item)
            .filter(|d| match domain {
                Some(dom) => d.entity_id.starts_with(&format!("{}.", dom)),
                None => true,
            })
            .collect();
        Ok(devices)
    }

    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String> {
        validate_entity_id(entity_id)?;
        // OpenHAB items are named by the part after the dot
        let item_name = entity_id
            .split('.')
            .nth(1)
            .ok_or_else(|| format!("Invalid entity_id: {}", entity_id))?;
        let item = self.get(&format!("/rest/items/{}", item_name))?;
        if item.get("name").is_none() {
            return Err(format!("Item not found: {}", item_name));
        }
        Ok(Self::parse_item(&item))
    }

    fn call_service(
        &self,
        _domain: &str,
        service: &str,
        entity_id: &str,
        data: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        validate_entity_id(entity_id)?;
        let item_name = entity_id
            .split('.')
            .nth(1)
            .ok_or_else(|| format!("Invalid entity_id: {}", entity_id))?;

        let command = match service {
            "turn_on" => "ON".to_string(),
            "turn_off" => "OFF".to_string(),
            "toggle" => "TOGGLE".to_string(),
            "set_temperature" | "set_value" => {
                data.and_then(|d| d.get("value").or(d.get("temperature")))
                    .map(|v| v.to_string().trim_matches('"').to_string())
                    .unwrap_or_else(|| "0".into())
            }
            "open" => "UP".to_string(),
            "close" => "DOWN".to_string(),
            "stop" => "STOP".to_string(),
            _ => data
                .map(|d| d.to_string())
                .unwrap_or_else(|| service.to_uppercase()),
        };

        self.post_command(item_name, &command)
    }

    fn list_scenes(&self) -> Result<Vec<DeviceState>, String> {
        // OpenHAB scenes are items tagged with "Scene"
        let all = self.list_devices(None)?;
        Ok(all
            .into_iter()
            .filter(|d| {
                d.attributes
                    .get("tags")
                    .and_then(|t| t.as_array())
                    .map(|tags| tags.iter().any(|t| t.as_str() == Some("Scene")))
                    .unwrap_or(false)
            })
            .collect())
    }

    fn list_automations(&self) -> Result<Vec<DeviceState>, String> {
        // OpenHAB automations are rules, accessible via different API
        // For simplicity, return items tagged with "Automation"
        let all = self.list_devices(None)?;
        Ok(all
            .into_iter()
            .filter(|d| {
                d.attributes
                    .get("tags")
                    .and_then(|t| t.as_array())
                    .map(|tags| tags.iter().any(|t| t.as_str() == Some("Automation")))
                    .unwrap_or(false)
            })
            .collect())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openhab_config_default() {
        let config = OpenHabConfig::default();
        assert_eq!(config.base_url, "http://openhab.local:8080");
        assert_eq!(config.timeout_secs, 10);
    }

    #[test]
    fn test_parse_item_switch() {
        let item = serde_json::json!({
            "name": "LivingRoom_Light",
            "label": "Living Room Light",
            "type": "Switch",
            "state": "ON",
            "tags": ["Lighting"]
        });
        let device = OpenHabBackend::parse_item(&item);
        assert_eq!(device.entity_id, "switch.livingroom_light");
        assert_eq!(device.name, "Living Room Light");
        assert_eq!(device.state, "ON");
    }

    #[test]
    fn test_parse_item_dimmer() {
        let item = serde_json::json!({
            "name": "Bedroom_Dimmer",
            "label": "Bedroom",
            "type": "Dimmer",
            "state": "75"
        });
        let device = OpenHabBackend::parse_item(&item);
        assert_eq!(device.entity_id, "light.bedroom_dimmer");
        assert_eq!(device.state, "75");
    }

    #[test]
    fn test_parse_item_number_sensor() {
        let item = serde_json::json!({
            "name": "Temperature_Outside",
            "label": "Outside Temperature",
            "type": "Number:Temperature",
            "state": "23.5 °C"
        });
        let device = OpenHabBackend::parse_item(&item);
        assert_eq!(device.entity_id, "sensor.temperature_outside");
        assert_eq!(device.state, "23.5 °C");
    }

    #[test]
    fn test_parse_item_rollershutter() {
        let item = serde_json::json!({
            "name": "Kitchen_Blinds",
            "label": "Kitchen Blinds",
            "type": "Rollershutter",
            "state": "50"
        });
        let device = OpenHabBackend::parse_item(&item);
        assert_eq!(device.entity_id, "cover.kitchen_blinds");
    }

    #[test]
    fn test_openhab_ssrf_blocked() {
        let mut config = OpenHabConfig::default();
        config.base_url = "http://169.254.169.254".into();
        let result = OpenHabBackend::new(config);
        assert!(result.is_err());
    }
}
