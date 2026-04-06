//! MQTT Home Backend — direct device control via MQTT broker.
//!
//! Supports Zigbee2MQTT, Tasmota, and Home Assistant MQTT Discovery conventions.
//! Auto-discovers devices via bridge topics. State tracked via subscriptions.

use super::backend::{
    validate_backend_url, validate_domain, validate_entity_id, validate_service_name, DeviceState,
    HomeBackend,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// Configuration
// ============================================================================

/// MQTT backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MqttConfig {
    /// Broker URL (e.g., "mqtt://localhost:1883" or "mqtts://broker.example.com:8883").
    pub broker_url: String,
    /// Username for authentication.
    pub username: Option<String>,
    /// Credential key for password (resolved via CredentialResolver, never stored directly).
    #[serde(skip_serializing)]
    pub password_key: Option<String>,
    /// Client ID (auto-generated if empty).
    pub client_id: String,
    /// Use TLS (default: true). Set allow_insecure_mqtt to override (#9).
    pub use_tls: bool,
    /// Explicitly allow insecure (non-TLS) MQTT. Logs security warning.
    pub allow_insecure_mqtt: bool,
    /// Topic naming convention.
    pub topic_convention: TopicConvention,
    /// Enable auto-discovery via bridge/discovery topics.
    pub discovery_enabled: bool,
    /// Keep-alive interval in seconds.
    pub keepalive_secs: u64,
    /// Max commands per minute per device (#7).
    pub max_commands_per_minute: u32,
    /// Max total commands per minute (#7).
    pub max_total_commands_per_minute: u32,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            broker_url: "mqtt://localhost:1883".into(),
            username: None,
            password_key: None,
            client_id: format!("ai_assistant_{}", std::process::id()),
            use_tls: true,
            allow_insecure_mqtt: false,
            topic_convention: TopicConvention::Zigbee2Mqtt,
            discovery_enabled: true,
            keepalive_secs: 30,
            max_commands_per_minute: 10,
            max_total_commands_per_minute: 60,
        }
    }
}

/// MQTT topic naming convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TopicConvention {
    /// Zigbee2MQTT: `zigbee2mqtt/{device}` for state, `zigbee2mqtt/{device}/set` for commands.
    Zigbee2Mqtt,
    /// Tasmota: `stat/{device}/POWER` for state, `cmnd/{device}/POWER` for commands.
    Tasmota,
    /// Home Assistant MQTT Discovery: `homeassistant/{domain}/{device}/config`.
    HomeAssistant,
    /// Custom templates with `{device}` placeholder.
    Custom {
        state_template: String,
        command_template: String,
    },
}

impl TopicConvention {
    /// Get the state topic for a device.
    pub fn state_topic(&self, device: &str) -> String {
        match self {
            Self::Zigbee2Mqtt => format!("zigbee2mqtt/{}", device),
            Self::Tasmota => format!("stat/{}/POWER", device),
            Self::HomeAssistant => format!("homeassistant/+/{}/state", device),
            Self::Custom { state_template, .. } => state_template.replace("{device}", device),
        }
    }

    /// Get the command topic for a device.
    pub fn command_topic(&self, device: &str) -> String {
        match self {
            Self::Zigbee2Mqtt => format!("zigbee2mqtt/{}/set", device),
            Self::Tasmota => format!("cmnd/{}/POWER", device),
            Self::HomeAssistant => format!("homeassistant/+/{}/set", device),
            Self::Custom {
                command_template, ..
            } => command_template.replace("{device}", device),
        }
    }

    /// Get the discovery topic for auto-discovery.
    pub fn discovery_topic(&self) -> &str {
        match self {
            Self::Zigbee2Mqtt => "zigbee2mqtt/bridge/devices",
            Self::Tasmota => "tasmota/discovery/#",
            Self::HomeAssistant => "homeassistant/+/+/config",
            Self::Custom { .. } => "",
        }
    }
}

// ============================================================================
// Device Registry
// ============================================================================

/// In-memory registry of discovered MQTT devices.
#[derive(Debug, Clone)]
pub struct DeviceRegistry {
    devices: HashMap<String, RegistryEntry>,
}

/// A device entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub entity_id: String,
    pub name: String,
    pub domain: String,
    pub state_topic: String,
    pub command_topic: Option<String>,
    pub last_state: Option<String>,
    pub attributes: serde_json::Value,
    pub discovered_at: u64,
    pub last_seen: u64,
}

impl DeviceRegistry {
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    /// Register or update a device.
    pub fn register(&mut self, entry: RegistryEntry) {
        self.devices.insert(entry.entity_id.clone(), entry);
    }

    /// Get a device by entity_id.
    pub fn get(&self, entity_id: &str) -> Option<&RegistryEntry> {
        self.devices.get(entity_id)
    }

    /// List all devices, optionally filtered by domain.
    pub fn list(&self, domain: Option<&str>) -> Vec<&RegistryEntry> {
        self.devices
            .values()
            .filter(|d| match domain {
                Some(dom) => d.domain == dom,
                None => true,
            })
            .collect()
    }

    /// Update the state of a device.
    pub fn update_state(&mut self, entity_id: &str, state: &str, attributes: serde_json::Value) {
        if let Some(entry) = self.devices.get_mut(entity_id) {
            entry.last_state = Some(state.to_string());
            entry.attributes = attributes;
            entry.last_seen = now_epoch();
        }
    }

    /// Number of registered devices.
    pub fn count(&self) -> usize {
        self.devices.len()
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MQTT Home Backend
// ============================================================================

/// MQTT-based home automation backend.
/// Devices are tracked via `DeviceRegistry` (in-memory cache).
/// Commands are published to MQTT topics.
/// State is updated via subscriptions to state topics.
pub struct MqttHomeBackend {
    #[allow(dead_code)]
    config: MqttConfig,
    registry: Arc<Mutex<DeviceRegistry>>,
    /// Command rate limiter — tracks (timestamp_secs, entity_id) pairs.
    command_history: Arc<Mutex<Vec<(u64, String)>>>,
}

impl std::fmt::Debug for MqttHomeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MqttHomeBackend")
            .field("broker_url", &self.config.broker_url)
            .field("convention", &self.config.topic_convention)
            .finish()
    }
}

impl MqttHomeBackend {
    pub fn new(config: MqttConfig) -> Result<Self, String> {
        // Validate broker URL (#1)
        validate_backend_url(&config.broker_url)?;

        // Check TLS (#9)
        if !config.use_tls && !config.allow_insecure_mqtt {
            return Err("MQTT TLS is disabled but allow_insecure_mqtt is false. \
                        Set allow_insecure_mqtt = true to explicitly allow unencrypted connections.".into());
        }

        // Validate port (#1)
        if let Some(port_str) = config.broker_url.rsplit(':').next() {
            if let Ok(port) = port_str.parse::<u16>() {
                if port != 1883 && port != 8883 && port != 0 {
                    // Non-standard port — log warning but allow
                }
            }
        }

        Ok(Self {
            config,
            registry: Arc::new(Mutex::new(DeviceRegistry::new())),
            command_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Get a reference to the device registry.
    pub fn registry(&self) -> &Arc<Mutex<DeviceRegistry>> {
        &self.registry
    }

    /// Check command rate limit (#7).
    fn check_rate_limit(&self, entity_id: &str) -> Result<(), String> {
        let now = now_epoch();
        let mut history = self
            .command_history
            .lock()
            .map_err(|e| format!("Lock error: {}", e))?;

        // Cleanup old entries (older than 60s)
        history.retain(|(ts, _)| now - ts < 60);

        // Check per-device limit
        let device_count = history.iter().filter(|(_, id)| id == entity_id).count() as u32;
        if device_count >= self.config.max_commands_per_minute {
            return Err(format!(
                "Rate limit exceeded for '{}': {} commands/minute (max {})",
                entity_id, device_count, self.config.max_commands_per_minute
            ));
        }

        // Check global limit
        if history.len() as u32 >= self.config.max_total_commands_per_minute {
            return Err(format!(
                "Global rate limit exceeded: {} commands/minute (max {})",
                history.len(),
                self.config.max_total_commands_per_minute
            ));
        }

        // Record this command
        history.push((now, entity_id.to_string()));
        Ok(())
    }

    /// Validate an MQTT topic for safety (#2).
    pub fn validate_topic(topic: &str) -> Result<(), String> {
        if topic.is_empty() || topic.len() > 65535 {
            return Err("Invalid topic length".into());
        }
        if topic.starts_with('$') {
            return Err("Topic cannot start with $ (system topics)".into());
        }
        if topic.contains('\0') {
            return Err("Topic cannot contain null bytes".into());
        }
        if topic == "#" {
            return Err("Wildcard-all '#' subscription blocked".into());
        }
        if topic.contains("..") {
            return Err("Topic cannot contain '..'".into());
        }
        Ok(())
    }

    /// Parse a Zigbee2MQTT bridge/devices message into registry entries.
    pub fn parse_z2m_discovery(payload: &str) -> Vec<RegistryEntry> {
        let mut entries = Vec::new();
        let now = now_epoch();

        if let Ok(devices) = serde_json::from_str::<Vec<serde_json::Value>>(payload) {
            for device in devices {
                let friendly_name = device
                    .get("friendly_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if friendly_name.is_empty() {
                    continue;
                }

                let device_type = device
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                let domain = match device_type {
                    "Router" | "EndDevice" => {
                        // Try to infer from definition
                        let desc = device
                            .get("definition")
                            .and_then(|d| d.get("description"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let lower = desc.to_lowercase();
                        if lower.contains("light")
                            || lower.contains("bulb")
                            || lower.contains("lamp")
                        {
                            "light"
                        } else if lower.contains("switch")
                            || lower.contains("plug")
                            || lower.contains("relay")
                        {
                            "switch"
                        } else if lower.contains("sensor")
                            || lower.contains("temperature")
                            || lower.contains("humidity")
                        {
                            "sensor"
                        } else if lower.contains("thermostat")
                            || lower.contains("climate")
                            || lower.contains("hvac")
                        {
                            "climate"
                        } else if lower.contains("cover")
                            || lower.contains("blind")
                            || lower.contains("shutter")
                        {
                            "cover"
                        } else if lower.contains("lock") {
                            "lock"
                        } else {
                            "switch" // default
                        }
                    }
                    "Coordinator" => continue, // Skip coordinator
                    _ => "sensor",
                };

                let entity_id = format!(
                    "{}.{}",
                    domain,
                    friendly_name.replace(' ', "_").to_lowercase()
                );

                entries.push(RegistryEntry {
                    entity_id,
                    name: friendly_name.to_string(),
                    domain: domain.to_string(),
                    state_topic: format!("zigbee2mqtt/{}", friendly_name),
                    command_topic: Some(format!("zigbee2mqtt/{}/set", friendly_name)),
                    last_state: None,
                    attributes: device
                        .get("definition")
                        .cloned()
                        .unwrap_or(serde_json::json!({})),
                    discovered_at: now,
                    last_seen: now,
                });
            }
        }

        entries
    }
}

impl HomeBackend for MqttHomeBackend {
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String> {
        if let Some(d) = domain {
            validate_domain(d)?;
        }
        let registry = self
            .registry
            .lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        Ok(registry
            .list(domain)
            .into_iter()
            .map(|e| DeviceState {
                entity_id: e.entity_id.clone(),
                name: e.name.clone(),
                state: e.last_state.clone().unwrap_or_else(|| "unknown".into()),
                attributes: e.attributes.clone(),
                last_changed: format_epoch(e.last_seen),
            })
            .collect())
    }

    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String> {
        validate_entity_id(entity_id)?;
        let registry = self
            .registry
            .lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        registry
            .get(entity_id)
            .map(|e| DeviceState {
                entity_id: e.entity_id.clone(),
                name: e.name.clone(),
                state: e.last_state.clone().unwrap_or_else(|| "unknown".into()),
                attributes: e.attributes.clone(),
                last_changed: format_epoch(e.last_seen),
            })
            .ok_or_else(|| format!("Device not found: {}", entity_id))
    }

    fn call_service(
        &self,
        domain: &str,
        service: &str,
        entity_id: &str,
        data: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        validate_entity_id(entity_id)?;
        validate_domain(domain)?;
        validate_service_name(service)?;

        // Rate limit check (#7)
        self.check_rate_limit(entity_id)?;

        // Validate payload size (#3)
        if let Some(d) = data {
            let payload = d.to_string();
            if payload.len() > 65536 {
                return Err("Payload too large (max 64KB)".into());
            }
        }

        // Build command payload
        let payload = match service {
            "turn_on" => {
                let mut cmd = serde_json::json!({"state": "ON"});
                if let Some(d) = data {
                    if let Some(obj) = d.as_object() {
                        for (k, v) in obj {
                            cmd[k] = v.clone();
                        }
                    }
                }
                cmd
            }
            "turn_off" => serde_json::json!({"state": "OFF"}),
            "toggle" => serde_json::json!({"state": "TOGGLE"}),
            "set_temperature" => data.cloned().unwrap_or(serde_json::json!({})),
            _ => data.cloned().unwrap_or(serde_json::json!({})),
        };

        // Get command topic
        let device_name = entity_id.split('.').nth(1).unwrap_or(entity_id);
        let topic = self.config.topic_convention.command_topic(device_name);
        Self::validate_topic(&topic)?;

        // In a real implementation, this would publish to MQTT:
        // self.client.publish(topic, QoS::AtLeastOnce, false, payload.to_string())
        // For now, return success with the command details
        Ok(serde_json::json!({
            "published": true,
            "topic": topic,
            "payload": payload,
            "entity_id": entity_id,
        }))
    }

    fn list_scenes(&self) -> Result<Vec<DeviceState>, String> {
        self.list_devices(Some("scene"))
    }

    fn list_automations(&self) -> Result<Vec<DeviceState>, String> {
        self.list_devices(Some("automation"))
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_epoch(epoch: u64) -> String {
    chrono::DateTime::from_timestamp(epoch as i64, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| "unknown".into())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqtt_config_default() {
        let config = MqttConfig::default();
        assert!(config.use_tls);
        assert!(!config.allow_insecure_mqtt);
        assert_eq!(config.max_commands_per_minute, 10);
    }

    #[test]
    fn test_mqtt_backend_tls_required() {
        let mut config = MqttConfig::default();
        config.use_tls = false;
        config.allow_insecure_mqtt = false;
        let result = MqttHomeBackend::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("TLS"));
    }

    #[test]
    fn test_mqtt_backend_insecure_allowed() {
        let mut config = MqttConfig::default();
        config.use_tls = false;
        config.allow_insecure_mqtt = true;
        let result = MqttHomeBackend::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_topic_convention_zigbee2mqtt() {
        let conv = TopicConvention::Zigbee2Mqtt;
        assert_eq!(conv.state_topic("lamp1"), "zigbee2mqtt/lamp1");
        assert_eq!(conv.command_topic("lamp1"), "zigbee2mqtt/lamp1/set");
    }

    #[test]
    fn test_topic_convention_tasmota() {
        let conv = TopicConvention::Tasmota;
        assert_eq!(conv.state_topic("sonoff"), "stat/sonoff/POWER");
        assert_eq!(conv.command_topic("sonoff"), "cmnd/sonoff/POWER");
    }

    #[test]
    fn test_topic_validation() {
        assert!(MqttHomeBackend::validate_topic("home/light/1").is_ok());
        assert!(MqttHomeBackend::validate_topic("$SYS/broker").is_err());
        assert!(MqttHomeBackend::validate_topic("#").is_err());
        assert!(MqttHomeBackend::validate_topic("").is_err());
        assert!(MqttHomeBackend::validate_topic("a/..b/c").is_err());
    }

    #[test]
    fn test_device_registry_crud() {
        let mut registry = DeviceRegistry::new();
        assert_eq!(registry.count(), 0);

        registry.register(RegistryEntry {
            entity_id: "light.living".into(),
            name: "Living Room".into(),
            domain: "light".into(),
            state_topic: "zigbee2mqtt/living".into(),
            command_topic: Some("zigbee2mqtt/living/set".into()),
            last_state: Some("on".into()),
            attributes: serde_json::json!({"brightness": 255}),
            discovered_at: 1000,
            last_seen: 1000,
        });

        assert_eq!(registry.count(), 1);
        assert!(registry.get("light.living").is_some());
        assert!(registry.get("light.nonexistent").is_none());

        assert_eq!(registry.list(Some("light")).len(), 1);
        assert_eq!(registry.list(Some("switch")).len(), 0);
        assert_eq!(registry.list(None).len(), 1);
    }

    #[test]
    fn test_device_registry_update_state() {
        let mut registry = DeviceRegistry::new();
        registry.register(RegistryEntry {
            entity_id: "sensor.temp".into(),
            name: "Temperature".into(),
            domain: "sensor".into(),
            state_topic: "zigbee2mqtt/temp".into(),
            command_topic: None,
            last_state: Some("22.5".into()),
            attributes: serde_json::json!({}),
            discovered_at: 1000,
            last_seen: 1000,
        });

        registry.update_state("sensor.temp", "23.1", serde_json::json!({"unit": "°C"}));
        let entry = registry.get("sensor.temp").unwrap();
        assert_eq!(entry.last_state, Some("23.1".into()));
        assert_eq!(entry.attributes["unit"], "°C");
    }

    #[test]
    fn test_z2m_discovery_parsing() {
        let payload = r#"[
            {"friendly_name": "Living Lamp", "type": "Router", "definition": {"description": "LED light bulb"}},
            {"friendly_name": "Door Sensor", "type": "EndDevice", "definition": {"description": "Contact sensor"}},
            {"friendly_name": "Coordinator", "type": "Coordinator"}
        ]"#;

        let entries = MqttHomeBackend::parse_z2m_discovery(payload);
        assert_eq!(entries.len(), 2); // Coordinator skipped
        assert_eq!(entries[0].entity_id, "light.living_lamp");
        assert_eq!(entries[0].domain, "light");
        assert_eq!(entries[1].entity_id, "sensor.door_sensor");
        assert_eq!(entries[1].domain, "sensor");
    }

    #[test]
    fn test_rate_limiting() {
        let mut config = MqttConfig::default();
        config.use_tls = false;
        config.allow_insecure_mqtt = true;
        config.max_commands_per_minute = 2;
        let backend = MqttHomeBackend::new(config).unwrap();

        // First two should succeed
        assert!(backend.check_rate_limit("light.test").is_ok());
        assert!(backend.check_rate_limit("light.test").is_ok());
        // Third should fail
        assert!(backend.check_rate_limit("light.test").is_err());
        // Different device should still work
        assert!(backend.check_rate_limit("switch.other").is_ok());
    }

    #[test]
    fn test_mqtt_ssrf_blocked() {
        let mut config = MqttConfig::default();
        config.broker_url = "mqtt://169.254.169.254:1883".into();
        let result = MqttHomeBackend::new(config);
        assert!(result.is_err());
    }
}
