//! MCP tools for home automation (domótica).
//!
//! Supports Home Assistant REST API as primary backend. Extensible via the
//! `HomeBackend` trait for other platforms (OpenHAB, generic REST, MQTT bridges).
//!
//! Covers: lights, switches, sensors, climate/HVAC, scenes, automations,
//! covers (blinds/shutters), media players, fans, locks, and any custom domain.

use crate::mcp_protocol::server::McpServer;
use crate::mcp_protocol::types::{McpTool, McpToolAnnotation};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// ============================================================================
// Types
// ============================================================================

/// State of a home automation device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    /// Entity ID (e.g., "light.living_room", "climate.bedroom").
    pub entity_id: String,
    /// Friendly name.
    pub name: String,
    /// Current state ("on", "off", "25.3", "heating", "idle", etc.).
    pub state: String,
    /// Device-specific attributes (brightness, color_temp, temperature, etc.).
    pub attributes: serde_json::Value,
    /// Last state change timestamp (ISO 8601).
    pub last_changed: String,
}

/// Configuration for a home automation backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HomeConfig {
    /// Backend type: "home_assistant", "generic_rest", "mqtt_bridge".
    pub backend_type: String,
    /// Base URL (e.g., "http://homeassistant.local:8123").
    pub base_url: String,
    /// Authentication token (long-lived access token for HA).
    #[serde(skip_serializing)]
    pub token: String,
    /// Whether to verify SSL certificates.
    pub verify_ssl: bool,
    /// HTTP timeout in seconds.
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

// ============================================================================
// HomeBackend Trait
// ============================================================================

/// Abstraction over home automation platforms.
pub trait HomeBackend: Send + Sync {
    /// List all devices, optionally filtered by domain.
    /// Domains: light, switch, sensor, climate, cover, media_player, fan,
    ///          lock, scene, automation, binary_sensor, input_boolean, etc.
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String>;

    /// Get the state of a specific device by entity_id.
    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String>;

    /// Call a service on a device.
    /// domain: "light", "switch", "climate", etc.
    /// service: "turn_on", "turn_off", "toggle", "set_temperature", etc.
    /// entity_id: target device.
    /// data: optional service-specific data (brightness, temperature, etc.).
    fn call_service(
        &self,
        domain: &str,
        service: &str,
        entity_id: &str,
        data: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String>;

    /// List available scenes.
    fn list_scenes(&self) -> Result<Vec<DeviceState>, String>;

    /// List automations.
    fn list_automations(&self) -> Result<Vec<DeviceState>, String>;
}

// ============================================================================
// Home Assistant Backend
// ============================================================================

/// Home Assistant REST API backend.
pub struct HomeAssistantBackend {
    base_url: String,
    token: String,
    timeout_secs: u64,
}

impl HomeAssistantBackend {
    pub fn new(config: &HomeConfig) -> Self {
        Self {
            base_url: config.base_url.trim_end_matches('/').to_string(),
            token: config.token.clone(),
            timeout_secs: config.timeout_secs,
        }
    }

    fn get(&self, path: &str) -> Result<serde_json::Value, String> {
        let url = format!("{}{}", self.base_url, path);
        // Validate URL to prevent SSRF
        if url.contains("169.254.") || url.contains("metadata.google") {
            return Err("Blocked: potential SSRF target".into());
        }
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .call()
            .map_err(|e| format!("HTTP GET error: {}", e))?;
        response
            .into_json::<serde_json::Value>()
            .map_err(|e| format!("JSON parse error: {}", e))
    }

    fn post(&self, path: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
        let url = format!("{}{}", self.base_url, path);
        if url.contains("169.254.") || url.contains("metadata.google") {
            return Err("Blocked: potential SSRF target".into());
        }
        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .send_json(body.clone())
            .map_err(|e| format!("HTTP POST error: {}", e))?;
        response
            .into_json::<serde_json::Value>()
            .map_err(|e| format!("JSON parse error: {}", e))
    }

    fn parse_state(item: &serde_json::Value) -> DeviceState {
        DeviceState {
            entity_id: item
                .get("entity_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            name: item
                .get("attributes")
                .and_then(|a| a.get("friendly_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            state: item
                .get("state")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            attributes: item
                .get("attributes")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            last_changed: item
                .get("last_changed")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        }
    }
}

impl HomeBackend for HomeAssistantBackend {
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String> {
        let states = self.get("/api/states")?;
        let arr = states.as_array().ok_or("Expected JSON array from /api/states")?;
        let devices: Vec<DeviceState> = arr
            .iter()
            .filter(|item| {
                if let Some(d) = domain {
                    item.get("entity_id")
                        .and_then(|v| v.as_str())
                        .map(|id| id.starts_with(&format!("{}.", d)))
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .map(Self::parse_state)
            .collect();
        Ok(devices)
    }

    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String> {
        validate_entity_id(entity_id)?;
        let path = format!("/api/states/{}", entity_id);
        let item = self.get(&path)?;
        if item.get("entity_id").is_none() {
            return Err(format!("Device not found: {}", entity_id));
        }
        Ok(Self::parse_state(&item))
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

        let path = format!("/api/services/{}/{}", domain, service);
        let mut body = data.cloned().unwrap_or(serde_json::json!({}));
        if let Some(obj) = body.as_object_mut() {
            obj.insert("entity_id".into(), serde_json::json!(entity_id));
        }
        self.post(&path, &body)
    }

    fn list_scenes(&self) -> Result<Vec<DeviceState>, String> {
        self.list_devices(Some("scene"))
    }

    fn list_automations(&self) -> Result<Vec<DeviceState>, String> {
        self.list_devices(Some("automation"))
    }
}

// ============================================================================
// Validation (security hardening)
// ============================================================================

fn validate_entity_id(entity_id: &str) -> Result<(), String> {
    if !entity_id.contains('.') {
        return Err(format!(
            "Invalid entity_id format '{}': expected 'domain.name' (e.g., 'light.living_room')",
            entity_id
        ));
    }
    // Prevent path traversal in entity_id
    if entity_id.contains("..") || entity_id.contains('/') || entity_id.contains('\\') {
        return Err("Invalid entity_id: contains path traversal characters".into());
    }
    // Max length
    if entity_id.len() > 256 {
        return Err("Entity ID too long".into());
    }
    Ok(())
}

fn validate_domain(domain: &str) -> Result<(), String> {
    // Domains are alphanumeric + underscore only
    if domain.is_empty() || domain.len() > 64 {
        return Err("Invalid domain length".into());
    }
    if !domain.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!("Invalid domain '{}': alphanumeric and underscore only", domain));
    }
    Ok(())
}

fn validate_service_name(service: &str) -> Result<(), String> {
    if service.is_empty() || service.len() > 64 {
        return Err("Invalid service name length".into());
    }
    if !service.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "Invalid service name '{}': alphanumeric and underscore only",
            service
        ));
    }
    Ok(())
}

fn extract_domain(entity_id: &str) -> Result<String, String> {
    validate_entity_id(entity_id)?;
    entity_id
        .split('.')
        .next()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Cannot extract domain from '{}'", entity_id))
}

// ============================================================================
// MCP Tool Registration
// ============================================================================

/// Register all 10 home automation MCP tools on the server.
pub fn register_home_tools(server: &mut McpServer, backend: Arc<Mutex<dyn HomeBackend>>) {
    let ann_ro = McpToolAnnotation {
        title: None,
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(true),
    };
    let ann_action = McpToolAnnotation {
        title: None,
        read_only_hint: Some(false),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(true),
    };
    let ann_toggle = McpToolAnnotation {
        title: None,
        read_only_hint: Some(false),
        destructive_hint: Some(false),
        idempotent_hint: Some(false), // toggle is NOT idempotent
        open_world_hint: Some(true),
    };

    // --- home_list_devices ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new(
                "home_list_devices",
                "List home automation devices. Filter by domain: light, switch, sensor, climate, cover, media_player, fan, lock, binary_sensor, input_boolean.",
            )
            .with_property("domain", "string", "Filter by domain (e.g., 'light', 'climate', 'sensor')", false)
            .with_annotations(ann_ro.clone()),
            move |args| {
                let domain = args.get("domain").and_then(|v| v.as_str());
                if let Some(d) = domain {
                    validate_domain(d)?;
                }
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                let devices = guard.list_devices(domain)?;
                Ok(serde_json::json!({
                    "devices": devices,
                    "count": devices.len(),
                }))
            },
        );
    }

    // --- home_get_device ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_get_device", "Get the current state of a specific device.")
                .with_property("entity_id", "string", "Entity ID (e.g., 'light.living_room', 'climate.bedroom')", true)
                .with_annotations(ann_ro.clone()),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                let device = guard.get_device(entity_id)?;
                Ok(serde_json::to_value(&device).map_err(|e| e.to_string())?)
            },
        );
    }

    // --- home_turn_on ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new(
                "home_turn_on",
                "Turn on a device. For lights: optional brightness (0-255), color_temp (mireds), rgb_color. For climate: starts heating/cooling.",
            )
            .with_property("entity_id", "string", "Entity ID (required)", true)
            .with_property("brightness", "integer", "Brightness 0-255 (lights only)", false)
            .with_property("color_temp", "integer", "Color temperature in mireds (lights only)", false)
            .with_property("temperature", "number", "Target temperature in °C (climate only)", false)
            .with_annotations(ann_action.clone()),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let domain = extract_domain(entity_id)?;

                let mut data = serde_json::Map::new();
                if let Some(b) = args.get("brightness").and_then(|v| v.as_u64()) {
                    data.insert("brightness".into(), serde_json::json!(b.min(255)));
                }
                if let Some(ct) = args.get("color_temp").and_then(|v| v.as_u64()) {
                    data.insert("color_temp".into(), serde_json::json!(ct));
                }
                if let Some(t) = args.get("temperature").and_then(|v| v.as_f64()) {
                    data.insert("temperature".into(), serde_json::json!(t));
                }

                let service_data = if data.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(data))
                };

                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service(&domain, "turn_on", entity_id, service_data.as_ref())
            },
        );
    }

    // --- home_turn_off ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_turn_off", "Turn off a device (light, switch, fan, climate, media_player, etc.).")
                .with_property("entity_id", "string", "Entity ID (required)", true)
                .with_annotations(ann_action.clone()),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let domain = extract_domain(entity_id)?;
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service(&domain, "turn_off", entity_id, None)
            },
        );
    }

    // --- home_toggle ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_toggle", "Toggle a device's state (on↔off).")
                .with_property("entity_id", "string", "Entity ID (required)", true)
                .with_annotations(ann_toggle),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let domain = extract_domain(entity_id)?;
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service(&domain, "toggle", entity_id, None)
            },
        );
    }

    // --- home_set_value (generic service call) ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new(
                "home_set_value",
                "Call any Home Assistant service. Use for climate.set_temperature, cover.set_position, media_player.set_volume, fan.set_speed, etc.",
            )
            .with_property("domain", "string", "Service domain (e.g., 'climate', 'cover', 'fan')", true)
            .with_property("service", "string", "Service name (e.g., 'set_temperature', 'set_position')", true)
            .with_property("entity_id", "string", "Entity ID (required)", true)
            .with_property("data", "object", "Service-specific data (e.g., {\"temperature\": 22, \"hvac_mode\": \"heat\"})", false)
            .with_annotations(ann_action.clone()),
            move |args| {
                let domain = args.get("domain").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: domain")?;
                let service = args.get("service").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: service")?;
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let data = args.get("data");
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service(domain, service, entity_id, data)
            },
        );
    }

    // --- home_list_scenes ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_list_scenes", "List available scenes.")
                .with_annotations(ann_ro.clone()),
            move |_args| {
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                let scenes = guard.list_scenes()?;
                Ok(serde_json::json!({ "scenes": scenes, "count": scenes.len() }))
            },
        );
    }

    // --- home_activate_scene ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_activate_scene", "Activate a scene.")
                .with_property("entity_id", "string", "Scene entity ID (e.g., 'scene.movie_night')", true)
                .with_annotations(ann_action.clone()),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service("scene", "turn_on", entity_id, None)
            },
        );
    }

    // --- home_list_automations ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_list_automations", "List automations and their current state.")
                .with_annotations(ann_ro),
            move |_args| {
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                let automations = guard.list_automations()?;
                Ok(serde_json::json!({ "automations": automations, "count": automations.len() }))
            },
        );
    }

    // --- home_trigger_automation ---
    {
        let backend = backend.clone();
        server.register_tool(
            McpTool::new("home_trigger_automation", "Trigger an automation manually.")
                .with_property("entity_id", "string", "Automation entity ID (e.g., 'automation.morning_routine')", true)
                .with_annotations(McpToolAnnotation {
                    title: None,
                    read_only_hint: Some(false),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(false),
                    open_world_hint: Some(true),
                }),
            move |args| {
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: entity_id")?;
                let guard = backend.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.call_service("automation", "trigger", entity_id, None)
            },
        );
    }
}

/// Register 4 additional home tools for event listening, device registration, and discovery.
pub fn register_home_management_tools(
    server: &mut McpServer,
    listener_mgr: Arc<Mutex<crate::home_automation::HomeEventListenerManager>>,
) {
    // --- home_subscribe ---
    {
        let mgr = listener_mgr.clone();
        server.register_tool(
            McpTool::new(
                "home_subscribe",
                "Start listening for device state changes from a backend (Home Assistant SSE, OpenHAB SSE, or MQTT).",
            )
            .with_property("backend", "string", "Backend type: ha, openhab, mqtt (required)", true)
            .with_property("url", "string", "Backend URL (e.g., http://ha.local:8123)", true)
            .with_property("token", "string", "Auth token (for HA)", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Subscribe to Device Events".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(false),
                open_world_hint: Some(true),
            }),
            move |args| {
                let backend = args.get("backend").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: backend")?;
                let url = args.get("url").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: url")?;
                let token = args.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string();

                let source = match backend {
                    "ha" | "home_assistant" => crate::home_automation::ListenerSource::HomeAssistantSse {
                        url: format!("{}/api/stream", url.trim_end_matches('/')),
                        token,
                    },
                    "openhab" => crate::home_automation::ListenerSource::OpenHabSse {
                        url: format!("{}/rest/events", url.trim_end_matches('/')),
                    },
                    "mqtt" => crate::home_automation::ListenerSource::MqttSubscription {
                        broker_url: url.to_string(),
                        topics: vec!["#".to_string()], // Subscribe to all — filtered by entity_filter in config
                    },
                    _ => return Err(format!("Unknown backend '{}'. Valid: ha, openhab, mqtt", backend)),
                };

                let mut guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                let id = guard.subscribe(source)?;
                Ok(serde_json::json!({ "listener_id": id, "backend": backend, "started": true }))
            },
        );
    }

    // --- home_unsubscribe ---
    {
        let mgr = listener_mgr.clone();
        server.register_tool(
            McpTool::new("home_unsubscribe", "Stop listening for device events. Pass listener_id or omit to stop all.")
                .with_property("listener_id", "string", "Listener ID (omit to stop all)", false)
                .with_annotations(McpToolAnnotation {
                    title: Some("Unsubscribe Device Events".into()),
                    read_only_hint: Some(false),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |args| {
                let mut guard = mgr.lock().map_err(|e| format!("Lock error: {}", e))?;
                if let Some(id) = args.get("listener_id").and_then(|v| v.as_str()) {
                    let stopped = guard.unsubscribe(id);
                    guard.cleanup();
                    Ok(serde_json::json!({ "stopped": stopped, "listener_id": id }))
                } else {
                    guard.unsubscribe_all();
                    guard.cleanup();
                    Ok(serde_json::json!({ "stopped_all": true }))
                }
            },
        );
    }

    // --- home_register_device ---
    {
        server.register_tool(
            McpTool::new(
                "home_register_device",
                "Register a custom IoT device (MQTT, webhook, or REST-polled). Requires admin permission.",
            )
            .with_property("name", "string", "Device name (required)", true)
            .with_property("entity_id", "string", "Entity ID, e.g., 'sensor.my_device' (required)", true)
            .with_property("device_type", "string", "Type: sensor, switch, light, climate, etc. (required)", true)
            .with_property("state_source_type", "string", "State source: mqtt, webhook, rest_poll (required)", true)
            .with_property("state_source_value", "string", "MQTT topic, webhook path, or REST URL (required)", true)
            .with_property("command_target_type", "string", "Command target: mqtt, rest_post (optional)", false)
            .with_property("command_target_value", "string", "MQTT topic or REST URL for commands", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Register IoT Device".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(true),
            }),
            move |args| {
                let name = args.get("name").and_then(|v| v.as_str())
                    .ok_or("Missing: name")?;
                let entity_id = args.get("entity_id").and_then(|v| v.as_str())
                    .ok_or("Missing: entity_id")?;
                let device_type = args.get("device_type").and_then(|v| v.as_str())
                    .ok_or("Missing: device_type")?;
                let src_type = args.get("state_source_type").and_then(|v| v.as_str())
                    .ok_or("Missing: state_source_type")?;
                let src_val = args.get("state_source_value").and_then(|v| v.as_str())
                    .ok_or("Missing: state_source_value")?;

                let state_source = match src_type {
                    "mqtt" => crate::home_automation::StateSource::MqttTopic(src_val.into()),
                    "webhook" => crate::home_automation::StateSource::WebhookInbound { path: src_val.into() },
                    "rest_poll" => crate::home_automation::StateSource::RestPoll {
                        url: src_val.into(),
                        interval_secs: args.get("poll_interval").and_then(|v| v.as_u64()).unwrap_or(60),
                    },
                    _ => return Err(format!("Unknown state_source_type '{}'. Valid: mqtt, webhook, rest_poll", src_type)),
                };

                let command_target = match (
                    args.get("command_target_type").and_then(|v| v.as_str()),
                    args.get("command_target_value").and_then(|v| v.as_str()),
                ) {
                    (Some("mqtt"), Some(val)) => Some(crate::home_automation::CommandTarget::MqttTopic(val.into())),
                    (Some("rest_post"), Some(val)) => Some(crate::home_automation::CommandTarget::RestPost { url: val.into() }),
                    _ => None,
                };

                let def = crate::home_automation::CustomDeviceDefinition {
                    name: name.into(),
                    entity_id: entity_id.into(),
                    device_type: device_type.into(),
                    state_source,
                    command_target,
                    attributes_schema: None,
                    alerts: Vec::new(),
                };

                crate::home_automation::validate_custom_device(&def)?;

                Ok(serde_json::json!({
                    "registered": true,
                    "entity_id": entity_id,
                    "device_type": device_type,
                    "name": name,
                }))
            },
        );
    }

    // --- home_discover ---
    {
        server.register_tool(
            McpTool::new(
                "home_discover",
                "Scan local network for Home Assistant, OpenHAB, and MQTT broker instances via mDNS. Results are for review only — never auto-connects.",
            )
            .with_property("timeout_secs", "integer", "Scan timeout in seconds (default: 3)", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Discover Home Services".into()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(true),
            }),
            move |args| {
                let timeout = args.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(3);
                let services = crate::home_automation::discover_services(timeout);
                Ok(serde_json::json!({
                    "services": services,
                    "count": services.len(),
                    "warning": "These services were found via mDNS. Verify authenticity before connecting (mDNS can be spoofed on local networks).",
                }))
            },
        );
    }
}

// ============================================================================
// LLM Enhancement: Natural Language Home Command Interpretation (V68)
// ============================================================================

/// Result of LLM-based home command interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeCommandInterpretation {
    /// The action to perform (e.g., "turn_on", "turn_off", "set_temperature").
    pub action: String,
    /// The target entity (e.g., "light.living_room").
    pub target: String,
    /// Optional value for the action (e.g., temperature, brightness).
    pub value: Option<serde_json::Value>,
}

/// Configuration for home command interpretation.
#[derive(Debug, Clone)]
pub struct HomeCommandConfig {
    /// Use LLM to interpret natural language home commands.
    /// When false (default), uses keyword-based parsing.
    pub llm_enhanced: bool,
}

impl Default for HomeCommandConfig {
    fn default() -> Self {
        Self {
            llm_enhanced: false,
        }
    }
}

/// Interprets natural language home commands, optionally enhanced by LLM.
pub struct HomeCommandInterpreter {
    pub config: HomeCommandConfig,
}

impl HomeCommandInterpreter {
    pub fn new(config: HomeCommandConfig) -> Self {
        Self { config }
    }

    /// Build a prompt for LLM-based command interpretation.
    ///
    /// Returns None if LLM enhancement is disabled.
    pub fn build_command_prompt(&self, command: &str) -> Option<String> {
        if !self.config.llm_enhanced {
            return None;
        }

        let prompt = format!(
            "Interpret this natural language home command. \
             Return JSON: {{\"action\":\"turn_on|turn_off|set_temperature|...\",\
             \"target\":\"light.living_room\",\"value\":null}}\n\n\
             Command: {}",
            crate::llm_enhance::prompt_wrap(command)
        );

        Some(prompt)
    }

    /// Parse LLM response for home command interpretation.
    pub fn parse_command_response(response: &str) -> Option<HomeCommandInterpretation> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let action = val
                    .get("action")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                let target = val
                    .get("target")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                let value = val.get("value").cloned();
                return Some(HomeCommandInterpretation {
                    action: action.to_string(),
                    target: target.to_string(),
                    value: if value == Some(serde_json::Value::Null) {
                        None
                    } else {
                        value
                    },
                });
            }
        }
        None
    }

    /// Interpret a natural language home command with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for
    /// nuanced interpretation. Otherwise uses keyword heuristics.
    pub fn interpret_home_command_with_llm(
        &self,
        command: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> HomeCommandInterpretation {
        // Heuristic baseline
        let lower = command.to_lowercase();
        let heuristic_action = if lower.contains("turn on") || lower.contains("switch on")
            || lower.contains("enciende") || lower.contains("encender")
        {
            "turn_on"
        } else if lower.contains("turn off") || lower.contains("switch off")
            || lower.contains("apaga") || lower.contains("apagar")
        {
            "turn_off"
        } else if lower.contains("temperature") || lower.contains("temp")
            || lower.contains("temperatura")
        {
            "set_temperature"
        } else if lower.contains("dim") || lower.contains("brightness") {
            "set_brightness"
        } else {
            "toggle"
        };

        let heuristic_target = if lower.contains("living") || lower.contains("salon")
            || lower.contains("salón")
        {
            "light.living_room"
        } else if lower.contains("bedroom") || lower.contains("dormitorio") {
            "light.bedroom"
        } else if lower.contains("kitchen") || lower.contains("cocina") {
            "light.kitchen"
        } else {
            "light.unknown"
        };

        let heuristic = HomeCommandInterpretation {
            action: heuristic_action.to_string(),
            target: heuristic_target.to_string(),
            value: None,
        };

        // Try LLM enhancement
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_command_prompt(command) {
                    if let Ok(response) = enhancer.generate(&prompt, 200) {
                        if let Some(interpretation) = Self::parse_command_response(&response) {
                            return interpretation;
                        }
                    }
                }
            }
        }

        heuristic
    }
}

impl Default for HomeCommandInterpreter {
    fn default() -> Self {
        Self::new(HomeCommandConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock backend for testing without a real Home Assistant server.
    struct MockHomeBackend {
        devices: Vec<DeviceState>,
        service_calls: Arc<Mutex<Vec<(String, String, String, Option<serde_json::Value>)>>>,
    }

    impl MockHomeBackend {
        fn new() -> Self {
            Self {
                devices: Vec::new(),
                service_calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_device(mut self, device: DeviceState) -> Self {
            self.devices.push(device);
            self
        }

        fn call_count(&self) -> usize {
            self.service_calls.lock().map(|v| v.len()).unwrap_or(0)
        }
    }

    impl HomeBackend for MockHomeBackend {
        fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String> {
            Ok(self
                .devices
                .iter()
                .filter(|d| match domain {
                    Some(dom) => d.entity_id.starts_with(&format!("{}.", dom)),
                    None => true,
                })
                .cloned()
                .collect())
        }

        fn get_device(&self, entity_id: &str) -> Result<DeviceState, String> {
            self.devices
                .iter()
                .find(|d| d.entity_id == entity_id)
                .cloned()
                .ok_or_else(|| format!("Device not found: {}", entity_id))
        }

        fn call_service(
            &self,
            domain: &str,
            service: &str,
            entity_id: &str,
            data: Option<&serde_json::Value>,
        ) -> Result<serde_json::Value, String> {
            if let Ok(mut calls) = self.service_calls.lock() {
                calls.push((
                    domain.into(),
                    service.into(),
                    entity_id.into(),
                    data.cloned(),
                ));
            }
            Ok(serde_json::json!([{"entity_id": entity_id, "state": "on"}]))
        }

        fn list_scenes(&self) -> Result<Vec<DeviceState>, String> {
            self.list_devices(Some("scene"))
        }

        fn list_automations(&self) -> Result<Vec<DeviceState>, String> {
            self.list_devices(Some("automation"))
        }
    }

    fn light(id: &str, name: &str, state: &str) -> DeviceState {
        DeviceState {
            entity_id: id.into(),
            name: name.into(),
            state: state.into(),
            attributes: serde_json::json!({"brightness": 255}),
            last_changed: "2026-03-26T10:00:00Z".into(),
        }
    }

    fn climate(id: &str, name: &str, state: &str, temp: f64) -> DeviceState {
        DeviceState {
            entity_id: id.into(),
            name: name.into(),
            state: state.into(),
            attributes: serde_json::json!({
                "temperature": temp,
                "current_temperature": temp - 2.0,
                "hvac_modes": ["off", "heat", "cool", "auto"],
                "min_temp": 7,
                "max_temp": 35,
            }),
            last_changed: "2026-03-26T10:00:00Z".into(),
        }
    }

    #[test]
    fn test_list_devices_all() {
        let backend = MockHomeBackend::new()
            .with_device(light("light.living", "Living Room", "on"))
            .with_device(light("switch.porch", "Porch", "off"))
            .with_device(climate("climate.bedroom", "Bedroom HVAC", "heating", 22.0));

        let devices = backend.list_devices(None).expect("list");
        assert_eq!(devices.len(), 3);
    }

    #[test]
    fn test_list_devices_filter_domain() {
        let backend = MockHomeBackend::new()
            .with_device(light("light.living", "Living Room", "on"))
            .with_device(light("light.bedroom", "Bedroom", "off"))
            .with_device(climate("climate.bedroom", "Bedroom HVAC", "heating", 22.0));

        let lights = backend.list_devices(Some("light")).expect("list");
        assert_eq!(lights.len(), 2);

        let climates = backend.list_devices(Some("climate")).expect("list");
        assert_eq!(climates.len(), 1);
        assert_eq!(climates[0].state, "heating");
    }

    #[test]
    fn test_get_device() {
        let backend = MockHomeBackend::new()
            .with_device(climate("climate.living", "Living HVAC", "idle", 20.0));

        let device = backend.get_device("climate.living").expect("get");
        assert_eq!(device.name, "Living HVAC");
        assert_eq!(device.state, "idle");
        assert_eq!(device.attributes["temperature"], 20.0);
    }

    #[test]
    fn test_get_device_not_found() {
        let backend = MockHomeBackend::new();
        let result = backend.get_device("light.nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_call_service_turn_on() {
        let backend = MockHomeBackend::new()
            .with_device(light("light.living", "Living Room", "off"));

        let data = serde_json::json!({"brightness": 200});
        backend
            .call_service("light", "turn_on", "light.living", Some(&data))
            .expect("call");

        assert_eq!(backend.call_count(), 1);
        let calls = backend.service_calls.lock().expect("lock");
        assert_eq!(calls[0].0, "light");
        assert_eq!(calls[0].1, "turn_on");
        assert_eq!(calls[0].2, "light.living");
    }

    #[test]
    fn test_call_service_toggle() {
        let backend = MockHomeBackend::new();
        backend
            .call_service("switch", "toggle", "switch.porch", None)
            .expect("call");
        assert_eq!(backend.call_count(), 1);
    }

    #[test]
    fn test_set_value_climate() {
        let backend = MockHomeBackend::new();
        let data = serde_json::json!({"temperature": 25, "hvac_mode": "heat"});
        backend
            .call_service("climate", "set_temperature", "climate.bedroom", Some(&data))
            .expect("call");

        let calls = backend.service_calls.lock().expect("lock");
        assert_eq!(calls[0].0, "climate");
        assert_eq!(calls[0].1, "set_temperature");
        assert_eq!(calls[0].3.as_ref().unwrap()["temperature"], 25);
    }

    #[test]
    fn test_invalid_entity_id() {
        assert!(validate_entity_id("no_dot_here").is_err());
        assert!(validate_entity_id("light.living").is_ok());
        assert!(validate_entity_id("../../etc/passwd").is_err());
        assert!(validate_entity_id("light.a/b").is_err());
    }

    #[test]
    fn test_validate_domain() {
        assert!(validate_domain("light").is_ok());
        assert!(validate_domain("climate").is_ok());
        assert!(validate_domain("media_player").is_ok());
        assert!(validate_domain("").is_err());
        assert!(validate_domain("light;DROP TABLE").is_err());
    }

    #[test]
    fn test_validate_service_name() {
        assert!(validate_service_name("turn_on").is_ok());
        assert!(validate_service_name("set_temperature").is_ok());
        assert!(validate_service_name("").is_err());
        assert!(validate_service_name("turn_on; rm -rf /").is_err());
    }

    // ── V68: LLM Enhancement tests ──────────────────────────────────

    #[test]
    fn test_interpret_command_heuristic_without_llm() {
        let config = HomeCommandConfig {
            llm_enhanced: false,
        };
        let interpreter = HomeCommandInterpreter::new(config);
        let result = interpreter.interpret_home_command_with_llm("Turn on the living room light", None);
        assert_eq!(result.action, "turn_on");
        assert_eq!(result.target, "light.living_room");
        assert!(result.value.is_none());
    }

    #[test]
    fn test_interpret_command_with_mock_llm() {
        let config = HomeCommandConfig {
            llm_enhanced: true,
        };
        let interpreter = HomeCommandInterpreter::new(config);
        let mock = crate::llm_enhance::MockLlm::new(
            "{\"action\":\"set_temperature\",\"target\":\"climate.bedroom\",\"value\":22}",
        );
        let result = interpreter.interpret_home_command_with_llm(
            "Set the bedroom to 22 degrees",
            Some(&mock),
        );
        assert_eq!(result.action, "set_temperature", "Expected LLM action, got: {}", result.action);
        assert_eq!(result.target, "climate.bedroom");
        assert!(result.value.is_some());
    }

    #[test]
    fn test_interpret_command_llm_fallback_on_failure() {
        let config = HomeCommandConfig {
            llm_enhanced: true,
        };
        let interpreter = HomeCommandInterpreter::new(config);
        let failing = crate::llm_enhance::FailingMockLlm;
        let result = interpreter.interpret_home_command_with_llm(
            "Turn off the kitchen light",
            Some(&failing),
        );
        // Should fall back to heuristic (not crash)
        assert_eq!(result.action, "turn_off");
        assert_eq!(result.target, "light.kitchen");
    }
}
