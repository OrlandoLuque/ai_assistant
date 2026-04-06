//! HomeEventListener — background event listener for device state changes.
//!
//! Connects to Home Assistant SSE, OpenHAB SSE, or MQTT subscriptions
//! and forwards device state changes to the EventBus.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Configuration for the event listener.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EventListenerConfig {
    /// Whether to auto-start on initialization.
    pub auto_start: bool,
    /// Reconnect delay in milliseconds after disconnect.
    pub reconnect_delay_ms: u64,
    /// Max reconnection attempts before giving up (0 = infinite).
    pub max_reconnect_attempts: u32,
    /// Entity ID patterns to forward (empty = all).
    pub entity_filter: Vec<String>,
}

impl Default for EventListenerConfig {
    fn default() -> Self {
        Self {
            auto_start: false,
            reconnect_delay_ms: 5000,
            max_reconnect_attempts: 0,
            entity_filter: Vec::new(),
        }
    }
}

/// Handle to a running event listener.
pub struct ListenerHandle {
    /// Unique listener ID.
    pub id: String,
    /// Source being listened to.
    pub source: ListenerSource,
    /// Cancellation flag.
    cancel: Arc<AtomicBool>,
    /// Whether the listener is currently connected.
    connected: Arc<AtomicBool>,
}

impl ListenerHandle {
    /// Create a new listener handle.
    pub fn new(id: String, source: ListenerSource) -> Self {
        Self {
            id,
            source,
            cancel: Arc::new(AtomicBool::new(false)),
            connected: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancel this listener.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get the cancellation token for sharing with background tasks.
    pub fn cancel_token(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// Get the connection status flag for sharing.
    pub fn connected_flag(&self) -> Arc<AtomicBool> {
        self.connected.clone()
    }
}

/// Source type for event listener.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ListenerSource {
    /// Home Assistant SSE stream.
    HomeAssistantSse { url: String, token: String },
    /// OpenHAB SSE stream.
    OpenHabSse { url: String },
    /// MQTT topic subscription.
    MqttSubscription {
        broker_url: String,
        topics: Vec<String>,
    },
}

impl std::fmt::Display for ListenerSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HomeAssistantSse { url, .. } => write!(f, "HA SSE: {}", url),
            Self::OpenHabSse { url } => write!(f, "OpenHAB SSE: {}", url),
            Self::MqttSubscription { broker_url, topics } => {
                write!(f, "MQTT: {} ({} topics)", broker_url, topics.len())
            }
        }
    }
}

/// Manages active event listeners.
pub struct HomeEventListenerManager {
    listeners: Vec<ListenerHandle>,
    config: EventListenerConfig,
}

impl HomeEventListenerManager {
    pub fn new(config: EventListenerConfig) -> Self {
        Self {
            listeners: Vec::new(),
            config,
        }
    }

    /// Start a new listener. Returns the listener ID.
    pub fn subscribe(&mut self, source: ListenerSource) -> Result<String, String> {
        // Validate source
        match &source {
            ListenerSource::HomeAssistantSse { url, .. } => {
                super::backend::validate_backend_url(url)?;
            }
            ListenerSource::OpenHabSse { url } => {
                super::backend::validate_backend_url(url)?;
            }
            ListenerSource::MqttSubscription { broker_url, topics } => {
                super::backend::validate_backend_url(broker_url)?;
                for topic in topics {
                    super::mqtt_backend::MqttHomeBackend::validate_topic(topic)?;
                }
            }
        }

        let id = format!("listener-{}", now_epoch());
        let handle = ListenerHandle::new(id.clone(), source);

        // In a full implementation, this would spawn a tokio task:
        // tokio::spawn(async move { listen_loop(handle.cancel_token(), ...).await });
        // For now, the handle is stored for status tracking.

        self.listeners.push(handle);
        Ok(id)
    }

    /// Stop a listener by ID.
    pub fn unsubscribe(&mut self, listener_id: &str) -> bool {
        if let Some(handle) = self.listeners.iter().find(|h| h.id == listener_id) {
            handle.cancel();
            true
        } else {
            false
        }
    }

    /// Stop all listeners.
    pub fn unsubscribe_all(&mut self) {
        for handle in &self.listeners {
            handle.cancel();
        }
    }

    /// List active listeners.
    pub fn active_listeners(&self) -> Vec<ListenerStatus> {
        self.listeners
            .iter()
            .filter(|h| !h.is_cancelled())
            .map(|h| ListenerStatus {
                id: h.id.clone(),
                source: h.source.to_string(),
                connected: h.is_connected(),
            })
            .collect()
    }

    /// Cleanup cancelled listeners.
    pub fn cleanup(&mut self) {
        self.listeners.retain(|h| !h.is_cancelled());
    }

    /// Get the listener config.
    pub fn config(&self) -> &EventListenerConfig {
        &self.config
    }
}

impl Default for HomeEventListenerManager {
    fn default() -> Self {
        Self::new(EventListenerConfig::default())
    }
}

/// Status of an active listener.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenerStatus {
    pub id: String,
    pub source: String,
    pub connected: bool,
}

/// Parse a Home Assistant SSE state_changed event.
pub fn parse_ha_state_changed(sse_data: &str) -> Option<DeviceStateChange> {
    let json: serde_json::Value = serde_json::from_str(sse_data).ok()?;

    if json.get("event_type")?.as_str()? != "state_changed" {
        return None;
    }

    let data = json.get("data")?;
    let entity_id = data.get("entity_id")?.as_str()?.to_string();

    let old_state = data
        .get("old_state")
        .and_then(|s| s.get("state"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let new_state = data
        .get("new_state")
        .and_then(|s| s.get("state"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let attributes = data
        .get("new_state")
        .and_then(|s| s.get("attributes"))
        .cloned()
        .unwrap_or(serde_json::json!({}));

    Some(DeviceStateChange {
        entity_id,
        old_state,
        new_state,
        attributes,
    })
}

/// Parse an OpenHAB SSE ItemStateChangedEvent.
pub fn parse_openhab_state_changed(sse_data: &str) -> Option<DeviceStateChange> {
    let json: serde_json::Value = serde_json::from_str(sse_data).ok()?;

    let event_type = json.get("type")?.as_str()?;
    if event_type != "ItemStateChangedEvent" {
        return None;
    }

    let topic = json.get("topic")?.as_str()?;
    // Topic format: "openhab/items/{item_name}/statechanged"
    let parts: Vec<&str> = topic.split('/').collect();
    let item_name = parts.get(2)?;

    let payload_str = json.get("payload")?.as_str()?;
    let payload: serde_json::Value = serde_json::from_str(payload_str).ok()?;

    let old_state = payload
        .get("oldType")
        .and_then(|v| v.as_str())
        .map(|t| {
            payload
                .get("oldValue")
                .and_then(|v| v.as_str())
                .unwrap_or(t)
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());

    let new_state = payload
        .get("type")
        .and_then(|v| v.as_str())
        .map(|t| {
            payload
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or(t)
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());

    Some(DeviceStateChange {
        entity_id: format!("sensor.{}", item_name.to_lowercase()),
        old_state,
        new_state,
        attributes: serde_json::json!({}),
    })
}

/// A device state change event (parsed from SSE/MQTT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStateChange {
    pub entity_id: String,
    pub old_state: String,
    pub new_state: String,
    pub attributes: serde_json::Value,
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listener_handle_lifecycle() {
        let handle = ListenerHandle::new(
            "test-1".into(),
            ListenerSource::HomeAssistantSse {
                url: "http://ha.local:8123/api/stream".into(),
                token: "test".into(),
            },
        );
        assert!(!handle.is_cancelled());
        assert!(!handle.is_connected());
        handle.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_manager_subscribe_unsubscribe() {
        let mut mgr = HomeEventListenerManager::default();
        let id = mgr
            .subscribe(ListenerSource::OpenHabSse {
                url: "http://openhab.local:8080/rest/events".into(),
            })
            .expect("subscribe");
        assert_eq!(mgr.active_listeners().len(), 1);

        assert!(mgr.unsubscribe(&id));
        mgr.cleanup();
        assert_eq!(mgr.active_listeners().len(), 0);
    }

    #[test]
    fn test_manager_ssrf_blocked() {
        let mut mgr = HomeEventListenerManager::default();
        let result = mgr.subscribe(ListenerSource::HomeAssistantSse {
            url: "http://169.254.169.254/api/stream".into(),
            token: "x".into(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ha_state_changed() {
        let sse = r#"{"event_type":"state_changed","data":{"entity_id":"light.living","old_state":{"state":"off"},"new_state":{"state":"on","attributes":{"brightness":255}}}}"#;
        let change = parse_ha_state_changed(sse).expect("parse");
        assert_eq!(change.entity_id, "light.living");
        assert_eq!(change.old_state, "off");
        assert_eq!(change.new_state, "on");
        assert_eq!(change.attributes["brightness"], 255);
    }

    #[test]
    fn test_parse_ha_ignores_non_state_changed() {
        let sse = r#"{"event_type":"call_service","data":{}}"#;
        assert!(parse_ha_state_changed(sse).is_none());
    }

    #[test]
    fn test_parse_openhab_state_changed() {
        let sse = r#"{"type":"ItemStateChangedEvent","topic":"openhab/items/LivingTemp/statechanged","payload":"{\"type\":\"Decimal\",\"value\":\"23.5\",\"oldType\":\"Decimal\",\"oldValue\":\"22.0\"}"}"#;
        let change = parse_openhab_state_changed(sse).expect("parse");
        assert_eq!(change.entity_id, "sensor.livingtemp");
        assert_eq!(change.old_state, "22.0");
        assert_eq!(change.new_state, "23.5");
    }

    #[test]
    fn test_listener_source_display() {
        let src = ListenerSource::MqttSubscription {
            broker_url: "mqtt://localhost:1883".into(),
            topics: vec!["home/#".into(), "sensors/#".into()],
        };
        assert!(src.to_string().contains("2 topics"));
    }
}
