//! Custom IoT device registration — user-defined devices with schemas and alerts.

use crate::event_source::EventAction;
use serde::{Deserialize, Serialize};

/// Maximum custom devices per user (#8).
pub const MAX_CUSTOM_DEVICES: usize = 50;

/// A user-defined IoT device.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CustomDeviceDefinition {
    /// Friendly name.
    pub name: String,
    /// Entity ID (e.g., "sensor.my_greenhouse_temp").
    pub entity_id: String,
    /// Device type: "sensor", "switch", "light", "climate", etc.
    pub device_type: String,
    /// Where to read state from.
    pub state_source: StateSource,
    /// Where to send commands (None for read-only sensors).
    pub command_target: Option<CommandTarget>,
    /// JSON Schema for expected attributes (informative, not enforced).
    pub attributes_schema: Option<serde_json::Value>,
    /// Threshold-based alerts.
    pub alerts: Vec<ThresholdAlert>,
}

/// Source of device state data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StateSource {
    /// Subscribe to MQTT topic for state updates.
    MqttTopic(String),
    /// Receive state via webhook POST to ai_assistant.
    WebhookInbound { path: String },
    /// Poll a REST API endpoint periodically.
    RestPoll { url: String, interval_secs: u64 },
}

/// Target for sending commands to the device.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandTarget {
    /// Publish command to MQTT topic.
    MqttTopic(String),
    /// POST command to REST endpoint.
    RestPost { url: String },
}

/// Alert that fires when a device attribute crosses a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAlert {
    /// Attribute to monitor (e.g., "temperature", "humidity", "battery").
    pub attribute: String,
    /// Condition to trigger.
    pub condition: AlertCondition,
    /// What to do when triggered.
    pub action: EventAction,
    /// Min seconds between firings.
    pub cooldown_secs: u64,
    /// Template for the alert message.
    pub message_template: String,
}

/// Condition for threshold alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AlertCondition {
    /// Trigger when value exceeds threshold.
    Above(f64),
    /// Trigger when value drops below threshold.
    Below(f64),
    /// Trigger when value equals a string.
    Equals(String),
    /// Trigger when value changes from previous.
    Changed,
}

impl AlertCondition {
    /// Check if the condition is met.
    pub fn check(&self, value: &str, previous: Option<&str>) -> bool {
        match self {
            Self::Above(threshold) => {
                value.parse::<f64>().map(|v| v > *threshold).unwrap_or(false)
            }
            Self::Below(threshold) => {
                value.parse::<f64>().map(|v| v < *threshold).unwrap_or(false)
            }
            Self::Equals(expected) => value == expected,
            Self::Changed => previous.map(|p| p != value).unwrap_or(true),
        }
    }
}

/// Validate a custom device definition for security.
pub fn validate_custom_device(def: &CustomDeviceDefinition) -> Result<(), String> {
    // Entity ID validation
    super::backend::validate_entity_id(&def.entity_id)?;

    // Device type validation
    super::backend::validate_domain(&def.device_type)?;

    // Validate state source
    match &def.state_source {
        StateSource::MqttTopic(topic) => {
            crate::event_source::validate_mqtt_topic_safe(topic)?;
        }
        StateSource::WebhookInbound { path } => {
            if path.contains("..") || path.contains("//") {
                return Err("Webhook path contains traversal characters".into());
            }
            if path.len() > 256 {
                return Err("Webhook path too long".into());
            }
        }
        StateSource::RestPoll { url, interval_secs } => {
            crate::event_source::validate_url(url, "State source URL")?;
            if *interval_secs < 10 {
                return Err("Poll interval too short (min 10s)".into());
            }
        }
    }

    // Validate command target
    if let Some(target) = &def.command_target {
        match target {
            CommandTarget::MqttTopic(topic) => {
                crate::event_source::validate_mqtt_topic_safe(topic)?;
            }
            CommandTarget::RestPost { url } => {
                crate::event_source::validate_url(url, "Command target URL")?;
            }
        }
    }

    // Validate alerts
    if def.alerts.len() > 20 {
        return Err("Too many alerts (max 20)".into());
    }
    for alert in &def.alerts {
        if alert.attribute.is_empty() || alert.attribute.len() > 64 {
            return Err("Invalid alert attribute name".into());
        }
        if alert.cooldown_secs < 10 {
            return Err("Alert cooldown too short (min 10s)".into());
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_device() -> CustomDeviceDefinition {
        CustomDeviceDefinition {
            name: "Greenhouse Temp".into(),
            entity_id: "sensor.greenhouse_temp".into(),
            device_type: "sensor".into(),
            state_source: StateSource::MqttTopic("sensors/greenhouse/temp".into()),
            command_target: None,
            attributes_schema: None,
            alerts: vec![ThresholdAlert {
                attribute: "temperature".into(),
                condition: AlertCondition::Above(35.0),
                action: EventAction::Both,
                cooldown_secs: 300,
                message_template: "Greenhouse temp is {{value}}°C!".into(),
            }],
        }
    }

    #[test]
    fn test_validate_custom_device_valid() {
        assert!(validate_custom_device(&sample_device()).is_ok());
    }

    #[test]
    fn test_validate_custom_device_bad_entity() {
        let mut dev = sample_device();
        dev.entity_id = "no_dot".into();
        assert!(validate_custom_device(&dev).is_err());
    }

    #[test]
    fn test_validate_custom_device_ssrf_target() {
        let mut dev = sample_device();
        dev.state_source = StateSource::RestPoll {
            url: "http://169.254.169.254/metadata".into(),
            interval_secs: 60,
        };
        assert!(validate_custom_device(&dev).is_err());
    }

    #[test]
    fn test_validate_custom_device_bad_mqtt_topic() {
        let mut dev = sample_device();
        dev.state_source = StateSource::MqttTopic("$SYS/broker".into());
        assert!(validate_custom_device(&dev).is_err());
    }

    #[test]
    fn test_alert_condition_above() {
        assert!(AlertCondition::Above(30.0).check("35.5", None));
        assert!(!AlertCondition::Above(30.0).check("25.0", None));
        assert!(!AlertCondition::Above(30.0).check("not a number", None));
    }

    #[test]
    fn test_alert_condition_below() {
        assert!(AlertCondition::Below(10.0).check("5.0", None));
        assert!(!AlertCondition::Below(10.0).check("15.0", None));
    }

    #[test]
    fn test_alert_condition_equals() {
        assert!(AlertCondition::Equals("on".into()).check("on", None));
        assert!(!AlertCondition::Equals("on".into()).check("off", None));
    }

    #[test]
    fn test_alert_condition_changed() {
        assert!(AlertCondition::Changed.check("new", Some("old")));
        assert!(!AlertCondition::Changed.check("same", Some("same")));
        assert!(AlertCondition::Changed.check("first", None)); // No previous = changed
    }

    #[test]
    fn test_validate_short_poll_interval() {
        let mut dev = sample_device();
        dev.state_source = StateSource::RestPoll {
            url: "https://api.example.com/sensor".into(),
            interval_secs: 5, // Too short
        };
        assert!(validate_custom_device(&dev).is_err());
    }

    #[test]
    fn test_validate_too_many_alerts() {
        let mut dev = sample_device();
        dev.alerts = (0..21)
            .map(|i| ThresholdAlert {
                attribute: format!("attr{}", i),
                condition: AlertCondition::Above(0.0),
                action: EventAction::Notify,
                cooldown_secs: 60,
                message_template: "alert".into(),
            })
            .collect();
        assert!(validate_custom_device(&dev).is_err());
    }
}
