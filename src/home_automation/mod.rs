//! Home automation module — multi-backend device control, IoT, events.
//!
//! Supports Home Assistant (REST), MQTT (Zigbee2MQTT/Tasmota), OpenHAB (REST),
//! and CoAP (industrial IoT). Feature-gated under `home-automation`.

pub mod backend;
pub mod mqtt_backend;
pub mod openhab_backend;

pub use backend::{
    DeviceState, HomeBackend, HomeConfig, extract_domain, validate_backend_url,
    validate_domain, validate_entity_id, validate_service_name,
};
pub use mqtt_backend::{
    DeviceRegistry, MqttConfig, MqttHomeBackend, RegistryEntry, TopicConvention,
};
pub use openhab_backend::{OpenHabBackend, OpenHabConfig};
