//! Home automation module — multi-backend device control, IoT, events.
//!
//! Supports Home Assistant (REST), MQTT (Zigbee2MQTT/Tasmota), OpenHAB (REST),
//! CoAP (industrial IoT), custom devices, and mDNS discovery.
//! Feature-gated under `home-automation`.

pub mod backend;
pub mod mqtt_backend;
pub mod openhab_backend;
pub mod coap_backend;
pub mod custom_device;
pub mod discovery;

pub use backend::{
    DeviceState, HomeBackend, HomeConfig, extract_domain, validate_backend_url,
    validate_domain, validate_entity_id, validate_service_name,
};
pub use mqtt_backend::{
    DeviceRegistry, MqttConfig, MqttHomeBackend, RegistryEntry, TopicConvention,
};
pub use openhab_backend::{OpenHabBackend, OpenHabConfig};
pub use coap_backend::{CoapBackend, CoapConfig, CoapDeviceEntry};
pub use custom_device::{
    AlertCondition, CommandTarget, CustomDeviceDefinition, StateSource, ThresholdAlert,
    validate_custom_device,
};
pub use discovery::{DiscoveredService, DiscoveredServiceType, discover_services};
