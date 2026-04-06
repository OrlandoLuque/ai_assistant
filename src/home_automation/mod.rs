//! Home automation module — multi-backend device control, IoT, events.
//!
//! Supports Home Assistant (REST), MQTT (Zigbee2MQTT/Tasmota), OpenHAB (REST),
//! CoAP (industrial IoT), custom devices, and mDNS discovery.
//! Feature-gated under `home-automation`.

pub mod backend;
pub mod coap_backend;
pub mod custom_device;
pub mod discovery;
pub mod event_listener;
pub mod mqtt_backend;
pub mod openhab_backend;

pub use backend::{
    extract_domain, validate_backend_url, validate_domain, validate_entity_id,
    validate_service_name, DeviceState, HomeBackend, HomeConfig,
};
pub use coap_backend::{CoapBackend, CoapConfig, CoapDeviceEntry};
pub use custom_device::{
    validate_custom_device, AlertCondition, CommandTarget, CustomDeviceDefinition, StateSource,
    ThresholdAlert,
};
pub use discovery::{discover_services, DiscoveredService, DiscoveredServiceType};
pub use event_listener::{
    parse_ha_state_changed, parse_openhab_state_changed, DeviceStateChange, EventListenerConfig,
    HomeEventListenerManager, ListenerSource, ListenerStatus,
};
pub use mqtt_backend::{
    DeviceRegistry, MqttConfig, MqttHomeBackend, RegistryEntry, TopicConvention,
};
pub use openhab_backend::{OpenHabBackend, OpenHabConfig};
