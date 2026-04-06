//! mDNS auto-discovery for home automation services on the local network.
//!
//! Scans for Home Assistant, OpenHAB, and MQTT brokers via mDNS/Zeroconf.
//! Results are returned as discovered services — never auto-connected (#23).

use serde::{Deserialize, Serialize};

/// A discovered service on the local network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredService {
    /// Type of service found.
    pub service_type: DiscoveredServiceType,
    /// Hostname.
    pub hostname: String,
    /// Port number.
    pub port: u16,
    /// IP addresses (may be multiple).
    pub ip_addresses: Vec<String>,
    /// TXT record key-value pairs.
    pub txt_records: std::collections::HashMap<String, String>,
}

/// Type of discovered service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DiscoveredServiceType {
    HomeAssistant,
    OpenHab,
    MqttBroker,
    Unknown,
}

impl std::fmt::Display for DiscoveredServiceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HomeAssistant => write!(f, "Home Assistant"),
            Self::OpenHab => write!(f, "OpenHAB"),
            Self::MqttBroker => write!(f, "MQTT Broker"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Service types to scan for.
pub const MDNS_SERVICES: &[(&str, DiscoveredServiceType)] = &[
    (
        "_home-assistant._tcp.local.",
        DiscoveredServiceType::HomeAssistant,
    ),
    (
        "_openhab-server._tcp.local.",
        DiscoveredServiceType::OpenHab,
    ),
    ("_mqtt._tcp.local.", DiscoveredServiceType::MqttBroker),
];

/// Perform mDNS discovery scan.
///
/// Returns found services. This is a **discovery only** operation —
/// no connections are made. The user must validate and explicitly
/// connect to any discovered service (#23: mDNS spoofing defense).
///
/// Note: In the current implementation, this returns an empty list
/// unless the `mdns-sd` crate is available and a real network scan
/// is performed. The function signature is stable for when the crate
/// is wired in.
pub fn discover_services(timeout_secs: u64) -> Vec<DiscoveredService> {
    let _ = timeout_secs;
    // TODO: Wire mdns-sd crate when home-automation feature includes it.
    // For now, return empty — the MCP tool `home_discover` will report
    // "No services found (mDNS scan not available in this build)".
    //
    // When wired:
    // let daemon = mdns_sd::ServiceDaemon::new().ok()?;
    // for (service_type, discovered_type) in MDNS_SERVICES {
    //     let receiver = daemon.browse(service_type).ok()?;
    //     // Collect for timeout_secs...
    // }
    Vec::new()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovered_service_type_display() {
        assert_eq!(
            DiscoveredServiceType::HomeAssistant.to_string(),
            "Home Assistant"
        );
        assert_eq!(DiscoveredServiceType::MqttBroker.to_string(), "MQTT Broker");
    }

    #[test]
    fn test_mdns_services_list() {
        assert_eq!(MDNS_SERVICES.len(), 3);
        assert_eq!(MDNS_SERVICES[0].1, DiscoveredServiceType::HomeAssistant);
    }

    #[test]
    fn test_discover_services_returns_empty() {
        // Without real mDNS daemon, returns empty
        let results = discover_services(3);
        assert!(results.is_empty());
    }
}
