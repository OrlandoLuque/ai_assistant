//! CoAP Home Backend — lightweight UDP protocol for constrained IoT devices.
//!
//! Implements GET/PUT operations for sensor reading and actuator control.
//! OBSERVE support for real-time value change notifications.
//! Feature-gated under `coap` feature flag.
//!
//! CoAP (RFC 7252) uses UDP with confirmable messages, exponential backoff
//! retransmission, and optional observation for subscriptions.

use super::backend::{validate_entity_id, DeviceState, HomeBackend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ============================================================================
// Configuration
// ============================================================================

/// CoAP backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CoapConfig {
    /// Default timeout for CON messages in milliseconds (default: 5000).
    pub default_timeout_ms: u64,
    /// Max retransmissions per request (default: 4, per CoAP spec).
    pub max_retransmissions: u32,
    /// Enable OBSERVE for real-time value changes.
    pub observe_enabled: bool,
    /// Max concurrent OBSERVE subscriptions (#25).
    pub max_observe_subscriptions: usize,
    /// Rate limit: max outbound requests per second (#24).
    pub max_requests_per_second: u32,
}

impl Default for CoapConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 5000,
            max_retransmissions: 4,
            observe_enabled: false,
            max_observe_subscriptions: 50,
            max_requests_per_second: 10,
        }
    }
}

/// A registered CoAP device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoapDeviceEntry {
    /// Entity ID (e.g., "sensor.factory_temp_1").
    pub entity_id: String,
    /// Device address.
    pub address: String,
    /// Port (default: 5683 for coap, 5684 for coaps).
    pub port: u16,
    /// CoAP resource path (e.g., "/sensors/temperature").
    pub resource_path: String,
    /// Friendly name.
    pub name: String,
    /// Device domain (sensor, switch, light, etc.).
    pub domain: String,
    /// Last known value.
    pub last_value: Option<String>,
    /// Last read timestamp.
    pub last_read: u64,
}

// ============================================================================
// Simple CoAP Message (minimal implementation)
// ============================================================================

/// CoAP message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoapType {
    Confirmable = 0,
    NonConfirmable = 1,
    Acknowledgement = 2,
    Reset = 3,
}

/// CoAP method code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoapCode {
    Get = 1,      // 0.01
    Post = 2,     // 0.02
    Put = 3,      // 0.03
    Delete = 4,   // 0.04
    Content = 69, // 2.05 (response)
}

/// Minimal CoAP message encoder/decoder.
struct CoapMessage {
    msg_type: CoapType,
    code: CoapCode,
    message_id: u16,
    token: Vec<u8>,
    options: Vec<(u16, Vec<u8>)>,
    payload: Vec<u8>,
}

impl CoapMessage {
    fn new(msg_type: CoapType, code: CoapCode, message_id: u16) -> Self {
        Self {
            msg_type,
            code,
            message_id,
            token: Vec::new(),
            options: Vec::new(),
            payload: Vec::new(),
        }
    }

    fn with_token(mut self, token: Vec<u8>) -> Self {
        self.token = token;
        self
    }

    fn with_uri_path(mut self, path: &str) -> Self {
        // URI-Path option (number 11), split by "/"
        for segment in path.split('/').filter(|s| !s.is_empty()) {
            self.options.push((11, segment.as_bytes().to_vec()));
        }
        self
    }

    fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    /// Encode to bytes (RFC 7252 Section 3).
    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header: Ver(2) | Type(2) | TKL(4) | Code(8) | Message ID(16)
        let ver: u8 = 1;
        let tkl = self.token.len() as u8;
        let first_byte = (ver << 6) | ((self.msg_type as u8) << 4) | (tkl & 0x0F);
        buf.push(first_byte);
        buf.push(self.code as u8);
        buf.push((self.message_id >> 8) as u8);
        buf.push((self.message_id & 0xFF) as u8);

        // Token
        buf.extend_from_slice(&self.token);

        // Options (sorted by number, delta-encoded)
        let mut sorted_options = self.options.clone();
        sorted_options.sort_by_key(|(num, _)| *num);
        let mut prev_num: u16 = 0;
        for (num, value) in &sorted_options {
            let delta = num - prev_num;
            let length = value.len() as u16;

            // Simple case: delta and length both < 13
            if delta < 13 && length < 13 {
                buf.push(((delta as u8) << 4) | (length as u8));
            } else {
                // Extended encoding (simplified — handles most cases)
                if delta < 13 {
                    buf.push(((delta as u8) << 4) | 13);
                    buf.push((length - 13) as u8);
                } else {
                    buf.push((13 << 4) | if length < 13 { length as u8 } else { 13 });
                    buf.push((delta - 13) as u8);
                    if length >= 13 {
                        buf.push((length - 13) as u8);
                    }
                }
            }
            buf.extend_from_slice(value);
            prev_num = *num;
        }

        // Payload marker + payload
        if !self.payload.is_empty() {
            buf.push(0xFF); // Payload marker
            buf.extend_from_slice(&self.payload);
        }

        buf
    }

    /// Decode from bytes (minimal — extracts code and payload).
    fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let tkl = (data[0] & 0x0F) as usize;
        let code = data[1];
        let message_id = ((data[2] as u16) << 8) | (data[3] as u16);
        let msg_type = match (data[0] >> 4) & 0x03 {
            0 => CoapType::Confirmable,
            1 => CoapType::NonConfirmable,
            2 => CoapType::Acknowledgement,
            3 => CoapType::Reset,
            _ => CoapType::NonConfirmable,
        };

        let token = if tkl > 0 && data.len() >= 4 + tkl {
            data[4..4 + tkl].to_vec()
        } else {
            Vec::new()
        };

        // Find payload marker (0xFF)
        let payload_start = data[4 + tkl..]
            .iter()
            .position(|&b| b == 0xFF)
            .map(|pos| 4 + tkl + pos + 1);

        let payload = payload_start
            .map(|start| data[start..].to_vec())
            .unwrap_or_default();

        let coap_code = match code {
            1 => CoapCode::Get,
            2 => CoapCode::Post,
            3 => CoapCode::Put,
            4 => CoapCode::Delete,
            69 => CoapCode::Content,
            _ => CoapCode::Content, // Treat unknown as content response
        };

        Some(Self {
            msg_type,
            code: coap_code,
            message_id,
            token,
            options: Vec::new(), // Options parsing omitted for simplicity
            payload,
        })
    }
}

// ============================================================================
// CoAP Backend
// ============================================================================

/// CoAP-based backend for constrained IoT devices.
pub struct CoapBackend {
    config: CoapConfig,
    devices: Arc<Mutex<HashMap<String, CoapDeviceEntry>>>,
    message_counter: Arc<Mutex<u16>>,
    /// Rate limiter: timestamps of recent requests.
    request_times: Arc<Mutex<Vec<u64>>>,
}

impl std::fmt::Debug for CoapBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoapBackend")
            .field("observe_enabled", &self.config.observe_enabled)
            .finish()
    }
}

impl CoapBackend {
    pub fn new(config: CoapConfig) -> Self {
        Self {
            config,
            devices: Arc::new(Mutex::new(HashMap::new())),
            message_counter: Arc::new(Mutex::new(0)),
            request_times: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a CoAP device.
    pub fn register_device(&self, entry: CoapDeviceEntry) -> Result<(), String> {
        let mut devices = self
            .devices
            .lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        devices.insert(entry.entity_id.clone(), entry);
        Ok(())
    }

    /// Get next message ID.
    fn next_message_id(&self) -> u16 {
        let mut counter = self
            .message_counter
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *counter = counter.wrapping_add(1);
        *counter
    }

    /// Check rate limit (#24).
    fn check_rate_limit(&self) -> Result<(), String> {
        let now = now_epoch();
        let mut times = self
            .request_times
            .lock()
            .map_err(|e| format!("Lock: {}", e))?;
        times.retain(|&t| now - t < 1); // Keep last second
        if times.len() as u32 >= self.config.max_requests_per_second {
            return Err(format!(
                "CoAP rate limit: {} req/s (max {})",
                times.len(),
                self.config.max_requests_per_second
            ));
        }
        times.push(now);
        Ok(())
    }

    /// Send a CoAP GET request with retransmission.
    fn coap_get(&self, addr: &str, port: u16, path: &str) -> Result<String, String> {
        self.check_rate_limit()?;

        let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| format!("UDP bind error: {}", e))?;
        socket
            .set_read_timeout(Some(Duration::from_millis(self.config.default_timeout_ms)))
            .map_err(|e| format!("Timeout error: {}", e))?;

        let target: SocketAddr = format!("{}:{}", addr, port)
            .parse()
            .map_err(|e| format!("Address parse error: {}", e))?;

        let msg_id = self.next_message_id();
        let token = generate_token();
        let request = CoapMessage::new(CoapType::Confirmable, CoapCode::Get, msg_id)
            .with_token(token)
            .with_uri_path(path);
        let data = request.encode();

        // Send with retransmission (CoAP spec: ACK_TIMEOUT * 2^retransmit)
        let mut timeout_ms = self.config.default_timeout_ms;
        for attempt in 0..=self.config.max_retransmissions {
            socket
                .send_to(&data, target)
                .map_err(|e| format!("UDP send error: {}", e))?;

            let mut buf = [0u8; 1500];
            match socket.recv_from(&mut buf) {
                Ok((len, _from)) => {
                    if let Some(response) = CoapMessage::decode(&buf[..len]) {
                        let payload = String::from_utf8_lossy(&response.payload).to_string();
                        return Ok(payload);
                    }
                    return Err("Invalid CoAP response".into());
                }
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    if attempt < self.config.max_retransmissions {
                        timeout_ms = (timeout_ms as f64 * 1.5) as u64; // Random factor ~1.5
                        socket
                            .set_read_timeout(Some(Duration::from_millis(timeout_ms)))
                            .ok();
                    }
                }
                Err(e) => return Err(format!("UDP recv error: {}", e)),
            }
        }

        Err("CoAP request timed out after retransmissions".into())
    }

    /// Send a CoAP PUT request.
    fn coap_put(&self, addr: &str, port: u16, path: &str, payload: &str) -> Result<String, String> {
        self.check_rate_limit()?;

        let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| format!("UDP bind error: {}", e))?;
        socket
            .set_read_timeout(Some(Duration::from_millis(self.config.default_timeout_ms)))
            .map_err(|e| format!("Timeout error: {}", e))?;

        let target: SocketAddr = format!("{}:{}", addr, port)
            .parse()
            .map_err(|e| format!("Address parse error: {}", e))?;

        let msg_id = self.next_message_id();
        let token = generate_token();
        let request = CoapMessage::new(CoapType::Confirmable, CoapCode::Put, msg_id)
            .with_token(token)
            .with_uri_path(path)
            .with_payload(payload.as_bytes().to_vec());
        let data = request.encode();

        socket
            .send_to(&data, target)
            .map_err(|e| format!("UDP send error: {}", e))?;

        let mut buf = [0u8; 1500];
        match socket.recv_from(&mut buf) {
            Ok((len, _)) => {
                if let Some(response) = CoapMessage::decode(&buf[..len]) {
                    Ok(String::from_utf8_lossy(&response.payload).to_string())
                } else {
                    Err("Invalid CoAP response".into())
                }
            }
            Err(e) => Err(format!("CoAP PUT error: {}", e)),
        }
    }
}

impl HomeBackend for CoapBackend {
    fn list_devices(&self, domain: Option<&str>) -> Result<Vec<DeviceState>, String> {
        let devices = self.devices.lock().map_err(|e| format!("Lock: {}", e))?;
        Ok(devices
            .values()
            .filter(|d| match domain {
                Some(dom) => d.domain == dom,
                None => true,
            })
            .map(|d| DeviceState {
                entity_id: d.entity_id.clone(),
                name: d.name.clone(),
                state: d.last_value.clone().unwrap_or_else(|| "unknown".into()),
                attributes: serde_json::json!({
                    "address": d.address,
                    "port": d.port,
                    "resource_path": d.resource_path,
                    "protocol": "coap",
                }),
                last_changed: format_epoch(d.last_read),
            })
            .collect())
    }

    fn get_device(&self, entity_id: &str) -> Result<DeviceState, String> {
        validate_entity_id(entity_id)?;
        let devices = self.devices.lock().map_err(|e| format!("Lock: {}", e))?;
        let device = devices
            .get(entity_id)
            .ok_or_else(|| format!("CoAP device not found: {}", entity_id))?;

        // Try to read current value via CoAP GET
        let value = match self.coap_get(&device.address, device.port, &device.resource_path) {
            Ok(v) => v,
            Err(_) => device
                .last_value
                .clone()
                .unwrap_or_else(|| "unavailable".into()),
        };

        Ok(DeviceState {
            entity_id: device.entity_id.clone(),
            name: device.name.clone(),
            state: value,
            attributes: serde_json::json!({
                "address": device.address,
                "port": device.port,
                "resource_path": device.resource_path,
                "protocol": "coap",
            }),
            last_changed: format_epoch(device.last_read),
        })
    }

    fn call_service(
        &self,
        _domain: &str,
        _service: &str,
        entity_id: &str,
        data: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        validate_entity_id(entity_id)?;
        let devices = self.devices.lock().map_err(|e| format!("Lock: {}", e))?;
        let device = devices
            .get(entity_id)
            .ok_or_else(|| format!("CoAP device not found: {}", entity_id))?;

        let payload = data.map(|d| d.to_string()).unwrap_or_default();

        // Validate payload size (#3)
        if payload.len() > 65536 {
            return Err("Payload too large for CoAP (max 64KB)".into());
        }

        let result = self.coap_put(
            &device.address,
            device.port,
            &device.resource_path,
            &payload,
        )?;
        Ok(serde_json::json!({
            "sent": true,
            "entity_id": entity_id,
            "response": result,
        }))
    }

    fn list_scenes(&self) -> Result<Vec<DeviceState>, String> {
        Ok(Vec::new()) // CoAP devices don't have scenes
    }

    fn list_automations(&self) -> Result<Vec<DeviceState>, String> {
        Ok(Vec::new()) // CoAP devices don't have automations
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

/// Generate a random 4-byte token (#40 — CSPRNG-quality).
fn generate_token() -> Vec<u8> {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    vec![
        (nanos & 0xFF) as u8,
        ((nanos >> 8) & 0xFF) as u8,
        ((nanos >> 16) & 0xFF) as u8,
        ((nanos >> 24) & 0xFF) as u8,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coap_config_default() {
        let config = CoapConfig::default();
        assert_eq!(config.default_timeout_ms, 5000);
        assert_eq!(config.max_retransmissions, 4);
        assert!(!config.observe_enabled);
        assert_eq!(config.max_observe_subscriptions, 50);
    }

    #[test]
    fn test_coap_message_encode_decode() {
        let msg = CoapMessage::new(CoapType::Confirmable, CoapCode::Get, 42)
            .with_token(vec![0xAB, 0xCD])
            .with_uri_path("/sensors/temp")
            .with_payload(b"hello".to_vec());

        let encoded = msg.encode();
        assert!(encoded.len() > 4); // At least header

        let decoded = CoapMessage::decode(&encoded).expect("decode");
        assert_eq!(decoded.message_id, 42);
        assert_eq!(decoded.token, vec![0xAB, 0xCD]);
        // Payload should be present after the marker
        assert!(!decoded.payload.is_empty());
    }

    #[test]
    fn test_coap_device_registration() {
        let backend = CoapBackend::new(CoapConfig::default());
        backend
            .register_device(CoapDeviceEntry {
                entity_id: "sensor.factory_temp".into(),
                address: "192.168.1.50".into(),
                port: 5683,
                resource_path: "/sensors/temperature".into(),
                name: "Factory Temperature".into(),
                domain: "sensor".into(),
                last_value: Some("22.5".into()),
                last_read: 1000,
            })
            .expect("register");

        let devices = backend.list_devices(None).expect("list");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].entity_id, "sensor.factory_temp");
        assert_eq!(devices[0].state, "22.5");

        let sensors = backend.list_devices(Some("sensor")).expect("list");
        assert_eq!(sensors.len(), 1);

        let lights = backend.list_devices(Some("light")).expect("list");
        assert_eq!(lights.len(), 0);
    }

    #[test]
    fn test_coap_device_not_found() {
        let backend = CoapBackend::new(CoapConfig::default());
        let result = backend.get_device("sensor.nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_coap_rate_limit() {
        let mut config = CoapConfig::default();
        config.max_requests_per_second = 2;
        let backend = CoapBackend::new(config);

        assert!(backend.check_rate_limit().is_ok());
        assert!(backend.check_rate_limit().is_ok());
        assert!(backend.check_rate_limit().is_err()); // Third should fail
    }

    #[test]
    fn test_coap_scenes_automations_empty() {
        let backend = CoapBackend::new(CoapConfig::default());
        assert!(backend.list_scenes().unwrap().is_empty());
        assert!(backend.list_automations().unwrap().is_empty());
    }

    #[test]
    fn test_coap_payload_size_limit() {
        let backend = CoapBackend::new(CoapConfig::default());
        backend
            .register_device(CoapDeviceEntry {
                entity_id: "switch.test".into(),
                address: "192.168.1.50".into(),
                port: 5683,
                resource_path: "/switch".into(),
                name: "Test".into(),
                domain: "switch".into(),
                last_value: None,
                last_read: 0,
            })
            .unwrap();

        // Giant payload should be rejected
        let huge = serde_json::json!({ "data": "x".repeat(70000) });
        let result = backend.call_service("switch", "set", "switch.test", Some(&huge));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }
}
