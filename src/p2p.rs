//! Peer-to-Peer Networking Module
//!
//! This module provides P2P functionality for the AI assistant:
//! - Peer discovery via DHT
//! - NAT traversal (STUN/TURN/ICE/UPnP)
//! - Secure peer communication
//! - Knowledge sharing between peers
//! - Reputation system for trust
//!
//! This module is optional and gated behind the "p2p" feature flag.
//! It does NOT affect local functionality when disabled.

use std::collections::HashMap;
use std::net::{SocketAddr, IpAddr, Ipv4Addr};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

#[cfg(feature = "distributed")]
use crate::distributed::{NodeId, Dht, DhtNode};

// =============================================================================
// P2P CONFIGURATION
// =============================================================================

/// How to handle data received from peers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerDataTrust {
    /// Ignore all peer data (P2P for discovery only)
    Ignore,
    /// Store in volatile memory only (lost on restart)
    VolatileOnly,
    /// Store but mark clearly as "peer_sourced"
    MarkAndStore,
    /// Only accept after N peers agree (consensus)
    ConsensusRequired(usize),
}

impl Default for PeerDataTrust {
    fn default() -> Self {
        PeerDataTrust::VolatileOnly
    }
}

/// TURN server configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnConfig {
    pub url: String,
    pub username: Option<String>,
    pub credential: Option<String>,
}

/// P2P network configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P2PConfig {
    /// Enable P2P networking
    pub enabled: bool,

    /// How to handle data from peers
    pub peer_data_trust: PeerDataTrust,

    /// Store peer data in local graphs?
    pub store_peer_data_locally: bool,

    // --- NAT Traversal ---

    /// STUN servers for NAT discovery
    pub stun_servers: Vec<String>,

    /// TURN servers for relay (when direct connection fails)
    pub turn_servers: Vec<TurnConfig>,

    /// Enable UPnP port mapping
    pub enable_upnp: bool,

    /// Enable NAT-PMP port mapping
    pub enable_nat_pmp: bool,

    /// Port to listen on (0 = auto-select)
    pub listen_port: u16,

    // --- Security ---

    /// Require peer authentication
    pub require_auth: bool,

    /// Maximum peers to connect to
    pub max_peers: usize,

    /// Rate limit: max messages per minute from a peer
    pub rate_limit_per_minute: usize,

    /// Minimum reputation to accept data from peer
    pub min_reputation: f32,

    // --- Bootstrap ---

    /// Bootstrap nodes for initial peer discovery
    pub bootstrap_nodes: Vec<String>,

    /// Enable mDNS for LAN discovery
    pub enable_mdns: bool,
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            peer_data_trust: PeerDataTrust::VolatileOnly,
            store_peer_data_locally: false,
            stun_servers: vec![
                "stun:stun.l.google.com:19302".to_string(),
                "stun:stun1.l.google.com:19302".to_string(),
            ],
            turn_servers: Vec::new(),
            enable_upnp: true,
            enable_nat_pmp: true,
            listen_port: 0,
            require_auth: true,
            max_peers: 50,
            rate_limit_per_minute: 60,
            min_reputation: 0.3,
            bootstrap_nodes: Vec::new(),
            enable_mdns: true,
        }
    }
}

// =============================================================================
// NAT TRAVERSAL
// =============================================================================

/// Type of NAT detected
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NatType {
    /// No NAT (public IP)
    None,
    /// Full cone NAT (easiest to traverse)
    FullCone,
    /// Restricted cone NAT
    RestrictedCone,
    /// Port restricted cone NAT
    PortRestrictedCone,
    /// Symmetric NAT (hardest to traverse, may need TURN)
    Symmetric,
    /// Unknown NAT type
    Unknown,
}

impl NatType {
    /// Whether direct P2P is likely possible
    pub fn can_direct_connect(&self) -> bool {
        matches!(self, NatType::None | NatType::FullCone | NatType::RestrictedCone)
    }

    /// Whether TURN relay is recommended
    pub fn needs_relay(&self) -> bool {
        matches!(self, NatType::Symmetric | NatType::Unknown)
    }
}

/// Result of NAT discovery
#[derive(Clone, Debug)]
pub struct NatDiscoveryResult {
    /// Detected NAT type
    pub nat_type: NatType,
    /// Public IP address (if discovered)
    pub public_ip: Option<IpAddr>,
    /// Public port (if mapped)
    pub public_port: Option<u16>,
    /// Local address
    pub local_addr: SocketAddr,
    /// Whether UPnP mapping succeeded
    pub upnp_success: bool,
    /// Whether NAT-PMP mapping succeeded
    pub nat_pmp_success: bool,
}

/// Parse STUN MAPPED-ADDRESS or XOR-MAPPED-ADDRESS from attributes
fn parse_stun_mapped_address(attrs: &[u8], magic: &[u8]) -> Option<(IpAddr, u16)> {
    let mut offset = 0;
    while offset + 4 <= attrs.len() {
        let attr_type = u16::from_be_bytes([attrs[offset], attrs[offset + 1]]);
        let attr_len = u16::from_be_bytes([attrs[offset + 2], attrs[offset + 3]]) as usize;
        let value_start = offset + 4;

        if value_start + attr_len > attrs.len() {
            break;
        }

        let value = &attrs[value_start..value_start + attr_len];

        match attr_type {
            0x0001 if value.len() >= 8 && value[1] == 0x01 => {
                // MAPPED-ADDRESS: family=IPv4
                let port = u16::from_be_bytes([value[2], value[3]]);
                let ip = IpAddr::V4(Ipv4Addr::new(value[4], value[5], value[6], value[7]));
                return Some((ip, port));
            }
            0x0020 if value.len() >= 8 && value[1] == 0x01 => {
                // XOR-MAPPED-ADDRESS: XOR with magic cookie
                let port = u16::from_be_bytes([value[2], value[3]]) ^ u16::from_be_bytes([magic[0], magic[1]]);
                let ip_bytes = [
                    value[4] ^ magic[0], value[5] ^ magic[1],
                    value[6] ^ magic[2], value[7] ^ magic[3],
                ];
                let ip = IpAddr::V4(Ipv4Addr::new(ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]));
                return Some((ip, port));
            }
            _ => {}
        }

        // Attributes are padded to 4-byte boundaries
        offset = value_start + ((attr_len + 3) & !3);
    }
    None
}

/// Fetch UPnP control URL from the device description XML
fn fetch_upnp_control_url(location: &str) -> Result<String, String> {
    use std::io::Read;

    // Parse host:port from location URL
    let url_parts: Vec<&str> = location.splitn(4, '/').collect();
    if url_parts.len() < 4 {
        return Err("Invalid location URL".to_string());
    }
    let host_port = url_parts[2];
    let path = format!("/{}", url_parts[3..].join("/"));

    let addr: SocketAddr = host_port.parse()
        .map_err(|_| format!("Cannot parse UPnP host: {}", host_port))?;
    let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .map_err(|e| format!("UPnP describe connect: {}", e))?;

    use std::io::Write;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, host_port
    );
    stream.write_all(request.as_bytes())
        .map_err(|e| format!("UPnP describe write: {}", e))?;

    let mut resp = String::new();
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let _ = stream.read_to_string(&mut resp);

    // Find the controlURL for WANIPConnection or WANPPPConnection
    let base_url = format!("http://{}", host_port);
    for line in resp.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<controlURL>") {
            if let Some(url) = trimmed.strip_prefix("<controlURL>")
                .and_then(|s| s.strip_suffix("</controlURL>"))
            {
                if url.starts_with("http") {
                    return Ok(url.to_string());
                }
                return Ok(format!("{}{}", base_url, url));
            }
        }
    }

    Err("No controlURL found in UPnP description".to_string())
}

/// NAT traversal manager
pub struct NatTraversal {
    config: P2PConfig,
    discovery_result: Option<NatDiscoveryResult>,
    upnp_mapping: Option<u16>,
}

impl NatTraversal {
    pub fn new(config: P2PConfig) -> Self {
        Self {
            config,
            discovery_result: None,
            upnp_mapping: None,
        }
    }

    /// Discover NAT type and local address using STUN.
    ///
    /// Sends a STUN Binding Request to each configured STUN server (UDP) and
    /// compares the reported external address. If the external address matches
    /// across servers, we have full-cone NAT; if ports differ, symmetric NAT.
    pub fn discover_nat(&mut self) -> Result<NatDiscoveryResult, String> {
        use std::net::UdpSocket;

        // Bind a local UDP socket on an ephemeral port
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| format!("Failed to bind UDP socket: {}", e))?;
        socket.set_read_timeout(Some(Duration::from_secs(3))).ok();

        let local_addr = socket.local_addr()
            .map_err(|e| format!("Failed to get local addr: {}", e))?;

        let mut external_addrs: Vec<(IpAddr, u16)> = Vec::new();

        // STUN Binding Request (RFC 5389 minimal): type=0x0001, length=0, magic=0x2112A442, txn_id=12 bytes
        let mut stun_request = [0u8; 20];
        stun_request[0] = 0x00; stun_request[1] = 0x01; // Binding Request
        stun_request[2] = 0x00; stun_request[3] = 0x00; // Length = 0
        stun_request[4] = 0x21; stun_request[5] = 0x12; // Magic cookie
        stun_request[6] = 0xA4; stun_request[7] = 0x42;
        // Transaction ID: use timestamp-based bytes
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        for i in 0..12 {
            stun_request[8 + i] = ((ts >> (i * 4)) & 0xFF) as u8;
        }

        for server_str in &self.config.stun_servers {
            // Parse "stun:host:port" or "host:port"
            let addr_str = server_str.strip_prefix("stun:").unwrap_or(server_str);
            let dest: SocketAddr = match addr_str.parse() {
                Ok(a) => a,
                Err(_) => {
                    // Try adding default STUN port
                    match format!("{}:3478", addr_str).parse() {
                        Ok(a) => a,
                        Err(_) => continue,
                    }
                }
            };

            if socket.send_to(&stun_request, dest).is_ok() {
                let mut buf = [0u8; 256];
                if let Ok((n, _)) = socket.recv_from(&mut buf) {
                    if n >= 20 && buf[0] == 0x01 && buf[1] == 0x01 {
                        // Parse XOR-MAPPED-ADDRESS (0x0020) or MAPPED-ADDRESS (0x0001)
                        if let Some((ip, port)) = parse_stun_mapped_address(&buf[20..n], &stun_request[4..8]) {
                            external_addrs.push((ip, port));
                        }
                    }
                }
            }
        }

        let (nat_type, public_ip, public_port) = if external_addrs.is_empty() {
            (NatType::Unknown, None, None)
        } else if external_addrs.len() == 1 {
            let (ip, port) = external_addrs[0];
            (NatType::Unknown, Some(ip), Some(port))
        } else {
            let (ip0, port0) = external_addrs[0];
            let all_same_port = external_addrs.iter().all(|(_, p)| *p == port0);
            let all_same_ip = external_addrs.iter().all(|(ip, _)| *ip == ip0);

            if all_same_ip && all_same_port {
                // Same external IP and port for different destinations → full cone or no NAT
                if ip0 == local_addr.ip() {
                    (NatType::None, Some(ip0), Some(port0))
                } else {
                    (NatType::FullCone, Some(ip0), Some(port0))
                }
            } else if all_same_ip {
                // Same IP but different ports → symmetric NAT
                (NatType::Symmetric, Some(ip0), Some(port0))
            } else {
                (NatType::Unknown, Some(ip0), Some(port0))
            }
        };

        let result = NatDiscoveryResult {
            nat_type,
            public_ip,
            public_port,
            local_addr,
            upnp_success: false,
            nat_pmp_success: false,
        };

        self.discovery_result = Some(result.clone());
        Ok(result)
    }

    /// Try to open a port via UPnP IGD (SSDP discovery + SOAP AddPortMapping).
    pub fn try_upnp_mapping(&mut self, internal_port: u16, external_port: u16) -> Result<u16, String> {
        if !self.config.enable_upnp {
            return Err("UPnP disabled".to_string());
        }

        use std::net::UdpSocket;
        use std::io::Read;

        // Step 1: SSDP M-SEARCH for IGD device
        let ssdp_request = format!(
            "M-SEARCH * HTTP/1.1\r\n\
             HOST: 239.255.255.250:1900\r\n\
             MAN: \"ssdp:discover\"\r\n\
             MX: 2\r\n\
             ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
        );

        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| format!("SSDP bind failed: {}", e))?;
        socket.set_read_timeout(Some(Duration::from_secs(3))).ok();

        let ssdp_addr: SocketAddr = "239.255.255.250:1900".parse().expect("valid SSDP multicast address");
        socket.send_to(ssdp_request.as_bytes(), ssdp_addr)
            .map_err(|e| format!("SSDP send failed: {}", e))?;

        let mut buf = [0u8; 2048];
        let (n, _) = socket.recv_from(&mut buf)
            .map_err(|e| format!("SSDP no response: {}", e))?;

        let response = String::from_utf8_lossy(&buf[..n]);

        // Extract LOCATION header
        let location = response.lines()
            .find(|line| line.to_uppercase().starts_with("LOCATION:"))
            .and_then(|line| line.splitn(2, ':').nth(1))
            .map(|s| s.trim().to_string())
            .ok_or("No LOCATION in SSDP response")?;

        // Step 2: Fetch device description to find control URL
        let control_url = fetch_upnp_control_url(&location)?;

        // Step 3: SOAP AddPortMapping
        let local_ip = socket.local_addr()
            .map_err(|e| format!("local addr: {}", e))?
            .ip();

        let soap_body = format!(
            r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
<s:Body>
<u:AddPortMapping xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
<NewRemoteHost></NewRemoteHost>
<NewExternalPort>{}</NewExternalPort>
<NewProtocol>UDP</NewProtocol>
<NewInternalPort>{}</NewInternalPort>
<NewInternalClient>{}</NewInternalClient>
<NewEnabled>1</NewEnabled>
<NewPortMappingDescription>AI Assistant P2P</NewPortMappingDescription>
<NewLeaseDuration>3600</NewLeaseDuration>
</u:AddPortMapping>
</s:Body>
</s:Envelope>"#,
            external_port, internal_port, local_ip
        );

        // Parse host:port from control URL for TCP connection
        let url_parts: Vec<&str> = control_url.splitn(4, '/').collect();
        if url_parts.len() < 4 {
            return Err("Invalid control URL".to_string());
        }
        let host_port = url_parts[2];
        let path = format!("/{}", url_parts[3..].join("/"));

        let http_request = format!(
            "POST {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Type: text/xml; charset=\"utf-8\"\r\n\
             Content-Length: {}\r\n\
             SOAPAction: \"urn:schemas-upnp-org:service:WANIPConnection:1#AddPortMapping\"\r\n\
             \r\n{}",
            path, host_port, soap_body.len(), soap_body
        );

        let stream_addr: SocketAddr = host_port.parse()
            .map_err(|_| format!("Cannot parse UPnP host: {}", host_port))?;
        let mut stream = std::net::TcpStream::connect_timeout(&stream_addr, Duration::from_secs(5))
            .map_err(|e| format!("UPnP TCP connect failed: {}", e))?;

        use std::io::Write;
        stream.write_all(http_request.as_bytes())
            .map_err(|e| format!("UPnP write failed: {}", e))?;

        let mut resp = String::new();
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let _ = stream.read_to_string(&mut resp);

        if resp.contains("200 OK") || resp.contains("<AddPortMappingResponse") {
            self.upnp_mapping = Some(external_port);
            Ok(external_port)
        } else {
            Err(format!("UPnP AddPortMapping failed: {}", resp.lines().next().unwrap_or("")))
        }
    }

    /// Try to open a port via NAT-PMP (RFC 6886).
    /// Sends a UDP mapping request to the default gateway on port 5351.
    pub fn try_nat_pmp_mapping(&mut self, internal_port: u16) -> Result<u16, String> {
        if !self.config.enable_nat_pmp {
            return Err("NAT-PMP disabled".to_string());
        }

        use std::net::UdpSocket;

        // NAT-PMP requires sending to the default gateway at port 5351
        // Try common gateway addresses
        let gateway_candidates = ["192.168.1.1:5351", "192.168.0.1:5351", "10.0.0.1:5351"];

        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| format!("NAT-PMP bind failed: {}", e))?;
        socket.set_read_timeout(Some(Duration::from_secs(3))).ok();

        for gateway in &gateway_candidates {
            let addr: SocketAddr = match gateway.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };

            // NAT-PMP mapping request: version=0, opcode=1 (UDP), reserved=0,
            // internal_port (2 bytes), external_port=0 (2 bytes, 0=same), lifetime=3600 (4 bytes)
            let mut request = [0u8; 12];
            request[0] = 0; // Version
            request[1] = 1; // Opcode: Map UDP
            // request[2..4] = 0 (reserved)
            request[4] = (internal_port >> 8) as u8;
            request[5] = (internal_port & 0xFF) as u8;
            // request[6..8] = 0 (suggested external port = 0 means "same as internal")
            // Lifetime: 3600 seconds
            request[8] = 0; request[9] = 0; request[10] = 0x0E; request[11] = 0x10;

            if socket.send_to(&request, addr).is_err() {
                continue;
            }

            let mut buf = [0u8; 16];
            if let Ok((n, _)) = socket.recv_from(&mut buf) {
                if n >= 16 && buf[1] == 129 { // Response opcode = 128 + request opcode
                    let result_code = u16::from_be_bytes([buf[2], buf[3]]);
                    if result_code == 0 {
                        let mapped_port = u16::from_be_bytes([buf[10], buf[11]]);
                        return Ok(mapped_port);
                    }
                }
            }
        }

        Err("NAT-PMP: no gateway responded".to_string())
    }

    /// Get the best address to share with peers
    pub fn get_connectable_address(&self) -> Option<SocketAddr> {
        if let Some(ref result) = self.discovery_result {
            if let (Some(ip), Some(port)) = (result.public_ip, result.public_port) {
                return Some(SocketAddr::new(ip, port));
            }
        }
        None
    }
}

// =============================================================================
// ICE (Interactive Connectivity Establishment)
// =============================================================================

/// ICE candidate type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IceCandidateType {
    /// Direct host address
    Host,
    /// Server reflexive (from STUN)
    ServerReflexive,
    /// Peer reflexive (discovered during connectivity checks)
    PeerReflexive,
    /// Relay (from TURN)
    Relay,
}

/// An ICE candidate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IceCandidate {
    pub candidate_type: IceCandidateType,
    pub address: SocketAddr,
    pub priority: u32,
    pub foundation: String,
}

impl IceCandidate {
    pub fn host(address: SocketAddr) -> Self {
        Self {
            candidate_type: IceCandidateType::Host,
            address,
            priority: 126 << 24, // Host candidates have highest priority
            foundation: format!("host_{}", address),
        }
    }

    pub fn server_reflexive(address: SocketAddr, base: SocketAddr) -> Self {
        Self {
            candidate_type: IceCandidateType::ServerReflexive,
            address,
            priority: 100 << 24,
            foundation: format!("srflx_{}_{}", address, base),
        }
    }

    pub fn relay(address: SocketAddr, server: &str) -> Self {
        Self {
            candidate_type: IceCandidateType::Relay,
            address,
            priority: 0, // Relay has lowest priority
            foundation: format!("relay_{}_{}", address, server),
        }
    }
}

/// ICE connection state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IceState {
    /// Gathering candidates
    Gathering,
    /// Checking connectivity
    Checking,
    /// Connected
    Connected,
    /// Connection failed
    Failed,
    /// Connection closed
    Closed,
}

/// ICE agent for establishing peer connections
pub struct IceAgent {
    local_candidates: Vec<IceCandidate>,
    remote_candidates: Vec<IceCandidate>,
    state: IceState,
    selected_pair: Option<(IceCandidate, IceCandidate)>,
}

impl IceAgent {
    pub fn new() -> Self {
        Self {
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            state: IceState::Gathering,
            selected_pair: None,
        }
    }

    /// Add a local candidate
    pub fn add_local_candidate(&mut self, candidate: IceCandidate) {
        self.local_candidates.push(candidate);
    }

    /// Add a remote candidate
    pub fn add_remote_candidate(&mut self, candidate: IceCandidate) {
        self.remote_candidates.push(candidate);
    }

    /// Get local candidates to send to remote peer
    pub fn get_local_candidates(&self) -> &[IceCandidate] {
        &self.local_candidates
    }

    /// Start connectivity checks between local and remote candidate pairs.
    ///
    /// Tries each pair (local, remote) ordered by combined priority. For each pair,
    /// sends a UDP probe and waits for a response. The first responsive pair is selected.
    pub fn start_checks(&mut self) {
        use std::net::UdpSocket;

        self.state = IceState::Checking;

        // Build candidate pairs sorted by combined priority (descending)
        let mut pairs: Vec<(IceCandidate, IceCandidate)> = Vec::new();
        for local in &self.local_candidates {
            for remote in &self.remote_candidates {
                pairs.push((local.clone(), remote.clone()));
            }
        }
        pairs.sort_by(|a, b| {
            let pa = a.0.priority.saturating_add(a.1.priority);
            let pb = b.0.priority.saturating_add(b.1.priority);
            pb.cmp(&pa)
        });

        // Try each pair with a short timeout
        for (local, remote) in &pairs {
            if let Ok(socket) = UdpSocket::bind(local.address) {
                socket.set_read_timeout(Some(Duration::from_millis(500))).ok();
                socket.set_write_timeout(Some(Duration::from_millis(500))).ok();

                // Send a 4-byte connectivity probe
                let probe = b"ICE\x01";
                if socket.send_to(probe, remote.address).is_ok() {
                    let mut buf = [0u8; 16];
                    if let Ok((n, _from)) = socket.recv_from(&mut buf) {
                        if n >= 4 && &buf[..3] == b"ICE" {
                            self.selected_pair = Some((local.clone(), remote.clone()));
                            self.state = IceState::Connected;
                            return;
                        }
                    }
                }
            }
        }

        // No pair succeeded
        if self.selected_pair.is_none() {
            self.state = IceState::Failed;
        }
    }

    /// Get current state
    pub fn state(&self) -> IceState {
        self.state
    }

    /// Get selected candidate pair (if connected)
    pub fn selected_pair(&self) -> Option<&(IceCandidate, IceCandidate)> {
        self.selected_pair.as_ref()
    }
}

impl Default for IceAgent {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PEER REPUTATION
// =============================================================================

/// Reputation tracking for a peer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerReputation {
    /// Peer identifier
    pub peer_id: String,
    /// Current reputation score (0.0 to 1.0)
    pub score: f32,
    /// Number of successful interactions
    pub successful_interactions: u64,
    /// Number of failed interactions
    pub failed_interactions: u64,
    /// Number of data points contributed
    pub contributions: u64,
    /// Number of data points that were verified correct
    pub verified_correct: u64,
    /// Number of data points that were verified incorrect
    pub verified_incorrect: u64,
    /// Last interaction time
    pub last_interaction: u64,
    /// Whether peer is currently banned
    pub banned: bool,
    /// Reason for ban (if banned)
    pub ban_reason: Option<String>,
}

impl PeerReputation {
    pub fn new(peer_id: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            score: 0.5, // Neutral starting score
            successful_interactions: 0,
            failed_interactions: 0,
            contributions: 0,
            verified_correct: 0,
            verified_incorrect: 0,
            last_interaction: 0,
            banned: false,
            ban_reason: None,
        }
    }

    /// Record a successful interaction
    pub fn record_success(&mut self) {
        self.successful_interactions += 1;
        self.update_score();
        self.touch();
    }

    /// Record a failed interaction
    pub fn record_failure(&mut self) {
        self.failed_interactions += 1;
        self.update_score();
        self.touch();
    }

    /// Record that a contribution was verified correct
    pub fn record_correct_contribution(&mut self) {
        self.contributions += 1;
        self.verified_correct += 1;
        self.update_score();
        self.touch();
    }

    /// Record that a contribution was verified incorrect
    pub fn record_incorrect_contribution(&mut self) {
        self.contributions += 1;
        self.verified_incorrect += 1;
        self.update_score();
        self.touch();

        // Auto-ban if too many incorrect contributions
        if self.verified_incorrect > 10 && self.accuracy() < 0.3 {
            self.ban("Too many incorrect contributions");
        }
    }

    /// Calculate accuracy rate
    pub fn accuracy(&self) -> f32 {
        if self.contributions == 0 {
            return 0.5;
        }
        self.verified_correct as f32 / self.contributions as f32
    }

    /// Update reputation score based on history
    fn update_score(&mut self) {
        let total_interactions = self.successful_interactions + self.failed_interactions;
        if total_interactions == 0 {
            return;
        }

        // Base score from interaction success rate
        let interaction_score = self.successful_interactions as f32 / total_interactions as f32;

        // Contribution accuracy score
        let accuracy_score = self.accuracy();

        // Weighted combination
        self.score = (interaction_score * 0.4 + accuracy_score * 0.6).clamp(0.0, 1.0);
    }

    /// Update last interaction time
    fn touch(&mut self) {
        self.last_interaction = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Ban this peer
    pub fn ban(&mut self, reason: &str) {
        self.banned = true;
        self.ban_reason = Some(reason.to_string());
        self.score = 0.0;
    }

    /// Unban this peer
    pub fn unban(&mut self) {
        self.banned = false;
        self.ban_reason = None;
        self.score = 0.1; // Start with low score after unban
    }

    /// Check if peer is trusted (above minimum reputation)
    pub fn is_trusted(&self, min_reputation: f32) -> bool {
        !self.banned && self.score >= min_reputation
    }
}

/// Reputation system for all known peers
pub struct ReputationSystem {
    peers: HashMap<String, PeerReputation>,
    min_reputation: f32,
}

impl ReputationSystem {
    pub fn new(min_reputation: f32) -> Self {
        Self {
            peers: HashMap::new(),
            min_reputation,
        }
    }

    /// Get or create reputation for a peer
    pub fn get_or_create(&mut self, peer_id: &str) -> &mut PeerReputation {
        if !self.peers.contains_key(peer_id) {
            self.peers.insert(peer_id.to_string(), PeerReputation::new(peer_id));
        }
        self.peers.get_mut(peer_id).expect("key just inserted")
    }

    /// Check if a peer is trusted
    pub fn is_trusted(&self, peer_id: &str) -> bool {
        self.peers
            .get(peer_id)
            .map(|r| r.is_trusted(self.min_reputation))
            .unwrap_or(false)
    }

    /// Get all banned peers
    pub fn get_banned(&self) -> Vec<&PeerReputation> {
        self.peers.values().filter(|r| r.banned).collect()
    }

    /// Get top N trusted peers
    pub fn get_top_peers(&self, n: usize) -> Vec<&PeerReputation> {
        let mut peers: Vec<_> = self.peers.values()
            .filter(|r| !r.banned)
            .collect();
        peers.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        peers.truncate(n);
        peers
    }
}

// =============================================================================
// PEER MESSAGE TYPES
// =============================================================================

/// Types of messages that can be sent between peers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PeerMessage {
    /// Ping to check liveness
    Ping { timestamp: u64 },
    /// Response to ping
    Pong { timestamp: u64, peer_id: String },

    /// Request peer list
    GetPeers,
    /// Response with peer list
    Peers { peers: Vec<PeerInfo> },

    /// Share knowledge data
    ShareKnowledge { data: KnowledgeShare },
    /// Acknowledge knowledge receipt
    AckKnowledge { id: String, accepted: bool },

    /// Query for knowledge
    QueryKnowledge { query: String },
    /// Response to knowledge query
    QueryResponse { query: String, results: Vec<KnowledgeShare> },

    /// Report contradiction
    ReportContradiction { contradiction: ContradictionReport },

    /// Request consensus on a value
    ConsensusRequest { entity: String, attribute: String, value: String },
    /// Vote in consensus
    ConsensusVote { request_id: String, agree: bool },
}

/// Information about a peer for sharing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub id: String,
    pub address: SocketAddr,
    pub reputation: f32,
    pub last_seen: u64,
}

/// Knowledge data shared between peers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeShare {
    pub id: String,
    pub entity: String,
    pub attribute: String,
    pub value: String,
    pub source: String,
    pub timestamp: u64,
    pub signature: Option<Vec<u8>>,
}

/// Contradiction report from peer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContradictionReport {
    pub id: String,
    pub entity: String,
    pub attribute: String,
    pub value_a: String,
    pub source_a: String,
    pub value_b: String,
    pub source_b: String,
    pub reporter_id: String,
    pub timestamp: u64,
}

// =============================================================================
// P2P NETWORK MANAGER
// =============================================================================

/// Connection state with a peer
#[derive(Clone)]
pub struct PeerConnection {
    pub peer_id: String,
    pub address: SocketAddr,
    pub connected_at: Instant,
    pub last_message: Instant,
    pub ice_agent: Option<Arc<Mutex<IceAgent>>>,
    pub messages_sent: u64,
    pub messages_received: u64,
}

impl std::fmt::Debug for PeerConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerConnection")
            .field("peer_id", &self.peer_id)
            .field("address", &self.address)
            .field("messages_sent", &self.messages_sent)
            .field("messages_received", &self.messages_received)
            .finish()
    }
}

/// P2P network manager
pub struct P2PManager {
    /// Configuration
    pub config: P2PConfig,
    /// NAT traversal
    nat: NatTraversal,
    /// Reputation system
    reputation: ReputationSystem,
    /// Connected peers
    connections: HashMap<String, PeerConnection>,
    /// Pending ICE negotiations
    pending_ice: HashMap<String, IceAgent>,
    /// Volatile peer data (not persisted)
    volatile_data: HashMap<String, Vec<KnowledgeShare>>,
    /// Our peer ID
    local_peer_id: String,
    /// Whether we're running
    running: bool,
    /// Outgoing messages buffered for delivery
    outgoing_messages: Vec<(String, PeerMessage)>,
    /// Consensus votes collected per request_id
    consensus_votes: HashMap<String, Vec<bool>>,
}

impl P2PManager {
    pub fn new(config: P2PConfig) -> Self {
        let local_peer_id = format!("peer_{}", SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos());

        Self {
            nat: NatTraversal::new(config.clone()),
            reputation: ReputationSystem::new(config.min_reputation),
            connections: HashMap::new(),
            pending_ice: HashMap::new(),
            volatile_data: HashMap::new(),
            local_peer_id,
            running: false,
            outgoing_messages: Vec::new(),
            consensus_votes: HashMap::new(),
            config,
        }
    }

    /// Start the P2P manager
    pub fn start(&mut self) -> Result<(), String> {
        if !self.config.enabled {
            return Err("P2P is disabled".to_string());
        }

        // Discover NAT type
        let nat_result = self.nat.discover_nat()?;
        println!("[P2P] NAT type: {:?}", nat_result.nat_type);

        // Try to open port
        if self.config.enable_upnp {
            let port = if self.config.listen_port > 0 {
                self.config.listen_port
            } else {
                // Auto-select port
                12345
            };
            if let Ok(mapped_port) = self.nat.try_upnp_mapping(port, port) {
                println!("[P2P] UPnP mapping successful: port {}", mapped_port);
            }
        }

        // Connect to bootstrap nodes via TCP
        for bootstrap in &self.config.bootstrap_nodes.clone() {
            let addr: SocketAddr = match bootstrap.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };

            match std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5)) {
                Ok(mut stream) => {
                    use std::io::{Read, Write};

                    // Send a Ping message
                    let ping = PeerMessage::Ping {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };
                    if let Ok(data) = serde_json::to_vec(&ping) {
                        let len = (data.len() as u32).to_be_bytes();
                        let _ = stream.write_all(&len);
                        let _ = stream.write_all(&data);
                        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

                        // Read response length + data
                        let mut len_buf = [0u8; 4];
                        if stream.read_exact(&mut len_buf).is_ok() {
                            let resp_len = u32::from_be_bytes(len_buf) as usize;
                            if resp_len <= 65536 {
                                let mut resp_buf = vec![0u8; resp_len];
                                if stream.read_exact(&mut resp_buf).is_ok() {
                                    if let Ok(PeerMessage::Pong { peer_id, .. }) = serde_json::from_slice(&resp_buf) {
                                        let conn = PeerConnection {
                                            peer_id: peer_id.clone(),
                                            address: addr,
                                            connected_at: Instant::now(),
                                            last_message: Instant::now(),
                                            ice_agent: None,
                                            messages_sent: 1,
                                            messages_received: 1,
                                        };
                                        self.connections.insert(peer_id.clone(), conn);
                                        self.reputation.get_or_create(&peer_id).record_success();
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Bootstrap node unreachable, skip
                }
            }
        }

        self.running = true;
        Ok(())
    }

    /// Stop the P2P manager
    pub fn stop(&mut self) {
        self.running = false;
        self.connections.clear();
    }

    /// Get our peer ID
    pub fn local_peer_id(&self) -> &str {
        &self.local_peer_id
    }

    /// Get number of connected peers
    pub fn peer_count(&self) -> usize {
        self.connections.len()
    }

    /// Handle incoming message from peer
    pub fn handle_message(&mut self, from_peer: &str, message: PeerMessage) -> Option<PeerMessage> {
        // Check reputation before processing
        if !self.reputation.is_trusted(from_peer) {
            println!("[P2P] Ignoring message from untrusted peer: {}", from_peer);
            return None;
        }

        match message {
            PeerMessage::Ping { timestamp } => {
                self.reputation.get_or_create(from_peer).record_success();
                Some(PeerMessage::Pong {
                    timestamp,
                    peer_id: self.local_peer_id.clone(),
                })
            }

            PeerMessage::Pong { timestamp: _, peer_id: _ } => {
                self.reputation.get_or_create(from_peer).record_success();
                None
            }

            PeerMessage::GetPeers => {
                let peers: Vec<PeerInfo> = self.connections
                    .values()
                    .filter(|c| c.peer_id != from_peer)
                    .take(20)
                    .map(|c| PeerInfo {
                        id: c.peer_id.clone(),
                        address: c.address,
                        reputation: self.reputation
                            .peers.get(&c.peer_id)
                            .map(|r| r.score)
                            .unwrap_or(0.5),
                        last_seen: c.last_message.elapsed().as_secs(),
                    })
                    .collect();
                Some(PeerMessage::Peers { peers })
            }

            PeerMessage::ShareKnowledge { data } => {
                self.handle_shared_knowledge(from_peer, data)
            }

            PeerMessage::QueryKnowledge { query } => {
                // Search volatile data for matches
                let query_lower = query.to_lowercase();
                let results: Vec<KnowledgeShare> = self.volatile_data.values()
                    .flat_map(|entries| entries.iter())
                    .filter(|ks| {
                        ks.entity.to_lowercase().contains(&query_lower)
                            || ks.attribute.to_lowercase().contains(&query_lower)
                            || ks.value.to_lowercase().contains(&query_lower)
                    })
                    .cloned()
                    .collect();

                Some(PeerMessage::QueryResponse { query, results })
            }

            PeerMessage::Peers { peers } => {
                // Add unknown peers to our connections (without establishing TCP)
                for peer_info in &peers {
                    if peer_info.id != self.local_peer_id
                        && !self.connections.contains_key(&peer_info.id)
                        && self.connections.len() < self.config.max_peers
                    {
                        let conn = PeerConnection {
                            peer_id: peer_info.id.clone(),
                            address: peer_info.address,
                            connected_at: Instant::now(),
                            last_message: Instant::now(),
                            ice_agent: None,
                            messages_sent: 0,
                            messages_received: 0,
                        };
                        self.connections.insert(peer_info.id.clone(), conn);
                    }
                }
                None
            }

            PeerMessage::AckKnowledge { id, accepted } => {
                self.reputation.get_or_create(from_peer).record_success();
                if accepted {
                    self.reputation.get_or_create(from_peer).record_correct_contribution();
                }
                let _ = (id, accepted); // logged implicitly via reputation
                None
            }

            PeerMessage::QueryResponse { query: _, results } => {
                // Store query results in volatile data
                for ks in results {
                    self.volatile_data
                        .entry(ks.entity.clone())
                        .or_default()
                        .push(ks);
                }
                None
            }

            PeerMessage::ReportContradiction { contradiction } => {
                // Store the contradiction report in volatile data for review
                let report_ks = KnowledgeShare {
                    id: contradiction.id.clone(),
                    entity: contradiction.entity.clone(),
                    attribute: format!("_contradiction:{}", contradiction.attribute),
                    value: format!("{} vs {}", contradiction.value_a, contradiction.value_b),
                    source: format!("peer:{}", from_peer),
                    timestamp: contradiction.timestamp,
                    signature: None,
                };
                self.volatile_data
                    .entry(contradiction.entity)
                    .or_default()
                    .push(report_ks);
                None
            }

            PeerMessage::ConsensusRequest { entity, attribute, value } => {
                // Check if we have local data that matches
                let agree = self.volatile_data
                    .get(&entity)
                    .map(|entries| {
                        entries.iter().any(|ks| ks.attribute == attribute && ks.value == value)
                    })
                    .unwrap_or(false);

                let request_id = format!("{}:{}:{}", entity, attribute, value);
                Some(PeerMessage::ConsensusVote { request_id, agree })
            }

            PeerMessage::ConsensusVote { request_id, agree } => {
                // Tally the vote
                self.consensus_votes
                    .entry(request_id)
                    .or_default()
                    .push(agree);
                None
            }
        }
    }

    /// Handle shared knowledge from peer
    fn handle_shared_knowledge(&mut self, from_peer: &str, data: KnowledgeShare) -> Option<PeerMessage> {
        let accepted = match self.config.peer_data_trust {
            PeerDataTrust::Ignore => false,

            PeerDataTrust::VolatileOnly => {
                // Store in volatile memory only
                self.volatile_data
                    .entry(data.entity.clone())
                    .or_insert_with(Vec::new)
                    .push(data.clone());
                true
            }

            PeerDataTrust::MarkAndStore => {
                // Store in volatile data marked with source attribution
                let mut marked = data.clone();
                marked.source = format!("peer:{}", from_peer);
                self.volatile_data
                    .entry(data.entity.clone())
                    .or_default()
                    .push(marked);
                true
            }

            PeerDataTrust::ConsensusRequired(n) => {
                // Count how many peers have sent the same entity+attribute+value
                let matching_count = self.volatile_data
                    .get(&data.entity)
                    .map(|entries| {
                        entries.iter()
                            .filter(|ks| ks.attribute == data.attribute && ks.value == data.value)
                            .count()
                    })
                    .unwrap_or(0);

                // Store in volatile regardless (for counting)
                self.volatile_data
                    .entry(data.entity.clone())
                    .or_default()
                    .push(data.clone());

                // Accept if we have enough confirmations (including this one)
                matching_count + 1 >= n
            }
        };

        if accepted {
            self.reputation.get_or_create(from_peer).contributions += 1;
        }

        Some(PeerMessage::AckKnowledge {
            id: data.id,
            accepted,
        })
    }

    /// Share knowledge with all connected peers.
    /// Messages are buffered in `outgoing_messages` for delivery.
    pub fn broadcast_knowledge(&mut self, data: KnowledgeShare) {
        let msg = PeerMessage::ShareKnowledge { data };
        let peer_ids: Vec<String> = self.connections.keys().cloned().collect();
        for peer_id in peer_ids {
            self.outgoing_messages.push((peer_id, msg.clone()));
        }
    }

    /// Query peers for knowledge. Returns results from local volatile data that match.
    /// Also buffers QueryKnowledge messages to connected peers for async retrieval.
    pub fn query_peers(&mut self, query: &str) -> Vec<KnowledgeShare> {
        // Buffer queries to all connected peers
        let msg = PeerMessage::QueryKnowledge { query: query.to_string() };
        let peer_ids: Vec<String> = self.connections.keys().cloned().collect();
        for peer_id in peer_ids {
            self.outgoing_messages.push((peer_id, msg.clone()));
        }

        // Return matching results from volatile data
        let query_lower = query.to_lowercase();
        self.volatile_data.values()
            .flat_map(|entries| entries.iter())
            .filter(|ks| {
                ks.entity.to_lowercase().contains(&query_lower)
                    || ks.attribute.to_lowercase().contains(&query_lower)
                    || ks.value.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect()
    }

    /// Drain buffered outgoing messages
    pub fn drain_outgoing(&mut self) -> Vec<(String, PeerMessage)> {
        std::mem::take(&mut self.outgoing_messages)
    }

    /// Get volatile data (peer-shared, not persisted)
    pub fn get_volatile_data(&self, entity: &str) -> Vec<&KnowledgeShare> {
        self.volatile_data
            .get(entity)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Clear all volatile data
    pub fn clear_volatile_data(&mut self) {
        self.volatile_data.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> P2PStats {
        P2PStats {
            enabled: self.config.enabled,
            running: self.running,
            peer_count: self.connections.len(),
            volatile_entries: self.volatile_data.values().map(|v| v.len()).sum(),
            total_peers_known: self.reputation.peers.len(),
            banned_peers: self.reputation.get_banned().len(),
        }
    }
}

/// P2P statistics
#[derive(Clone, Debug, Default)]
pub struct P2PStats {
    pub enabled: bool,
    pub running: bool,
    pub peer_count: usize,
    pub volatile_entries: usize,
    pub total_peers_known: usize,
    pub banned_peers: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_reputation() {
        let mut rep = PeerReputation::new("peer1");
        assert_eq!(rep.score, 0.5);

        rep.record_success();
        rep.record_success();
        rep.record_failure();

        // 2 successes, 1 failure = 66% success rate
        assert!(rep.score > 0.5);

        rep.record_correct_contribution();
        rep.record_correct_contribution();
        rep.record_incorrect_contribution();

        // 2/3 = 66% accuracy
        assert!((rep.accuracy() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_reputation_system() {
        let mut system = ReputationSystem::new(0.5);

        // New peer starts at 0.5
        let rep = system.get_or_create("peer1");
        assert_eq!(rep.score, 0.5);

        // Good peer
        let rep = system.get_or_create("peer1");
        for _ in 0..10 {
            rep.record_success();
            rep.record_correct_contribution();
        }

        assert!(system.is_trusted("peer1"));

        // Bad peer gets banned
        let rep = system.get_or_create("peer2");
        for _ in 0..20 {
            rep.record_incorrect_contribution();
        }

        assert!(!system.is_trusted("peer2"));
        assert_eq!(system.get_banned().len(), 1);
    }

    #[test]
    fn test_ice_candidate() {
        let host = IceCandidate::host("192.168.1.1:12345".parse().unwrap());
        assert!(matches!(host.candidate_type, IceCandidateType::Host));
        assert!(host.priority > 0);

        let relay = IceCandidate::relay("1.2.3.4:443".parse().unwrap(), "turn.example.com");
        assert!(matches!(relay.candidate_type, IceCandidateType::Relay));
        assert_eq!(relay.priority, 0);
    }

    #[test]
    fn test_p2p_config_default() {
        let config = P2PConfig::default();
        assert!(!config.enabled);
        assert!(config.enable_upnp);
        assert_eq!(config.max_peers, 50);
        assert!(!config.stun_servers.is_empty());
    }

    #[test]
    fn test_peer_data_trust() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            peer_data_trust: PeerDataTrust::VolatileOnly,
            ..Default::default()
        });

        // Simulate receiving knowledge
        let data = KnowledgeShare {
            id: "test1".to_string(),
            entity: "Sabre".to_string(),
            attribute: "shields".to_string(),
            value: "2".to_string(),
            source: "peer_data".to_string(),
            timestamp: 0,
            signature: None,
        };

        // Add peer to reputation first
        manager.reputation.get_or_create("test_peer").record_success();

        manager.handle_shared_knowledge("test_peer", data);

        let volatile = manager.get_volatile_data("Sabre");
        assert_eq!(volatile.len(), 1);

        // Clear should remove it
        manager.clear_volatile_data();
        assert!(manager.get_volatile_data("Sabre").is_empty());
    }

    #[test]
    fn test_handle_ping_pong() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });

        // Ensure peer is trusted first
        manager.reputation.get_or_create("peer1").record_success();

        let response = manager.handle_message("peer1", PeerMessage::Ping { timestamp: 12345 });
        match response {
            Some(PeerMessage::Pong { timestamp, peer_id }) => {
                assert_eq!(timestamp, 12345);
                assert_eq!(peer_id, manager.local_peer_id);
            }
            other => panic!("Expected Pong, got {:?}", other),
        }
    }

    #[test]
    fn test_handle_get_peers() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });

        // Add some connections
        manager.connections.insert("peer_a".to_string(), PeerConnection {
            peer_id: "peer_a".to_string(),
            address: "10.0.0.1:5000".parse().unwrap(),
            connected_at: Instant::now(),
            last_message: Instant::now(),
            ice_agent: None,
            messages_sent: 0,
            messages_received: 0,
        });

        manager.reputation.get_or_create("peer_b").record_success();
        let response = manager.handle_message("peer_b", PeerMessage::GetPeers);
        match response {
            Some(PeerMessage::Peers { peers }) => {
                // peer_a should be in the list, peer_b (requester) should not
                assert!(peers.iter().any(|p| p.id == "peer_a"));
                assert!(!peers.iter().any(|p| p.id == "peer_b"));
            }
            other => panic!("Expected Peers, got {:?}", other),
        }
    }

    #[test]
    fn test_handle_peers_adds_connections() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            max_peers: 10,
            ..Default::default()
        });

        manager.reputation.get_or_create("peer1").record_success();

        let peers = vec![
            PeerInfo { id: "new_peer_1".to_string(), address: "10.0.0.2:5000".parse().unwrap(), reputation: 0.8, last_seen: 0 },
            PeerInfo { id: "new_peer_2".to_string(), address: "10.0.0.3:5000".parse().unwrap(), reputation: 0.7, last_seen: 0 },
        ];

        manager.handle_message("peer1", PeerMessage::Peers { peers });

        assert_eq!(manager.connections.len(), 2);
        assert!(manager.connections.contains_key("new_peer_1"));
        assert!(manager.connections.contains_key("new_peer_2"));
    }

    #[test]
    fn test_handle_query_knowledge() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            peer_data_trust: PeerDataTrust::VolatileOnly,
            ..Default::default()
        });

        // Store some data
        let data = KnowledgeShare {
            id: "k1".to_string(),
            entity: "Rust".to_string(),
            attribute: "type".to_string(),
            value: "programming language".to_string(),
            source: "local".to_string(),
            timestamp: 0,
            signature: None,
        };
        manager.volatile_data.entry("Rust".to_string()).or_default().push(data);

        manager.reputation.get_or_create("peer1").record_success();
        let response = manager.handle_message("peer1", PeerMessage::QueryKnowledge { query: "Rust".to_string() });

        match response {
            Some(PeerMessage::QueryResponse { query, results }) => {
                assert_eq!(query, "Rust");
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].entity, "Rust");
            }
            other => panic!("Expected QueryResponse, got {:?}", other),
        }
    }

    #[test]
    fn test_broadcast_knowledge_buffers() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });

        // Add 3 peers
        for i in 0..3 {
            manager.connections.insert(format!("peer_{}", i), PeerConnection {
                peer_id: format!("peer_{}", i),
                address: format!("10.0.0.{}:5000", i + 1).parse().unwrap(),
                connected_at: Instant::now(),
                last_message: Instant::now(),
                ice_agent: None,
                messages_sent: 0,
                messages_received: 0,
            });
        }

        let data = KnowledgeShare {
            id: "b1".to_string(),
            entity: "Test".to_string(),
            attribute: "attr".to_string(),
            value: "val".to_string(),
            source: "local".to_string(),
            timestamp: 0,
            signature: None,
        };

        manager.broadcast_knowledge(data);
        let outgoing = manager.drain_outgoing();
        assert_eq!(outgoing.len(), 3);
    }

    #[test]
    fn test_query_peers_returns_volatile() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });

        // Add data to volatile store
        let data = KnowledgeShare {
            id: "q1".to_string(),
            entity: "Python".to_string(),
            attribute: "version".to_string(),
            value: "3.12".to_string(),
            source: "peer:someone".to_string(),
            timestamp: 0,
            signature: None,
        };
        manager.volatile_data.entry("Python".to_string()).or_default().push(data);

        let results = manager.query_peers("python");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "3.12");
    }

    #[test]
    fn test_mark_and_store_trust() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            peer_data_trust: PeerDataTrust::MarkAndStore,
            ..Default::default()
        });

        manager.reputation.get_or_create("peer1").record_success();

        let data = KnowledgeShare {
            id: "m1".to_string(),
            entity: "AI".to_string(),
            attribute: "field".to_string(),
            value: "computer science".to_string(),
            source: "original".to_string(),
            timestamp: 0,
            signature: None,
        };

        let response = manager.handle_shared_knowledge("peer1", data);
        assert!(matches!(response, Some(PeerMessage::AckKnowledge { accepted: true, .. })));

        // Data should be stored with peer attribution
        let stored = manager.get_volatile_data("AI");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].source, "peer:peer1");
    }

    #[test]
    fn test_consensus_required_trust() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            peer_data_trust: PeerDataTrust::ConsensusRequired(2),
            ..Default::default()
        });

        manager.reputation.get_or_create("peer1").record_success();
        manager.reputation.get_or_create("peer2").record_success();

        let make_data = |id: &str| KnowledgeShare {
            id: id.to_string(),
            entity: "Fact".to_string(),
            attribute: "truth".to_string(),
            value: "42".to_string(),
            source: "test".to_string(),
            timestamp: 0,
            signature: None,
        };

        // First submission: not enough confirmations yet (0 existing + 1 = 1 < 2)
        let resp1 = manager.handle_shared_knowledge("peer1", make_data("c1"));
        assert!(matches!(resp1, Some(PeerMessage::AckKnowledge { accepted: false, .. })));

        // Second submission: now we have 1 existing + 1 = 2 >= 2
        let resp2 = manager.handle_shared_knowledge("peer2", make_data("c2"));
        assert!(matches!(resp2, Some(PeerMessage::AckKnowledge { accepted: true, .. })));
    }

    #[test]
    fn test_consensus_vote_handling() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            peer_data_trust: PeerDataTrust::VolatileOnly,
            ..Default::default()
        });

        manager.reputation.get_or_create("peer1").record_success();
        manager.reputation.get_or_create("peer2").record_success();

        // Receive consensus request — should vote based on local data
        let resp = manager.handle_message("peer1", PeerMessage::ConsensusRequest {
            entity: "X".to_string(),
            attribute: "a".to_string(),
            value: "1".to_string(),
        });
        assert!(matches!(resp, Some(PeerMessage::ConsensusVote { agree: false, .. })));

        // Now store matching data and try again
        manager.volatile_data.entry("X".to_string()).or_default().push(KnowledgeShare {
            id: "x1".to_string(), entity: "X".to_string(), attribute: "a".to_string(),
            value: "1".to_string(), source: "local".to_string(), timestamp: 0, signature: None,
        });

        let resp2 = manager.handle_message("peer2", PeerMessage::ConsensusRequest {
            entity: "X".to_string(),
            attribute: "a".to_string(),
            value: "1".to_string(),
        });
        assert!(matches!(resp2, Some(PeerMessage::ConsensusVote { agree: true, .. })));

        // Receive votes and check tally
        manager.handle_message("peer1", PeerMessage::ConsensusVote {
            request_id: "X:a:1".to_string(),
            agree: true,
        });
        manager.handle_message("peer2", PeerMessage::ConsensusVote {
            request_id: "X:a:1".to_string(),
            agree: false,
        });

        let votes = manager.consensus_votes.get("X:a:1").unwrap();
        assert_eq!(votes.len(), 2);
        assert_eq!(votes.iter().filter(|v| **v).count(), 1);
    }

    #[test]
    fn test_contradiction_report() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });

        manager.reputation.get_or_create("peer1").record_success();

        let contradiction = ContradictionReport {
            id: "cr1".to_string(),
            entity: "Mars".to_string(),
            attribute: "distance".to_string(),
            value_a: "225M km".to_string(),
            source_a: "src_a".to_string(),
            value_b: "300M km".to_string(),
            source_b: "src_b".to_string(),
            reporter_id: "peer1".to_string(),
            timestamp: 1234,
        };

        let resp = manager.handle_message("peer1", PeerMessage::ReportContradiction { contradiction });
        assert!(resp.is_none());

        // Should be stored in volatile data
        let stored = manager.get_volatile_data("Mars");
        assert_eq!(stored.len(), 1);
        assert!(stored[0].attribute.contains("_contradiction"));
    }

    #[test]
    fn test_untrusted_peer_ignored() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            min_reputation: 0.5,
            ..Default::default()
        });

        // Make peer2 banned
        let rep = manager.reputation.get_or_create("bad_peer");
        for _ in 0..20 {
            rep.record_incorrect_contribution();
        }

        // Messages from banned peer should be ignored
        let resp = manager.handle_message("bad_peer", PeerMessage::Ping { timestamp: 0 });
        assert!(resp.is_none());
    }

    #[test]
    fn test_stun_mapped_address_parsing() {
        // Construct a mock STUN XOR-MAPPED-ADDRESS attribute
        // Type: 0x0020, Length: 8, Family: 0x01 (IPv4)
        // XOR Port: port ^ 0x2112 = 5000 ^ 0x2112 = 0x0AA2
        // XOR IP: ip ^ 0x2112A442
        let magic = [0x21, 0x12, 0xA4, 0x42];
        let port: u16 = 5000;
        let ip = Ipv4Addr::new(203, 0, 113, 1);

        let xor_port = port ^ u16::from_be_bytes([magic[0], magic[1]]);
        let ip_octets = ip.octets();
        let xor_ip = [
            ip_octets[0] ^ magic[0],
            ip_octets[1] ^ magic[1],
            ip_octets[2] ^ magic[2],
            ip_octets[3] ^ magic[3],
        ];

        let mut attr = Vec::new();
        attr.extend_from_slice(&[0x00, 0x20]); // Type
        attr.extend_from_slice(&[0x00, 0x08]); // Length
        attr.push(0x00); // Reserved
        attr.push(0x01); // Family: IPv4
        attr.extend_from_slice(&xor_port.to_be_bytes());
        attr.extend_from_slice(&xor_ip);

        let result = parse_stun_mapped_address(&attr, &magic);
        assert!(result.is_some());
        let (parsed_ip, parsed_port) = result.unwrap();
        assert_eq!(parsed_port, 5000);
        assert_eq!(parsed_ip, IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)));
    }

    #[test]
    fn test_ice_start_checks_no_candidates() {
        let mut agent = IceAgent::new();
        agent.start_checks();
        // No candidates → should fail
        assert_eq!(agent.state(), IceState::Failed);
        assert!(agent.selected_pair().is_none());
    }

    #[test]
    fn test_query_response_stores_results() {
        let mut manager = P2PManager::new(P2PConfig {
            enabled: true,
            ..Default::default()
        });
        manager.reputation.get_or_create("peer1").record_success();

        let results = vec![KnowledgeShare {
            id: "r1".to_string(),
            entity: "Topic".to_string(),
            attribute: "info".to_string(),
            value: "data".to_string(),
            source: "peer1".to_string(),
            timestamp: 0,
            signature: None,
        }];

        manager.handle_message("peer1", PeerMessage::QueryResponse {
            query: "Topic".to_string(),
            results,
        });

        assert_eq!(manager.get_volatile_data("Topic").len(), 1);
    }
}
