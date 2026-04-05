//! GroupQueue host + client — distributes slot assignments over TCP/JSON.
//!
//! One peer runs the host (`ai_virtual_mic_host` binary); other peers connect
//! as clients and receive their slot, priority, and override permission from
//! the host. The host is the authoritative source of the priority table.
//!
//! Protocol: newline-delimited JSON over TCP. Each message is one line.
//!
//! ## Messages
//!
//! **Client → Host**
//! - `{"type":"join","name":"Lander","version":1}`
//! - `{"type":"heartbeat"}`
//! - `{"type":"leave"}`
//!
//! **Host → Client**
//! - `{"type":"joined","slot":0,"table":[...], "preset":"squad"}`
//! - `{"type":"rejected","reason":"full"}`
//! - `{"type":"table_update","table":[...]}`
//! - `{"type":"ping"}`
//!
//! ## Limits
//!
//! - Max slots per preset (default 8).
//! - Heartbeat interval 5 s; client dropped after 15 s silence.
//! - LAN / trusted network only. No encryption, no NAT traversal.

use crate::audio_priority_protocol::{PriorityTable, SlotId, Priority, SlotAssignment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// Wire types
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SlotAssignmentWire {
    pub slot: u8,
    pub priority: u8,
    pub can_override: bool,
    pub display_name: String,
}

impl From<&SlotAssignment> for SlotAssignmentWire {
    fn from(a: &SlotAssignment) -> Self {
        Self {
            slot: a.slot.as_u8(),
            priority: a.priority.as_u8(),
            can_override: a.can_override,
            display_name: a.display_name.clone(),
        }
    }
}

impl SlotAssignmentWire {
    pub fn to_assignment(&self) -> SlotAssignment {
        SlotAssignment {
            slot: SlotId(self.slot.min(SlotId::MAX)),
            priority: Priority::new(self.priority),
            can_override: self.can_override,
            display_name: self.display_name.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Join { name: String, version: u32 },
    Heartbeat,
    Leave,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostMessage {
    Joined { slot: u8, table: Vec<SlotAssignmentWire>, preset: String, slot_count: u8 },
    Rejected { reason: String },
    TableUpdate { table: Vec<SlotAssignmentWire> },
    Ping,
    Goodbye,
}

// ============================================================================
// Host preset
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HostPreset {
    Flat,
    Squad { callouts: u8 },
    Meeting,
}

impl HostPreset {
    pub fn build_table(&self, slot_count: u8) -> PriorityTable {
        match self {
            HostPreset::Flat => PriorityTable::flat(slot_count),
            HostPreset::Squad { callouts } => PriorityTable::squad(slot_count, *callouts),
            HostPreset::Meeting => PriorityTable::meeting(slot_count),
        }
    }

    pub fn as_label(&self) -> &'static str {
        match self { HostPreset::Flat => "flat", HostPreset::Squad { .. } => "squad", HostPreset::Meeting => "meeting" }
    }
}

// ============================================================================
// Host
// ============================================================================

#[derive(Clone, Debug)]
pub struct HostConfig {
    pub bind_addr: SocketAddr,
    pub slot_count: u8,
    pub preset: HostPreset,
    pub heartbeat_timeout: Duration,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9876".parse().unwrap(),
            slot_count: 8,
            preset: HostPreset::Flat,
            heartbeat_timeout: Duration::from_secs(15),
        }
    }
}

struct ConnectedClient {
    /// Slot assigned to this client (one per connection).
    slot: SlotId,
    /// Name the client reported.
    name: String,
    /// Last heartbeat time.
    last_seen: Instant,
    /// Writer half for pushing updates.
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    addr: SocketAddr,
}

pub struct GroupQueueHost {
    config: HostConfig,
    /// Priority table, keyed by slot. Always has all slots (names empty if unclaimed).
    table: Arc<Mutex<PriorityTable>>,
    /// Connected clients, keyed by slot.
    clients: Arc<Mutex<HashMap<SlotId, ConnectedClient>>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl GroupQueueHost {
    pub fn new(config: HostConfig) -> Self {
        let table = config.preset.build_table(config.slot_count);
        Self {
            config,
            table: Arc::new(Mutex::new(table)),
            clients: Arc::new(Mutex::new(HashMap::new())),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Start the server loop (blocking). Use in its own thread.
    pub fn run(&self) -> std::io::Result<()> {
        let listener = TcpListener::bind(self.config.bind_addr)?;
        listener.set_nonblocking(true)?;
        eprintln!("[host] listening on {} · slots={} · preset={}",
            self.config.bind_addr, self.config.slot_count, self.config.preset.as_label());
        let clients = self.clients.clone();
        let table = self.table.clone();
        let timeout = self.config.heartbeat_timeout;
        let shutdown = self.shutdown.clone();

        // Heartbeat reaper
        let clients_rp = clients.clone();
        let table_rp = table.clone();
        let shutdown_rp = shutdown.clone();
        std::thread::spawn(move || {
            while !shutdown_rp.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(2));
                Self::reap_dead_clients(&clients_rp, &table_rp, timeout);
            }
        });

        loop {
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
            match listener.accept() {
                Ok((stream, addr)) => {
                    let clients_c = clients.clone();
                    let table_c = table.clone();
                    let slot_count = self.config.slot_count;
                    let preset = self.config.preset.clone();
                    std::thread::spawn(move || {
                        if let Err(e) = Self::handle_client(stream, addr, clients_c, table_c, slot_count, preset) {
                            eprintln!("[host] client {} error: {}", addr, e);
                        }
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn reap_dead_clients(
        clients: &Arc<Mutex<HashMap<SlotId, ConnectedClient>>>,
        table: &Arc<Mutex<PriorityTable>>,
        timeout: Duration,
    ) {
        let now = Instant::now();
        let dropped: Vec<SlotId> = {
            let cl = clients.lock().unwrap_or_else(|e| e.into_inner());
            cl.iter().filter(|(_, c)| now.duration_since(c.last_seen) > timeout).map(|(k, _)| *k).collect()
        };
        if dropped.is_empty() { return; }
        let mut cl = clients.lock().unwrap_or_else(|e| e.into_inner());
        let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
        for slot in &dropped {
            eprintln!("[host] dropping slot {} (heartbeat timeout)", slot.as_u8());
            cl.remove(slot);
            if let Some(a) = t.get(*slot) {
                let mut updated = a.clone();
                updated.display_name = String::new(); // free
                t.assign(updated);
            }
        }
        // Broadcast updated table
        drop(cl);
        Self::broadcast_table(clients, &t);
    }

    fn broadcast_table(clients: &Arc<Mutex<HashMap<SlotId, ConnectedClient>>>, table: &PriorityTable) {
        let wire: Vec<SlotAssignmentWire> = table.assigned_slots().iter()
            .filter_map(|s| table.get(*s))
            .map(SlotAssignmentWire::from)
            .collect();
        let msg = HostMessage::TableUpdate { table: wire };
        let Ok(line) = serde_json::to_string(&msg) else { return; };
        let clients_l = clients.lock().unwrap_or_else(|e| e.into_inner());
        for (_, client) in clients_l.iter() {
            if let Ok(mut w) = client.writer.lock() {
                let _ = writeln!(w, "{}", line);
                let _ = w.flush();
            }
        }
    }

    fn handle_client(
        stream: TcpStream,
        addr: SocketAddr,
        clients: Arc<Mutex<HashMap<SlotId, ConnectedClient>>>,
        table: Arc<Mutex<PriorityTable>>,
        slot_count: u8,
        preset: HostPreset,
    ) -> std::io::Result<()> {
        stream.set_nonblocking(false)?;
        let reader = BufReader::new(stream.try_clone()?);
        let writer: Box<dyn Write + Send> = Box::new(stream);
        let writer = Arc::new(Mutex::new(writer));
        let mut my_slot: Option<SlotId> = None;

        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.is_empty() { continue; }
            let msg: ClientMessage = match serde_json::from_str(&line) {
                Ok(m) => m,
                Err(e) => { eprintln!("[host] bad msg from {}: {}", addr, e); continue; }
            };
            match msg {
                ClientMessage::Join { name, .. } => {
                    // Find first slot not currently claimed by a connected client
                    let free = {
                        let cl = clients.lock().unwrap_or_else(|e| e.into_inner());
                        (0..slot_count).map(SlotId).find(|s| !cl.contains_key(s))
                    };
                    let slot = match free {
                        Some(s) => s,
                        None => {
                            let reject = HostMessage::Rejected { reason: "All slots full".into() };
                            if let (Ok(mut w), Ok(line)) = (writer.lock(), serde_json::to_string(&reject)) {
                                let _ = writeln!(w, "{}", line);
                            }
                            return Ok(());
                        }
                    };
                    // Assign name to that slot
                    {
                        let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(a) = t.get(slot) {
                            let mut updated = a.clone();
                            updated.display_name = name.clone();
                            t.assign(updated);
                        }
                    }
                    my_slot = Some(slot);
                    // Send Joined + broadcast TableUpdate
                    {
                        let t = table.lock().unwrap_or_else(|e| e.into_inner());
                        let wire: Vec<SlotAssignmentWire> = t.assigned_slots().iter()
                            .filter_map(|s| t.get(*s)).map(SlotAssignmentWire::from).collect();
                        let resp = HostMessage::Joined {
                            slot: slot.as_u8(), table: wire, preset: preset.as_label().into(), slot_count,
                        };
                        if let (Ok(mut w), Ok(line)) = (writer.lock(), serde_json::to_string(&resp)) {
                            let _ = writeln!(w, "{}", line);
                            let _ = w.flush();
                        }
                    }
                    // Register client
                    {
                        let mut cl = clients.lock().unwrap_or_else(|e| e.into_inner());
                        cl.insert(slot, ConnectedClient {
                            slot, name: name.clone(), last_seen: Instant::now(),
                            writer: writer.clone(), addr,
                        });
                    }
                    // Broadcast
                    let t = table.lock().unwrap_or_else(|e| e.into_inner());
                    Self::broadcast_table(&clients, &t);
                    eprintln!("[host] slot {} assigned to {} ({})", slot.as_u8(), name, addr);
                }
                ClientMessage::Heartbeat => {
                    if let Some(s) = my_slot {
                        let mut cl = clients.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(c) = cl.get_mut(&s) { c.last_seen = Instant::now(); }
                    }
                }
                ClientMessage::Leave => {
                    break;
                }
            }
        }
        // Cleanup on disconnect
        if let Some(s) = my_slot {
            let mut cl = clients.lock().unwrap_or_else(|e| e.into_inner());
            cl.remove(&s);
            let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(a) = t.get(s) {
                let mut updated = a.clone();
                updated.display_name = String::new();
                t.assign(updated);
            }
            drop(cl);
            Self::broadcast_table(&clients, &t);
            eprintln!("[host] slot {} disconnected", s.as_u8());
        }
        Ok(())
    }

    /// Snapshot of the current priority table (for host GUI / logging).
    pub fn snapshot_table(&self) -> PriorityTable {
        self.table.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Number of connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// ============================================================================
// Host client — connects to a remote host and keeps a shared PriorityTable
// ============================================================================

#[derive(Clone, Debug)]
pub struct ClientStatus {
    pub connected: bool,
    pub my_slot: Option<SlotId>,
    pub error: Option<String>,
    pub slot_count: u8,
    pub preset: String,
}

impl Default for ClientStatus {
    fn default() -> Self {
        Self { connected: false, my_slot: None, error: None, slot_count: 8, preset: "flat".into() }
    }
}

pub struct GroupQueueHostClient {
    status: Arc<Mutex<ClientStatus>>,
    table: Arc<Mutex<PriorityTable>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl GroupQueueHostClient {
    pub fn new() -> Self {
        Self {
            status: Arc::new(Mutex::new(ClientStatus::default())),
            table: Arc::new(Mutex::new(PriorityTable::flat(8))),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn status(&self) -> ClientStatus {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn snapshot_table(&self) -> PriorityTable {
        self.table.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Connect in a background thread. Spawns reader + heartbeat loops.
    pub fn connect(&self, addr: SocketAddr, name: String) {
        let status = self.status.clone();
        let table = self.table.clone();
        let shutdown = self.shutdown.clone();
        std::thread::spawn(move || {
            let stream = match TcpStream::connect_timeout(&addr, Duration::from_secs(5)) {
                Ok(s) => s,
                Err(e) => {
                    let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                    s.connected = false;
                    s.error = Some(format!("connect failed: {}", e));
                    return;
                }
            };
            if let Err(e) = stream.set_read_timeout(Some(Duration::from_secs(30))) {
                eprintln!("[client] set_read_timeout failed: {}", e);
            }
            let reader = match stream.try_clone() {
                Ok(s) => BufReader::new(s),
                Err(e) => {
                    let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                    s.error = Some(format!("clone failed: {}", e));
                    return;
                }
            };
            let writer = Arc::new(Mutex::new(stream));

            // Send Join
            let join = ClientMessage::Join { name: name.clone(), version: 1 };
            if let Ok(line) = serde_json::to_string(&join) {
                if let Ok(mut w) = writer.lock() {
                    let _ = writeln!(w, "{}", line);
                    let _ = w.flush();
                }
            }

            // Spawn heartbeat sender
            let writer_hb = writer.clone();
            let shutdown_hb = shutdown.clone();
            std::thread::spawn(move || {
                while !shutdown_hb.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_secs(5));
                    if let Ok(line) = serde_json::to_string(&ClientMessage::Heartbeat) {
                        if let Ok(mut w) = writer_hb.lock() {
                            if writeln!(w, "{}", line).is_err() { break; }
                            let _ = w.flush();
                        }
                    }
                }
            });

            // Read loop
            for line_res in reader.lines() {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
                let line = match line_res {
                    Ok(l) => l,
                    Err(e) => {
                        let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                        s.connected = false;
                        s.error = Some(format!("read: {}", e));
                        break;
                    }
                };
                if line.is_empty() { continue; }
                let msg: HostMessage = match serde_json::from_str(&line) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("[client] bad msg: {}", e); continue; }
                };
                match msg {
                    HostMessage::Joined { slot, table: wire, preset, slot_count } => {
                        {
                            let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
                            *t = PriorityTable::new();
                            for a in &wire { t.assign(a.to_assignment()); }
                        }
                        {
                            let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                            s.connected = true; s.error = None;
                            s.my_slot = Some(SlotId(slot.min(SlotId::MAX)));
                            s.preset = preset; s.slot_count = slot_count;
                        }
                    }
                    HostMessage::Rejected { reason } => {
                        let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                        s.connected = false;
                        s.error = Some(format!("rejected: {}", reason));
                        break;
                    }
                    HostMessage::TableUpdate { table: wire } => {
                        let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
                        *t = PriorityTable::new();
                        for a in &wire { t.assign(a.to_assignment()); }
                    }
                    HostMessage::Ping => {}
                    HostMessage::Goodbye => {
                        let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
                        s.connected = false;
                        break;
                    }
                }
            }
            let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
            s.connected = false;
        });
    }
}

impl Default for GroupQueueHostClient { fn default() -> Self { Self::new() } }

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_roundtrip() {
        let a = SlotAssignment {
            slot: SlotId(3), priority: Priority(7), can_override: true,
            display_name: "Alice".into(),
        };
        let wire = SlotAssignmentWire::from(&a);
        let back = wire.to_assignment();
        assert_eq!(back.slot, a.slot);
        assert_eq!(back.priority, a.priority);
        assert_eq!(back.can_override, a.can_override);
        assert_eq!(back.display_name, a.display_name);
    }

    #[test]
    fn client_message_serde_roundtrip() {
        let m = ClientMessage::Join { name: "Bob".into(), version: 1 };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"type\":\"join\""));
        let back: ClientMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn host_message_serde_roundtrip() {
        let m = HostMessage::Joined {
            slot: 2, table: vec![], preset: "flat".into(), slot_count: 8,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"type\":\"joined\""));
        let back: HostMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn host_preset_flat_builds_table() {
        let p = HostPreset::Flat;
        let t = p.build_table(4);
        assert_eq!(t.priority_of(SlotId(0)), Priority(5));
        assert_eq!(t.priority_of(SlotId(3)), Priority(5));
    }

    #[test]
    fn host_preset_squad_has_leaders() {
        let p = HostPreset::Squad { callouts: 2 };
        let t = p.build_table(6);
        assert_eq!(t.priority_of(SlotId(0)), Priority(10));
        assert_eq!(t.priority_of(SlotId(1)), Priority(10));
        assert_eq!(t.priority_of(SlotId(2)), Priority(7));
        assert_eq!(t.priority_of(SlotId(3)), Priority(7));
        assert_eq!(t.priority_of(SlotId(4)), Priority(3));
    }

    #[test]
    fn client_status_default_is_disconnected() {
        let s = ClientStatus::default();
        assert!(!s.connected);
        assert!(s.my_slot.is_none());
    }

    #[test]
    fn host_config_default_valid() {
        let c = HostConfig::default();
        assert_eq!(c.bind_addr.port(), 9876);
        assert_eq!(c.slot_count, 8);
    }

    #[test]
    fn host_creates_with_preset() {
        let cfg = HostConfig { preset: HostPreset::Squad { callouts: 3 }, ..HostConfig::default() };
        let h = GroupQueueHost::new(cfg);
        assert_eq!(h.client_count(), 0);
        let table = h.snapshot_table();
        assert_eq!(table.priority_of(SlotId(0)), Priority(10));
    }

    #[test]
    fn host_end_to_end_join_and_table_update() {
        // Launch host on random port
        let cfg = HostConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(), ..HostConfig::default()
        };
        // Re-bind to get actual port
        let listener = TcpListener::bind(cfg.bind_addr).unwrap();
        let actual_addr = listener.local_addr().unwrap();
        drop(listener);
        let cfg = HostConfig { bind_addr: actual_addr, ..cfg };
        let host = Arc::new(GroupQueueHost::new(cfg.clone()));
        let host_run = host.clone();
        std::thread::spawn(move || { let _ = host_run.run(); });
        std::thread::sleep(Duration::from_millis(200));

        // Connect client
        let client = GroupQueueHostClient::new();
        client.connect(actual_addr, "TestClient".into());
        // Wait for connection
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(100));
            if client.status().connected { break; }
        }
        let status = client.status();
        assert!(status.connected, "client should connect, error: {:?}", status.error);
        assert_eq!(status.my_slot, Some(SlotId(0)));
        let t = client.snapshot_table();
        assert_eq!(t.get(SlotId(0)).unwrap().display_name, "TestClient");

        client.shutdown();
        host.shutdown();
    }
}
