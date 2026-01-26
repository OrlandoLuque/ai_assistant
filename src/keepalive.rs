//! Connection keepalive management
//!
//! This module provides utilities for maintaining persistent connections
//! to LLM providers, reducing connection overhead and latency.
//!
//! # Features
//!
//! - **Heartbeat monitoring**: Periodic health checks
//! - **Auto-reconnection**: Automatic reconnection on failure
//! - **Connection pooling**: Manage multiple connections
//! - **Provider-specific**: Different strategies per provider
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::keepalive::{KeepaliveManager, KeepaliveConfig, ConnectionState};
//!
//! let manager = KeepaliveManager::new(KeepaliveConfig::default());
//!
//! // Register a connection
//! manager.register("ollama", "http://localhost:11434");
//!
//! // Start keepalive monitoring
//! manager.start();
//!
//! // Check connection state
//! let state = manager.get_state("ollama");
//! assert_eq!(state, Some(ConnectionState::Connected));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected
    Disconnected,
    /// Attempting to connect
    Connecting,
    /// Connected and healthy
    Connected,
    /// Connection degraded (high latency)
    Degraded,
    /// Connection failed, waiting to retry
    Failed,
    /// Connection is being closed
    Closing,
}

impl ConnectionState {
    /// Check if connection is usable
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Connected | Self::Degraded)
    }

    /// Check if connection needs attention
    pub fn needs_action(&self) -> bool {
        matches!(self, Self::Disconnected | Self::Failed)
    }
}

/// Keepalive configuration
#[derive(Debug, Clone)]
pub struct KeepaliveConfig {
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Timeout for heartbeat requests
    pub heartbeat_timeout: Duration,
    /// Number of failed heartbeats before marking as failed
    pub failure_threshold: u32,
    /// Delay before reconnection attempt
    pub reconnect_delay: Duration,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Latency threshold for degraded state (ms)
    pub degraded_latency_ms: u64,
    /// Enable automatic reconnection
    pub auto_reconnect: bool,
}

impl Default for KeepaliveConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(30),
            heartbeat_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 10,
            degraded_latency_ms: 1000,
            auto_reconnect: true,
        }
    }
}

impl KeepaliveConfig {
    /// Create config for aggressive keepalive
    pub fn aggressive() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(2),
            failure_threshold: 2,
            reconnect_delay: Duration::from_secs(2),
            ..Default::default()
        }
    }

    /// Create config for relaxed keepalive
    pub fn relaxed() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(60),
            heartbeat_timeout: Duration::from_secs(10),
            failure_threshold: 5,
            reconnect_delay: Duration::from_secs(10),
            ..Default::default()
        }
    }
}

/// Connection information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Provider name
    pub provider: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Current state
    pub state: ConnectionState,
    /// Last successful heartbeat
    pub last_heartbeat: Option<Instant>,
    /// Last heartbeat latency
    pub last_latency: Option<Duration>,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Total reconnection attempts
    pub reconnect_attempts: u32,
    /// Connection established time
    pub connected_since: Option<Instant>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ConnectionInfo {
    fn new(provider: String, endpoint: String) -> Self {
        Self {
            provider,
            endpoint,
            state: ConnectionState::Disconnected,
            last_heartbeat: None,
            last_latency: None,
            consecutive_failures: 0,
            reconnect_attempts: 0,
            connected_since: None,
            metadata: HashMap::new(),
        }
    }

    /// Get uptime if connected
    pub fn uptime(&self) -> Option<Duration> {
        self.connected_since.map(|t| t.elapsed())
    }

    /// Get time since last heartbeat
    pub fn time_since_heartbeat(&self) -> Option<Duration> {
        self.last_heartbeat.map(|t| t.elapsed())
    }
}

/// Heartbeat result
#[derive(Debug, Clone)]
pub struct HeartbeatResult {
    /// Success or failure
    pub success: bool,
    /// Response latency
    pub latency: Duration,
    /// Error message if failed
    pub error: Option<String>,
    /// Response details
    pub details: Option<String>,
}

/// Keepalive event
#[derive(Debug, Clone)]
pub enum KeepaliveEvent {
    /// Connection established
    Connected { provider: String },
    /// Connection lost
    Disconnected { provider: String, reason: String },
    /// Connection degraded
    Degraded { provider: String, latency_ms: u64 },
    /// Connection recovered from degraded
    Recovered { provider: String },
    /// Heartbeat received
    Heartbeat { provider: String, latency_ms: u64 },
    /// Reconnection attempt
    Reconnecting { provider: String, attempt: u32 },
    /// Reconnection failed
    ReconnectFailed { provider: String, reason: String },
}

/// Event callback type
pub type EventCallback = Box<dyn Fn(KeepaliveEvent) + Send + Sync>;

/// Keepalive manager
pub struct KeepaliveManager {
    config: KeepaliveConfig,
    connections: RwLock<HashMap<String, ConnectionInfo>>,
    running: Arc<Mutex<bool>>,
    callbacks: RwLock<Vec<EventCallback>>,
}

impl KeepaliveManager {
    /// Create a new keepalive manager
    pub fn new(config: KeepaliveConfig) -> Self {
        Self {
            config,
            connections: RwLock::new(HashMap::new()),
            running: Arc::new(Mutex::new(false)),
            callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Register a connection to monitor
    pub fn register(&self, provider: &str, endpoint: &str) {
        let mut connections = self.connections.write().unwrap();
        connections.insert(
            provider.to_string(),
            ConnectionInfo::new(provider.to_string(), endpoint.to_string()),
        );
    }

    /// Unregister a connection
    pub fn unregister(&self, provider: &str) {
        let mut connections = self.connections.write().unwrap();
        connections.remove(provider);
    }

    /// Get connection state
    pub fn get_state(&self, provider: &str) -> Option<ConnectionState> {
        let connections = self.connections.read().unwrap();
        connections.get(provider).map(|c| c.state)
    }

    /// Get connection info
    pub fn get_info(&self, provider: &str) -> Option<ConnectionInfo> {
        let connections = self.connections.read().unwrap();
        connections.get(provider).cloned()
    }

    /// Get all connections
    pub fn all_connections(&self) -> Vec<ConnectionInfo> {
        let connections = self.connections.read().unwrap();
        connections.values().cloned().collect()
    }

    /// Mark connection as connected
    pub fn mark_connected(&self, provider: &str) {
        let mut connections = self.connections.write().unwrap();
        if let Some(conn) = connections.get_mut(provider) {
            conn.state = ConnectionState::Connected;
            conn.connected_since = Some(Instant::now());
            conn.consecutive_failures = 0;
            conn.last_heartbeat = Some(Instant::now());
        }

        self.emit_event(KeepaliveEvent::Connected {
            provider: provider.to_string(),
        });
    }

    /// Mark connection as disconnected
    pub fn mark_disconnected(&self, provider: &str, reason: &str) {
        let mut connections = self.connections.write().unwrap();
        if let Some(conn) = connections.get_mut(provider) {
            conn.state = ConnectionState::Disconnected;
            conn.connected_since = None;
        }

        self.emit_event(KeepaliveEvent::Disconnected {
            provider: provider.to_string(),
            reason: reason.to_string(),
        });
    }

    /// Record heartbeat result
    pub fn record_heartbeat(&self, provider: &str, result: HeartbeatResult) {
        let mut connections = self.connections.write().unwrap();
        if let Some(conn) = connections.get_mut(provider) {
            if result.success {
                conn.last_heartbeat = Some(Instant::now());
                conn.last_latency = Some(result.latency);
                conn.consecutive_failures = 0;

                // Check for degraded state
                let latency_ms = result.latency.as_millis() as u64;
                if latency_ms > self.config.degraded_latency_ms {
                    if conn.state == ConnectionState::Connected {
                        conn.state = ConnectionState::Degraded;
                        drop(connections);
                        self.emit_event(KeepaliveEvent::Degraded {
                            provider: provider.to_string(),
                            latency_ms,
                        });
                        return;
                    }
                } else if conn.state == ConnectionState::Degraded {
                    conn.state = ConnectionState::Connected;
                    drop(connections);
                    self.emit_event(KeepaliveEvent::Recovered {
                        provider: provider.to_string(),
                    });
                    return;
                }

                drop(connections);
                self.emit_event(KeepaliveEvent::Heartbeat {
                    provider: provider.to_string(),
                    latency_ms,
                });
            } else {
                conn.consecutive_failures += 1;

                if conn.consecutive_failures >= self.config.failure_threshold {
                    conn.state = ConnectionState::Failed;
                    drop(connections);
                    self.emit_event(KeepaliveEvent::Disconnected {
                        provider: provider.to_string(),
                        reason: result.error.unwrap_or_else(|| "Heartbeat failed".to_string()),
                    });
                }
            }
        }
    }

    /// Attempt reconnection
    pub fn attempt_reconnect(&self, provider: &str) -> bool {
        let should_reconnect = {
            let mut connections = self.connections.write().unwrap();
            if let Some(conn) = connections.get_mut(provider) {
                if conn.reconnect_attempts >= self.config.max_reconnect_attempts {
                    return false;
                }

                conn.reconnect_attempts += 1;
                conn.state = ConnectionState::Connecting;
                true
            } else {
                false
            }
        };

        if should_reconnect {
            let attempts = {
                let connections = self.connections.read().unwrap();
                connections.get(provider).map(|c| c.reconnect_attempts).unwrap_or(0)
            };

            self.emit_event(KeepaliveEvent::Reconnecting {
                provider: provider.to_string(),
                attempt: attempts,
            });
        }

        should_reconnect
    }

    /// Reset reconnection counter
    pub fn reset_reconnect_counter(&self, provider: &str) {
        let mut connections = self.connections.write().unwrap();
        if let Some(conn) = connections.get_mut(provider) {
            conn.reconnect_attempts = 0;
        }
    }

    /// Add event callback
    pub fn on_event(&self, callback: EventCallback) {
        let mut callbacks = self.callbacks.write().unwrap();
        callbacks.push(callback);
    }

    fn emit_event(&self, event: KeepaliveEvent) {
        let callbacks = self.callbacks.read().unwrap();
        for callback in callbacks.iter() {
            callback(event.clone());
        }
    }

    /// Check if any connection needs attention
    pub fn needs_attention(&self) -> Vec<String> {
        let connections = self.connections.read().unwrap();
        connections.iter()
            .filter(|(_, conn)| conn.state.needs_action())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get healthy connections
    pub fn healthy_connections(&self) -> Vec<String> {
        let connections = self.connections.read().unwrap();
        connections.iter()
            .filter(|(_, conn)| conn.state == ConnectionState::Connected)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get summary statistics
    pub fn stats(&self) -> KeepaliveStats {
        let connections = self.connections.read().unwrap();

        let mut stats = KeepaliveStats::default();
        stats.total_connections = connections.len();

        for conn in connections.values() {
            match conn.state {
                ConnectionState::Connected => stats.connected += 1,
                ConnectionState::Degraded => stats.degraded += 1,
                ConnectionState::Failed | ConnectionState::Disconnected => stats.disconnected += 1,
                ConnectionState::Connecting => stats.connecting += 1,
                ConnectionState::Closing => {}
            }

            if let Some(latency) = conn.last_latency {
                stats.total_latency_ms += latency.as_millis() as u64;
                stats.latency_samples += 1;
            }
        }

        if stats.latency_samples > 0 {
            stats.avg_latency_ms = stats.total_latency_ms / stats.latency_samples;
        }

        stats
    }

    /// Perform a heartbeat check on a provider
    pub fn heartbeat(&self, provider: &str) -> Option<HeartbeatResult> {
        let endpoint = {
            let connections = self.connections.read().unwrap();
            connections.get(provider).map(|c| c.endpoint.clone())
        }?;

        let start = Instant::now();
        let result = self.do_heartbeat(&endpoint);
        let latency = start.elapsed();

        let heartbeat_result = HeartbeatResult {
            success: result.is_ok(),
            latency,
            error: result.err(),
            details: None,
        };

        self.record_heartbeat(provider, heartbeat_result.clone());

        Some(heartbeat_result)
    }

    fn do_heartbeat(&self, endpoint: &str) -> Result<(), String> {
        // Simple HTTP GET to check if endpoint is alive
        let url = if endpoint.ends_with('/') {
            format!("{}api/version", endpoint)
        } else {
            format!("{}/api/version", endpoint)
        };

        match ureq::get(&url)
            .timeout(self.config.heartbeat_timeout)
            .call()
        {
            Ok(_) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Start the keepalive monitoring loop
    pub fn start(&self) -> KeepaliveHandle {
        let running = self.running.clone();
        *running.lock().unwrap() = true;

        KeepaliveHandle { running }
    }

    /// Stop the keepalive monitoring
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }

    /// Check if monitoring is running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

impl Default for KeepaliveManager {
    fn default() -> Self {
        Self::new(KeepaliveConfig::default())
    }
}

/// Handle for stopping keepalive
pub struct KeepaliveHandle {
    running: Arc<Mutex<bool>>,
}

impl KeepaliveHandle {
    /// Stop the keepalive monitoring
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }

    /// Check if still running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

impl Drop for KeepaliveHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Keepalive statistics
#[derive(Debug, Clone, Default)]
pub struct KeepaliveStats {
    /// Total connections monitored
    pub total_connections: usize,
    /// Currently connected
    pub connected: usize,
    /// Currently degraded
    pub degraded: usize,
    /// Currently disconnected/failed
    pub disconnected: usize,
    /// Currently connecting
    pub connecting: usize,
    /// Average latency (ms)
    pub avg_latency_ms: u64,
    /// Total latency samples
    pub latency_samples: u64,
    /// Total latency (for calculation)
    total_latency_ms: u64,
}

/// Simple connection monitor for single provider
pub struct ConnectionMonitor {
    endpoint: String,
    state: Arc<RwLock<ConnectionState>>,
    last_check: Arc<RwLock<Option<Instant>>>,
    config: KeepaliveConfig,
}

impl ConnectionMonitor {
    /// Create a new connection monitor
    pub fn new(endpoint: &str, config: KeepaliveConfig) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            last_check: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Get current state
    pub fn state(&self) -> ConnectionState {
        *self.state.read().unwrap()
    }

    /// Check connection now
    pub fn check(&self) -> bool {
        let url = if self.endpoint.ends_with('/') {
            format!("{}api/version", self.endpoint)
        } else {
            format!("{}/api/version", self.endpoint)
        };

        let result = ureq::get(&url)
            .timeout(self.config.heartbeat_timeout)
            .call();

        let success = result.is_ok();

        {
            let mut state = self.state.write().unwrap();
            *state = if success {
                ConnectionState::Connected
            } else {
                ConnectionState::Disconnected
            };
        }

        {
            let mut last_check = self.last_check.write().unwrap();
            *last_check = Some(Instant::now());
        }

        success
    }

    /// Check if monitoring should run
    pub fn should_check(&self) -> bool {
        let last_check = self.last_check.read().unwrap();
        match *last_check {
            None => true,
            Some(last) => last.elapsed() >= self.config.heartbeat_interval,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_state() {
        assert!(ConnectionState::Connected.is_usable());
        assert!(ConnectionState::Degraded.is_usable());
        assert!(!ConnectionState::Disconnected.is_usable());

        assert!(ConnectionState::Disconnected.needs_action());
        assert!(ConnectionState::Failed.needs_action());
        assert!(!ConnectionState::Connected.needs_action());
    }

    #[test]
    fn test_manager_registration() {
        let manager = KeepaliveManager::default();

        manager.register("test", "http://localhost:8080");
        assert_eq!(manager.get_state("test"), Some(ConnectionState::Disconnected));

        manager.mark_connected("test");
        assert_eq!(manager.get_state("test"), Some(ConnectionState::Connected));

        manager.unregister("test");
        assert_eq!(manager.get_state("test"), None);
    }

    #[test]
    fn test_heartbeat_recording() {
        let manager = KeepaliveManager::default();
        manager.register("test", "http://localhost:8080");
        manager.mark_connected("test");

        // Successful heartbeat
        manager.record_heartbeat("test", HeartbeatResult {
            success: true,
            latency: Duration::from_millis(50),
            error: None,
            details: None,
        });

        let info = manager.get_info("test").unwrap();
        assert_eq!(info.consecutive_failures, 0);
        assert!(info.last_latency.is_some());
    }

    #[test]
    fn test_failure_threshold() {
        let config = KeepaliveConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let manager = KeepaliveManager::new(config);
        manager.register("test", "http://localhost:8080");
        manager.mark_connected("test");

        // First failure
        manager.record_heartbeat("test", HeartbeatResult {
            success: false,
            latency: Duration::from_millis(5000),
            error: Some("timeout".to_string()),
            details: None,
        });
        assert_eq!(manager.get_state("test"), Some(ConnectionState::Connected));

        // Second failure - should mark as failed
        manager.record_heartbeat("test", HeartbeatResult {
            success: false,
            latency: Duration::from_millis(5000),
            error: Some("timeout".to_string()),
            details: None,
        });
        assert_eq!(manager.get_state("test"), Some(ConnectionState::Failed));
    }

    #[test]
    fn test_stats() {
        let manager = KeepaliveManager::default();
        manager.register("p1", "http://localhost:8080");
        manager.register("p2", "http://localhost:8081");
        manager.mark_connected("p1");

        let stats = manager.stats();
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.connected, 1);
        assert_eq!(stats.disconnected, 1);
    }
}
