//! # Cluster Health (Phase 11)
//!
//! Health management for cluster nodes:
//! - **Readiness probe**: `/health/ready` — node is synced and accepting traffic
//! - **Liveness probe**: `/health/live` — node process is alive
//! - **Graceful drain**: Stop accepting new requests, finish in-flight, leave ring
//! - **Circuit breaker**: Track per-peer failures, open after N consecutive failures
//! - **Backpressure**: Return 503 when request queue exceeds threshold

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::distributed::NodeId;
use crate::failure_detector::HeartbeatManager;

// ============================================================================
// ClusterHealthManager
// ============================================================================

/// Manages node health state, drain mode, and circuit breakers.
pub struct ClusterHealthManager {
    /// Node identifier.
    node_id: String,
    /// Whether this node is ready to accept traffic.
    ready: AtomicBool,
    /// Whether this node is in drain mode (shutting down gracefully).
    draining: AtomicBool,
    /// Reference to the heartbeat manager for peer health.
    #[allow(dead_code)]
    heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
    /// Per-peer circuit breakers.
    circuit_breakers: DashMap<String, CircuitBreaker>,
    /// In-flight request count for backpressure.
    in_flight: AtomicU64,
    /// Maximum in-flight requests before backpressure.
    max_in_flight: u64,
    /// Time when the node started.
    started_at: Instant,
}

impl ClusterHealthManager {
    /// Create a new health manager.
    pub fn new(
        node_id: String,
        heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
    ) -> Self {
        Self {
            node_id,
            ready: AtomicBool::new(true),
            draining: AtomicBool::new(false),
            heartbeat_mgr,
            circuit_breakers: DashMap::new(),
            in_flight: AtomicU64::new(0),
            max_in_flight: 10000,
            started_at: Instant::now(),
        }
    }

    /// Check if the node is ready (synced, not draining, not overloaded).
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
            && !self.draining.load(Ordering::Relaxed)
            && !self.is_overloaded()
    }

    /// Check if the node process is alive (always true if this code runs).
    pub fn is_alive(&self) -> bool {
        true
    }

    /// Check if we're in drain mode.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Relaxed)
    }

    /// Check if we're overloaded (too many in-flight requests).
    pub fn is_overloaded(&self) -> bool {
        self.in_flight.load(Ordering::Relaxed) >= self.max_in_flight
    }

    /// Set the node as ready/not ready.
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Relaxed);
    }

    /// Start graceful drain (stop accepting new traffic).
    pub fn start_drain(&self) {
        self.draining.store(true, Ordering::Relaxed);
        log::info!("Node {} entering drain mode", self.node_id);
    }

    /// Stop drain mode (resume accepting traffic).
    pub fn stop_drain(&self) {
        self.draining.store(false, Ordering::Relaxed);
        log::info!("Node {} exiting drain mode", self.node_id);
    }

    /// Track a new in-flight request. Returns false if backpressure should apply.
    pub fn track_request(&self) -> bool {
        if self.is_draining() {
            return false;
        }
        let count = self.in_flight.fetch_add(1, Ordering::Relaxed);
        if count >= self.max_in_flight {
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Mark a request as complete.
    pub fn complete_request(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the current in-flight request count.
    pub fn in_flight_count(&self) -> u64 {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Record a peer failure (for circuit breaker).
    pub fn record_peer_failure(&self, peer_id: &NodeId) {
        let key = peer_id.to_hex();
        let mut entry = self.circuit_breakers
            .entry(key)
            .or_insert_with(CircuitBreaker::new);
        entry.record_failure();
    }

    /// Record a peer success (for circuit breaker).
    pub fn record_peer_success(&self, peer_id: &NodeId) {
        let key = peer_id.to_hex();
        if let Some(mut entry) = self.circuit_breakers.get_mut(&key) {
            entry.record_success();
        }
    }

    /// Check if the circuit breaker for a peer is open (should not send traffic).
    pub fn is_circuit_open(&self, peer_id: &NodeId) -> bool {
        let key = peer_id.to_hex();
        self.circuit_breakers
            .get(&key)
            .map(|cb| cb.is_open())
            .unwrap_or(false)
    }

    /// Check if the circuit is half-open (can try one probe request).
    pub fn is_circuit_half_open(&self, peer_id: &NodeId) -> bool {
        let key = peer_id.to_hex();
        self.circuit_breakers
            .get(&key)
            .map(|cb| cb.is_half_open())
            .unwrap_or(false)
    }

    /// Get uptime of this node.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get a health summary for the readiness probe response.
    pub fn readiness_info(&self) -> ReadinessInfo {
        ReadinessInfo {
            ready: self.is_ready(),
            draining: self.is_draining(),
            overloaded: self.is_overloaded(),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }

    /// Get liveness info.
    pub fn liveness_info(&self) -> LivenessInfo {
        LivenessInfo {
            alive: true,
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Per-peer circuit breaker for isolating failing peers.
///
/// States:
/// - **Closed**: Normal operation. After N consecutive failures → Open.
/// - **Open**: No traffic sent. After timeout → Half-Open.
/// - **Half-Open**: One probe request allowed. Success → Closed, Failure → Open.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Consecutive failure count.
    failure_count: u32,
    /// Threshold for opening the circuit.
    failure_threshold: u32,
    /// Current state.
    state: CircuitState,
    /// When the circuit was opened (for half-open timeout).
    opened_at: Option<Instant>,
    /// Timeout before transitioning from Open → Half-Open.
    recovery_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker (closed state).
    pub fn new() -> Self {
        Self {
            failure_count: 0,
            failure_threshold: 5,
            state: CircuitState::Closed,
            opened_at: None,
            recovery_timeout: Duration::from_secs(30),
        }
    }

    /// Create with custom thresholds.
    pub fn with_config(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            failure_count: 0,
            failure_threshold,
            state: CircuitState::Closed,
            opened_at: None,
            recovery_timeout,
        }
    }

    /// Record a failure.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
            self.opened_at = Some(Instant::now());
        }
    }

    /// Record a success.
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = CircuitState::Closed;
        self.opened_at = None;
    }

    /// Check if the circuit is open (don't send traffic).
    pub fn is_open(&self) -> bool {
        match self.state {
            CircuitState::Open => {
                // Check if enough time has passed for half-open
                if let Some(opened) = self.opened_at {
                    if opened.elapsed() >= self.recovery_timeout {
                        return false; // Should be half-open now
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Check if the circuit is half-open (can probe).
    pub fn is_half_open(&self) -> bool {
        match self.state {
            CircuitState::Open => {
                if let Some(opened) = self.opened_at {
                    opened.elapsed() >= self.recovery_timeout
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
            _ => false,
        }
    }

    /// Check if the circuit is closed (normal operation).
    pub fn is_closed(&self) -> bool {
        self.state == CircuitState::Closed
    }

    /// Get consecutive failure count.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Response types
// ============================================================================

/// Readiness probe response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadinessInfo {
    pub ready: bool,
    pub draining: bool,
    pub overloaded: bool,
    pub in_flight: u64,
    pub uptime_secs: u64,
}

/// Liveness probe response.
#[derive(Debug, Serialize, Deserialize)]
pub struct LivenessInfo {
    pub alive: bool,
    pub uptime_secs: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::failure_detector::HeartbeatConfig;

    fn make_health_manager() -> ClusterHealthManager {
        let hb = HeartbeatManager::new(HeartbeatConfig {
            interval: Duration::from_secs(1),
            phi_threshold: 12.0,
            max_samples: 100,
            suspicious_threshold: 6.0,
        });
        ClusterHealthManager::new("test-node".to_string(), Arc::new(RwLock::new(hb)))
    }

    #[test]
    fn test_health_manager_default_ready() {
        let hm = make_health_manager();
        assert!(hm.is_ready());
        assert!(hm.is_alive());
        assert!(!hm.is_draining());
        assert!(!hm.is_overloaded());
    }

    #[test]
    fn test_health_manager_drain_mode() {
        let hm = make_health_manager();
        assert!(hm.is_ready());
        hm.start_drain();
        assert!(!hm.is_ready());
        assert!(hm.is_draining());
        hm.stop_drain();
        assert!(hm.is_ready());
    }

    #[test]
    fn test_health_manager_set_ready() {
        let hm = make_health_manager();
        hm.set_ready(false);
        assert!(!hm.is_ready());
        hm.set_ready(true);
        assert!(hm.is_ready());
    }

    #[test]
    fn test_track_request() {
        let hm = make_health_manager();
        assert!(hm.track_request());
        assert_eq!(hm.in_flight_count(), 1);
        hm.complete_request();
        assert_eq!(hm.in_flight_count(), 0);
    }

    #[test]
    fn test_drain_rejects_requests() {
        let hm = make_health_manager();
        hm.start_drain();
        assert!(!hm.track_request());
    }

    #[test]
    fn test_circuit_breaker_closed_by_default() {
        let cb = CircuitBreaker::new();
        assert!(cb.is_closed());
        assert!(!cb.is_open());
        assert!(!cb.is_half_open());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::with_config(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        cb.record_failure();
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let mut cb = CircuitBreaker::with_config(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert!(cb.is_closed());
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_after_timeout() {
        let mut cb = CircuitBreaker::with_config(2, Duration::from_millis(1));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        std::thread::sleep(Duration::from_millis(5));
        assert!(cb.is_half_open());
    }

    #[test]
    fn test_peer_circuit_breaker() {
        let hm = make_health_manager();
        let peer = NodeId::from_string("peer1");
        assert!(!hm.is_circuit_open(&peer));

        for _ in 0..5 {
            hm.record_peer_failure(&peer);
        }
        assert!(hm.is_circuit_open(&peer));

        hm.record_peer_success(&peer);
        assert!(!hm.is_circuit_open(&peer));
    }

    #[test]
    fn test_readiness_info() {
        let hm = make_health_manager();
        let info = hm.readiness_info();
        assert!(info.ready);
        assert!(!info.draining);
        assert!(!info.overloaded);
        assert_eq!(info.in_flight, 0);
    }

    #[test]
    fn test_liveness_info() {
        let hm = make_health_manager();
        let info = hm.liveness_info();
        assert!(info.alive);
    }

    #[test]
    fn test_readiness_info_serialization() {
        let info = ReadinessInfo {
            ready: true,
            draining: false,
            overloaded: false,
            in_flight: 42,
            uptime_secs: 100,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"ready\":true"));
        assert!(json.contains("\"in_flight\":42"));
    }
}
