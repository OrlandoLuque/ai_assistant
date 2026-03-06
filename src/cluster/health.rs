//! # Cluster Health (Phase 11)
//!
//! Health management for cluster nodes:
//! - **Readiness probe**: `/health/ready` — node is synced and accepting traffic
//! - **Liveness probe**: `/health/live` — node process is alive
//! - **Graceful drain**: Stop accepting new requests, finish in-flight, leave ring
//! - **Circuit breaker**: Track per-peer failures, open after N consecutive failures
//! - **Backpressure**: Return 503 when request queue exceeds threshold
//! - **Latency tracking**: Sliding window P95/P99, marks degraded when slow
//! - **System load monitoring**: CPU% and memory%, marks degraded when saturated

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

/// Manages node health state, drain mode, circuit breakers, latency tracking,
/// and system load monitoring.
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
    /// Latency tracker for degradation detection.
    latency_tracker: LatencyTracker,
    /// System load monitor for CPU/memory pressure detection.
    system_load: SystemLoadMonitor,
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
            latency_tracker: LatencyTracker::new(
                1000,                            // window: last 1000 requests
                Duration::from_secs(5),          // P95 threshold: 5s
                Duration::from_secs(10),         // P99 threshold: 10s
            ),
            system_load: SystemLoadMonitor::new(0.90, 0.85), // CPU 90%, memory 85%
        }
    }

    /// Create with custom degradation thresholds.
    pub fn with_thresholds(
        node_id: String,
        heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
        latency_p95_threshold: Duration,
        latency_p99_threshold: Duration,
        cpu_threshold: f64,
        memory_threshold: f64,
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
            latency_tracker: LatencyTracker::new(1000, latency_p95_threshold, latency_p99_threshold),
            system_load: SystemLoadMonitor::new(cpu_threshold, memory_threshold),
        }
    }

    /// Check if the node is ready (synced, not draining, not overloaded, not degraded).
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
            && !self.draining.load(Ordering::Relaxed)
            && !self.is_overloaded()
            && !self.is_degraded()
    }

    /// Check if the node process is alive (always true if this code runs).
    pub fn is_alive(&self) -> bool {
        true
    }

    /// Check if the node is degraded (high latency or high system load).
    pub fn is_degraded(&self) -> bool {
        self.latency_tracker.is_degraded() || self.system_load.is_overloaded()
    }

    /// Get the current degradation reason, if any.
    pub fn degradation_reason(&self) -> Option<String> {
        let mut reasons = Vec::new();
        if let Some(p95) = self.latency_tracker.p95() {
            if p95 >= self.latency_tracker.p95_threshold {
                reasons.push(format!("latency_p95={:.0}ms (threshold={:.0}ms)",
                    p95.as_millis(), self.latency_tracker.p95_threshold.as_millis()));
            }
        }
        if let Some(p99) = self.latency_tracker.p99() {
            if p99 >= self.latency_tracker.p99_threshold {
                reasons.push(format!("latency_p99={:.0}ms (threshold={:.0}ms)",
                    p99.as_millis(), self.latency_tracker.p99_threshold.as_millis()));
            }
        }
        let load = self.system_load.current_load();
        if load.cpu_percent >= self.system_load.cpu_threshold * 100.0 {
            reasons.push(format!("cpu={:.1}% (threshold={:.0}%)",
                load.cpu_percent, self.system_load.cpu_threshold * 100.0));
        }
        if load.memory_percent >= self.system_load.memory_threshold * 100.0 {
            reasons.push(format!("memory={:.1}% (threshold={:.0}%)",
                load.memory_percent, self.system_load.memory_threshold * 100.0));
        }
        if reasons.is_empty() { None } else { Some(reasons.join(", ")) }
    }

    /// Record a completed request latency.
    pub fn record_latency(&self, latency: Duration) {
        self.latency_tracker.record(latency);
    }

    /// Get a reference to the latency tracker.
    pub fn latency_tracker(&self) -> &LatencyTracker {
        &self.latency_tracker
    }

    /// Get a reference to the system load monitor.
    pub fn system_load(&self) -> &SystemLoadMonitor {
        &self.system_load
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
            degraded: self.is_degraded(),
            degradation_reason: self.degradation_reason(),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            latency_p95_ms: self.latency_tracker.p95().map(|d| d.as_millis() as u64),
            latency_p99_ms: self.latency_tracker.p99().map(|d| d.as_millis() as u64),
            cpu_percent: self.system_load.current_load().cpu_percent,
            memory_percent: self.system_load.current_load().memory_percent,
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
// Latency Tracker
// ============================================================================

/// Sliding-window latency tracker using a lock-free ring buffer.
///
/// Records request latencies and computes P95/P99 percentiles on demand.
/// When percentiles exceed configured thresholds, the node is considered degraded.
pub struct LatencyTracker {
    /// Ring buffer of latency samples (microseconds for precision, stored as u64).
    samples: Vec<AtomicU64>,
    /// Current write position in the ring buffer.
    write_pos: AtomicU64,
    /// Number of samples recorded (saturates at buffer size).
    count: AtomicU64,
    /// P95 latency threshold — above this, node is degraded.
    pub p95_threshold: Duration,
    /// P99 latency threshold — above this, node is degraded.
    pub p99_threshold: Duration,
}

impl LatencyTracker {
    /// Create a new tracker with the given window size and thresholds.
    pub fn new(window_size: usize, p95_threshold: Duration, p99_threshold: Duration) -> Self {
        let mut samples = Vec::with_capacity(window_size);
        for _ in 0..window_size {
            samples.push(AtomicU64::new(0));
        }
        Self {
            samples,
            write_pos: AtomicU64::new(0),
            count: AtomicU64::new(0),
            p95_threshold,
            p99_threshold,
        }
    }

    /// Record a request latency.
    pub fn record(&self, latency: Duration) {
        let micros = latency.as_micros() as u64;
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed) as usize % self.samples.len();
        self.samples[pos].store(micros, Ordering::Relaxed);
        // Saturate count at window size.
        let prev = self.count.fetch_add(1, Ordering::Relaxed);
        if prev >= self.samples.len() as u64 {
            self.count.store(self.samples.len() as u64, Ordering::Relaxed);
        }
    }

    /// Get the number of samples recorded (capped at window size).
    pub fn sample_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed).min(self.samples.len() as u64)
    }

    /// Compute a percentile (0.0 to 1.0) from the current samples.
    /// Returns `None` if no samples have been recorded.
    fn percentile(&self, pct: f64) -> Option<Duration> {
        let n = self.sample_count() as usize;
        if n == 0 {
            return None;
        }
        // Collect current samples into a sortable vec.
        let mut vals: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            vals.push(self.samples[i].load(Ordering::Relaxed));
        }
        vals.sort_unstable();
        let idx = ((pct * n as f64) as usize).min(n - 1);
        Some(Duration::from_micros(vals[idx]))
    }

    /// Compute P50 (median) latency.
    pub fn p50(&self) -> Option<Duration> {
        self.percentile(0.50)
    }

    /// Compute P95 latency.
    pub fn p95(&self) -> Option<Duration> {
        self.percentile(0.95)
    }

    /// Compute P99 latency.
    pub fn p99(&self) -> Option<Duration> {
        self.percentile(0.99)
    }

    /// Check if latency indicates degradation.
    pub fn is_degraded(&self) -> bool {
        if let Some(p95) = self.p95() {
            if p95 >= self.p95_threshold {
                return true;
            }
        }
        if let Some(p99) = self.p99() {
            if p99 >= self.p99_threshold {
                return true;
            }
        }
        false
    }

    /// Get a snapshot of current latency stats.
    pub fn stats(&self) -> LatencyStats {
        LatencyStats {
            sample_count: self.sample_count(),
            p50_ms: self.p50().map(|d| d.as_millis() as u64),
            p95_ms: self.p95().map(|d| d.as_millis() as u64),
            p99_ms: self.p99().map(|d| d.as_millis() as u64),
            degraded: self.is_degraded(),
        }
    }
}

/// Snapshot of latency statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub sample_count: u64,
    pub p50_ms: Option<u64>,
    pub p95_ms: Option<u64>,
    pub p99_ms: Option<u64>,
    pub degraded: bool,
}

// ============================================================================
// System Load Monitor
// ============================================================================

/// Monitors CPU and memory usage of the host system.
///
/// Uses OS-specific APIs:
/// - **Linux/macOS**: Reads `/proc/stat` and `/proc/meminfo` (Linux) or `sysctl` (macOS)
/// - **Windows**: Uses `GetSystemTimes` and `GlobalMemoryStatusEx`
/// - **Fallback**: Returns 0% if the OS is unsupported
///
/// Thresholds are configurable. When exceeded, the node is marked degraded.
pub struct SystemLoadMonitor {
    /// CPU usage threshold (0.0 to 1.0, e.g., 0.90 = 90%).
    pub cpu_threshold: f64,
    /// Memory usage threshold (0.0 to 1.0, e.g., 0.85 = 85%).
    pub memory_threshold: f64,
    /// Last CPU measurement (for delta calculation).
    last_cpu: std::sync::Mutex<Option<CpuSample>>,
}

/// A point-in-time CPU sample for delta computation.
#[derive(Clone)]
struct CpuSample {
    /// Total CPU time (user + system + idle + ...).
    total: u64,
    /// Idle CPU time.
    idle: u64,
}

/// Current system load snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoad {
    /// CPU usage as a percentage (0.0 to 100.0).
    pub cpu_percent: f64,
    /// Memory usage as a percentage (0.0 to 100.0).
    pub memory_percent: f64,
}

impl SystemLoadMonitor {
    /// Create a new monitor with the given thresholds.
    pub fn new(cpu_threshold: f64, memory_threshold: f64) -> Self {
        let monitor = Self {
            cpu_threshold,
            memory_threshold,
            last_cpu: std::sync::Mutex::new(None),
        };
        // Take an initial CPU sample so the first delta has a baseline.
        if let Some(sample) = Self::read_cpu_sample() {
            *monitor.last_cpu.lock().unwrap_or_else(|e| e.into_inner()) = Some(sample);
        }
        monitor
    }

    /// Check if the system is overloaded (CPU or memory above threshold).
    pub fn is_overloaded(&self) -> bool {
        let load = self.current_load();
        load.cpu_percent >= self.cpu_threshold * 100.0
            || load.memory_percent >= self.memory_threshold * 100.0
    }

    /// Get the current system load.
    pub fn current_load(&self) -> SystemLoad {
        SystemLoad {
            cpu_percent: self.cpu_percent(),
            memory_percent: Self::memory_percent(),
        }
    }

    /// Read CPU usage as a percentage (delta between two samples).
    ///
    /// Returns 0.0 if the delta is too small for a meaningful reading
    /// (avoids noisy 100% spikes on very short measurement intervals).
    fn cpu_percent(&self) -> f64 {
        let new_sample = match Self::read_cpu_sample() {
            Some(s) => s,
            None => return 0.0,
        };
        let mut guard = self.last_cpu.lock().unwrap_or_else(|e| e.into_inner());
        let pct = if let Some(ref prev) = *guard {
            let total_delta = new_sample.total.saturating_sub(prev.total);
            let idle_delta = new_sample.idle.saturating_sub(prev.idle);
            // Require a minimum delta to avoid noisy readings on short intervals.
            // On Windows, FILETIME ticks are 100ns units; on Linux, jiffies (~10ms).
            // A delta of <1000 means the interval was too short for meaningful data.
            if total_delta < 1000 {
                0.0
            } else {
                ((total_delta - idle_delta) as f64 / total_delta as f64) * 100.0
            }
        } else {
            0.0
        };
        *guard = Some(new_sample);
        pct
    }

    /// Read a CPU sample from the OS.
    #[cfg(target_os = "linux")]
    fn read_cpu_sample() -> Option<CpuSample> {
        let content = std::fs::read_to_string("/proc/stat").ok()?;
        let line = content.lines().next()?;
        if !line.starts_with("cpu ") {
            return None;
        }
        let fields: Vec<u64> = line.split_whitespace()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();
        if fields.len() < 4 {
            return None;
        }
        // fields: user, nice, system, idle, iowait, irq, softirq, steal, ...
        let total: u64 = fields.iter().sum();
        let idle = fields[3] + fields.get(4).copied().unwrap_or(0); // idle + iowait
        Some(CpuSample { total, idle })
    }

    #[cfg(target_os = "windows")]
    fn read_cpu_sample() -> Option<CpuSample> {
        // Use GetSystemTimes via windows-sys or manual FFI.
        // GetSystemTimes returns FILETIME for idle, kernel, user.
        #[repr(C)]
        struct FileTime { lo: u32, hi: u32 }
        extern "system" {
            fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        }
        unsafe {
            let (mut idle_ft, mut kernel_ft, mut user_ft) = (
                FileTime { lo: 0, hi: 0 },
                FileTime { lo: 0, hi: 0 },
                FileTime { lo: 0, hi: 0 },
            );
            if GetSystemTimes(&mut idle_ft, &mut kernel_ft, &mut user_ft) == 0 {
                return None;
            }
            let idle = (idle_ft.hi as u64) << 32 | idle_ft.lo as u64;
            let kernel = (kernel_ft.hi as u64) << 32 | kernel_ft.lo as u64;
            let user = (user_ft.hi as u64) << 32 | user_ft.lo as u64;
            // kernel includes idle time
            let total = kernel + user;
            Some(CpuSample { total, idle })
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn read_cpu_sample() -> Option<CpuSample> {
        None // Unsupported OS — returns 0% CPU
    }

    /// Read memory usage percentage.
    #[cfg(target_os = "linux")]
    fn memory_percent() -> f64 {
        let content = match std::fs::read_to_string("/proc/meminfo") {
            Ok(c) => c,
            Err(_) => return 0.0,
        };
        let mut total_kb: u64 = 0;
        let mut available_kb: u64 = 0;
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                available_kb = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }
        if total_kb == 0 { return 0.0; }
        ((total_kb - available_kb) as f64 / total_kb as f64) * 100.0
    }

    #[cfg(target_os = "windows")]
    fn memory_percent() -> f64 {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
        }
        unsafe {
            let mut status = std::mem::zeroed::<MemoryStatusEx>();
            status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
            if GlobalMemoryStatusEx(&mut status) == 0 {
                return 0.0;
            }
            status.memory_load as f64 // Already a percentage (0-100)
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn memory_percent() -> f64 {
        0.0 // Unsupported OS
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
    pub degraded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degradation_reason: Option<String>,
    pub in_flight: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p95_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p99_ms: Option<u64>,
    pub cpu_percent: f64,
    pub memory_percent: f64,
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
        // Use permissive thresholds (100%) so system load doesn't interfere with tests.
        ClusterHealthManager::with_thresholds(
            "test-node".to_string(),
            Arc::new(RwLock::new(hb)),
            Duration::from_secs(60),   // P95 threshold: 60s (won't trigger)
            Duration::from_secs(120),  // P99 threshold: 120s (won't trigger)
            1.0,                        // CPU: 100% (won't trigger)
            1.0,                        // Memory: 100% (won't trigger)
        )
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
        assert!(!info.degraded);
        assert!(info.degradation_reason.is_none());
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
            degraded: false,
            degradation_reason: None,
            in_flight: 42,
            latency_p95_ms: Some(150),
            latency_p99_ms: Some(500),
            cpu_percent: 45.2,
            memory_percent: 62.1,
            uptime_secs: 100,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"ready\":true"));
        assert!(json.contains("\"in_flight\":42"));
        assert!(json.contains("\"cpu_percent\":"));
        assert!(json.contains("\"memory_percent\":"));
    }

    // ================================================================
    // Latency Tracker tests
    // ================================================================

    #[test]
    fn test_latency_tracker_empty() {
        let lt = LatencyTracker::new(100, Duration::from_secs(5), Duration::from_secs(10));
        assert_eq!(lt.sample_count(), 0);
        assert!(lt.p50().is_none());
        assert!(lt.p95().is_none());
        assert!(lt.p99().is_none());
        assert!(!lt.is_degraded());
    }

    #[test]
    fn test_latency_tracker_single_sample() {
        let lt = LatencyTracker::new(100, Duration::from_secs(5), Duration::from_secs(10));
        lt.record(Duration::from_millis(100));
        assert_eq!(lt.sample_count(), 1);
        assert!(lt.p50().is_some());
    }

    #[test]
    fn test_latency_tracker_percentiles() {
        let lt = LatencyTracker::new(100, Duration::from_secs(5), Duration::from_secs(10));
        // Record 100 samples from 1ms to 100ms
        for i in 1..=100 {
            lt.record(Duration::from_millis(i));
        }
        assert_eq!(lt.sample_count(), 100);

        let p50 = lt.p50().unwrap();
        let p95 = lt.p95().unwrap();
        let p99 = lt.p99().unwrap();

        // P50 should be around 50ms
        assert!(p50.as_millis() >= 45 && p50.as_millis() <= 55,
            "P50 was {}ms", p50.as_millis());
        // P95 should be around 95ms
        assert!(p95.as_millis() >= 90 && p95.as_millis() <= 100,
            "P95 was {}ms", p95.as_millis());
        // P99 should be around 99ms
        assert!(p99.as_millis() >= 95 && p99.as_millis() <= 100,
            "P99 was {}ms", p99.as_millis());
    }

    #[test]
    fn test_latency_tracker_degraded_when_slow() {
        // P95 threshold = 50ms
        let lt = LatencyTracker::new(100, Duration::from_millis(50), Duration::from_secs(10));
        // Record all fast samples
        for _ in 0..100 {
            lt.record(Duration::from_millis(10));
        }
        assert!(!lt.is_degraded());

        // Now record slow samples to push P95 above threshold
        for _ in 0..100 {
            lt.record(Duration::from_millis(200));
        }
        assert!(lt.is_degraded());
    }

    #[test]
    fn test_latency_tracker_ring_buffer_wraps() {
        let lt = LatencyTracker::new(10, Duration::from_secs(5), Duration::from_secs(10));
        // Write more than window size
        for i in 0..50 {
            lt.record(Duration::from_millis(i));
        }
        // Count should be capped at window size
        assert_eq!(lt.sample_count(), 10);
    }

    #[test]
    fn test_latency_stats_snapshot() {
        let lt = LatencyTracker::new(100, Duration::from_secs(5), Duration::from_secs(10));
        for i in 1..=20 {
            lt.record(Duration::from_millis(i));
        }
        let stats = lt.stats();
        assert_eq!(stats.sample_count, 20);
        assert!(stats.p50_ms.is_some());
        assert!(stats.p95_ms.is_some());
        assert!(!stats.degraded);
    }

    // ================================================================
    // System Load Monitor tests
    // ================================================================

    #[test]
    fn test_system_load_monitor_creation() {
        let monitor = SystemLoadMonitor::new(0.90, 0.85);
        assert!((monitor.cpu_threshold - 0.90).abs() < f64::EPSILON);
        assert!((monitor.memory_threshold - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_system_load_returns_values() {
        let monitor = SystemLoadMonitor::new(0.90, 0.85);
        let load = monitor.current_load();
        // CPU and memory should be non-negative percentages
        assert!(load.cpu_percent >= 0.0);
        assert!(load.memory_percent >= 0.0);
        assert!(load.cpu_percent <= 100.0);
        // Memory percent should be reasonable (0-100)
        assert!(load.memory_percent <= 100.0);
    }

    #[test]
    fn test_system_load_serialization() {
        let load = SystemLoad {
            cpu_percent: 45.5,
            memory_percent: 72.3,
        };
        let json = serde_json::to_string(&load).unwrap();
        assert!(json.contains("45.5"));
        assert!(json.contains("72.3"));
    }

    #[test]
    fn test_system_load_not_overloaded_with_high_threshold() {
        // With 100% thresholds, should never be overloaded
        let monitor = SystemLoadMonitor::new(1.0, 1.0);
        assert!(!monitor.is_overloaded());
    }

    // ================================================================
    // Integration: degradation in health manager
    // ================================================================

    #[test]
    fn test_health_manager_not_degraded_by_default() {
        let hm = make_health_manager();
        assert!(!hm.is_degraded());
        assert!(hm.is_ready());
        assert!(hm.degradation_reason().is_none());
    }

    #[test]
    fn test_health_manager_degraded_by_latency() {
        let hb = HeartbeatManager::new(HeartbeatConfig {
            interval: Duration::from_secs(1),
            phi_threshold: 12.0,
            max_samples: 100,
            suspicious_threshold: 6.0,
        });
        let hm = ClusterHealthManager::with_thresholds(
            "test".to_string(),
            Arc::new(RwLock::new(hb)),
            Duration::from_millis(50),  // very low P95 threshold
            Duration::from_millis(100), // very low P99 threshold
            1.0,  // CPU threshold 100% (won't trigger)
            1.0,  // Mem threshold 100% (won't trigger)
        );
        // Fast requests → not degraded
        for _ in 0..100 {
            hm.record_latency(Duration::from_millis(10));
        }
        assert!(!hm.is_degraded());
        assert!(hm.is_ready());

        // Slow requests → degraded
        for _ in 0..100 {
            hm.record_latency(Duration::from_millis(200));
        }
        assert!(hm.is_degraded());
        assert!(!hm.is_ready());
        assert!(hm.degradation_reason().is_some());
        let reason = hm.degradation_reason().unwrap();
        assert!(reason.contains("latency_p95"), "reason was: {}", reason);
    }

    #[test]
    fn test_health_manager_records_latency() {
        let hm = make_health_manager();
        hm.record_latency(Duration::from_millis(50));
        hm.record_latency(Duration::from_millis(100));
        assert_eq!(hm.latency_tracker().sample_count(), 2);
    }
}
