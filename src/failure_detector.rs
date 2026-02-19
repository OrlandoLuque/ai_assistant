//! Phi Accrual Failure Detector and heartbeat management.
//!
//! Implements the Phi Accrual Failure Detector algorithm (as used in Apache Cassandra)
//! for detecting node failures in a distributed system. Unlike fixed-timeout detectors,
//! this approach adapts to actual network conditions by tracking heartbeat interval
//! statistics and computing a suspicion level (phi) based on the probability that
//! a heartbeat is late.
//!
//! The phi value represents how many standard deviations the current silence period
//! is from the expected heartbeat interval. Higher phi = more suspicious.
//!
//! This module is gated behind the `distributed-network` feature.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[cfg(feature = "distributed")]
use crate::distributed::NodeId;

// =============================================================================
// Phi Accrual Failure Detector
// =============================================================================

/// Phi Accrual Failure Detector.
///
/// Tracks heartbeat intervals and computes a suspicion level (phi) based on
/// the statistical probability that a heartbeat is overdue. The phi value
/// is calculated as: `phi = -log10(1 - CDF(elapsed_time))` where CDF is
/// the cumulative distribution function of observed heartbeat intervals.
///
/// # Interpretation of phi values
/// - phi < 1: Very likely alive
/// - phi 1-3: Probably alive
/// - phi 3-5: Uncertain
/// - phi 5-8: Suspicious
/// - phi > 8: Very likely dead (default threshold)
///
/// # Example
/// ```ignore
/// let mut detector = PhiAccrualDetector::new(8.0);
/// detector.heartbeat(); // Record first heartbeat
/// // ... time passes ...
/// detector.heartbeat(); // Record another
/// let phi = detector.phi(); // Check suspicion level
/// ```
pub struct PhiAccrualDetector {
    /// Sliding window of recent heartbeat intervals (in seconds).
    intervals: VecDeque<f64>,
    /// Maximum number of samples to keep.
    max_samples: usize,
    /// Timestamp of the last heartbeat received.
    last_heartbeat: Option<Instant>,
    /// Phi threshold above which the node is considered dead.
    phi_threshold: f64,
    /// Minimum number of samples before phi can be computed.
    min_samples: usize,
}

impl PhiAccrualDetector {
    /// Create a new detector with the given phi threshold.
    ///
    /// # Arguments
    /// * `phi_threshold` - Phi value above which the node is considered dead.
    ///   Typical values: 8 (aggressive), 12 (conservative), 16 (very conservative).
    pub fn new(phi_threshold: f64) -> Self {
        Self::with_config(phi_threshold, 200, 5)
    }

    /// Create a detector with full configuration.
    ///
    /// # Arguments
    /// * `phi_threshold` - Dead threshold
    /// * `max_samples` - Maximum heartbeat intervals to track
    /// * `min_samples` - Minimum intervals before phi computation is reliable
    pub fn with_config(phi_threshold: f64, max_samples: usize, min_samples: usize) -> Self {
        Self {
            intervals: VecDeque::with_capacity(max_samples),
            max_samples: max_samples.max(1),
            last_heartbeat: None,
            phi_threshold: phi_threshold.max(0.1),
            min_samples: min_samples.max(1),
        }
    }

    /// Record the arrival of a heartbeat.
    ///
    /// If a previous heartbeat was recorded, the interval between the two
    /// is added to the sliding window.
    pub fn heartbeat(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_heartbeat {
            let interval = now.duration_since(last).as_secs_f64();
            if self.intervals.len() >= self.max_samples {
                self.intervals.pop_front();
            }
            self.intervals.push_back(interval);
        }
        self.last_heartbeat = Some(now);
    }

    /// Compute the current phi (suspicion) value.
    ///
    /// Returns 0.0 if no heartbeats have been recorded or if there aren't
    /// enough samples for a reliable computation. Returns a very high value
    /// if the elapsed time far exceeds the expected interval.
    ///
    /// The formula: `phi = -log10(1 - CDF_normal(elapsed, mean, std_dev))`
    pub fn phi(&self) -> f64 {
        let last = match self.last_heartbeat {
            Some(t) => t,
            None => return 0.0,
        };

        if self.intervals.len() < self.min_samples {
            return 0.0;
        }

        let elapsed = last.elapsed().as_secs_f64();
        let mean = self.mean();
        let std_dev = self.std_dev();

        if std_dev < 1e-10 {
            // All intervals identical; use simple comparison
            if elapsed > mean * 2.0 {
                return self.phi_threshold + 1.0;
            }
            return 0.0;
        }

        let p = Self::cdf_normal(elapsed, mean, std_dev);
        // Clamp p to avoid log10(0)
        let p_clamped = p.min(1.0 - 1e-15);

        if p_clamped <= 0.0 {
            return 0.0;
        }

        -((1.0 - p_clamped).log10())
    }

    /// Check if the node is considered alive (phi < threshold).
    pub fn is_alive(&self) -> bool {
        self.phi() < self.phi_threshold
    }

    /// Check if the node is suspicious (phi > threshold / 2).
    pub fn is_suspicious(&self) -> bool {
        let phi = self.phi();
        phi > self.phi_threshold * 0.5
    }

    /// Duration since the last heartbeat, if any.
    pub fn last_heartbeat_ago(&self) -> Option<Duration> {
        self.last_heartbeat.map(|t| t.elapsed())
    }

    /// Number of heartbeat interval samples collected.
    pub fn sample_count(&self) -> usize {
        self.intervals.len()
    }

    /// Current phi threshold.
    pub fn threshold(&self) -> f64 {
        self.phi_threshold
    }

    /// Mean of collected heartbeat intervals (seconds).
    fn mean(&self) -> f64 {
        if self.intervals.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.intervals.iter().sum();
        sum / self.intervals.len() as f64
    }

    /// Variance of collected heartbeat intervals.
    fn variance(&self) -> f64 {
        if self.intervals.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f64 = self.intervals.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq / (self.intervals.len() - 1) as f64
    }

    /// Standard deviation of collected heartbeat intervals.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Approximation of the cumulative distribution function (CDF)
    /// of the normal distribution.
    ///
    /// Uses the Abramowitz and Stegun approximation (error < 1.5e-7).
    ///
    /// # Arguments
    /// * `x` - The value to evaluate
    /// * `mean` - Distribution mean
    /// * `std_dev` - Distribution standard deviation
    fn cdf_normal(x: f64, mean: f64, std_dev: f64) -> f64 {
        let z = (x - mean) / std_dev;
        // Use the error function approximation
        0.5 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun, formula 7.1.26).
    /// Maximum error: 1.5e-7.
    fn erf(x: f64) -> f64 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        // Constants for the approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let t = 1.0 / (1.0 + p * x);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;

        let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();

        sign * y
    }
}

// =============================================================================
// Node Status
// =============================================================================

/// Status of a monitored node.
#[derive(Debug, Clone)]
pub enum NodeStatus {
    /// Node is responding normally.
    Alive,
    /// Node is suspicious — phi is above the suspicious threshold but below dead.
    Suspicious(f64),
    /// Node is considered dead — phi exceeds the threshold.
    Dead(f64),
    /// No heartbeat data available.
    Unknown,
}

impl NodeStatus {
    /// Whether the node is considered reachable (Alive or Unknown).
    pub fn is_reachable(&self) -> bool {
        matches!(self, NodeStatus::Alive | NodeStatus::Unknown)
    }
}

// =============================================================================
// Heartbeat Configuration
// =============================================================================

/// Configuration for the heartbeat manager.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// How often to send heartbeats.
    pub interval: Duration,
    /// Phi value above which a node is considered dead.
    pub phi_threshold: f64,
    /// Maximum number of heartbeat interval samples per peer.
    pub max_samples: usize,
    /// Phi value above which a node is considered suspicious (typically threshold/2).
    pub suspicious_threshold: f64,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(2),
            phi_threshold: 8.0,
            max_samples: 200,
            suspicious_threshold: 4.0,
        }
    }
}

// =============================================================================
// Heartbeat Manager
// =============================================================================

/// Manages failure detectors for multiple peer nodes.
///
/// Provides a centralized interface for recording heartbeats and
/// querying node status across all monitored peers.
pub struct HeartbeatManager {
    /// Per-node failure detectors.
    detectors: HashMap<NodeId, PhiAccrualDetector>,
    /// Configuration.
    config: HeartbeatConfig,
}

impl HeartbeatManager {
    /// Create a new heartbeat manager.
    pub fn new(config: HeartbeatConfig) -> Self {
        Self {
            detectors: HashMap::new(),
            config,
        }
    }

    /// Record a heartbeat from a node. Creates a detector if one doesn't exist.
    pub fn record_heartbeat(&mut self, node_id: &NodeId) {
        let detector = self.detectors.entry(*node_id).or_insert_with(|| {
            PhiAccrualDetector::with_config(
                self.config.phi_threshold,
                self.config.max_samples,
                5,
            )
        });
        detector.heartbeat();
    }

    /// Check the status of a specific node.
    pub fn check_node(&self, node_id: &NodeId) -> NodeStatus {
        match self.detectors.get(node_id) {
            None => NodeStatus::Unknown,
            Some(detector) => {
                let phi = detector.phi();
                if phi >= self.config.phi_threshold {
                    NodeStatus::Dead(phi)
                } else if phi >= self.config.suspicious_threshold {
                    NodeStatus::Suspicious(phi)
                } else {
                    NodeStatus::Alive
                }
            }
        }
    }

    /// Get all nodes considered dead (phi > threshold).
    pub fn get_dead_nodes(&self) -> Vec<NodeId> {
        self.detectors
            .iter()
            .filter(|(_, d)| d.phi() >= self.config.phi_threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all suspicious nodes with their phi values.
    pub fn get_suspicious_nodes(&self) -> Vec<(NodeId, f64)> {
        self.detectors
            .iter()
            .filter_map(|(id, d)| {
                let phi = d.phi();
                if phi >= self.config.suspicious_threshold && phi < self.config.phi_threshold {
                    Some((*id, phi))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Remove a node from monitoring.
    pub fn remove_node(&mut self, node_id: &NodeId) {
        self.detectors.remove(node_id);
    }

    /// Get status of all monitored nodes.
    pub fn all_statuses(&self) -> Vec<(NodeId, NodeStatus)> {
        self.detectors
            .keys()
            .map(|id| (*id, self.check_node(id)))
            .collect()
    }

    /// Number of monitored nodes.
    pub fn monitored_count(&self) -> usize {
        self.detectors.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_node(s: &str) -> NodeId {
        NodeId::from_string(s)
    }

    #[test]
    fn test_new_detector_is_alive() {
        let detector = PhiAccrualDetector::new(8.0);
        assert_eq!(detector.phi(), 0.0);
        assert!(detector.is_alive());
        assert!(!detector.is_suspicious());
        assert!(detector.last_heartbeat_ago().is_none());
    }

    #[test]
    fn test_single_heartbeat() {
        let mut detector = PhiAccrualDetector::new(8.0);
        detector.heartbeat();

        // One heartbeat: no intervals yet, phi should be 0
        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.phi(), 0.0);
        assert!(detector.last_heartbeat_ago().is_some());
    }

    #[test]
    fn test_regular_heartbeats_alive() {
        // Use a high threshold to tolerate CI/loaded system jitter
        let mut detector = PhiAccrualDetector::with_config(16.0, 50, 3);

        // Simulate regular heartbeats at ~20ms intervals (generous for loaded systems)
        for _ in 0..15 {
            detector.heartbeat();
            thread::sleep(Duration::from_millis(20));
        }

        // Should be alive with regular heartbeats
        assert!(detector.sample_count() >= 3);
        assert!(detector.is_alive(), "Regular heartbeats should keep node alive, phi={}", detector.phi());
    }

    #[test]
    fn test_missed_heartbeats_dead() {
        let mut detector = PhiAccrualDetector::with_config(8.0, 50, 3);

        // Record several rapid heartbeats (1ms intervals)
        for _ in 0..10 {
            detector.heartbeat();
            thread::sleep(Duration::from_millis(1));
        }

        // Now wait much longer than the expected interval
        thread::sleep(Duration::from_millis(200));

        // Phi should be very high
        let phi = detector.phi();
        assert!(phi > 5.0, "After long silence, phi should be high, got {}", phi);
    }

    #[test]
    fn test_erf_approximation() {
        // erf(0) = 0
        let erf_0 = PhiAccrualDetector::erf(0.0);
        assert!((erf_0).abs() < 0.001, "erf(0) should be ~0, got {}", erf_0);

        // erf(1) ≈ 0.8427
        let erf_1 = PhiAccrualDetector::erf(1.0);
        assert!((erf_1 - 0.8427).abs() < 0.01, "erf(1) should be ~0.8427, got {}", erf_1);

        // erf(-x) = -erf(x)
        let erf_neg = PhiAccrualDetector::erf(-1.0);
        assert!((erf_neg + erf_1).abs() < 0.001, "erf should be odd function");

        // erf(3) ≈ 0.9999
        let erf_3 = PhiAccrualDetector::erf(3.0);
        assert!(erf_3 > 0.999, "erf(3) should be ~1, got {}", erf_3);
    }

    #[test]
    fn test_cdf_normal() {
        // CDF at mean should be 0.5
        let cdf = PhiAccrualDetector::cdf_normal(10.0, 10.0, 2.0);
        assert!((cdf - 0.5).abs() < 0.01, "CDF at mean should be ~0.5, got {}", cdf);

        // CDF well above mean should approach 1.0
        let cdf_high = PhiAccrualDetector::cdf_normal(20.0, 10.0, 2.0);
        assert!(cdf_high > 0.99, "CDF far above mean should approach 1, got {}", cdf_high);

        // CDF well below mean should approach 0.0
        let cdf_low = PhiAccrualDetector::cdf_normal(0.0, 10.0, 2.0);
        assert!(cdf_low < 0.01, "CDF far below mean should approach 0, got {}", cdf_low);
    }

    #[test]
    fn test_not_enough_samples() {
        let mut detector = PhiAccrualDetector::with_config(8.0, 50, 5);

        // Record a few heartbeats (less than min_samples)
        for _ in 0..3 {
            detector.heartbeat();
            thread::sleep(Duration::from_millis(5));
        }

        // Should return phi=0 since not enough samples
        assert_eq!(detector.sample_count(), 2); // 3 heartbeats = 2 intervals
        assert_eq!(detector.phi(), 0.0, "Not enough samples, phi should be 0");
    }

    #[test]
    fn test_heartbeat_manager_basic() {
        let config = HeartbeatConfig::default();
        let mut mgr = HeartbeatManager::new(config);

        let node_a = make_node("node_a");
        let node_b = make_node("node_b");

        // Unknown node
        assert!(matches!(mgr.check_node(&node_a), NodeStatus::Unknown));

        // Record heartbeats
        mgr.record_heartbeat(&node_a);
        mgr.record_heartbeat(&node_b);

        assert_eq!(mgr.monitored_count(), 2);
        assert!(matches!(mgr.check_node(&node_a), NodeStatus::Alive));
    }

    #[test]
    fn test_heartbeat_manager_dead_node() {
        let config = HeartbeatConfig {
            phi_threshold: 8.0,
            suspicious_threshold: 4.0,
            max_samples: 50,
            interval: Duration::from_millis(10),
        };
        let mut mgr = HeartbeatManager::new(config);

        let node_a = make_node("node_a");

        // Rapid heartbeats to build up short-interval statistics
        for _ in 0..10 {
            mgr.record_heartbeat(&node_a);
            thread::sleep(Duration::from_millis(1));
        }

        // Wait long enough for phi to rise
        thread::sleep(Duration::from_millis(200));

        let _dead = mgr.get_dead_nodes();
        // phi should be high enough to trigger dead detection
        let status = mgr.check_node(&node_a);
        match status {
            NodeStatus::Dead(phi) => assert!(phi > 5.0, "phi should be high, got {}", phi),
            NodeStatus::Suspicious(phi) => {
                // May also be suspicious if threshold is borderline
                assert!(phi > 3.0, "phi should be suspicious, got {}", phi);
            }
            other => panic!("Expected Dead or Suspicious, got {:?}", other),
        }
    }

    #[test]
    fn test_heartbeat_manager_remove() {
        let mut mgr = HeartbeatManager::new(HeartbeatConfig::default());
        let node_a = make_node("node_a");

        mgr.record_heartbeat(&node_a);
        assert_eq!(mgr.monitored_count(), 1);

        mgr.remove_node(&node_a);
        assert_eq!(mgr.monitored_count(), 0);
        assert!(matches!(mgr.check_node(&node_a), NodeStatus::Unknown));
    }

    #[test]
    fn test_heartbeat_manager_all_statuses() {
        let mut mgr = HeartbeatManager::new(HeartbeatConfig::default());
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");

        mgr.record_heartbeat(&node_a);
        mgr.record_heartbeat(&node_b);

        let statuses = mgr.all_statuses();
        assert_eq!(statuses.len(), 2);
    }

    #[test]
    fn test_node_status_is_reachable() {
        assert!(NodeStatus::Alive.is_reachable());
        assert!(NodeStatus::Unknown.is_reachable());
        assert!(!NodeStatus::Dead(10.0).is_reachable());
        assert!(!NodeStatus::Suspicious(5.0).is_reachable());
    }

    #[test]
    fn test_detector_threshold() {
        let detector = PhiAccrualDetector::new(12.0);
        assert_eq!(detector.threshold(), 12.0);
    }
}
