//! # Cluster Metrics (Phase 12)
//!
//! Prometheus-compatible metrics for cluster observability.
//!
//! Exposes counters and gauges for:
//! - Active nodes, peer connections
//! - CRDT sync operations, merge counts, state sizes
//! - Rate limit syncs
//! - Ring rebalances
//! - Circuit breaker events
//! - Session migrations

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

// ============================================================================
// ClusterMetrics
// ============================================================================

/// Prometheus-compatible cluster metrics using atomic counters.
pub struct ClusterMetrics {
    node_id: String,

    // Gauge: current active nodes
    active_nodes: AtomicU64,

    // Counters
    crdt_syncs_total: AtomicU64,
    crdt_merges_total: AtomicU64,
    rate_limit_syncs_total: AtomicU64,
    ring_rebalances_total: AtomicU64,
    circuit_breaker_opens_total: AtomicU64,
    sessions_migrated_total: AtomicU64,

    // Gauge: CRDT state sizes (bytes)
    crdt_state_bytes: AtomicU64,

    // Gauge: sync lag (millis)
    sync_lag_ms: AtomicU64,
}

impl ClusterMetrics {
    /// Create a new metrics instance for a node.
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            active_nodes: AtomicU64::new(0),
            crdt_syncs_total: AtomicU64::new(0),
            crdt_merges_total: AtomicU64::new(0),
            rate_limit_syncs_total: AtomicU64::new(0),
            ring_rebalances_total: AtomicU64::new(0),
            circuit_breaker_opens_total: AtomicU64::new(0),
            sessions_migrated_total: AtomicU64::new(0),
            crdt_state_bytes: AtomicU64::new(0),
            sync_lag_ms: AtomicU64::new(0),
        }
    }

    // ========== Setters (Gauges) ==========

    /// Set the number of active nodes in the cluster.
    pub fn set_active_nodes(&self, count: usize) {
        self.active_nodes.store(count as u64, Ordering::Relaxed);
    }

    /// Set the total CRDT state size in bytes.
    pub fn set_crdt_state_bytes(&self, bytes: u64) {
        self.crdt_state_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Set the sync lag in milliseconds.
    pub fn set_sync_lag_ms(&self, ms: u64) {
        self.sync_lag_ms.store(ms, Ordering::Relaxed);
    }

    // ========== Incrementors (Counters) ==========

    /// Record a CRDT sync operation.
    pub fn record_crdt_sync(&self) {
        self.crdt_syncs_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a CRDT merge.
    pub fn record_crdt_merge(&self) {
        self.crdt_merges_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rate limit sync.
    pub fn record_rate_limit_sync(&self) {
        self.rate_limit_syncs_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a hash ring rebalance.
    pub fn record_ring_rebalance(&self) {
        self.ring_rebalances_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a circuit breaker open event.
    pub fn record_circuit_breaker_open(&self) {
        self.circuit_breaker_opens_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a session migration.
    pub fn record_session_migration(&self) {
        self.sessions_migrated_total.fetch_add(1, Ordering::Relaxed);
    }

    // ========== Getters ==========

    pub fn active_nodes(&self) -> u64 { self.active_nodes.load(Ordering::Relaxed) }
    pub fn crdt_syncs_total(&self) -> u64 { self.crdt_syncs_total.load(Ordering::Relaxed) }
    pub fn crdt_merges_total(&self) -> u64 { self.crdt_merges_total.load(Ordering::Relaxed) }
    pub fn rate_limit_syncs_total(&self) -> u64 { self.rate_limit_syncs_total.load(Ordering::Relaxed) }
    pub fn ring_rebalances_total(&self) -> u64 { self.ring_rebalances_total.load(Ordering::Relaxed) }
    pub fn circuit_breaker_opens_total(&self) -> u64 { self.circuit_breaker_opens_total.load(Ordering::Relaxed) }
    pub fn sessions_migrated_total(&self) -> u64 { self.sessions_migrated_total.load(Ordering::Relaxed) }
    pub fn crdt_state_bytes(&self) -> u64 { self.crdt_state_bytes.load(Ordering::Relaxed) }
    pub fn sync_lag_ms(&self) -> u64 { self.sync_lag_ms.load(Ordering::Relaxed) }

    /// Take a snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            active_nodes: self.active_nodes(),
            crdt_syncs_total: self.crdt_syncs_total(),
            crdt_merges_total: self.crdt_merges_total(),
            rate_limit_syncs_total: self.rate_limit_syncs_total(),
            ring_rebalances_total: self.ring_rebalances_total(),
            circuit_breaker_opens_total: self.circuit_breaker_opens_total(),
            sessions_migrated_total: self.sessions_migrated_total(),
            crdt_state_bytes: self.crdt_state_bytes(),
            sync_lag_ms: self.sync_lag_ms(),
        }
    }

    /// Format metrics in Prometheus exposition format.
    pub fn to_prometheus(&self) -> String {
        let node = &self.node_id;
        format!(
            concat!(
                "# HELP ai_cluster_nodes_active Number of active nodes in the cluster\n",
                "# TYPE ai_cluster_nodes_active gauge\n",
                "ai_cluster_nodes_active{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_sync_lag_seconds Sync lag to peers\n",
                "# TYPE ai_cluster_sync_lag_seconds gauge\n",
                "ai_cluster_sync_lag_seconds{{node=\"{}\"}} {:.3}\n",
                "# HELP ai_cluster_crdt_syncs_total Total CRDT sync operations\n",
                "# TYPE ai_cluster_crdt_syncs_total counter\n",
                "ai_cluster_crdt_syncs_total{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_crdt_merges_total Total CRDT merge operations\n",
                "# TYPE ai_cluster_crdt_merges_total counter\n",
                "ai_cluster_crdt_merges_total{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_crdt_state_bytes Total CRDT state size in bytes\n",
                "# TYPE ai_cluster_crdt_state_bytes gauge\n",
                "ai_cluster_crdt_state_bytes{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_rate_limit_syncs_total Total rate limit sync operations\n",
                "# TYPE ai_cluster_rate_limit_syncs_total counter\n",
                "ai_cluster_rate_limit_syncs_total{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_ring_rebalances_total Total hash ring rebalances\n",
                "# TYPE ai_cluster_ring_rebalances_total counter\n",
                "ai_cluster_ring_rebalances_total{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_circuit_breaker_opens_total Total circuit breaker opens\n",
                "# TYPE ai_cluster_circuit_breaker_opens_total counter\n",
                "ai_cluster_circuit_breaker_opens_total{{node=\"{}\"}} {}\n",
                "# HELP ai_cluster_sessions_migrated_total Total sessions migrated\n",
                "# TYPE ai_cluster_sessions_migrated_total counter\n",
                "ai_cluster_sessions_migrated_total{{node=\"{}\"}} {}\n",
            ),
            node, self.active_nodes(),
            node, self.sync_lag_ms() as f64 / 1000.0,
            node, self.crdt_syncs_total(),
            node, self.crdt_merges_total(),
            node, self.crdt_state_bytes(),
            node, self.rate_limit_syncs_total(),
            node, self.ring_rebalances_total(),
            node, self.circuit_breaker_opens_total(),
            node, self.sessions_migrated_total(),
        )
    }
}

// ============================================================================
// MetricsSnapshot
// ============================================================================

/// A point-in-time snapshot of all cluster metrics (serializable).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub active_nodes: u64,
    pub crdt_syncs_total: u64,
    pub crdt_merges_total: u64,
    pub rate_limit_syncs_total: u64,
    pub ring_rebalances_total: u64,
    pub circuit_breaker_opens_total: u64,
    pub sessions_migrated_total: u64,
    pub crdt_state_bytes: u64,
    pub sync_lag_ms: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_metrics_new_zeros() {
        let m = ClusterMetrics::new("node1".to_string());
        assert_eq!(m.active_nodes(), 0);
        assert_eq!(m.crdt_syncs_total(), 0);
        assert_eq!(m.crdt_merges_total(), 0);
        assert_eq!(m.ring_rebalances_total(), 0);
        assert_eq!(m.sync_lag_ms(), 0);
    }

    #[test]
    fn test_metrics_set_gauges() {
        let m = ClusterMetrics::new("node1".to_string());
        m.set_active_nodes(5);
        assert_eq!(m.active_nodes(), 5);
        m.set_crdt_state_bytes(4096);
        assert_eq!(m.crdt_state_bytes(), 4096);
        m.set_sync_lag_ms(150);
        assert_eq!(m.sync_lag_ms(), 150);
    }

    #[test]
    fn test_metrics_increment_counters() {
        let m = ClusterMetrics::new("node1".to_string());
        m.record_crdt_sync();
        m.record_crdt_sync();
        assert_eq!(m.crdt_syncs_total(), 2);

        m.record_crdt_merge();
        assert_eq!(m.crdt_merges_total(), 1);

        m.record_rate_limit_sync();
        assert_eq!(m.rate_limit_syncs_total(), 1);

        m.record_ring_rebalance();
        assert_eq!(m.ring_rebalances_total(), 1);

        m.record_circuit_breaker_open();
        assert_eq!(m.circuit_breaker_opens_total(), 1);

        m.record_session_migration();
        assert_eq!(m.sessions_migrated_total(), 1);
    }

    #[test]
    fn test_metrics_snapshot() {
        let m = ClusterMetrics::new("node1".to_string());
        m.set_active_nodes(3);
        m.record_crdt_sync();
        m.record_ring_rebalance();

        let snap = m.snapshot();
        assert_eq!(snap.active_nodes, 3);
        assert_eq!(snap.crdt_syncs_total, 1);
        assert_eq!(snap.ring_rebalances_total, 1);
    }

    #[test]
    fn test_metrics_snapshot_serialization() {
        let snap = MetricsSnapshot {
            active_nodes: 5,
            crdt_syncs_total: 100,
            crdt_merges_total: 50,
            rate_limit_syncs_total: 20,
            ring_rebalances_total: 3,
            circuit_breaker_opens_total: 1,
            sessions_migrated_total: 10,
            crdt_state_bytes: 8192,
            sync_lag_ms: 200,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.active_nodes, 5);
        assert_eq!(parsed.crdt_state_bytes, 8192);
    }

    #[test]
    fn test_prometheus_format() {
        let m = ClusterMetrics::new("test-node".to_string());
        m.set_active_nodes(3);
        m.record_crdt_sync();

        let output = m.to_prometheus();
        assert!(output.contains("ai_cluster_nodes_active{node=\"test-node\"} 3"));
        assert!(output.contains("ai_cluster_crdt_syncs_total{node=\"test-node\"} 1"));
        assert!(output.contains("# TYPE ai_cluster_nodes_active gauge"));
        assert!(output.contains("# TYPE ai_cluster_crdt_syncs_total counter"));
    }

    #[test]
    fn test_prometheus_sync_lag_format() {
        let m = ClusterMetrics::new("n1".to_string());
        m.set_sync_lag_ms(1500);
        let output = m.to_prometheus();
        assert!(output.contains("ai_cluster_sync_lag_seconds{node=\"n1\"} 1.500"));
    }

    #[test]
    fn test_metrics_concurrent_access() {
        let m = Arc::new(ClusterMetrics::new("node1".to_string()));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let m = m.clone();
            let handle = std::thread::spawn(move || {
                for _ in 0..100 {
                    m.record_crdt_sync();
                    m.record_crdt_merge();
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(m.crdt_syncs_total(), 400);
        assert_eq!(m.crdt_merges_total(), 400);
    }

    #[test]
    fn test_snapshot_default() {
        let snap = MetricsSnapshot::default();
        assert_eq!(snap.active_nodes, 0);
        assert_eq!(snap.crdt_syncs_total, 0);
        assert_eq!(snap.sync_lag_ms, 0);
    }
}
