//! # Distributed Rate Limiter (Phase 9)
//!
//! Two-layer rate limiting for cluster deployments:
//!
//! ```text
//! Layer 1 (LOCAL, instant):  DashMap<IpAddr, SlidingWindowCounter>
//!     ↓ (if passes)
//! Layer 2 (GLOBAL, eventual): PNCounter CRDTs synced via ClusterManager
//!     ↓ (if global total >= limit → block on ALL nodes)
//! ```
//!
//! Layer 1 provides immediate per-node rate limiting with no network overhead.
//! Layer 2 synchronizes counters across the cluster for global limits.

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::distributed::PNCounter;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the 2-layer distributed rate limiter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedRateLimitConfig {
    /// Local per-IP rate limit (requests per window).
    pub local_limit: u64,
    /// Local sliding window duration.
    pub local_window_secs: u64,
    /// Global cluster-wide rate limit.
    pub global_limit: u64,
    /// How often to sync local counts to the CRDT (seconds).
    pub sync_interval_secs: u64,
}

impl Default for DistributedRateLimitConfig {
    fn default() -> Self {
        Self {
            local_limit: 100,
            local_window_secs: 60,
            global_limit: 1000,
            sync_interval_secs: 10,
        }
    }
}

// ============================================================================
// Local sliding window counter
// ============================================================================

/// Per-IP sliding window counter for Layer 1 (local, instant).
struct LocalCounter {
    /// Request timestamps within the current window.
    timestamps: Vec<Instant>,
    /// Window duration.
    window: Duration,
}

impl LocalCounter {
    fn new(window: Duration) -> Self {
        Self {
            timestamps: Vec::new(),
            window,
        }
    }

    /// Record a request and return the count within the window.
    fn record(&mut self) -> u64 {
        let now = Instant::now();
        let cutoff = now - self.window;
        self.timestamps.retain(|t| *t > cutoff);
        self.timestamps.push(now);
        self.timestamps.len() as u64
    }

    /// Get the current count without recording.
    fn count(&mut self) -> u64 {
        let cutoff = Instant::now() - self.window;
        self.timestamps.retain(|t| *t > cutoff);
        self.timestamps.len() as u64
    }
}

// ============================================================================
// DistributedRateLimiter
// ============================================================================

/// Two-layer distributed rate limiter.
///
/// Layer 1: Local `DashMap<IpAddr, LocalCounter>` — instant decisions, no network I/O.
/// Layer 2: `PNCounter` CRDT synced via `ClusterManager` — global cluster-wide limit.
pub struct DistributedRateLimiter {
    /// Node identifier for CRDT operations.
    node_id: String,
    /// Layer 1: Per-IP local counters.
    local_counters: DashMap<IpAddr, LocalCounter>,
    /// Layer 2: Global CRDT counter (shared with ClusterState).
    global_counter: Arc<RwLock<PNCounter>>,
    /// Configuration.
    config: DistributedRateLimitConfig,
    /// Local requests since last sync (accumulated for CRDT push).
    local_since_sync: AtomicU64,
}

impl DistributedRateLimiter {
    /// Create a new distributed rate limiter.
    pub fn new(node_id: String, global_counter: Arc<RwLock<PNCounter>>) -> Self {
        Self {
            node_id,
            local_counters: DashMap::new(),
            global_counter,
            config: DistributedRateLimitConfig::default(),
            local_since_sync: AtomicU64::new(0),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        node_id: String,
        global_counter: Arc<RwLock<PNCounter>>,
        config: DistributedRateLimitConfig,
    ) -> Self {
        Self {
            node_id,
            local_counters: DashMap::new(),
            global_counter,
            config,
            local_since_sync: AtomicU64::new(0),
        }
    }

    /// Check if a request from the given IP should be allowed.
    ///
    /// Returns `Ok(())` if allowed, `Err(retry_after_secs)` if rate limited.
    pub fn check(&self, ip: &IpAddr) -> Result<(), u64> {
        // Layer 1: Local per-IP check (instant)
        let window = Duration::from_secs(self.config.local_window_secs);
        let count = {
            let mut entry = self.local_counters.entry(*ip).or_insert_with(|| LocalCounter::new(window));
            entry.record()
        };

        if count > self.config.local_limit {
            return Err(self.config.local_window_secs);
        }

        // Track local requests for global sync
        self.local_since_sync.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Check the global rate limit (async — reads CRDT).
    ///
    /// Call this after `check()` passes Layer 1.
    pub async fn check_global(&self) -> Result<(), u64> {
        let counter = self.global_counter.read().await;
        let global_count = counter.value();
        if global_count as u64 > self.config.global_limit {
            Err(self.config.sync_interval_secs)
        } else {
            Ok(())
        }
    }

    /// Sync local counts to the global CRDT.
    ///
    /// Called periodically by the cluster sync loop.
    pub async fn sync_to_global(&self) {
        let local = self.local_since_sync.swap(0, Ordering::Relaxed);
        if local > 0 {
            let mut counter = self.global_counter.write().await;
            counter.positive.increment_by(&self.node_id, local);
        }
    }

    /// Reset local counters for an IP.
    pub fn reset_ip(&self, ip: &IpAddr) {
        self.local_counters.remove(ip);
    }

    /// Cleanup expired entries from local counters.
    pub fn cleanup(&self) {
        let mut to_remove = Vec::new();
        for mut entry in self.local_counters.iter_mut() {
            if entry.value_mut().count() == 0 {
                to_remove.push(*entry.key());
            }
        }
        for ip in to_remove {
            self.local_counters.remove(&ip);
        }
    }

    /// Get local counter count for diagnostics.
    pub fn local_counter_count(&self) -> usize {
        self.local_counters.len()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DistributedRateLimitConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter() -> DistributedRateLimiter {
        let counter = Arc::new(RwLock::new(PNCounter::new()));
        DistributedRateLimiter::with_config(
            "test-node".to_string(),
            counter,
            DistributedRateLimitConfig {
                local_limit: 5,
                local_window_secs: 60,
                global_limit: 100,
                sync_interval_secs: 10,
            },
        )
    }

    #[test]
    fn test_local_allows_under_limit() {
        let limiter = make_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..5 {
            assert!(limiter.check(&ip).is_ok());
        }
    }

    #[test]
    fn test_local_blocks_over_limit() {
        let limiter = make_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..5 {
            let _ = limiter.check(&ip);
        }
        // 6th request should be blocked
        assert!(limiter.check(&ip).is_err());
    }

    #[test]
    fn test_local_independent_ips() {
        let limiter = make_limiter();
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        for _ in 0..5 {
            assert!(limiter.check(&ip1).is_ok());
        }
        // ip2 should still be allowed
        assert!(limiter.check(&ip2).is_ok());
    }

    #[test]
    fn test_reset_ip() {
        let limiter = make_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..5 {
            let _ = limiter.check(&ip);
        }
        assert!(limiter.check(&ip).is_err());
        limiter.reset_ip(&ip);
        assert!(limiter.check(&ip).is_ok());
    }

    #[test]
    fn test_global_check() {
        let counter = Arc::new(RwLock::new(PNCounter::new()));
        let limiter = DistributedRateLimiter::with_config(
            "test-node".to_string(),
            counter.clone(),
            DistributedRateLimitConfig {
                local_limit: 1000,
                local_window_secs: 60,
                global_limit: 10,
                sync_interval_secs: 10,
            },
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Under global limit
            assert!(limiter.check_global().await.is_ok());

            // Push over global limit
            counter.write().await.positive.increment_by("other-node", 20);
            assert!(limiter.check_global().await.is_err());
        });
    }

    #[test]
    fn test_sync_to_global() {
        let counter = Arc::new(RwLock::new(PNCounter::new()));
        let limiter = DistributedRateLimiter::new("test-node".to_string(), counter.clone());

        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..10 {
            let _ = limiter.check(&ip);
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            limiter.sync_to_global().await;
            let c = counter.read().await;
            assert_eq!(c.value(), 10);
        });
    }

    #[test]
    fn test_cleanup() {
        let limiter = make_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        let _ = limiter.check(&ip);
        assert_eq!(limiter.local_counter_count(), 1);
        // Cleanup won't remove because window hasn't expired
        limiter.cleanup();
        assert_eq!(limiter.local_counter_count(), 1);
    }

    #[test]
    fn test_config_defaults() {
        let config = DistributedRateLimitConfig::default();
        assert_eq!(config.local_limit, 100);
        assert_eq!(config.local_window_secs, 60);
        assert_eq!(config.global_limit, 1000);
        assert_eq!(config.sync_interval_secs, 10);
    }

    #[test]
    fn test_config_serialization() {
        let config = DistributedRateLimitConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: DistributedRateLimitConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.local_limit, config.local_limit);
    }

    #[test]
    fn test_error_returns_retry_after() {
        let limiter = make_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..6 {
            let _ = limiter.check(&ip);
        }
        match limiter.check(&ip) {
            Err(retry_after) => assert_eq!(retry_after, 60), // local_window_secs
            Ok(()) => panic!("should be rate limited"),
        }
    }

    #[test]
    fn test_concurrent_access() {
        let limiter = Arc::new(make_limiter());
        let mut handles = Vec::new();

        for i in 0..4 {
            let limiter = limiter.clone();
            let handle = std::thread::spawn(move || {
                let ip: IpAddr = format!("10.0.0.{}", i + 1).parse().unwrap();
                for _ in 0..3 {
                    let _ = limiter.check(&ip);
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(limiter.local_counter_count(), 4);
    }
}
