// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Bulkhead pattern for resource isolation.
//!
//! This module implements the Bulkhead pattern (also known as the Isolation pattern),
//! which limits the number of concurrent operations that can access a given resource.
//! By partitioning resources into isolated pools, a failure or overload in one area
//! does not cascade to others.
//!
//! # Overview
//!
//! - **`BulkheadConfig`** — defines concurrency limits and wait timeouts with presets
//!   for common workloads (chat, streaming, embeddings, background).
//! - **`Bulkhead`** — enforces the concurrency limit, issuing permits to callers.
//! - **`BulkheadPermit`** — RAII guard that automatically releases its slot on drop.
//! - **`BulkheadRegistry`** — manages multiple named bulkheads for multi-resource isolation.
//! - **`BulkheadStats`** — snapshot of usage and rejection counters.
//! - **`BulkheadError`** — error type covering full, timeout, and not-found conditions.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// BulkheadConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`Bulkhead`] instance.
#[derive(Debug, Clone)]
pub struct BulkheadConfig {
    /// Human-readable name for this bulkhead (used in stats and error messages).
    pub name: String,
    /// Maximum number of permits that can be held concurrently.
    pub max_concurrent: usize,
    /// Maximum time a caller will wait for a permit in [`Bulkhead::acquire_timeout`].
    pub max_wait: Duration,
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            max_concurrent: 10,
            max_wait: Duration::from_secs(5),
        }
    }
}

impl BulkheadConfig {
    /// Preset for interactive chat workloads (max 10 concurrent, 5 s wait).
    pub fn for_chat() -> Self {
        Self {
            name: "chat".to_string(),
            max_concurrent: 10,
            max_wait: Duration::from_secs(5),
        }
    }

    /// Preset for streaming workloads (max 5 concurrent, 30 s wait).
    pub fn for_streaming() -> Self {
        Self {
            name: "streaming".to_string(),
            max_concurrent: 5,
            max_wait: Duration::from_secs(30),
        }
    }

    /// Preset for embedding workloads (max 20 concurrent, 2 s wait).
    pub fn for_embeddings() -> Self {
        Self {
            name: "embeddings".to_string(),
            max_concurrent: 20,
            max_wait: Duration::from_secs(2),
        }
    }

    /// Preset for background / batch workloads (max 3 concurrent, 60 s wait).
    pub fn for_background() -> Self {
        Self {
            name: "background".to_string(),
            max_concurrent: 3,
            max_wait: Duration::from_secs(60),
        }
    }
}

// ---------------------------------------------------------------------------
// BulkheadState (private)
// ---------------------------------------------------------------------------

/// Internal mutable state protected by a [`Mutex`].
#[derive(Debug)]
struct BulkheadState {
    /// Number of permits currently held.
    active: usize,
    /// Lifetime counter of successfully acquired permits.
    total_accepted: u64,
    /// Lifetime counter of rejections due to a full bulkhead (non-blocking path).
    total_rejected: u64,
    /// Lifetime counter of timeouts in the blocking path.
    total_timed_out: u64,
}

impl BulkheadState {
    fn new() -> Self {
        Self {
            active: 0,
            total_accepted: 0,
            total_rejected: 0,
            total_timed_out: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BulkheadError
// ---------------------------------------------------------------------------

/// Errors returned by bulkhead operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BulkheadError {
    /// The bulkhead has reached its concurrency limit (non-blocking rejection).
    Full,
    /// The caller waited for a permit but the timeout expired.
    Timeout,
    /// The requested bulkhead name was not found in the registry.
    NotFound(String),
}

impl fmt::Display for BulkheadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BulkheadError::Full => write!(f, "bulkhead is full — no permits available"),
            BulkheadError::Timeout => {
                write!(f, "bulkhead acquire timed out waiting for a permit")
            }
            BulkheadError::NotFound(name) => {
                write!(f, "bulkhead not found: {name}")
            }
        }
    }
}

impl std::error::Error for BulkheadError {}

// ---------------------------------------------------------------------------
// BulkheadStats
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of a bulkhead's counters.
#[derive(Debug, Clone)]
pub struct BulkheadStats {
    /// Name of the bulkhead.
    pub name: String,
    /// Number of permits currently held.
    pub active: usize,
    /// Configured maximum concurrency.
    pub max_concurrent: usize,
    /// Lifetime count of successfully acquired permits.
    pub total_accepted: u64,
    /// Lifetime count of rejections (bulkhead was full).
    pub total_rejected: u64,
    /// Lifetime count of timed-out acquisition attempts.
    pub total_timed_out: u64,
    /// Current utilization as a percentage (0.0–100.0).
    pub utilization_percent: f64,
}

// ---------------------------------------------------------------------------
// BulkheadPermit
// ---------------------------------------------------------------------------

/// RAII permit returned by [`Bulkhead::try_acquire`] and [`Bulkhead::acquire_timeout`].
///
/// When the permit is dropped the active count is decremented and one waiting
/// thread (if any) is notified via the internal [`Condvar`].
pub struct BulkheadPermit {
    state: Arc<(Mutex<BulkheadState>, Condvar)>,
}

impl Drop for BulkheadPermit {
    fn drop(&mut self) {
        let (mutex, condvar) = &*self.state;
        let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
        guard.active = guard.active.saturating_sub(1);
        condvar.notify_one();
    }
}

// We intentionally do not derive Debug for BulkheadPermit because there is
// nothing useful to show and holding the lock for Debug would be undesirable.
impl fmt::Debug for BulkheadPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BulkheadPermit").finish()
    }
}

// ---------------------------------------------------------------------------
// Bulkhead
// ---------------------------------------------------------------------------

/// A concurrency-limiting bulkhead.
///
/// Create one via [`Bulkhead::new`], then call [`Bulkhead::try_acquire`] for
/// non-blocking access or [`Bulkhead::acquire_timeout`] for blocking access
/// with a deadline.
#[derive(Debug)]
pub struct Bulkhead {
    config: BulkheadConfig,
    state: Arc<(Mutex<BulkheadState>, Condvar)>,
}

impl Bulkhead {
    /// Create a new bulkhead with the given configuration.
    pub fn new(config: BulkheadConfig) -> Self {
        Self {
            config,
            state: Arc::new((Mutex::new(BulkheadState::new()), Condvar::new())),
        }
    }

    /// Try to acquire a permit without blocking.
    ///
    /// Returns [`BulkheadError::Full`] immediately if the concurrency limit
    /// has been reached.
    pub fn try_acquire(&self) -> Result<BulkheadPermit, BulkheadError> {
        let (mutex, _condvar) = &*self.state;
        let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());

        if guard.active >= self.config.max_concurrent {
            guard.total_rejected += 1;
            return Err(BulkheadError::Full);
        }

        guard.active += 1;
        guard.total_accepted += 1;

        Ok(BulkheadPermit {
            state: Arc::clone(&self.state),
        })
    }

    /// Acquire a permit, blocking up to `timeout` for one to become available.
    ///
    /// Returns [`BulkheadError::Timeout`] if no permit becomes available within
    /// the given duration.
    pub fn acquire_timeout(&self, timeout: Duration) -> Result<BulkheadPermit, BulkheadError> {
        let (mutex, condvar) = &*self.state;
        let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());

        let deadline = std::time::Instant::now() + timeout;

        while guard.active >= self.config.max_concurrent {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                guard.total_timed_out += 1;
                return Err(BulkheadError::Timeout);
            }

            let (new_guard, wait_result) = condvar
                .wait_timeout(guard, remaining)
                .unwrap_or_else(|e| e.into_inner());
            guard = new_guard;

            if wait_result.timed_out() && guard.active >= self.config.max_concurrent {
                guard.total_timed_out += 1;
                return Err(BulkheadError::Timeout);
            }
        }

        guard.active += 1;
        guard.total_accepted += 1;

        Ok(BulkheadPermit {
            state: Arc::clone(&self.state),
        })
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> BulkheadStats {
        let (mutex, _) = &*self.state;
        let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());

        let utilization_percent = if self.config.max_concurrent == 0 {
            0.0
        } else {
            (guard.active as f64 / self.config.max_concurrent as f64) * 100.0
        };

        BulkheadStats {
            name: self.config.name.clone(),
            active: guard.active,
            max_concurrent: self.config.max_concurrent,
            total_accepted: guard.total_accepted,
            total_rejected: guard.total_rejected,
            total_timed_out: guard.total_timed_out,
            utilization_percent,
        }
    }

    /// Return the number of currently active (held) permits.
    pub fn active_count(&self) -> usize {
        let (mutex, _) = &*self.state;
        let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
        guard.active
    }

    /// Return `true` if no more permits are available right now.
    pub fn is_full(&self) -> bool {
        self.active_count() >= self.config.max_concurrent
    }

    /// Return the configured name of this bulkhead.
    pub fn name(&self) -> &str {
        &self.config.name
    }
}

// ---------------------------------------------------------------------------
// BulkheadRegistry
// ---------------------------------------------------------------------------

/// A registry of named [`Bulkhead`] instances for managing multiple resource pools.
#[derive(Debug)]
pub struct BulkheadRegistry {
    bulkheads: HashMap<String, Arc<Bulkhead>>,
}

impl BulkheadRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            bulkheads: HashMap::new(),
        }
    }

    /// Register a new bulkhead under the given `name` with the provided config.
    ///
    /// If a bulkhead with the same name already exists it is replaced.
    pub fn register(&mut self, name: &str, config: BulkheadConfig) {
        let bulkhead = Arc::new(Bulkhead::new(BulkheadConfig {
            name: name.to_string(),
            ..config
        }));
        self.bulkheads.insert(name.to_string(), bulkhead);
    }

    /// Look up a bulkhead by name.
    pub fn get(&self, name: &str) -> Option<Arc<Bulkhead>> {
        self.bulkheads.get(name).cloned()
    }

    /// Convenience: look up a bulkhead by name and try to acquire a permit.
    ///
    /// Returns [`BulkheadError::NotFound`] if the name is not registered,
    /// or [`BulkheadError::Full`] if the bulkhead is at capacity.
    pub fn try_acquire(&self, name: &str) -> Result<BulkheadPermit, BulkheadError> {
        let bulkhead = self
            .bulkheads
            .get(name)
            .ok_or_else(|| BulkheadError::NotFound(name.to_string()))?;
        bulkhead.try_acquire()
    }

    /// Collect stats from every registered bulkhead.
    pub fn all_stats(&self) -> Vec<BulkheadStats> {
        self.bulkheads.values().map(|b| b.stats()).collect()
    }
}

impl Default for BulkheadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn test_try_acquire_success() {
        let bh = Bulkhead::new(BulkheadConfig {
            max_concurrent: 2,
            ..BulkheadConfig::default()
        });

        let _p1 = bh.try_acquire().expect("should acquire first permit");
        let _p2 = bh.try_acquire().expect("should acquire second permit");
        assert_eq!(bh.active_count(), 2);
    }

    #[test]
    fn test_try_acquire_full_rejection() {
        let bh = Bulkhead::new(BulkheadConfig {
            max_concurrent: 1,
            ..BulkheadConfig::default()
        });

        let _p1 = bh.try_acquire().expect("first acquire should succeed");
        let err = bh.try_acquire().expect_err("second acquire should fail");
        assert_eq!(err, BulkheadError::Full);

        let stats = bh.stats();
        assert_eq!(stats.total_rejected, 1);
    }

    #[test]
    fn test_acquire_timeout_success() {
        let bh = Arc::new(Bulkhead::new(BulkheadConfig {
            max_concurrent: 1,
            ..BulkheadConfig::default()
        }));

        // Acquire the only slot.
        let permit = bh.try_acquire().expect("initial acquire should succeed");

        let bh_clone = Arc::clone(&bh);
        let handle = thread::spawn(move || {
            // This should block, then succeed once the permit is released.
            bh_clone
                .acquire_timeout(Duration::from_secs(2))
                .expect("should acquire after release")
        });

        // Give the spawned thread time to start waiting, then release.
        thread::sleep(Duration::from_millis(50));
        drop(permit);

        let _permit2 = handle.join().expect("thread should succeed");
        assert_eq!(bh.active_count(), 1);
    }

    #[test]
    fn test_acquire_timeout_expired() {
        let bh = Bulkhead::new(BulkheadConfig {
            max_concurrent: 1,
            ..BulkheadConfig::default()
        });

        let _p1 = bh.try_acquire().expect("first acquire should succeed");

        let start = Instant::now();
        let err = bh
            .acquire_timeout(Duration::from_millis(100))
            .expect_err("should time out");
        let elapsed = start.elapsed();

        assert_eq!(err, BulkheadError::Timeout);
        assert!(
            elapsed >= Duration::from_millis(80),
            "should have waited at least ~100 ms, waited {:?}",
            elapsed
        );

        let stats = bh.stats();
        assert_eq!(stats.total_timed_out, 1);
    }

    #[test]
    fn test_permit_auto_release_on_drop() {
        let bh = Bulkhead::new(BulkheadConfig {
            max_concurrent: 1,
            ..BulkheadConfig::default()
        });

        {
            let _p = bh.try_acquire().expect("should acquire");
            assert_eq!(bh.active_count(), 1);
        }
        // Permit dropped — slot should be free.
        assert_eq!(bh.active_count(), 0);

        let _p2 = bh.try_acquire().expect("should acquire again after drop");
        assert_eq!(bh.active_count(), 1);
    }

    #[test]
    fn test_concurrent_acquire_release() {
        let bh = Arc::new(Bulkhead::new(BulkheadConfig {
            max_concurrent: 4,
            ..BulkheadConfig::default()
        }));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let bh_clone = Arc::clone(&bh);
            handles.push(thread::spawn(move || {
                let permit = bh_clone
                    .acquire_timeout(Duration::from_secs(5))
                    .expect("should eventually acquire");
                // Hold the permit briefly.
                thread::sleep(Duration::from_millis(20));
                drop(permit);
            }));
        }

        for h in handles {
            h.join().expect("worker thread should not panic");
        }

        assert_eq!(bh.active_count(), 0);
        let stats = bh.stats();
        assert_eq!(stats.total_accepted, 8);
    }

    #[test]
    fn test_stats_tracking() {
        let bh = Bulkhead::new(BulkheadConfig {
            name: "test-stats".to_string(),
            max_concurrent: 1,
            max_wait: Duration::from_millis(10),
        });

        // Successful acquire.
        let p = bh.try_acquire().expect("should succeed");

        // Rejected acquire.
        let _ = bh.try_acquire();

        // Timed-out acquire.
        let _ = bh.acquire_timeout(Duration::from_millis(10));

        drop(p);

        let stats = bh.stats();
        assert_eq!(stats.name, "test-stats");
        assert_eq!(stats.total_accepted, 1);
        assert_eq!(stats.total_rejected, 1);
        assert_eq!(stats.total_timed_out, 1);
        assert_eq!(stats.active, 0);
        assert_eq!(stats.max_concurrent, 1);
        assert!((stats.utilization_percent - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_full() {
        let bh = Bulkhead::new(BulkheadConfig {
            max_concurrent: 2,
            ..BulkheadConfig::default()
        });

        assert!(!bh.is_full());

        let _p1 = bh.try_acquire().expect("first");
        assert!(!bh.is_full());

        let _p2 = bh.try_acquire().expect("second");
        assert!(bh.is_full());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = BulkheadRegistry::new();
        reg.register("api", BulkheadConfig::for_chat());

        let bh = reg.get("api");
        assert!(bh.is_some());
        assert_eq!(bh.expect("should exist").name(), "api");

        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_try_acquire() {
        let mut reg = BulkheadRegistry::new();
        reg.register("svc", BulkheadConfig {
            max_concurrent: 1,
            ..BulkheadConfig::default()
        });

        let _p = reg.try_acquire("svc").expect("should acquire from registry");

        let err = reg
            .try_acquire("svc")
            .expect_err("should be full");
        assert_eq!(err, BulkheadError::Full);
    }

    #[test]
    fn test_registry_not_found() {
        let reg = BulkheadRegistry::new();
        let err = reg.try_acquire("missing").expect_err("should not find");
        assert_eq!(err, BulkheadError::NotFound("missing".to_string()));
    }

    #[test]
    fn test_registry_all_stats() {
        let mut reg = BulkheadRegistry::new();
        reg.register("a", BulkheadConfig::for_chat());
        reg.register("b", BulkheadConfig::for_streaming());

        let stats = reg.all_stats();
        assert_eq!(stats.len(), 2);

        let names: Vec<&str> = stats.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn test_preset_chat() {
        let cfg = BulkheadConfig::for_chat();
        assert_eq!(cfg.name, "chat");
        assert_eq!(cfg.max_concurrent, 10);
        assert_eq!(cfg.max_wait, Duration::from_secs(5));
    }

    #[test]
    fn test_preset_streaming() {
        let cfg = BulkheadConfig::for_streaming();
        assert_eq!(cfg.name, "streaming");
        assert_eq!(cfg.max_concurrent, 5);
        assert_eq!(cfg.max_wait, Duration::from_secs(30));
    }

    #[test]
    fn test_preset_embeddings() {
        let cfg = BulkheadConfig::for_embeddings();
        assert_eq!(cfg.name, "embeddings");
        assert_eq!(cfg.max_concurrent, 20);
        assert_eq!(cfg.max_wait, Duration::from_secs(2));
    }

    #[test]
    fn test_preset_background() {
        let cfg = BulkheadConfig::for_background();
        assert_eq!(cfg.name, "background");
        assert_eq!(cfg.max_concurrent, 3);
        assert_eq!(cfg.max_wait, Duration::from_secs(60));
    }

    #[test]
    fn test_default_config() {
        let cfg = BulkheadConfig::default();
        assert_eq!(cfg.name, "default");
        assert_eq!(cfg.max_concurrent, 10);
        assert_eq!(cfg.max_wait, Duration::from_secs(5));
    }

    #[test]
    fn test_error_display() {
        let full = BulkheadError::Full;
        assert_eq!(
            full.to_string(),
            "bulkhead is full — no permits available"
        );

        let timeout = BulkheadError::Timeout;
        assert_eq!(
            timeout.to_string(),
            "bulkhead acquire timed out waiting for a permit"
        );

        let not_found = BulkheadError::NotFound("xyz".to_string());
        assert_eq!(not_found.to_string(), "bulkhead not found: xyz");

        // Verify std::error::Error is implemented.
        let err: &dyn std::error::Error = &full;
        assert!(err.source().is_none());
    }
}
