// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Adaptive timeout calculation based on observed latency.
//!
//! Uses a lock-free ring buffer of [`AtomicU64`] samples to track request latencies,
//! then computes a configurable percentile (P50/P95/P99) and applies a multiplier
//! to derive a dynamic timeout clamped between configurable floor and ceiling values.
//!
//! This is useful for services that communicate with backends of varying speed:
//! instead of a static timeout, the timeout adapts to observed conditions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

// ============================================================================
// Percentile
// ============================================================================

/// Which latency percentile to use for timeout calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Percentile {
    /// 50th percentile (median).
    P50,
    /// 95th percentile.
    P95,
    /// 99th percentile.
    P99,
}

impl Percentile {
    /// Return the fractional value (0.0 to 1.0) for this percentile.
    fn as_fraction(&self) -> f64 {
        match self {
            Percentile::P50 => 0.50,
            Percentile::P95 => 0.95,
            Percentile::P99 => 0.99,
        }
    }
}

// ============================================================================
// AdaptiveTimeoutConfig
// ============================================================================

/// Configuration for [`AdaptiveTimeout`].
#[derive(Debug, Clone)]
pub struct AdaptiveTimeoutConfig {
    /// Minimum timeout (floor). The adaptive timeout will never go below this.
    pub min_timeout: Duration,
    /// Maximum timeout (ceiling). The adaptive timeout will never exceed this.
    pub max_timeout: Duration,
    /// Timeout returned before any latency samples have been recorded.
    pub initial_timeout: Duration,
    /// Multiplier applied to the computed percentile value.
    /// For example, 2.0 means "timeout = 2x the observed P95 latency".
    pub multiplier: f64,
    /// Which percentile to track for the timeout calculation.
    pub percentile: Percentile,
    /// Size of the ring buffer (number of latency samples to keep).
    pub window_size: usize,
    /// EWMA sensitivity / alpha (0.0 to 1.0). Reserved for future smoothing.
    pub sensitivity: f64,
}

impl AdaptiveTimeoutConfig {
    /// Conservative preset: 3.0x P99, floor 1s, ceiling 120s.
    ///
    /// Best for critical paths where you would rather wait longer than
    /// time out prematurely.
    pub fn conservative() -> Self {
        Self {
            min_timeout: Duration::from_secs(1),
            max_timeout: Duration::from_secs(120),
            initial_timeout: Duration::from_secs(30),
            multiplier: 3.0,
            percentile: Percentile::P99,
            window_size: 100,
            sensitivity: 0.3,
        }
    }

    /// Responsive preset: 2.0x P95, floor 500ms, ceiling 60s.
    ///
    /// A balanced default suitable for most request/response workloads.
    pub fn responsive() -> Self {
        Self {
            min_timeout: Duration::from_millis(500),
            max_timeout: Duration::from_secs(60),
            initial_timeout: Duration::from_secs(10),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        }
    }

    /// Aggressive preset: 1.5x P95, floor 200ms, ceiling 30s.
    ///
    /// Best for latency-sensitive paths where fast failure is preferred
    /// over waiting.
    pub fn aggressive() -> Self {
        Self {
            min_timeout: Duration::from_millis(200),
            max_timeout: Duration::from_secs(30),
            initial_timeout: Duration::from_secs(5),
            multiplier: 1.5,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        }
    }
}

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            min_timeout: Duration::from_millis(500),
            max_timeout: Duration::from_secs(60),
            initial_timeout: Duration::from_secs(10),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        }
    }
}

// ============================================================================
// AdaptiveTimeoutStats
// ============================================================================

/// Snapshot of adaptive timeout statistics.
#[derive(Debug, Clone)]
pub struct AdaptiveTimeoutStats {
    /// The current computed timeout.
    pub current_timeout: Duration,
    /// 50th percentile latency, if samples are available.
    pub p50: Option<Duration>,
    /// 95th percentile latency, if samples are available.
    pub p95: Option<Duration>,
    /// 99th percentile latency, if samples are available.
    pub p99: Option<Duration>,
    /// Number of latency samples recorded (capped at window size).
    pub sample_count: u64,
}

// ============================================================================
// AdaptiveTimeout
// ============================================================================

/// Adaptive timeout calculator backed by a lock-free ring buffer.
///
/// Records observed latencies into a fixed-size ring buffer of [`AtomicU64`] values
/// (stored as microseconds). On each new sample, recalculates the timeout as:
///
/// ```text
/// timeout = clamp(percentile_value * multiplier, min_timeout, max_timeout)
/// ```
///
/// All operations are lock-free and use [`Ordering::Relaxed`] — acceptable for
/// statistics that tolerate slightly stale reads.
pub struct AdaptiveTimeout {
    /// Configuration.
    config: AdaptiveTimeoutConfig,
    /// Ring buffer storing latency samples in microseconds. A value of 0 means
    /// the slot is empty (no sample recorded there yet).
    samples: Vec<AtomicU64>,
    /// Next write position in the ring buffer (wraps via modulo).
    write_pos: AtomicU64,
    /// Total number of samples recorded (saturates at window_size).
    count: AtomicU64,
    /// Current computed timeout stored as microseconds for atomic access.
    current_timeout_us: AtomicU64,
}

impl AdaptiveTimeout {
    /// Create a new adaptive timeout calculator with the given configuration.
    pub fn new(config: AdaptiveTimeoutConfig) -> Self {
        let initial_us = config.initial_timeout.as_micros() as u64;
        let window_size = config.window_size;
        let mut samples = Vec::with_capacity(window_size);
        for _ in 0..window_size {
            samples.push(AtomicU64::new(0));
        }
        Self {
            config,
            samples,
            write_pos: AtomicU64::new(0),
            count: AtomicU64::new(0),
            current_timeout_us: AtomicU64::new(initial_us),
        }
    }

    /// Record an observed latency and recalculate the timeout.
    ///
    /// Writes the latency into the ring buffer at `write_pos % window_size`,
    /// increments the write position and sample count, then triggers a
    /// recalculation.
    pub fn record(&self, latency: Duration) {
        let micros = latency.as_micros() as u64;
        // Ensure we never store 0 (reserved for "empty slot"). A 0-microsecond
        // latency is stored as 1 microsecond — negligible rounding for stats.
        let micros = if micros == 0 { 1 } else { micros };

        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed) as usize % self.samples.len();
        self.samples[pos].store(micros, Ordering::Relaxed);

        // Saturate count at window_size.
        let prev = self.count.fetch_add(1, Ordering::Relaxed);
        if prev >= self.samples.len() as u64 {
            self.count.store(self.samples.len() as u64, Ordering::Relaxed);
        }

        self.recalculate();
    }

    /// Return the current adaptive timeout as a [`Duration`].
    ///
    /// Before any samples are recorded this returns `initial_timeout`.
    pub fn current_timeout(&self) -> Duration {
        Duration::from_micros(self.current_timeout_us.load(Ordering::Relaxed))
    }

    /// Return the number of samples recorded (capped at `window_size`).
    pub fn sample_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed).min(self.samples.len() as u64)
    }

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> AdaptiveTimeoutStats {
        let sorted = self.collect_sorted_samples();
        AdaptiveTimeoutStats {
            current_timeout: self.current_timeout(),
            p50: Self::compute_percentile(&sorted, &Percentile::P50)
                .map(Duration::from_micros),
            p95: Self::compute_percentile(&sorted, &Percentile::P95)
                .map(Duration::from_micros),
            p99: Self::compute_percentile(&sorted, &Percentile::P99)
                .map(Duration::from_micros),
            sample_count: self.sample_count(),
        }
    }

    /// Reset the tracker: zero all samples, reset the count, and restore
    /// the initial timeout.
    pub fn reset(&self) {
        for slot in &self.samples {
            slot.store(0, Ordering::Relaxed);
        }
        self.write_pos.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.current_timeout_us.store(
            self.config.initial_timeout.as_micros() as u64,
            Ordering::Relaxed,
        );
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Recalculate the timeout from the current ring buffer contents.
    ///
    /// Collects non-zero samples, sorts them, computes the configured
    /// percentile, applies the multiplier, and clamps the result to
    /// `[min_timeout, max_timeout]`.
    fn recalculate(&self) {
        let sorted = self.collect_sorted_samples();

        if let Some(pct_value) = Self::compute_percentile(&sorted, &self.config.percentile) {
            let raw_timeout_us = (pct_value as f64 * self.config.multiplier) as u64;
            let min_us = self.config.min_timeout.as_micros() as u64;
            let max_us = self.config.max_timeout.as_micros() as u64;
            let clamped = raw_timeout_us.clamp(min_us, max_us);
            self.current_timeout_us.store(clamped, Ordering::Relaxed);
        }
        // If no samples, leave current_timeout_us unchanged (initial_timeout).
    }

    /// Collect all non-zero samples from the ring buffer into a sorted `Vec`.
    fn collect_sorted_samples(&self) -> Vec<u64> {
        let n = self.sample_count() as usize;
        let mut vals: Vec<u64> = Vec::with_capacity(n);
        for i in 0..self.samples.len() {
            let v = self.samples[i].load(Ordering::Relaxed);
            if v != 0 {
                vals.push(v);
            }
        }
        vals.sort_unstable();
        vals
    }

    /// Compute the value at the given percentile from a pre-sorted slice.
    ///
    /// Returns `None` if `sorted_samples` is empty.
    fn compute_percentile(sorted_samples: &[u64], percentile: &Percentile) -> Option<u64> {
        if sorted_samples.is_empty() {
            return None;
        }
        let n = sorted_samples.len();
        let idx = ((percentile.as_fraction() * n as f64) as usize).min(n - 1);
        Some(sorted_samples[idx])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_initial_timeout_used_before_samples() {
        let config = AdaptiveTimeoutConfig {
            initial_timeout: Duration::from_secs(10),
            ..AdaptiveTimeoutConfig::default()
        };
        let at = AdaptiveTimeout::new(config);
        assert_eq!(at.current_timeout(), Duration::from_secs(10));
        assert_eq!(at.sample_count(), 0);
    }

    #[test]
    fn test_timeout_adapts_to_low_latency() {
        let config = AdaptiveTimeoutConfig {
            min_timeout: Duration::from_millis(10),
            max_timeout: Duration::from_secs(60),
            initial_timeout: Duration::from_secs(10),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        };
        let at = AdaptiveTimeout::new(config);

        // Record 100 low-latency samples (5ms each).
        for _ in 0..100 {
            at.record(Duration::from_millis(5));
        }

        let timeout = at.current_timeout();
        // Expected: 5ms * 2.0 = 10ms, clamped to [10ms, 60s] = 10ms.
        assert!(
            timeout <= Duration::from_millis(15),
            "Timeout should adapt low: got {:?}",
            timeout
        );
        assert!(
            timeout >= Duration::from_millis(10),
            "Timeout should respect min: got {:?}",
            timeout
        );
    }

    #[test]
    fn test_timeout_adapts_to_high_latency() {
        let config = AdaptiveTimeoutConfig {
            min_timeout: Duration::from_millis(100),
            max_timeout: Duration::from_secs(60),
            initial_timeout: Duration::from_secs(1),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        };
        let at = AdaptiveTimeout::new(config);

        // Record 100 high-latency samples (5s each).
        for _ in 0..100 {
            at.record(Duration::from_secs(5));
        }

        let timeout = at.current_timeout();
        // Expected: 5s * 2.0 = 10s, clamped to [100ms, 60s] = 10s.
        assert!(
            timeout >= Duration::from_secs(9),
            "Timeout should adapt to high latency: got {:?}",
            timeout
        );
        assert!(
            timeout <= Duration::from_secs(11),
            "Timeout should be around 10s: got {:?}",
            timeout
        );
    }

    #[test]
    fn test_respects_min_bound() {
        let config = AdaptiveTimeoutConfig {
            min_timeout: Duration::from_secs(2),
            max_timeout: Duration::from_secs(60),
            initial_timeout: Duration::from_secs(10),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        };
        let at = AdaptiveTimeout::new(config);

        // Record very fast latencies (1 microsecond).
        for _ in 0..100 {
            at.record(Duration::from_micros(1));
        }

        let timeout = at.current_timeout();
        // 1us * 2.0 = 2us, clamped to min = 2s.
        assert_eq!(
            timeout,
            Duration::from_secs(2),
            "Timeout should be clamped to min_timeout"
        );
    }

    #[test]
    fn test_respects_max_bound() {
        let config = AdaptiveTimeoutConfig {
            min_timeout: Duration::from_millis(100),
            max_timeout: Duration::from_secs(30),
            initial_timeout: Duration::from_secs(10),
            multiplier: 2.0,
            percentile: Percentile::P95,
            window_size: 100,
            sensitivity: 0.3,
        };
        let at = AdaptiveTimeout::new(config);

        // Record very slow latencies (60s each).
        for _ in 0..100 {
            at.record(Duration::from_secs(60));
        }

        let timeout = at.current_timeout();
        // 60s * 2.0 = 120s, clamped to max = 30s.
        assert_eq!(
            timeout,
            Duration::from_secs(30),
            "Timeout should be clamped to max_timeout"
        );
    }

    #[test]
    fn test_reset_restores_initial() {
        let config = AdaptiveTimeoutConfig {
            initial_timeout: Duration::from_secs(10),
            ..AdaptiveTimeoutConfig::default()
        };
        let at = AdaptiveTimeout::new(config);

        // Record some samples.
        for _ in 0..50 {
            at.record(Duration::from_millis(100));
        }
        assert!(at.sample_count() > 0);

        // Reset.
        at.reset();
        assert_eq!(at.sample_count(), 0);
        assert_eq!(at.current_timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_sample_count() {
        let config = AdaptiveTimeoutConfig {
            window_size: 50,
            ..AdaptiveTimeoutConfig::default()
        };
        let at = AdaptiveTimeout::new(config);

        assert_eq!(at.sample_count(), 0);
        at.record(Duration::from_millis(10));
        assert_eq!(at.sample_count(), 1);
        at.record(Duration::from_millis(20));
        assert_eq!(at.sample_count(), 2);

        // Fill to capacity.
        for _ in 0..48 {
            at.record(Duration::from_millis(30));
        }
        assert_eq!(at.sample_count(), 50);

        // Exceed capacity — count saturates.
        at.record(Duration::from_millis(40));
        assert_eq!(at.sample_count(), 50);
    }

    #[test]
    fn test_stats_percentiles() {
        let config = AdaptiveTimeoutConfig {
            window_size: 100,
            ..AdaptiveTimeoutConfig::default()
        };
        let at = AdaptiveTimeout::new(config);

        // Record 100 samples: 1ms, 2ms, ..., 100ms.
        for i in 1..=100 {
            at.record(Duration::from_millis(i));
        }

        let stats = at.stats();
        assert_eq!(stats.sample_count, 100);
        assert!(stats.p50.is_some());
        assert!(stats.p95.is_some());
        assert!(stats.p99.is_some());

        let p50_ms = stats.p50.map(|d| d.as_millis()).unwrap_or(0);
        let p95_ms = stats.p95.map(|d| d.as_millis()).unwrap_or(0);
        let p99_ms = stats.p99.map(|d| d.as_millis()).unwrap_or(0);

        // P50 ~ 50ms, P95 ~ 95ms, P99 ~ 99ms.
        assert!(p50_ms >= 45 && p50_ms <= 55, "P50 was {}ms", p50_ms);
        assert!(p95_ms >= 90 && p95_ms <= 100, "P95 was {}ms", p95_ms);
        assert!(p99_ms >= 95 && p99_ms <= 100, "P99 was {}ms", p99_ms);
    }

    #[test]
    fn test_preset_conservative() {
        let config = AdaptiveTimeoutConfig::conservative();
        assert_eq!(config.min_timeout, Duration::from_secs(1));
        assert_eq!(config.max_timeout, Duration::from_secs(120));
        assert!((config.multiplier - 3.0).abs() < f64::EPSILON);
        assert_eq!(config.percentile, Percentile::P99);
    }

    #[test]
    fn test_preset_responsive() {
        let config = AdaptiveTimeoutConfig::responsive();
        assert_eq!(config.min_timeout, Duration::from_millis(500));
        assert_eq!(config.max_timeout, Duration::from_secs(60));
        assert!((config.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.percentile, Percentile::P95);
    }

    #[test]
    fn test_preset_aggressive() {
        let config = AdaptiveTimeoutConfig::aggressive();
        assert_eq!(config.min_timeout, Duration::from_millis(200));
        assert_eq!(config.max_timeout, Duration::from_secs(30));
        assert!((config.multiplier - 1.5).abs() < f64::EPSILON);
        assert_eq!(config.percentile, Percentile::P95);
    }

    #[test]
    fn test_default_config() {
        let config = AdaptiveTimeoutConfig::default();
        assert_eq!(config.min_timeout, Duration::from_millis(500));
        assert_eq!(config.max_timeout, Duration::from_secs(60));
        assert_eq!(config.initial_timeout, Duration::from_secs(10));
        assert!((config.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.percentile, Percentile::P95);
        assert_eq!(config.window_size, 100);
        assert!((config.sensitivity - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ring_buffer_wraps_around() {
        let config = AdaptiveTimeoutConfig {
            min_timeout: Duration::from_millis(1),
            max_timeout: Duration::from_secs(600),
            initial_timeout: Duration::from_secs(10),
            multiplier: 1.0,
            percentile: Percentile::P50,
            window_size: 10,
            sensitivity: 0.3,
        };
        let at = AdaptiveTimeout::new(config);

        // Fill buffer with 10ms samples.
        for _ in 0..10 {
            at.record(Duration::from_millis(10));
        }
        assert_eq!(at.sample_count(), 10);

        // Overwrite with 500ms samples — old 10ms values should be gone.
        for _ in 0..10 {
            at.record(Duration::from_millis(500));
        }
        assert_eq!(at.sample_count(), 10);

        // P50 with multiplier 1.0 should be around 500ms now (all old data overwritten).
        let timeout = at.current_timeout();
        assert!(
            timeout >= Duration::from_millis(450),
            "Expected ~500ms after wrap, got {:?}",
            timeout
        );
    }

    #[test]
    fn test_concurrent_recording() {
        let config = AdaptiveTimeoutConfig {
            window_size: 1000,
            ..AdaptiveTimeoutConfig::default()
        };
        let at = Arc::new(AdaptiveTimeout::new(config));

        let mut handles = Vec::new();
        for t in 0..4 {
            let at_clone = Arc::clone(&at);
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    let latency_ms = (t * 250 + i + 1) as u64;
                    at_clone.record(Duration::from_millis(latency_ms));
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread should not panic");
        }

        // All 1000 slots should be filled.
        assert_eq!(at.sample_count(), 1000);

        // The timeout should be some reasonable positive value (not zero, not initial).
        let timeout = at.current_timeout();
        assert!(
            timeout > Duration::from_millis(0),
            "Timeout should be positive after concurrent recording"
        );

        // Stats should report all percentiles.
        let stats = at.stats();
        assert!(stats.p50.is_some());
        assert!(stats.p95.is_some());
        assert!(stats.p99.is_some());
    }

    #[test]
    fn test_empty_samples_returns_initial() {
        let config = AdaptiveTimeoutConfig {
            initial_timeout: Duration::from_secs(7),
            ..AdaptiveTimeoutConfig::default()
        };
        let at = AdaptiveTimeout::new(config);

        assert_eq!(at.sample_count(), 0);
        assert_eq!(at.current_timeout(), Duration::from_secs(7));

        let stats = at.stats();
        assert!(stats.p50.is_none());
        assert!(stats.p95.is_none());
        assert!(stats.p99.is_none());
        assert_eq!(stats.current_timeout, Duration::from_secs(7));
    }
}
