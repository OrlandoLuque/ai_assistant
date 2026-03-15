// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Load shedding for graceful degradation under pressure.
//!
//! Provides intelligent request rejection based on system load signals
//! (CPU, memory, queue depth, latency) combined with request priority.
//! Integrates with [`RequestPriority`](crate::request_queue::RequestPriority).

use crate::request_queue::RequestPriority;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Strategy for deciding which requests to shed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheddingStrategy {
    /// Shed lowest-priority requests first.
    PriorityBased,
    /// Random shedding proportional to overload severity.
    Probabilistic,
    /// Shed requests that have waited longest (age-based).
    OldestFirst,
    /// Combine all signals: priority, age, and load metrics.
    Adaptive,
}

impl Default for SheddingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Configuration for load shedding behavior.
#[derive(Debug, Clone)]
pub struct LoadSheddingConfig {
    /// Shedding strategy to use.
    pub strategy: SheddingStrategy,
    /// CPU usage threshold (0.0–1.0) above which shedding activates.
    pub cpu_threshold: f64,
    /// Memory usage threshold (0.0–1.0) above which shedding activates.
    pub memory_threshold: f64,
    /// Queue depth threshold above which shedding activates.
    pub queue_depth_threshold: usize,
    /// P95 latency threshold above which shedding activates.
    pub latency_threshold: Duration,
    /// When true, `High`-priority requests are never shed.
    pub priority_protection: bool,
    /// Cooldown between shed decisions to avoid oscillation.
    pub cooldown: Duration,
}

impl Default for LoadSheddingConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

impl LoadSheddingConfig {
    /// Conservative: high thresholds, priority protection on.
    pub fn conservative() -> Self {
        Self {
            strategy: SheddingStrategy::Adaptive,
            cpu_threshold: 0.90,
            memory_threshold: 0.90,
            queue_depth_threshold: 500,
            latency_threshold: Duration::from_secs(30),
            priority_protection: true,
            cooldown: Duration::from_secs(5),
        }
    }

    /// Aggressive: lower thresholds, shed Low priority early.
    pub fn aggressive() -> Self {
        Self {
            strategy: SheddingStrategy::Adaptive,
            cpu_threshold: 0.70,
            memory_threshold: 0.80,
            queue_depth_threshold: 200,
            latency_threshold: Duration::from_secs(10),
            priority_protection: true,
            cooldown: Duration::from_secs(2),
        }
    }

    /// Disabled: always accepts. Useful for testing.
    pub fn disabled() -> Self {
        Self {
            strategy: SheddingStrategy::PriorityBased,
            cpu_threshold: 1.0,
            memory_threshold: 1.0,
            queue_depth_threshold: usize::MAX,
            latency_threshold: Duration::from_secs(u64::MAX),
            priority_protection: true,
            cooldown: Duration::ZERO,
        }
    }
}

// ============================================================================
// Load context and decisions
// ============================================================================

/// Snapshot of current system load used for shedding decisions.
#[derive(Debug, Clone)]
pub struct LoadContext {
    /// Current CPU utilization (0.0–1.0).
    pub cpu_load: f64,
    /// Current memory utilization (0.0–1.0).
    pub memory_load: f64,
    /// Current request queue depth.
    pub queue_depth: usize,
    /// Priority of the request being evaluated.
    pub priority: RequestPriority,
    /// How long the request has been waiting.
    pub request_age: Duration,
    /// Current P95 latency, if known.
    pub p95_latency: Option<Duration>,
}

/// Decision from load shedding evaluation.
#[derive(Debug, Clone)]
pub enum SheddingDecision {
    /// Accept the request for processing.
    Accept,
    /// Reject the request entirely.
    Shed {
        /// Human-readable reason for shedding.
        reason: String,
    },
    /// Accept but delay processing to reduce load.
    Throttle {
        /// Suggested delay before processing.
        delay: Duration,
    },
}

impl SheddingDecision {
    /// Returns `true` if the request was accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accept)
    }

    /// Returns `true` if the request was shed.
    pub fn is_shed(&self) -> bool {
        matches!(self, Self::Shed { .. })
    }

    /// Returns `true` if the request was throttled.
    pub fn is_throttled(&self) -> bool {
        matches!(self, Self::Throttle { .. })
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Accumulated load shedding statistics.
#[derive(Debug, Clone, Default)]
pub struct SheddingStats {
    /// Total requests evaluated.
    pub total_evaluated: u64,
    /// Total requests shed (rejected).
    pub total_shed: u64,
    /// Total requests throttled.
    pub total_throttled: u64,
    /// Shed count grouped by reason.
    pub shed_by_reason: HashMap<String, u64>,
}

impl SheddingStats {
    /// Percentage of evaluated requests that were shed.
    pub fn shed_rate(&self) -> f64 {
        if self.total_evaluated == 0 {
            0.0
        } else {
            self.total_shed as f64 / self.total_evaluated as f64
        }
    }
}

// ============================================================================
// Load Shedder
// ============================================================================

/// Evaluates incoming requests against system load and decides whether
/// to accept, shed, or throttle them.
pub struct LoadShedder {
    config: LoadSheddingConfig,
    stats: Mutex<SheddingStats>,
    last_shed: Mutex<Option<Instant>>,
}

impl LoadShedder {
    /// Create a new load shedder with the given configuration.
    pub fn new(config: LoadSheddingConfig) -> Self {
        Self {
            config,
            stats: Mutex::new(SheddingStats::default()),
            last_shed: Mutex::new(None),
        }
    }

    /// Evaluate whether a request should be accepted, shed, or throttled.
    pub fn evaluate(&self, context: &LoadContext) -> SheddingDecision {
        // Priority protection: High-priority requests never shed
        if self.config.priority_protection && context.priority == RequestPriority::High {
            return SheddingDecision::Accept;
        }

        // Check if any threshold is exceeded
        let cpu_exceeded = context.cpu_load >= self.config.cpu_threshold;
        let memory_exceeded = context.memory_load >= self.config.memory_threshold;
        let queue_exceeded = context.queue_depth >= self.config.queue_depth_threshold;
        let latency_exceeded = context
            .p95_latency
            .map_or(false, |l| l >= self.config.latency_threshold);

        let any_exceeded = cpu_exceeded || memory_exceeded || queue_exceeded || latency_exceeded;

        if !any_exceeded {
            return SheddingDecision::Accept;
        }

        // Build reason
        let mut reasons = Vec::new();
        if cpu_exceeded {
            reasons.push(format!("cpu={:.0}%", context.cpu_load * 100.0));
        }
        if memory_exceeded {
            reasons.push(format!("memory={:.0}%", context.memory_load * 100.0));
        }
        if queue_exceeded {
            reasons.push(format!("queue_depth={}", context.queue_depth));
        }
        if latency_exceeded {
            reasons.push(format!("p95={}ms", context.p95_latency.map_or(0, |l| l.as_millis() as u64)));
        }
        let reason = reasons.join(", ");

        match self.config.strategy {
            SheddingStrategy::PriorityBased => self.evaluate_priority_based(context, reason),
            SheddingStrategy::Probabilistic => {
                self.evaluate_probabilistic(context, cpu_exceeded, memory_exceeded, queue_exceeded, reason)
            }
            SheddingStrategy::OldestFirst => self.evaluate_oldest_first(context, reason),
            SheddingStrategy::Adaptive => self.evaluate_adaptive(
                context,
                cpu_exceeded,
                memory_exceeded,
                queue_exceeded,
                latency_exceeded,
                reason,
            ),
        }
    }

    /// Record a shedding decision in statistics.
    pub fn record_decision(&self, decision: &SheddingDecision) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.total_evaluated += 1;

        match decision {
            SheddingDecision::Shed { reason } => {
                stats.total_shed += 1;
                *stats.shed_by_reason.entry(reason.clone()).or_insert(0) += 1;
                drop(stats);
                let mut last = self.last_shed.lock().unwrap_or_else(|e| e.into_inner());
                *last = Some(Instant::now());
            }
            SheddingDecision::Throttle { .. } => {
                stats.total_throttled += 1;
            }
            SheddingDecision::Accept => {}
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> SheddingStats {
        self.stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Reset all statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = SheddingStats::default();
    }

    /// Check if the shedder is in cooldown (recently shed a request).
    pub fn is_in_cooldown(&self) -> bool {
        let last = self.last_shed.lock().unwrap_or_else(|e| e.into_inner());
        match *last {
            Some(t) => t.elapsed() < self.config.cooldown,
            None => false,
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &LoadSheddingConfig {
        &self.config
    }

    // ---- Strategy implementations ----

    fn evaluate_priority_based(&self, context: &LoadContext, reason: String) -> SheddingDecision {
        match context.priority {
            RequestPriority::Low => SheddingDecision::Shed { reason },
            RequestPriority::Normal => SheddingDecision::Throttle {
                delay: Duration::from_millis(500),
            },
            RequestPriority::High => SheddingDecision::Accept,
        }
    }

    fn evaluate_probabilistic(
        &self,
        _context: &LoadContext,
        cpu_exceeded: bool,
        memory_exceeded: bool,
        queue_exceeded: bool,
        reason: String,
    ) -> SheddingDecision {
        // Count how many thresholds are exceeded → higher = more likely to shed
        let exceeded_count =
            cpu_exceeded as u32 + memory_exceeded as u32 + queue_exceeded as u32;
        let shed_probability = match exceeded_count {
            0 => return SheddingDecision::Accept,
            1 => 0.25,
            2 => 0.50,
            _ => 0.75,
        };

        // Use a simple deterministic signal based on queue depth parity
        // (real production would use random, but we avoid rand dependency)
        let signal = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as f64
            / 1_000_000_000.0;

        if signal < shed_probability {
            SheddingDecision::Shed { reason }
        } else {
            SheddingDecision::Throttle {
                delay: Duration::from_millis(100 * exceeded_count as u64),
            }
        }
    }

    fn evaluate_oldest_first(&self, context: &LoadContext, reason: String) -> SheddingDecision {
        // Shed requests that have waited more than 2x the latency threshold
        let age_threshold = self.config.latency_threshold.saturating_mul(2);
        if context.request_age >= age_threshold {
            SheddingDecision::Shed { reason }
        } else if context.request_age >= self.config.latency_threshold {
            SheddingDecision::Throttle {
                delay: Duration::from_millis(200),
            }
        } else {
            SheddingDecision::Accept
        }
    }

    fn evaluate_adaptive(
        &self,
        context: &LoadContext,
        cpu_exceeded: bool,
        memory_exceeded: bool,
        queue_exceeded: bool,
        latency_exceeded: bool,
        reason: String,
    ) -> SheddingDecision {
        let exceeded_count = cpu_exceeded as u32
            + memory_exceeded as u32
            + queue_exceeded as u32
            + latency_exceeded as u32;

        // Severity score: combine overload count with priority
        let priority_factor = match context.priority {
            RequestPriority::High => 0.0,   // protected
            RequestPriority::Normal => 0.5,
            RequestPriority::Low => 1.0,
        };

        let severity = exceeded_count as f64 * priority_factor;

        if severity >= 2.0 {
            SheddingDecision::Shed { reason }
        } else if severity >= 1.0 {
            SheddingDecision::Throttle {
                delay: Duration::from_millis((severity * 500.0) as u64),
            }
        } else {
            SheddingDecision::Accept
        }
    }
}

impl Default for LoadShedder {
    fn default() -> Self {
        Self::new(LoadSheddingConfig::default())
    }
}

// ============================================================================
// Display implementations
// ============================================================================

impl std::fmt::Display for SheddingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PriorityBased => write!(f, "PriorityBased"),
            Self::Probabilistic => write!(f, "Probabilistic"),
            Self::OldestFirst => write!(f, "OldestFirst"),
            Self::Adaptive => write!(f, "Adaptive"),
        }
    }
}

impl std::fmt::Display for SheddingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Accept => write!(f, "Accept"),
            Self::Shed { reason } => write!(f, "Shed({})", reason),
            Self::Throttle { delay } => write!(f, "Throttle({}ms)", delay.as_millis()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_context() -> LoadContext {
        LoadContext {
            cpu_load: 0.50,
            memory_load: 0.50,
            queue_depth: 10,
            priority: RequestPriority::Normal,
            request_age: Duration::from_millis(100),
            p95_latency: Some(Duration::from_secs(1)),
        }
    }

    fn overloaded_context() -> LoadContext {
        LoadContext {
            cpu_load: 0.95,
            memory_load: 0.92,
            queue_depth: 600,
            priority: RequestPriority::Normal,
            request_age: Duration::from_millis(100),
            p95_latency: Some(Duration::from_secs(35)),
        }
    }

    #[test]
    fn test_accept_under_normal_load() {
        let shedder = LoadShedder::default();
        let decision = shedder.evaluate(&normal_context());
        assert!(decision.is_accepted());
    }

    #[test]
    fn test_high_priority_never_shed() {
        let shedder = LoadShedder::new(LoadSheddingConfig::aggressive());
        let mut ctx = overloaded_context();
        ctx.priority = RequestPriority::High;
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_accepted());
    }

    #[test]
    fn test_priority_based_low_shed() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            strategy: SheddingStrategy::PriorityBased,
            cpu_threshold: 0.80,
            memory_threshold: 0.80,
            queue_depth_threshold: 100,
            latency_threshold: Duration::from_secs(10),
            priority_protection: false,
            cooldown: Duration::ZERO,
        });

        let mut ctx = overloaded_context();
        ctx.priority = RequestPriority::Low;
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_shed());
    }

    #[test]
    fn test_priority_based_normal_throttle() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            strategy: SheddingStrategy::PriorityBased,
            cpu_threshold: 0.80,
            memory_threshold: 0.80,
            queue_depth_threshold: 100,
            latency_threshold: Duration::from_secs(10),
            priority_protection: false,
            cooldown: Duration::ZERO,
        });

        let mut ctx = overloaded_context();
        ctx.priority = RequestPriority::Normal;
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_throttled());
    }

    #[test]
    fn test_oldest_first_old_request_shed() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            strategy: SheddingStrategy::OldestFirst,
            cpu_threshold: 0.80,
            memory_threshold: 0.80,
            queue_depth_threshold: 100,
            latency_threshold: Duration::from_secs(10),
            priority_protection: false,
            cooldown: Duration::ZERO,
        });

        let mut ctx = overloaded_context();
        ctx.request_age = Duration::from_secs(25); // > 2x latency_threshold (20s)
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_shed());
    }

    #[test]
    fn test_oldest_first_medium_age_throttle() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            strategy: SheddingStrategy::OldestFirst,
            cpu_threshold: 0.80,
            memory_threshold: 0.80,
            queue_depth_threshold: 100,
            latency_threshold: Duration::from_secs(10),
            priority_protection: false,
            cooldown: Duration::ZERO,
        });

        let mut ctx = overloaded_context();
        ctx.request_age = Duration::from_secs(15); // > latency_threshold, < 2x
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_throttled());
    }

    #[test]
    fn test_adaptive_low_priority_overloaded_shed() {
        let shedder = LoadShedder::new(LoadSheddingConfig::aggressive());
        let mut ctx = overloaded_context();
        ctx.priority = RequestPriority::Low;
        let decision = shedder.evaluate(&ctx);
        // severity = 4 * 1.0 = 4.0 >= 2.0 → Shed
        assert!(decision.is_shed());
    }

    #[test]
    fn test_adaptive_normal_single_threshold() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            strategy: SheddingStrategy::Adaptive,
            cpu_threshold: 0.80,
            memory_threshold: 0.95,
            queue_depth_threshold: 1000,
            latency_threshold: Duration::from_secs(60),
            priority_protection: false,
            cooldown: Duration::ZERO,
        });

        // Only CPU exceeded
        let ctx = LoadContext {
            cpu_load: 0.85,
            memory_load: 0.50,
            queue_depth: 10,
            priority: RequestPriority::Normal,
            request_age: Duration::from_millis(100),
            p95_latency: Some(Duration::from_secs(1)),
        };

        let decision = shedder.evaluate(&ctx);
        // severity = 1 * 0.5 = 0.5 < 1.0 → Accept
        assert!(decision.is_accepted());
    }

    #[test]
    fn test_disabled_config_accepts_all() {
        let shedder = LoadShedder::new(LoadSheddingConfig::disabled());
        let ctx = overloaded_context();
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_accepted());
    }

    #[test]
    fn test_record_decision_stats() {
        let shedder = LoadShedder::default();

        shedder.record_decision(&SheddingDecision::Accept);
        shedder.record_decision(&SheddingDecision::Shed {
            reason: "cpu=95%".to_string(),
        });
        shedder.record_decision(&SheddingDecision::Throttle {
            delay: Duration::from_millis(100),
        });

        let stats = shedder.stats();
        assert_eq!(stats.total_evaluated, 3);
        assert_eq!(stats.total_shed, 1);
        assert_eq!(stats.total_throttled, 1);
        assert_eq!(*stats.shed_by_reason.get("cpu=95%").unwrap_or(&0), 1);
    }

    #[test]
    fn test_reset_stats() {
        let shedder = LoadShedder::default();
        shedder.record_decision(&SheddingDecision::Shed {
            reason: "test".to_string(),
        });
        assert_eq!(shedder.stats().total_shed, 1);

        shedder.reset_stats();
        let stats = shedder.stats();
        assert_eq!(stats.total_evaluated, 0);
        assert_eq!(stats.total_shed, 0);
    }

    #[test]
    fn test_cooldown() {
        let shedder = LoadShedder::new(LoadSheddingConfig {
            cooldown: Duration::from_secs(10),
            ..LoadSheddingConfig::default()
        });

        assert!(!shedder.is_in_cooldown());

        shedder.record_decision(&SheddingDecision::Shed {
            reason: "test".to_string(),
        });

        assert!(shedder.is_in_cooldown());
    }

    #[test]
    fn test_shed_rate() {
        let stats = SheddingStats {
            total_evaluated: 100,
            total_shed: 25,
            total_throttled: 10,
            shed_by_reason: HashMap::new(),
        };
        let rate = stats.shed_rate();
        assert!((rate - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shed_rate_zero_evaluated() {
        let stats = SheddingStats::default();
        assert!((stats.shed_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decision_display() {
        assert_eq!(SheddingDecision::Accept.to_string(), "Accept");
        assert_eq!(
            SheddingDecision::Shed {
                reason: "cpu".to_string()
            }
            .to_string(),
            "Shed(cpu)"
        );
        assert_eq!(
            SheddingDecision::Throttle {
                delay: Duration::from_millis(500)
            }
            .to_string(),
            "Throttle(500ms)"
        );
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(SheddingStrategy::PriorityBased.to_string(), "PriorityBased");
        assert_eq!(SheddingStrategy::Adaptive.to_string(), "Adaptive");
    }

    #[test]
    fn test_no_p95_latency() {
        let shedder = LoadShedder::default();
        let ctx = LoadContext {
            cpu_load: 0.50,
            memory_load: 0.50,
            queue_depth: 10,
            priority: RequestPriority::Normal,
            request_age: Duration::from_millis(100),
            p95_latency: None,
        };
        let decision = shedder.evaluate(&ctx);
        assert!(decision.is_accepted());
    }
}
