// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Chaos engineering and fault injection framework.
//!
//! Provides a configurable fault injector that can simulate various failure modes
//! (latency, errors, timeouts, connection resets, rate limiting, partial/corrupt
//! responses) against named targets or message-matching patterns. Designed for
//! deterministic testing via a seeded xorshift64 PRNG.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The kind of fault to inject.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FaultType {
    /// Simulate additional latency between `min` and `max`.
    Latency {
        /// Minimum latency to add.
        min: Duration,
        /// Maximum latency to add.
        max: Duration,
    },
    /// Return an error with the given message.
    Error {
        /// Human-readable error description.
        message: String,
    },
    /// Simulate a timeout (no response).
    Timeout,
    /// Simulate a connection reset.
    ConnectionReset,
    /// Simulate rate-limiting with a `Retry-After` hint.
    RateLimited {
        /// Suggested retry delay.
        retry_after: Duration,
    },
    /// Truncate the response by `truncate_percent` (0.0–1.0).
    PartialResponse {
        /// Fraction of the response to drop.
        truncate_percent: f64,
    },
    /// Return a garbled / corrupted response.
    CorruptResponse,
}

/// Determines which requests a [`FaultRule`] applies to.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FaultTarget {
    /// Matches every request regardless of target name or message content.
    All,
    /// Matches only when the target name equals the contained string.
    Named(String),
    /// Matches when the message contains the given substring.
    Matching(String),
}

/// A single fault-injection rule.
#[derive(Clone, Debug)]
pub struct FaultRule {
    /// Human-readable name (also used as the key in stats).
    pub name: String,
    /// What kind of fault to inject.
    pub fault_type: FaultType,
    /// Which requests this rule applies to.
    pub target: FaultTarget,
    /// Probability of injection per check (0.0 = never, 1.0 = always).
    pub probability: f64,
    /// Optional cap on the total number of injections for this rule.
    pub max_injections: Option<u64>,
    /// Whether the rule is currently active.
    pub enabled: bool,
}

impl FaultRule {
    /// Create a latency-injection rule.
    pub fn latency(
        name: &str,
        target: FaultTarget,
        min: Duration,
        max: Duration,
        probability: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            fault_type: FaultType::Latency { min, max },
            target,
            probability,
            max_injections: None,
            enabled: true,
        }
    }

    /// Create an error-injection rule.
    pub fn error(name: &str, target: FaultTarget, message: &str, probability: f64) -> Self {
        Self {
            name: name.to_string(),
            fault_type: FaultType::Error {
                message: message.to_string(),
            },
            target,
            probability,
            max_injections: None,
            enabled: true,
        }
    }

    /// Create a timeout-injection rule.
    pub fn timeout(name: &str, target: FaultTarget, probability: f64) -> Self {
        Self {
            name: name.to_string(),
            fault_type: FaultType::Timeout,
            target,
            probability,
            max_injections: None,
            enabled: true,
        }
    }

    /// Create a rate-limit injection rule.
    pub fn rate_limit(
        name: &str,
        target: FaultTarget,
        retry_after: Duration,
        probability: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            fault_type: FaultType::RateLimited { retry_after },
            target,
            probability,
            max_injections: None,
            enabled: true,
        }
    }
}

/// The outcome of a single [`FaultInjector::check`] call.
#[derive(Clone, Debug)]
pub struct FaultResult {
    /// Whether a fault was actually injected.
    pub injected: bool,
    /// The type of fault that was injected, if any.
    pub fault_type: Option<FaultType>,
    /// Name of the rule that fired, if any.
    pub rule_name: Option<String>,
}

/// Aggregate statistics for a [`FaultInjector`].
#[derive(Clone, Debug, Default)]
pub struct FaultStats {
    /// Total number of [`FaultInjector::check`] invocations.
    pub total_checked: u64,
    /// Total number of faults actually injected.
    pub total_injected: u64,
    /// Per-rule injection counts keyed by rule name.
    pub by_rule: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// FaultInjector
// ---------------------------------------------------------------------------

/// A configurable chaos-engineering fault injector.
///
/// Rules are evaluated in insertion order; the first enabled, matching rule
/// whose probability roll succeeds (and whose `max_injections` cap has not been
/// reached) wins.
pub struct FaultInjector {
    rules: Vec<FaultRule>,
    stats: Mutex<FaultStats>,
    injections_count: Mutex<HashMap<String, u64>>,
    rng_state: Mutex<u64>,
    #[allow(dead_code)]
    seeded: bool,
}

impl FaultInjector {
    /// Create a new injector with a time-based (non-deterministic) seed.
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        // Avoid a zero seed which would make xorshift produce only zeros.
        let seed = if seed == 0 { 1 } else { seed };
        Self {
            rules: Vec::new(),
            stats: Mutex::new(FaultStats::default()),
            injections_count: Mutex::new(HashMap::new()),
            rng_state: Mutex::new(seed),
            seeded: false,
        }
    }

    /// Create a new injector with a fixed seed for deterministic testing.
    pub fn with_seed(seed: u64) -> Self {
        let seed = if seed == 0 { 1 } else { seed };
        Self {
            rules: Vec::new(),
            stats: Mutex::new(FaultStats::default()),
            injections_count: Mutex::new(HashMap::new()),
            rng_state: Mutex::new(seed),
            seeded: true,
        }
    }

    /// Append a rule. Rules are evaluated in insertion order.
    pub fn add_rule(&mut self, rule: FaultRule) -> &mut Self {
        self.rules.push(rule);
        self
    }

    /// Remove a rule by name. Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.name != name);
        self.rules.len() < before
    }

    /// Enable a rule by name. Returns `true` if the rule was found.
    pub fn enable_rule(&mut self, name: &str) -> bool {
        for rule in &mut self.rules {
            if rule.name == name {
                rule.enabled = true;
                return true;
            }
        }
        false
    }

    /// Disable a rule by name. Returns `true` if the rule was found.
    pub fn disable_rule(&mut self, name: &str) -> bool {
        for rule in &mut self.rules {
            if rule.name == name {
                rule.enabled = false;
                return true;
            }
        }
        false
    }

    /// Evaluate all enabled rules against `target_name` / `message`.
    ///
    /// The first matching rule whose probability roll succeeds and whose
    /// `max_injections` cap has not been reached is returned. Stats are updated
    /// regardless of injection outcome.
    pub fn check(&self, target_name: &str, message: &str) -> FaultResult {
        // Increment total_checked
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_checked += 1;
        }

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            if !Self::matches_target(rule, target_name, message) {
                continue;
            }
            if !self.should_inject(rule.probability) {
                continue;
            }

            // Check max_injections cap
            if let Some(max) = rule.max_injections {
                let mut counts = self.injections_count.lock().unwrap_or_else(|e| e.into_inner());
                let current = counts.get(&rule.name).copied().unwrap_or(0);
                if current >= max {
                    continue;
                }
                *counts.entry(rule.name.clone()).or_insert(0) += 1;
            } else {
                let mut counts = self.injections_count.lock().unwrap_or_else(|e| e.into_inner());
                *counts.entry(rule.name.clone()).or_insert(0) += 1;
            }

            // Update stats
            {
                let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                stats.total_injected += 1;
                *stats.by_rule.entry(rule.name.clone()).or_insert(0) += 1;
            }

            return FaultResult {
                injected: true,
                fault_type: Some(rule.fault_type.clone()),
                rule_name: Some(rule.name.clone()),
            };
        }

        FaultResult {
            injected: false,
            fault_type: None,
            rule_name: None,
        }
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> FaultStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Reset all statistics to zero.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = FaultStats::default();
        let mut counts = self.injections_count.lock().unwrap_or_else(|e| e.into_inner());
        counts.clear();
    }

    /// Remove all rules.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Return the number of rules currently registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    // -- private helpers ----------------------------------------------------

    fn matches_target(rule: &FaultRule, target_name: &str, message: &str) -> bool {
        match &rule.target {
            FaultTarget::All => true,
            FaultTarget::Named(name) => target_name == name,
            FaultTarget::Matching(substring) => message.contains(substring.as_str()),
        }
    }

    /// Xorshift64 PRNG producing a value in `[0.0, 1.0)`.
    fn next_random(&self) -> f64 {
        let mut state = self.rng_state.lock().unwrap_or_else(|e| e.into_inner());
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x as f64) / (u64::MAX as f64)
    }

    fn should_inject(&self, probability: f64) -> bool {
        self.next_random() < probability
    }
}

impl Default for FaultInjector {
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

    // -- helpers ------------------------------------------------------------

    fn seeded_injector() -> FaultInjector {
        FaultInjector::with_seed(42)
    }

    // -- tests --------------------------------------------------------------

    #[test]
    fn test_empty_injector_no_injection() {
        let injector = seeded_injector();
        let result = injector.check("any", "any message");
        assert!(!result.injected);
        assert!(result.fault_type.is_none());
        assert!(result.rule_name.is_none());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mut inj1 = FaultInjector::with_seed(123);
        let mut inj2 = FaultInjector::with_seed(123);

        let rule = FaultRule::timeout("t", FaultTarget::All, 0.5);
        inj1.add_rule(rule.clone());
        inj2.add_rule(rule);

        let mut results1 = Vec::new();
        let mut results2 = Vec::new();
        for _ in 0..20 {
            results1.push(inj1.check("x", "m").injected);
            results2.push(inj2.check("x", "m").injected);
        }
        assert_eq!(results1, results2);
    }

    #[test]
    fn test_probability_zero_never_injects() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("never", FaultTarget::All, 0.0));
        for _ in 0..100 {
            assert!(!injector.check("a", "b").injected);
        }
    }

    #[test]
    fn test_probability_one_always_injects() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("always", FaultTarget::All, 1.0));
        for _ in 0..100 {
            assert!(injector.check("a", "b").injected);
        }
    }

    #[test]
    fn test_target_all_matches_everything() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("all", FaultTarget::All, 1.0));
        assert!(injector.check("foo", "bar").injected);
        assert!(injector.check("baz", "qux").injected);
        assert!(injector.check("", "").injected);
    }

    #[test]
    fn test_target_named_matches_only_name() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout(
            "named",
            FaultTarget::Named("target-a".to_string()),
            1.0,
        ));
        assert!(injector.check("target-a", "any message").injected);
        assert!(!injector.check("target-b", "any message").injected);
        assert!(!injector.check("", "target-a in message").injected);
    }

    #[test]
    fn test_target_matching_substring() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout(
            "match",
            FaultTarget::Matching("error".to_string()),
            1.0,
        ));
        assert!(injector.check("any", "this is an error case").injected);
        assert!(injector.check("any", "error").injected);
        assert!(!injector.check("any", "everything is fine").injected);
    }

    #[test]
    fn test_max_injections_cap() {
        let mut injector = seeded_injector();
        let mut rule = FaultRule::timeout("capped", FaultTarget::All, 1.0);
        rule.max_injections = Some(3);
        injector.add_rule(rule);

        let mut injected_count = 0u64;
        for _ in 0..10 {
            if injector.check("a", "b").injected {
                injected_count += 1;
            }
        }
        assert_eq!(injected_count, 3);
    }

    #[test]
    fn test_enable_disable_rule() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("toggle", FaultTarget::All, 1.0));

        // Starts enabled
        assert!(injector.check("a", "b").injected);

        // Disable
        assert!(injector.disable_rule("toggle"));
        assert!(!injector.check("a", "b").injected);

        // Enable again
        assert!(injector.enable_rule("toggle"));
        assert!(injector.check("a", "b").injected);

        // Non-existent rule returns false
        assert!(!injector.disable_rule("nonexistent"));
        assert!(!injector.enable_rule("nonexistent"));
    }

    #[test]
    fn test_remove_rule() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("removeme", FaultTarget::All, 1.0));
        assert_eq!(injector.rule_count(), 1);

        assert!(injector.remove_rule("removeme"));
        assert_eq!(injector.rule_count(), 0);
        assert!(!injector.check("a", "b").injected);

        // Removing non-existent returns false
        assert!(!injector.remove_rule("removeme"));
    }

    #[test]
    fn test_stats_tracking() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("s", FaultTarget::All, 1.0));

        for _ in 0..5 {
            injector.check("a", "b");
        }
        let stats = injector.stats();
        assert_eq!(stats.total_checked, 5);
        assert_eq!(stats.total_injected, 5);
    }

    #[test]
    fn test_stats_by_rule() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout(
            "rule-a",
            FaultTarget::Named("a".to_string()),
            1.0,
        ));
        injector.add_rule(FaultRule::timeout(
            "rule-b",
            FaultTarget::Named("b".to_string()),
            1.0,
        ));

        injector.check("a", "m");
        injector.check("a", "m");
        injector.check("b", "m");

        let stats = injector.stats();
        assert_eq!(stats.by_rule.get("rule-a").copied().unwrap_or(0), 2);
        assert_eq!(stats.by_rule.get("rule-b").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_reset_stats() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("r", FaultTarget::All, 1.0));

        injector.check("a", "b");
        injector.check("a", "b");
        assert_eq!(injector.stats().total_injected, 2);

        injector.reset_stats();
        let stats = injector.stats();
        assert_eq!(stats.total_checked, 0);
        assert_eq!(stats.total_injected, 0);
        assert!(stats.by_rule.is_empty());
    }

    #[test]
    fn test_clear_rules() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::timeout("a", FaultTarget::All, 1.0));
        injector.add_rule(FaultRule::timeout("b", FaultTarget::All, 1.0));
        assert_eq!(injector.rule_count(), 2);

        injector.clear_rules();
        assert_eq!(injector.rule_count(), 0);
        assert!(!injector.check("x", "y").injected);
    }

    #[test]
    fn test_fault_rule_latency_builder() {
        let rule = FaultRule::latency(
            "lat",
            FaultTarget::All,
            Duration::from_millis(10),
            Duration::from_millis(100),
            0.5,
        );
        assert_eq!(rule.name, "lat");
        assert!(matches!(
            rule.fault_type,
            FaultType::Latency { min, max }
            if min == Duration::from_millis(10) && max == Duration::from_millis(100)
        ));
        assert!((rule.probability - 0.5).abs() < f64::EPSILON);
        assert!(rule.enabled);
        assert!(rule.max_injections.is_none());
    }

    #[test]
    fn test_fault_rule_error_builder() {
        let rule = FaultRule::error("err", FaultTarget::All, "boom", 0.8);
        assert_eq!(rule.name, "err");
        assert!(matches!(
            &rule.fault_type,
            FaultType::Error { message } if message == "boom"
        ));
        assert!((rule.probability - 0.8).abs() < f64::EPSILON);
        assert!(rule.enabled);
    }

    #[test]
    fn test_fault_rule_timeout_builder() {
        let rule = FaultRule::timeout("to", FaultTarget::All, 0.3);
        assert_eq!(rule.name, "to");
        assert!(matches!(rule.fault_type, FaultType::Timeout));
        assert!((rule.probability - 0.3).abs() < f64::EPSILON);
        assert!(rule.enabled);
    }

    #[test]
    fn test_fault_rule_rate_limit_builder() {
        let rule = FaultRule::rate_limit(
            "rl",
            FaultTarget::All,
            Duration::from_secs(30),
            0.1,
        );
        assert_eq!(rule.name, "rl");
        assert!(matches!(
            rule.fault_type,
            FaultType::RateLimited { retry_after }
            if retry_after == Duration::from_secs(30)
        ));
        assert!((rule.probability - 0.1).abs() < f64::EPSILON);
        assert!(rule.enabled);
    }

    #[test]
    fn test_multiple_rules_first_match_wins() {
        let mut injector = seeded_injector();
        injector.add_rule(FaultRule::error(
            "first",
            FaultTarget::All,
            "first error",
            1.0,
        ));
        injector.add_rule(FaultRule::timeout("second", FaultTarget::All, 1.0));

        let result = injector.check("any", "msg");
        assert!(result.injected);
        assert_eq!(result.rule_name.as_deref(), Some("first"));
        assert!(matches!(
            &result.fault_type,
            Some(FaultType::Error { message }) if message == "first error"
        ));
    }

    #[test]
    fn test_disabled_rule_skipped() {
        let mut injector = seeded_injector();
        let mut rule = FaultRule::timeout("disabled", FaultTarget::All, 1.0);
        rule.enabled = false;
        injector.add_rule(rule);

        assert!(!injector.check("a", "b").injected);
        assert_eq!(injector.stats().total_injected, 0);
        assert_eq!(injector.stats().total_checked, 1);
    }
}
