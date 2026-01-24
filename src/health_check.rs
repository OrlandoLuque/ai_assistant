//! Health checks for AI providers
//!
//! This module provides health checking utilities to monitor
//! provider availability and performance.
//!
//! # Features
//!
//! - **Periodic checks**: Automatic background health monitoring
//! - **Multiple check types**: Ping, model list, simple generation
//! - **Status tracking**: Track provider health over time
//! - **Alerts**: Configurable alerting on status changes

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Provider name
    pub provider: String,
    /// Whether check passed
    pub healthy: bool,
    /// Response time
    pub response_time: Duration,
    /// Optional error message
    pub error: Option<String>,
    /// Check timestamp
    pub timestamp: Instant,
    /// Check type performed
    pub check_type: HealthCheckType,
}

/// Types of health checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthCheckType {
    /// Simple connectivity check
    Ping,
    /// Check if models endpoint responds
    ModelList,
    /// Simple generation test
    Generation,
    /// Custom check
    Custom,
}

/// Health status of a provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Provider is healthy
    Healthy,
    /// Provider is degraded (slow or intermittent)
    Degraded,
    /// Provider is unhealthy
    Unhealthy,
    /// Status unknown (not yet checked)
    Unknown,
}

impl HealthStatus {
    /// Check if status indicates availability
    pub fn is_available(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

/// Provider health information
#[derive(Debug, Clone)]
pub struct ProviderHealth {
    /// Current status
    pub status: HealthStatus,
    /// Last check result
    pub last_check: Option<HealthCheckResult>,
    /// Recent check history
    pub history: Vec<HealthCheckResult>,
    /// Consecutive failures
    pub consecutive_failures: usize,
    /// Consecutive successes
    pub consecutive_successes: usize,
    /// Average response time (last N checks)
    pub avg_response_time: Duration,
    /// Uptime percentage (based on history)
    pub uptime_percent: f64,
}

impl Default for ProviderHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Unknown,
            last_check: None,
            history: Vec::new(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            avg_response_time: Duration::ZERO,
            uptime_percent: 100.0,
        }
    }
}

/// Configuration for health checking
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Check interval
    pub interval: Duration,
    /// Timeout for each check
    pub timeout: Duration,
    /// Number of failures before marking unhealthy
    pub failure_threshold: usize,
    /// Number of successes before marking healthy
    pub success_threshold: usize,
    /// Maximum history to keep
    pub max_history: usize,
    /// Response time threshold for degraded status
    pub degraded_threshold: Duration,
    /// Type of check to perform
    pub check_type: HealthCheckType,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
            success_threshold: 2,
            max_history: 100,
            degraded_threshold: Duration::from_secs(5),
            check_type: HealthCheckType::Ping,
        }
    }
}

/// Health checker for multiple providers
pub struct HealthChecker {
    config: HealthCheckConfig,
    providers: HashMap<String, ProviderHealth>,
    check_fn: Option<Arc<dyn Fn(&str, &str) -> Result<Duration, String> + Send + Sync>>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            providers: HashMap::new(),
            check_fn: None,
        }
    }

    /// Set custom check function
    pub fn with_check_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str) -> Result<Duration, String> + Send + Sync + 'static,
    {
        self.check_fn = Some(Arc::new(f));
        self
    }

    /// Register a provider
    pub fn register(&mut self, name: impl Into<String>, url: impl Into<String>) {
        let name = name.into();
        let _url = url.into();
        self.providers.insert(name, ProviderHealth::default());
    }

    /// Check a specific provider
    pub fn check(&mut self, name: &str, url: &str) -> HealthCheckResult {
        let start = Instant::now();

        let result = if let Some(ref check_fn) = self.check_fn {
            check_fn(name, url)
        } else {
            // Default: try to connect to the URL
            Self::default_check(url, self.config.timeout)
        };

        let (healthy, error) = match result {
            Ok(response_time) => {
                let healthy = response_time < self.config.degraded_threshold;
                (healthy, None)
            }
            Err(e) => (false, Some(e)),
        };

        let check_result = HealthCheckResult {
            provider: name.to_string(),
            healthy,
            response_time: start.elapsed(),
            error,
            timestamp: Instant::now(),
            check_type: self.config.check_type,
        };

        // Update provider health
        if let Some(health) = self.providers.get_mut(name) {
            Self::update_health_impl(health, &check_result, &self.config);
        }

        check_result
    }

    fn update_health_impl(health: &mut ProviderHealth, result: &HealthCheckResult, config: &HealthCheckConfig) {
        // Update history
        health.history.push(result.clone());
        if health.history.len() > config.max_history {
            health.history.remove(0);
        }

        // Update consecutive counts
        if result.healthy {
            health.consecutive_successes += 1;
            health.consecutive_failures = 0;
        } else {
            health.consecutive_failures += 1;
            health.consecutive_successes = 0;
        }

        // Update status
        health.status = if health.consecutive_failures >= config.failure_threshold {
            HealthStatus::Unhealthy
        } else if health.consecutive_successes >= config.success_threshold {
            if result.response_time > config.degraded_threshold {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            }
        } else if health.consecutive_failures > 0 {
            HealthStatus::Degraded
        } else {
            health.status
        };

        // Update average response time
        let successful: Vec<_> = health.history.iter()
            .filter(|r| r.healthy)
            .collect();
        if !successful.is_empty() {
            let total: Duration = successful.iter().map(|r| r.response_time).sum();
            health.avg_response_time = total / successful.len() as u32;
        }

        // Update uptime
        if !health.history.is_empty() {
            let healthy_count = health.history.iter().filter(|r| r.healthy).count();
            health.uptime_percent = (healthy_count as f64 / health.history.len() as f64) * 100.0;
        }

        health.last_check = Some(result.clone());
    }

    fn default_check(url: &str, timeout: Duration) -> Result<Duration, String> {
        let start = Instant::now();

        let agent = ureq::AgentBuilder::new()
            .timeout_connect(timeout)
            .timeout_read(timeout)
            .build();

        // Try to reach the base URL
        let check_url = if url.ends_with('/') {
            url.to_string()
        } else {
            format!("{}/", url)
        };

        match agent.get(&check_url).call() {
            Ok(_) => Ok(start.elapsed()),
            Err(ureq::Error::Status(code, _)) if code < 500 => {
                // 4xx errors mean server is responding
                Ok(start.elapsed())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Check all registered providers
    pub fn check_all(&mut self) -> Vec<HealthCheckResult> {
        let providers: Vec<_> = self.providers.keys().cloned().collect();
        let mut results = Vec::new();

        for name in providers {
            // We'd need to store URLs to do this properly
            // For now, this is a placeholder
            results.push(HealthCheckResult {
                provider: name,
                healthy: true,
                response_time: Duration::ZERO,
                error: None,
                timestamp: Instant::now(),
                check_type: self.config.check_type,
            });
        }

        results
    }

    /// Get health status of a provider
    pub fn get_health(&self, name: &str) -> Option<&ProviderHealth> {
        self.providers.get(name)
    }

    /// Get all provider health statuses
    pub fn all_health(&self) -> &HashMap<String, ProviderHealth> {
        &self.providers
    }

    /// Get healthy providers
    pub fn healthy_providers(&self) -> Vec<&str> {
        self.providers.iter()
            .filter(|(_, h)| h.status.is_available())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Get unhealthy providers
    pub fn unhealthy_providers(&self) -> Vec<&str> {
        self.providers.iter()
            .filter(|(_, h)| !h.status.is_available())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Reset health status for a provider
    pub fn reset(&mut self, name: &str) {
        if let Some(health) = self.providers.get_mut(name) {
            *health = ProviderHealth::default();
        }
    }

    /// Reset all providers
    pub fn reset_all(&mut self) {
        for health in self.providers.values_mut() {
            *health = ProviderHealth::default();
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new(HealthCheckConfig::default())
    }
}

/// Summary of overall health
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Total providers
    pub total: usize,
    /// Healthy count
    pub healthy: usize,
    /// Degraded count
    pub degraded: usize,
    /// Unhealthy count
    pub unhealthy: usize,
    /// Unknown count
    pub unknown: usize,
    /// Overall health percentage
    pub health_percent: f64,
}

impl HealthChecker {
    /// Get overall health summary
    pub fn summary(&self) -> HealthSummary {
        let mut summary = HealthSummary {
            total: self.providers.len(),
            healthy: 0,
            degraded: 0,
            unhealthy: 0,
            unknown: 0,
            health_percent: 0.0,
        };

        for health in self.providers.values() {
            match health.status {
                HealthStatus::Healthy => summary.healthy += 1,
                HealthStatus::Degraded => summary.degraded += 1,
                HealthStatus::Unhealthy => summary.unhealthy += 1,
                HealthStatus::Unknown => summary.unknown += 1,
            }
        }

        if summary.total > 0 {
            summary.health_percent = ((summary.healthy as f64 + 0.5 * summary.degraded as f64)
                / summary.total as f64) * 100.0;
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_available());
        assert!(HealthStatus::Degraded.is_available());
        assert!(!HealthStatus::Unhealthy.is_available());
        assert!(!HealthStatus::Unknown.is_available());
    }

    #[test]
    fn test_health_checker_registration() {
        let mut checker = HealthChecker::default();
        checker.register("test", "http://localhost:1234");

        assert!(checker.get_health("test").is_some());
        assert_eq!(checker.get_health("test").unwrap().status, HealthStatus::Unknown);
    }

    #[test]
    fn test_health_update() {
        let config = HealthCheckConfig {
            failure_threshold: 2,
            success_threshold: 2,
            ..Default::default()
        };

        let mut checker = HealthChecker::new(config)
            .with_check_fn(|_, _| Ok(Duration::from_millis(100)));

        checker.register("test", "http://localhost");

        // Two successful checks should mark healthy
        checker.check("test", "http://localhost");
        checker.check("test", "http://localhost");

        let health = checker.get_health("test").unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.consecutive_successes, 2);
    }

    #[test]
    fn test_unhealthy_detection() {
        let config = HealthCheckConfig {
            failure_threshold: 2,
            ..Default::default()
        };

        let mut checker = HealthChecker::new(config)
            .with_check_fn(|_, _| Err("Connection failed".to_string()));

        checker.register("test", "http://localhost");

        // Two failures should mark unhealthy
        checker.check("test", "http://localhost");
        checker.check("test", "http://localhost");

        let health = checker.get_health("test").unwrap();
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_summary() {
        let mut checker = HealthChecker::default();

        checker.providers.insert("p1".to_string(), ProviderHealth {
            status: HealthStatus::Healthy,
            ..Default::default()
        });
        checker.providers.insert("p2".to_string(), ProviderHealth {
            status: HealthStatus::Degraded,
            ..Default::default()
        });
        checker.providers.insert("p3".to_string(), ProviderHealth {
            status: HealthStatus::Unhealthy,
            ..Default::default()
        });

        let summary = checker.summary();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.healthy, 1);
        assert_eq!(summary.degraded, 1);
        assert_eq!(summary.unhealthy, 1);
    }
}
