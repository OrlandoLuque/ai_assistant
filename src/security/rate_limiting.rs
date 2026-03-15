//! Rate limiting for controlling request frequency

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: usize,
    /// Maximum tokens per minute (output)
    pub tokens_per_minute: usize,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Cooldown period after hitting limit (seconds)
    pub cooldown_seconds: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 30,
            tokens_per_minute: 10000,
            max_concurrent: 2,
            cooldown_seconds: 30,
        }
    }
}

/// Rate limiter for controlling request frequency
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    request_times: VecDeque<Instant>,
    token_counts: VecDeque<(Instant, usize)>,
    current_concurrent: usize,
    cooldown_until: Option<Instant>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            request_times: VecDeque::new(),
            token_counts: VecDeque::new(),
            current_concurrent: 0,
            cooldown_until: None,
        }
    }

    /// Check if a request is allowed
    pub fn check_allowed(&mut self) -> RateLimitResult {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        // Check cooldown
        if let Some(until) = self.cooldown_until {
            if now < until {
                let remaining = until.duration_since(now);
                return RateLimitResult::Denied {
                    reason: RateLimitReason::Cooldown,
                    retry_after_ms: remaining.as_millis() as u64,
                };
            }
            self.cooldown_until = None;
        }

        // Clean old entries
        while self
            .request_times
            .front()
            .map(|t| *t < one_minute_ago)
            .unwrap_or(false)
        {
            self.request_times.pop_front();
        }
        while self
            .token_counts
            .front()
            .map(|(t, _)| *t < one_minute_ago)
            .unwrap_or(false)
        {
            self.token_counts.pop_front();
        }

        // Check concurrent limit
        if self.current_concurrent >= self.config.max_concurrent {
            return RateLimitResult::Denied {
                reason: RateLimitReason::TooManyConcurrent,
                retry_after_ms: 1000,
            };
        }

        // Check requests per minute
        if self.request_times.len() >= self.config.requests_per_minute {
            self.trigger_cooldown();
            return RateLimitResult::Denied {
                reason: RateLimitReason::TooManyRequests,
                retry_after_ms: self.config.cooldown_seconds * 1000,
            };
        }

        // Check tokens per minute
        let total_tokens: usize = self.token_counts.iter().map(|(_, t)| *t).sum();
        if total_tokens >= self.config.tokens_per_minute {
            self.trigger_cooldown();
            return RateLimitResult::Denied {
                reason: RateLimitReason::TooManyTokens,
                retry_after_ms: self.config.cooldown_seconds * 1000,
            };
        }

        RateLimitResult::Allowed {
            requests_remaining: self.config.requests_per_minute - self.request_times.len(),
            tokens_remaining: self.config.tokens_per_minute.saturating_sub(total_tokens),
        }
    }

    /// Record a request start
    pub fn record_request_start(&mut self) {
        self.request_times.push_back(Instant::now());
        self.current_concurrent += 1;
    }

    /// Record a request completion
    pub fn record_request_end(&mut self, tokens_used: usize) {
        self.current_concurrent = self.current_concurrent.saturating_sub(1);
        self.token_counts.push_back((Instant::now(), tokens_used));
    }

    /// Cancel a request (didn't complete)
    pub fn cancel_request(&mut self) {
        self.current_concurrent = self.current_concurrent.saturating_sub(1);
    }

    fn trigger_cooldown(&mut self) {
        self.cooldown_until =
            Some(Instant::now() + Duration::from_secs(self.config.cooldown_seconds));
    }

    /// Get current usage statistics
    pub fn get_usage(&self) -> RateLimitUsage {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        let requests = self
            .request_times
            .iter()
            .filter(|t| **t >= one_minute_ago)
            .count();
        let tokens: usize = self
            .token_counts
            .iter()
            .filter(|(t, _)| *t >= one_minute_ago)
            .map(|(_, tok)| *tok)
            .sum();

        RateLimitUsage {
            requests_used: requests,
            requests_limit: self.config.requests_per_minute,
            tokens_used: tokens,
            tokens_limit: self.config.tokens_per_minute,
            concurrent_active: self.current_concurrent,
            concurrent_limit: self.config.max_concurrent,
            in_cooldown: self.cooldown_until.map(|u| u > now).unwrap_or(false),
        }
    }

    /// Get status for UI display
    pub fn get_status(&self) -> RateLimitStatus {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        let requests = self
            .request_times
            .iter()
            .filter(|t| **t >= one_minute_ago)
            .count();
        let tokens: usize = self
            .token_counts
            .iter()
            .filter(|(t, _)| *t >= one_minute_ago)
            .map(|(_, tok)| *tok)
            .sum();

        let cooldown_remaining = self.cooldown_until.and_then(|until| {
            if until > now {
                Some(until.duration_since(now).as_secs())
            } else {
                None
            }
        });

        RateLimitStatus {
            requests_remaining: self.config.requests_per_minute.saturating_sub(requests),
            requests_per_minute: self.config.requests_per_minute,
            tokens_remaining: self.config.tokens_per_minute.saturating_sub(tokens),
            tokens_per_minute: self.config.tokens_per_minute,
            cooldown_remaining,
        }
    }
}

/// Status for UI display
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub requests_remaining: usize,
    pub requests_per_minute: usize,
    pub tokens_remaining: usize,
    pub tokens_per_minute: usize,
    pub cooldown_remaining: Option<u64>,
}

/// Result of rate limit check
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RateLimitResult {
    Allowed {
        requests_remaining: usize,
        tokens_remaining: usize,
    },
    Denied {
        reason: RateLimitReason,
        retry_after_ms: u64,
    },
}

impl RateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed { .. })
    }
}

/// Reason for rate limit denial
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RateLimitReason {
    TooManyRequests,
    TooManyTokens,
    TooManyConcurrent,
    Cooldown,
}

impl std::fmt::Display for RateLimitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyRequests => write!(f, "Too many requests per minute"),
            Self::TooManyTokens => write!(f, "Too many tokens generated per minute"),
            Self::TooManyConcurrent => write!(f, "Too many concurrent requests"),
            Self::Cooldown => write!(f, "Rate limit cooldown active"),
        }
    }
}

/// Current rate limit usage
#[derive(Debug, Clone)]
pub struct RateLimitUsage {
    pub requests_used: usize,
    pub requests_limit: usize,
    pub tokens_used: usize,
    pub tokens_limit: usize,
    pub concurrent_active: usize,
    pub concurrent_limit: usize,
    pub in_cooldown: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let config = RateLimitConfig {
            requests_per_minute: 3,
            tokens_per_minute: 100,
            max_concurrent: 1,
            cooldown_seconds: 1,
        };
        let mut limiter = RateLimiter::new(config);

        // First request should be allowed
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();

        // Second concurrent should be denied
        assert!(!limiter.check_allowed().is_allowed());

        // Complete first request
        limiter.record_request_end(10);

        // Now second should be allowed
        assert!(limiter.check_allowed().is_allowed());
    }

    #[test]
    fn test_rpm_enforcement() {
        let config = RateLimitConfig {
            requests_per_minute: 2,
            tokens_per_minute: 10000,
            max_concurrent: 10,
            cooldown_seconds: 5,
        };
        let mut limiter = RateLimiter::new(config);

        // First request allowed
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();
        limiter.record_request_end(1);

        // Second request allowed
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();
        limiter.record_request_end(1);

        // Third request denied (2 RPM limit exceeded)
        let result = limiter.check_allowed();
        assert!(!result.is_allowed());
        match result {
            RateLimitResult::Denied { reason, .. } => {
                assert_eq!(reason, RateLimitReason::TooManyRequests);
            }
            _ => panic!("Expected Denied result"),
        }
    }

    #[test]
    fn test_tpm_enforcement() {
        let config = RateLimitConfig {
            requests_per_minute: 100,
            tokens_per_minute: 50,
            max_concurrent: 10,
            cooldown_seconds: 5,
        };
        let mut limiter = RateLimiter::new(config);

        // First request: 30 tokens
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();
        limiter.record_request_end(30);

        // Second request: 30 tokens (total 60, exceeds 50 TPM)
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();
        limiter.record_request_end(30);

        // Third request should be denied due to token limit
        let result = limiter.check_allowed();
        assert!(!result.is_allowed());
        match result {
            RateLimitResult::Denied { reason, .. } => {
                assert_eq!(reason, RateLimitReason::TooManyTokens);
            }
            _ => panic!("Expected Denied result"),
        }
    }

    #[test]
    fn test_cooldown_behavior() {
        let config = RateLimitConfig {
            requests_per_minute: 1,
            tokens_per_minute: 10000,
            max_concurrent: 10,
            cooldown_seconds: 60,
        };
        let mut limiter = RateLimiter::new(config);

        // Use up the single allowed request
        assert!(limiter.check_allowed().is_allowed());
        limiter.record_request_start();
        limiter.record_request_end(1);

        // This triggers cooldown
        let result = limiter.check_allowed();
        assert!(!result.is_allowed());

        // Verify we are now in cooldown via usage stats
        let usage = limiter.get_usage();
        assert!(usage.in_cooldown);

        // Subsequent check should report Cooldown reason
        let result2 = limiter.check_allowed();
        match result2 {
            RateLimitResult::Denied { reason, .. } => {
                assert_eq!(reason, RateLimitReason::Cooldown);
            }
            _ => panic!("Expected Cooldown denial"),
        }
    }

    #[test]
    fn test_cancel_request() {
        let config = RateLimitConfig {
            max_concurrent: 2,
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(config);
        limiter.record_request_start();
        limiter.record_request_start();
        assert_eq!(limiter.get_usage().concurrent_active, 2);
        limiter.cancel_request();
        assert_eq!(limiter.get_usage().concurrent_active, 1);
        limiter.cancel_request();
        assert_eq!(limiter.get_usage().concurrent_active, 0);
        // Saturating: can't go below 0
        limiter.cancel_request();
        assert_eq!(limiter.get_usage().concurrent_active, 0);
    }

    #[test]
    fn test_get_status() {
        let config = RateLimitConfig {
            requests_per_minute: 10,
            tokens_per_minute: 500,
            max_concurrent: 3,
            cooldown_seconds: 30,
        };
        let mut limiter = RateLimiter::new(config);
        let status = limiter.get_status();
        assert_eq!(status.requests_remaining, 10);
        assert_eq!(status.tokens_remaining, 500);
        assert_eq!(status.requests_per_minute, 10);
        assert_eq!(status.tokens_per_minute, 500);
        assert!(status.cooldown_remaining.is_none());

        limiter.record_request_start();
        limiter.record_request_end(100);
        let status2 = limiter.get_status();
        assert_eq!(status2.requests_remaining, 9);
        assert_eq!(status2.tokens_remaining, 400);
    }

    #[test]
    fn test_rate_limit_reason_display() {
        assert_eq!(
            RateLimitReason::TooManyRequests.to_string(),
            "Too many requests per minute"
        );
        assert_eq!(
            RateLimitReason::TooManyTokens.to_string(),
            "Too many tokens generated per minute"
        );
        assert_eq!(
            RateLimitReason::TooManyConcurrent.to_string(),
            "Too many concurrent requests"
        );
        assert_eq!(
            RateLimitReason::Cooldown.to_string(),
            "Rate limit cooldown active"
        );
    }

    #[test]
    fn test_rate_limit_result_is_allowed() {
        let allowed = RateLimitResult::Allowed {
            requests_remaining: 5,
            tokens_remaining: 100,
        };
        assert!(allowed.is_allowed());
        let denied = RateLimitResult::Denied {
            reason: RateLimitReason::TooManyRequests,
            retry_after_ms: 1000,
        };
        assert!(!denied.is_allowed());
    }

    #[test]
    fn test_default_config() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_minute, 30);
        assert_eq!(config.tokens_per_minute, 10000);
        assert_eq!(config.max_concurrent, 2);
        assert_eq!(config.cooldown_seconds, 30);
    }

    #[test]
    fn test_get_usage_stats() {
        let config = RateLimitConfig {
            requests_per_minute: 10,
            tokens_per_minute: 1000,
            max_concurrent: 5,
            cooldown_seconds: 30,
        };
        let mut limiter = RateLimiter::new(config);

        // Record two requests, one still active
        limiter.record_request_start();
        limiter.record_request_end(25);

        limiter.record_request_start();
        limiter.record_request_end(35);

        // Start a third that is still in-flight
        limiter.record_request_start();

        let usage = limiter.get_usage();
        assert_eq!(usage.requests_used, 3);
        assert_eq!(usage.tokens_used, 60); // 25 + 35
        assert_eq!(usage.concurrent_active, 1); // one still in-flight
        assert_eq!(usage.requests_limit, 10);
        assert_eq!(usage.tokens_limit, 1000);
        assert!(!usage.in_cooldown);
    }
}
