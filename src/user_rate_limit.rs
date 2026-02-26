//! Per-user rate limiting
//!
//! Rate limiting on a per-user basis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// User rate limit configuration
#[derive(Debug, Clone)]
pub struct UserRateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
    pub tokens_per_minute: u64,
    pub tokens_per_day: u64,
    pub burst_allowance: u32,
}

impl Default for UserRateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 20,
            requests_per_hour: 200,
            requests_per_day: 1000,
            tokens_per_minute: 50_000,
            tokens_per_day: 1_000_000,
            burst_allowance: 5,
        }
    }
}

/// User usage tracking
#[derive(Debug, Clone)]
struct UserUsage {
    minute_requests: Vec<Instant>,
    hour_requests: Vec<Instant>,
    day_requests: Vec<Instant>,
    minute_tokens: u64,
    day_tokens: u64,
    last_minute_reset: Instant,
    last_day_reset: Instant,
}

impl UserUsage {
    fn new() -> Self {
        Self {
            minute_requests: Vec::new(),
            hour_requests: Vec::new(),
            day_requests: Vec::new(),
            minute_tokens: 0,
            day_tokens: 0,
            last_minute_reset: Instant::now(),
            last_day_reset: Instant::now(),
        }
    }

    fn cleanup(&mut self) {
        let now = Instant::now();

        self.minute_requests
            .retain(|t| now.duration_since(*t) < Duration::from_secs(60));
        self.hour_requests
            .retain(|t| now.duration_since(*t) < Duration::from_secs(3600));
        self.day_requests
            .retain(|t| now.duration_since(*t) < Duration::from_secs(86400));

        if now.duration_since(self.last_minute_reset) >= Duration::from_secs(60) {
            self.minute_tokens = 0;
            self.last_minute_reset = now;
        }

        if now.duration_since(self.last_day_reset) >= Duration::from_secs(86400) {
            self.day_tokens = 0;
            self.last_day_reset = now;
        }
    }
}

/// Rate limit check result
#[derive(Debug, Clone)]
pub struct RateLimitCheckResult {
    pub allowed: bool,
    pub reason: Option<String>,
    pub retry_after: Option<Duration>,
    pub remaining_minute: u32,
    pub remaining_hour: u32,
    pub remaining_day: u32,
}

/// Per-user rate limiter
pub struct UserRateLimiter {
    users: HashMap<String, UserUsage>,
    default_config: UserRateLimitConfig,
    custom_configs: HashMap<String, UserRateLimitConfig>,
}

impl UserRateLimiter {
    pub fn new(config: UserRateLimitConfig) -> Self {
        Self {
            default_config: config,
            users: HashMap::new(),
            custom_configs: HashMap::new(),
        }
    }

    pub fn set_user_config(&mut self, user_id: &str, config: UserRateLimitConfig) {
        self.custom_configs.insert(user_id.to_string(), config);
    }

    pub fn check(&mut self, user_id: &str, tokens: u64) -> RateLimitCheckResult {
        // Clone config to avoid borrow issues
        let config = self
            .custom_configs
            .get(user_id)
            .unwrap_or(&self.default_config)
            .clone();

        let usage = self
            .users
            .entry(user_id.to_string())
            .or_insert_with(UserUsage::new);
        usage.cleanup();

        let minute_count = usage.minute_requests.len() as u32;
        let hour_count = usage.hour_requests.len() as u32;
        let day_count = usage.day_requests.len() as u32;

        // Check request limits
        if minute_count >= config.requests_per_minute + config.burst_allowance {
            return RateLimitCheckResult {
                allowed: false,
                reason: Some("Minute request limit exceeded".to_string()),
                retry_after: Some(Duration::from_secs(60)),
                remaining_minute: 0,
                remaining_hour: config.requests_per_hour.saturating_sub(hour_count),
                remaining_day: config.requests_per_day.saturating_sub(day_count),
            };
        }

        if hour_count >= config.requests_per_hour {
            return RateLimitCheckResult {
                allowed: false,
                reason: Some("Hourly request limit exceeded".to_string()),
                retry_after: Some(Duration::from_secs(3600)),
                remaining_minute: config.requests_per_minute.saturating_sub(minute_count),
                remaining_hour: 0,
                remaining_day: config.requests_per_day.saturating_sub(day_count),
            };
        }

        if day_count >= config.requests_per_day {
            return RateLimitCheckResult {
                allowed: false,
                reason: Some("Daily request limit exceeded".to_string()),
                retry_after: Some(Duration::from_secs(86400)),
                remaining_minute: config.requests_per_minute.saturating_sub(minute_count),
                remaining_hour: config.requests_per_hour.saturating_sub(hour_count),
                remaining_day: 0,
            };
        }

        // Check token limits
        if usage.minute_tokens + tokens > config.tokens_per_minute {
            return RateLimitCheckResult {
                allowed: false,
                reason: Some("Minute token limit exceeded".to_string()),
                retry_after: Some(Duration::from_secs(60)),
                remaining_minute: config.requests_per_minute.saturating_sub(minute_count),
                remaining_hour: config.requests_per_hour.saturating_sub(hour_count),
                remaining_day: config.requests_per_day.saturating_sub(day_count),
            };
        }

        if usage.day_tokens + tokens > config.tokens_per_day {
            return RateLimitCheckResult {
                allowed: false,
                reason: Some("Daily token limit exceeded".to_string()),
                retry_after: Some(Duration::from_secs(86400)),
                remaining_minute: config.requests_per_minute.saturating_sub(minute_count),
                remaining_hour: config.requests_per_hour.saturating_sub(hour_count),
                remaining_day: config.requests_per_day.saturating_sub(day_count),
            };
        }

        RateLimitCheckResult {
            allowed: true,
            reason: None,
            retry_after: None,
            remaining_minute: config.requests_per_minute.saturating_sub(minute_count + 1),
            remaining_hour: config.requests_per_hour.saturating_sub(hour_count + 1),
            remaining_day: config.requests_per_day.saturating_sub(day_count + 1),
        }
    }

    pub fn record(&mut self, user_id: &str, tokens: u64) {
        let usage = self
            .users
            .entry(user_id.to_string())
            .or_insert_with(UserUsage::new);
        let now = Instant::now();

        usage.minute_requests.push(now);
        usage.hour_requests.push(now);
        usage.day_requests.push(now);
        usage.minute_tokens += tokens;
        usage.day_tokens += tokens;
    }

    pub fn reset_user(&mut self, user_id: &str) {
        self.users.remove(user_id);
    }
}

impl Default for UserRateLimiter {
    fn default() -> Self {
        Self::new(UserRateLimitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit() {
        let mut limiter = UserRateLimiter::new(UserRateLimitConfig {
            requests_per_minute: 2,
            burst_allowance: 0,
            ..Default::default()
        });

        assert!(limiter.check("user1", 100).allowed);
        limiter.record("user1", 100);

        assert!(limiter.check("user1", 100).allowed);
        limiter.record("user1", 100);

        assert!(!limiter.check("user1", 100).allowed);
    }

    #[test]
    fn test_burst_allowance() {
        // RPM=2 with burst_allowance=1 means 3 total requests allowed in the minute window
        let mut limiter = UserRateLimiter::new(UserRateLimitConfig {
            requests_per_minute: 2,
            burst_allowance: 1,
            ..Default::default()
        });

        // First two within normal RPM
        assert!(limiter.check("user1", 10).allowed);
        limiter.record("user1", 10);
        assert!(limiter.check("user1", 10).allowed);
        limiter.record("user1", 10);

        // Third allowed by burst (burst_allowance=1 extra beyond RPM)
        assert!(limiter.check("user1", 10).allowed);
        limiter.record("user1", 10);

        // Fourth should be denied (2 RPM + 1 burst = 3 total)
        let result = limiter.check("user1", 10);
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("Minute"));
    }

    #[test]
    fn test_token_limit_enforcement() {
        let mut limiter = UserRateLimiter::new(UserRateLimitConfig {
            requests_per_minute: 100, // high RPM so we don't hit request limit
            tokens_per_minute: 500,
            burst_allowance: 0,
            ..Default::default()
        });

        // First request with 400 tokens - allowed
        assert!(limiter.check("user1", 400).allowed);
        limiter.record("user1", 400);

        // Second request with 200 tokens would exceed 500 TPM - denied
        let result = limiter.check("user1", 200);
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("token"));

        // But a small request under the remaining budget is still allowed
        assert!(limiter.check("user1", 50).allowed);
    }

    #[test]
    fn test_multi_user_isolation() {
        let mut limiter = UserRateLimiter::new(UserRateLimitConfig {
            requests_per_minute: 2,
            burst_allowance: 0,
            ..Default::default()
        });

        // Exhaust user1's limit
        limiter.record("user1", 10);
        limiter.record("user1", 10);
        assert!(!limiter.check("user1", 10).allowed);

        // user2 should still have full quota
        assert!(limiter.check("user2", 10).allowed);
        limiter.record("user2", 10);
        assert!(limiter.check("user2", 10).allowed);
    }
}
