//! Distributed rate limiting for multi-instance deployments
//!
//! Share rate limits across multiple application instances.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Distributed rate limit state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitState {
    pub requests: usize,
    pub tokens: usize,
    pub window_start: u64,
    pub last_update: u64,
}

/// Backend trait for distributed storage
pub trait RateLimitBackend: Send + Sync {
    fn get(&self, key: &str) -> Option<RateLimitState>;
    fn set(&self, key: &str, state: RateLimitState, ttl: Duration);
    fn increment(&self, key: &str, requests: usize, tokens: usize) -> RateLimitState;
}

/// In-memory backend (for single instance)
pub struct InMemoryBackend {
    data: Arc<Mutex<HashMap<String, (RateLimitState, Instant)>>>,
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimitBackend for InMemoryBackend {
    fn get(&self, key: &str) -> Option<RateLimitState> {
        let data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        data.get(key).map(|(s, _)| s.clone())
    }

    fn set(&self, key: &str, state: RateLimitState, _ttl: Duration) {
        let mut data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        data.insert(key.to_string(), (state, Instant::now()));
    }

    fn increment(&self, key: &str, requests: usize, tokens: usize) -> RateLimitState {
        let mut data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = data.entry(key.to_string()).or_insert_with(|| {
            (
                RateLimitState {
                    requests: 0,
                    tokens: 0,
                    window_start: now,
                    last_update: now,
                },
                Instant::now(),
            )
        });

        entry.0.requests += requests;
        entry.0.tokens += tokens;
        entry.0.last_update = now;
        entry.0.clone()
    }
}

/// Distributed rate limiter
pub struct DistributedRateLimiter {
    backend: Box<dyn RateLimitBackend>,
    requests_per_minute: usize,
    tokens_per_minute: usize,
    window_seconds: u64,
}

impl DistributedRateLimiter {
    pub fn new(backend: Box<dyn RateLimitBackend>, rpm: usize, tpm: usize) -> Self {
        Self {
            backend,
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
            window_seconds: 60,
        }
    }

    pub fn check(&self, key: &str) -> DistributedRateLimitResult {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(state) = self.backend.get(key) {
            if now - state.window_start < self.window_seconds {
                if state.requests >= self.requests_per_minute {
                    return DistributedRateLimitResult::Denied {
                        reason: "Too many requests".into(),
                        retry_after: Duration::from_secs(
                            self.window_seconds - (now - state.window_start),
                        ),
                    };
                }
                if state.tokens >= self.tokens_per_minute {
                    return DistributedRateLimitResult::Denied {
                        reason: "Token limit exceeded".into(),
                        retry_after: Duration::from_secs(
                            self.window_seconds - (now - state.window_start),
                        ),
                    };
                }
            }
        }

        DistributedRateLimitResult::Allowed {
            remaining_requests: self.requests_per_minute,
            remaining_tokens: self.tokens_per_minute,
        }
    }

    pub fn record(&self, key: &str, tokens: usize) {
        self.backend.increment(key, 1, tokens);
    }
}

#[derive(Debug, Clone)]
pub enum DistributedRateLimitResult {
    Allowed {
        remaining_requests: usize,
        remaining_tokens: usize,
    },
    Denied {
        reason: String,
        retry_after: Duration,
    },
}

impl DistributedRateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_backend() {
        let backend = InMemoryBackend::new();
        let limiter = DistributedRateLimiter::new(Box::new(backend), 10, 1000);

        assert!(limiter.check("user1").is_allowed());
        limiter.record("user1", 100);
        assert!(limiter.check("user1").is_allowed());
    }

    #[test]
    fn test_in_memory_backend_default() {
        let backend = InMemoryBackend::default();
        assert!(backend.get("nonexistent").is_none());
    }

    #[test]
    fn test_backend_get_set() {
        let backend = InMemoryBackend::new();
        let state = RateLimitState {
            requests: 5,
            tokens: 100,
            window_start: 1000,
            last_update: 1001,
        };
        backend.set("k1", state.clone(), Duration::from_secs(60));
        let retrieved = backend.get("k1").expect("Should find key");
        assert_eq!(retrieved.requests, 5);
        assert_eq!(retrieved.tokens, 100);
    }

    #[test]
    fn test_backend_increment() {
        let backend = InMemoryBackend::new();
        let s1 = backend.increment("user1", 1, 50);
        assert_eq!(s1.requests, 1);
        assert_eq!(s1.tokens, 50);
        let s2 = backend.increment("user1", 1, 30);
        assert_eq!(s2.requests, 2);
        assert_eq!(s2.tokens, 80);
    }

    #[test]
    fn test_separate_keys() {
        let backend = InMemoryBackend::new();
        let limiter = DistributedRateLimiter::new(Box::new(backend), 5, 1000);
        limiter.record("user_a", 100);
        limiter.record("user_b", 200);
        assert!(limiter.check("user_a").is_allowed());
        assert!(limiter.check("user_b").is_allowed());
    }

    #[test]
    fn test_result_is_allowed() {
        let allowed = DistributedRateLimitResult::Allowed {
            remaining_requests: 5,
            remaining_tokens: 100,
        };
        assert!(allowed.is_allowed());
        let denied = DistributedRateLimitResult::Denied {
            reason: "test".into(),
            retry_after: Duration::from_secs(10),
        };
        assert!(!denied.is_allowed());
    }

    #[test]
    fn test_token_limit() {
        let backend = InMemoryBackend::new();
        let limiter = DistributedRateLimiter::new(Box::new(backend), 100, 50);
        limiter.record("u1", 60);
        let result = limiter.check("u1");
        assert!(!result.is_allowed());
        match result {
            DistributedRateLimitResult::Denied { reason, .. } => {
                assert!(reason.contains("Token"));
            }
            _ => panic!("Expected Denied"),
        }
    }

    #[test]
    fn test_rate_limit_state_fields() {
        let state = RateLimitState {
            requests: 10,
            tokens: 500,
            window_start: 123456,
            last_update: 123460,
        };
        assert_eq!(state.requests, 10);
        assert_eq!(state.tokens, 500);
        assert_eq!(state.window_start, 123456);
        assert_eq!(state.last_update, 123460);
    }

    #[test]
    fn test_fresh_key_allowed() {
        let backend = InMemoryBackend::new();
        let limiter = DistributedRateLimiter::new(Box::new(backend), 1, 1);
        assert!(limiter.check("brand_new_key").is_allowed());
    }

    #[test]
    fn test_distributed_rpm_limit() {
        let backend = InMemoryBackend::new();
        let limiter = DistributedRateLimiter::new(Box::new(backend), 2, 100_000);

        // First request - allowed
        assert!(limiter.check("api_user").is_allowed());
        limiter.record("api_user", 10);

        // Second request - allowed (at limit)
        assert!(limiter.check("api_user").is_allowed());
        limiter.record("api_user", 10);

        // Third request - denied (rpm=2 exceeded)
        let result = limiter.check("api_user");
        assert!(!result.is_allowed());
        match result {
            DistributedRateLimitResult::Denied {
                reason,
                retry_after,
            } => {
                assert!(
                    reason.contains("requests"),
                    "Expected 'requests' in reason, got: {}",
                    reason
                );
                assert!(retry_after.as_secs() > 0);
            }
            _ => panic!("Expected Denied result"),
        }
    }
}
