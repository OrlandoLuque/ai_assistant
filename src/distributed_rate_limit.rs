//! Distributed rate limiting for multi-instance deployments
//!
//! Share rate limits across multiple application instances.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

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
        Self { data: Arc::new(Mutex::new(HashMap::new())) }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self { Self::new() }
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
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

        let entry = data.entry(key.to_string()).or_insert_with(|| {
            (RateLimitState {
                requests: 0, tokens: 0, window_start: now, last_update: now,
            }, Instant::now())
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
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

        if let Some(state) = self.backend.get(key) {
            if now - state.window_start < self.window_seconds {
                if state.requests >= self.requests_per_minute {
                    return DistributedRateLimitResult::Denied {
                        reason: "Too many requests".into(),
                        retry_after: Duration::from_secs(self.window_seconds - (now - state.window_start)),
                    };
                }
                if state.tokens >= self.tokens_per_minute {
                    return DistributedRateLimitResult::Denied {
                        reason: "Token limit exceeded".into(),
                        retry_after: Duration::from_secs(self.window_seconds - (now - state.window_start)),
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
    Allowed { remaining_requests: usize, remaining_tokens: usize },
    Denied { reason: String, retry_after: Duration },
}

impl DistributedRateLimitResult {
    pub fn is_allowed(&self) -> bool { matches!(self, Self::Allowed { .. }) }
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
}
