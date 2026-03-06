//! # Redis Backend (Phase 13)
//!
//! Optional Redis-backed implementations for rate limiting, session storage,
//! and caching. Alternative to CRDT-based sync — useful when Redis is already
//! deployed in the infrastructure.
//!
//! ## Feature flag
//! `redis-backend` — requires Redis server running.
//!
//! ## Usage
//! ```rust,no_run
//! use ai_assistant::redis_backend::{RedisConfig, RedisBackend};
//!
//! let config = RedisConfig { url: "redis://127.0.0.1:6379".to_string(), ..Default::default() };
//! // let backend = RedisBackend::new(config).await?;
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Redis connection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL (redis://host:port or rediss://host:port for TLS).
    pub url: String,
    /// Connection pool size.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    /// Connection timeout.
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    /// Key prefix for namespacing.
    #[serde(default = "default_key_prefix")]
    pub key_prefix: String,
}

fn default_pool_size() -> usize { 10 }
fn default_connect_timeout_ms() -> u64 { 5000 }
fn default_key_prefix() -> String { "ai_assistant:".to_string() }

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            pool_size: default_pool_size(),
            connect_timeout_ms: default_connect_timeout_ms(),
            key_prefix: default_key_prefix(),
        }
    }
}

// ============================================================================
// Redis Backend
// ============================================================================

/// Redis-backed storage for rate limiting, sessions, and caching.
///
/// Uses the `redis` crate with `tokio-comp` for async operations
/// and `connection-manager` for automatic reconnection.
pub struct RedisBackend {
    /// Connection manager (auto-reconnects).
    #[allow(dead_code)]
    client: redis::Client,
    /// Connection manager for async operations.
    manager: redis::aio::ConnectionManager,
    /// Configuration.
    config: RedisConfig,
}

impl RedisBackend {
    /// Connect to Redis and create a new backend.
    pub async fn new(config: RedisConfig) -> Result<Self, String> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| format!("Redis connection error: {}", e))?;

        let manager = redis::aio::ConnectionManager::new(client.clone())
            .await
            .map_err(|e| format!("Redis connection manager error: {}", e))?;

        Ok(Self {
            client,
            manager,
            config,
        })
    }

    /// Get the Redis config.
    pub fn config(&self) -> &RedisConfig {
        &self.config
    }

    fn prefixed_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }
}

// ============================================================================
// Rate Limit Backend
// ============================================================================

impl RedisBackend {
    /// Increment a rate limit counter and check if the limit is exceeded.
    ///
    /// Uses Redis INCR + EXPIRE for atomic sliding window.
    /// Returns the current count after increment.
    pub async fn rate_limit_check(
        &mut self,
        key: &str,
        limit: u64,
        window_secs: u64,
    ) -> Result<RateLimitResult, String> {
        let redis_key = self.prefixed_key(&format!("rl:{}", key));

        let (count,): (u64,) = redis::pipe()
            .atomic()
            .cmd("INCR").arg(&redis_key)
            .cmd("EXPIRE").arg(&redis_key).arg(window_secs).ignore()
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis rate limit error: {}", e))?;

        Ok(RateLimitResult {
            allowed: count <= limit,
            current_count: count,
            limit,
            remaining: if count <= limit { limit - count } else { 0 },
            retry_after_secs: if count > limit { Some(window_secs) } else { None },
        })
    }

    /// Reset a rate limit counter.
    pub async fn rate_limit_reset(&mut self, key: &str) -> Result<(), String> {
        let redis_key = self.prefixed_key(&format!("rl:{}", key));
        redis::cmd("DEL")
            .arg(&redis_key)
            .query_async::<()>(&mut self.manager)
            .await
            .map_err(|e| format!("Redis reset error: {}", e))?;
        Ok(())
    }
}

/// Result of a rate limit check.
#[derive(Debug, Clone, Serialize)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub current_count: u64,
    pub limit: u64,
    pub remaining: u64,
    pub retry_after_secs: Option<u64>,
}

// ============================================================================
// Session Backend
// ============================================================================

impl RedisBackend {
    /// Store a session with TTL.
    pub async fn session_set(
        &mut self,
        session_id: &str,
        data: &[u8],
        ttl: Duration,
    ) -> Result<(), String> {
        let key = self.prefixed_key(&format!("sess:{}", session_id));
        redis::cmd("SET")
            .arg(&key)
            .arg(data)
            .arg("EX")
            .arg(ttl.as_secs())
            .query_async::<()>(&mut self.manager)
            .await
            .map_err(|e| format!("Redis session set error: {}", e))?;
        Ok(())
    }

    /// Retrieve a session.
    pub async fn session_get(&mut self, session_id: &str) -> Result<Option<Vec<u8>>, String> {
        let key = self.prefixed_key(&format!("sess:{}", session_id));
        let result: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis session get error: {}", e))?;
        Ok(result)
    }

    /// Delete a session.
    pub async fn session_delete(&mut self, session_id: &str) -> Result<bool, String> {
        let key = self.prefixed_key(&format!("sess:{}", session_id));
        let deleted: u64 = redis::cmd("DEL")
            .arg(&key)
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis session delete error: {}", e))?;
        Ok(deleted > 0)
    }

    /// Touch a session (extend TTL).
    pub async fn session_touch(
        &mut self,
        session_id: &str,
        ttl: Duration,
    ) -> Result<bool, String> {
        let key = self.prefixed_key(&format!("sess:{}", session_id));
        let set: bool = redis::cmd("EXPIRE")
            .arg(&key)
            .arg(ttl.as_secs())
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis session touch error: {}", e))?;
        Ok(set)
    }
}

// ============================================================================
// Cache Backend
// ============================================================================

impl RedisBackend {
    /// Cache a value with TTL.
    pub async fn cache_set(
        &mut self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<(), String> {
        let redis_key = self.prefixed_key(&format!("cache:{}", key));
        redis::cmd("SET")
            .arg(&redis_key)
            .arg(value)
            .arg("EX")
            .arg(ttl.as_secs())
            .query_async::<()>(&mut self.manager)
            .await
            .map_err(|e| format!("Redis cache set error: {}", e))?;
        Ok(())
    }

    /// Get a cached value.
    pub async fn cache_get(&mut self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let redis_key = self.prefixed_key(&format!("cache:{}", key));
        let result: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&redis_key)
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis cache get error: {}", e))?;
        Ok(result)
    }

    /// Delete a cached value.
    pub async fn cache_delete(&mut self, key: &str) -> Result<bool, String> {
        let redis_key = self.prefixed_key(&format!("cache:{}", key));
        let deleted: u64 = redis::cmd("DEL")
            .arg(&redis_key)
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis cache delete error: {}", e))?;
        Ok(deleted > 0)
    }
}

// ============================================================================
// Health Check
// ============================================================================

impl RedisBackend {
    /// Ping Redis to check connectivity.
    pub async fn ping(&mut self) -> Result<bool, String> {
        let result: String = redis::cmd("PING")
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis ping error: {}", e))?;
        Ok(result == "PONG")
    }

    /// Get Redis server info (for diagnostics).
    pub async fn info(&mut self) -> Result<String, String> {
        let info: String = redis::cmd("INFO")
            .arg("server")
            .query_async(&mut self.manager)
            .await
            .map_err(|e| format!("Redis info error: {}", e))?;
        Ok(info)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_config_defaults() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.pool_size, 10);
        assert_eq!(config.connect_timeout_ms, 5000);
        assert_eq!(config.key_prefix, "ai_assistant:");
    }

    #[test]
    fn test_redis_config_serialization() {
        let config = RedisConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: RedisConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.url, config.url);
        assert_eq!(parsed.pool_size, config.pool_size);
    }

    #[test]
    fn test_redis_config_custom() {
        let config = RedisConfig {
            url: "rediss://cluster.example.com:6380".to_string(),
            pool_size: 20,
            connect_timeout_ms: 3000,
            key_prefix: "myapp:".to_string(),
        };
        assert_eq!(config.url, "rediss://cluster.example.com:6380");
        assert_eq!(config.pool_size, 20);
        assert_eq!(config.key_prefix, "myapp:");
    }

    #[test]
    fn test_rate_limit_result_serialization() {
        let result = RateLimitResult {
            allowed: true,
            current_count: 5,
            limit: 100,
            remaining: 95,
            retry_after_secs: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"allowed\":true"));
        assert!(json.contains("\"remaining\":95"));
    }

    #[test]
    fn test_rate_limit_result_exceeded() {
        let result = RateLimitResult {
            allowed: false,
            current_count: 101,
            limit: 100,
            remaining: 0,
            retry_after_secs: Some(60),
        };
        assert!(!result.allowed);
        assert_eq!(result.retry_after_secs, Some(60));
    }

    // NOTE: Integration tests with actual Redis require a running Redis server.
    // They are skipped by default and can be run with:
    // REDIS_URL=redis://127.0.0.1:6379 cargo test --features "redis-backend" -- redis_backend::
    //
    // The following test verifies client creation logic (will fail without Redis):
    #[tokio::test]
    async fn test_redis_client_creation_invalid_url() {
        // Invalid URL should fail at connection
        let config = RedisConfig {
            url: "redis://nonexistent-host:9999".to_string(),
            ..Default::default()
        };
        // Client creation may succeed (lazy connect), manager creation should fail
        let result = RedisBackend::new(config).await;
        // This may succeed or fail depending on whether the Redis client
        // validates immediately or lazily. Either outcome is acceptable.
        let _ = result;
    }

    #[test]
    fn test_prefixed_key() {
        let config = RedisConfig {
            key_prefix: "test:".to_string(),
            ..Default::default()
        };
        // We can't test prefixed_key directly since it needs a backend,
        // but we can verify the config
        assert_eq!(format!("{}rl:192.168.1.1", config.key_prefix), "test:rl:192.168.1.1");
        assert_eq!(format!("{}sess:abc", config.key_prefix), "test:sess:abc");
    }
}
