//! HTTP Connection Pool for efficient request handling
//!
//! This module provides connection pooling to reuse HTTP connections
//! across multiple requests, improving performance significantly.
//!
//! # Features
//!
//! - **Connection reuse**: Keep connections alive between requests
//! - **Per-host pools**: Separate pools for different hosts
//! - **Health checking**: Automatic connection validation
//! - **Metrics**: Track connection usage and performance
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::connection_pool::{ConnectionPool, PoolConfig};
//!
//! let config = PoolConfig::default();
//! let pool = ConnectionPool::new(config);
//!
//! // Get a connection for a host
//! let conn = pool.get("http://localhost:11434");
//!
//! // Use the connection...
//! // Connection is automatically returned to pool when dropped
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for the connection pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per host
    pub max_connections_per_host: usize,
    /// Maximum total connections
    pub max_total_connections: usize,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Idle connection timeout
    pub idle_timeout: Duration,
    /// Enable connection keep-alive
    pub keep_alive: bool,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 10,
            max_total_connections: 50,
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(300),
            idle_timeout: Duration::from_secs(60),
            keep_alive: true,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// A pooled HTTP connection
pub struct PooledConnection {
    host: String,
    agent: ureq::Agent,
    created_at: Instant,
    last_used: Instant,
    request_count: usize,
}

impl PooledConnection {
    fn new(host: String, config: &PoolConfig) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(config.connect_timeout)
            .timeout_read(config.read_timeout)
            .build();

        Self {
            host,
            agent,
            created_at: Instant::now(),
            last_used: Instant::now(),
            request_count: 0,
        }
    }

    /// Get the underlying agent
    pub fn agent(&self) -> &ureq::Agent {
        &self.agent
    }

    /// Check if connection is still valid
    pub fn is_valid(&self, idle_timeout: Duration) -> bool {
        self.last_used.elapsed() < idle_timeout
    }

    /// Mark as used
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
        self.request_count += 1;
    }

    /// Get connection age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get request count
    pub fn request_count(&self) -> usize {
        self.request_count
    }
}

/// Connection pool for managing HTTP connections
pub struct ConnectionPool {
    config: PoolConfig,
    pools: RwLock<HashMap<String, Arc<Mutex<Vec<PooledConnection>>>>>,
    metrics: Mutex<PoolMetrics>,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            pools: RwLock::new(HashMap::new()),
            metrics: Mutex::new(PoolMetrics::default()),
        }
    }

    /// Get or create a connection for a host
    pub fn get(&self, host: &str) -> PooledConnectionGuard<'_> {
        let normalized_host = Self::normalize_host(host);

        // Try to get an existing connection
        if let Some(conn) = self.try_get_existing(&normalized_host) {
            let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
            metrics.hits += 1;
            return PooledConnectionGuard {
                connection: Some(conn),
                pool: self,
                host: normalized_host,
            };
        }

        // Create a new connection
        let conn = PooledConnection::new(normalized_host.clone(), &self.config);
        let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        metrics.misses += 1;
        metrics.connections_created += 1;

        PooledConnectionGuard {
            connection: Some(conn),
            pool: self,
            host: normalized_host,
        }
    }

    fn try_get_existing(&self, host: &str) -> Option<PooledConnection> {
        let pools = self.pools.read().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = pools.get(host) {
            let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());

            // Find a valid connection
            while let Some(mut conn) = pool.pop() {
                if conn.is_valid(self.config.idle_timeout) {
                    conn.touch();
                    return Some(conn);
                }
                // Connection expired, discard
                let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
                metrics.connections_expired += 1;
            }
        }
        None
    }

    fn return_connection(&self, conn: PooledConnection) {
        let host = conn.host.clone();

        // Get or create pool for this host
        let pool = {
            let pools = self.pools.read().unwrap_or_else(|e| e.into_inner());
            pools.get(&host).cloned()
        };

        let pool = pool.unwrap_or_else(|| {
            let mut pools = self.pools.write().unwrap_or_else(|e| e.into_inner());
            pools.entry(host.clone())
                .or_insert_with(|| Arc::new(Mutex::new(Vec::new())))
                .clone()
        });

        let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());

        // Only keep if under limit and connection is valid
        if pool.len() < self.config.max_connections_per_host
            && conn.is_valid(self.config.idle_timeout)
        {
            pool.push(conn);
            let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
            metrics.connections_reused += 1;
        }
    }

    fn normalize_host(url: &str) -> String {
        // Extract host:port from URL
        if let Some(rest) = url.strip_prefix("http://") {
            rest.split('/').next().unwrap_or(rest).to_string()
        } else if let Some(rest) = url.strip_prefix("https://") {
            rest.split('/').next().unwrap_or(rest).to_string()
        } else {
            url.to_string()
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let pools = self.pools.read().unwrap_or_else(|e| e.into_inner());
        let metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());

        let mut total_connections = 0;
        let mut connections_by_host = HashMap::new();

        for (host, pool) in pools.iter() {
            let count = pool.lock().unwrap_or_else(|e| e.into_inner()).len();
            total_connections += count;
            connections_by_host.insert(host.clone(), count);
        }

        PoolStats {
            total_connections,
            connections_by_host,
            hits: metrics.hits,
            misses: metrics.misses,
            connections_created: metrics.connections_created,
            connections_reused: metrics.connections_reused,
            connections_expired: metrics.connections_expired,
            hit_rate: if metrics.hits + metrics.misses > 0 {
                metrics.hits as f64 / (metrics.hits + metrics.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clean up expired connections
    pub fn cleanup(&self) {
        let pools = self.pools.read().unwrap_or_else(|e| e.into_inner());
        let idle_timeout = self.config.idle_timeout;

        for pool in pools.values() {
            let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
            let before = pool.len();
            pool.retain(|conn| conn.is_valid(idle_timeout));
            let expired = before - pool.len();
            if expired > 0 {
                let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
                metrics.connections_expired += expired;
            }
        }
    }

    /// Clear all connections
    pub fn clear(&self) {
        let mut pools = self.pools.write().unwrap_or_else(|e| e.into_inner());
        pools.clear();
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new(PoolConfig::default())
    }
}

/// Guard for a pooled connection that returns it on drop
pub struct PooledConnectionGuard<'a> {
    connection: Option<PooledConnection>,
    pool: &'a ConnectionPool,
    #[allow(dead_code)]
    host: String,
}

impl<'a> PooledConnectionGuard<'a> {
    /// Get the agent for making requests
    pub fn agent(&self) -> &ureq::Agent {
        self.connection.as_ref().expect("connection must be set").agent()
    }

    /// Make a GET request
    pub fn get(&self, url: &str) -> ureq::Request {
        self.agent().get(url)
    }

    /// Make a POST request
    pub fn post(&self, url: &str) -> ureq::Request {
        self.agent().post(url)
    }

    /// Make a PUT request
    pub fn put(&self, url: &str) -> ureq::Request {
        self.agent().put(url)
    }

    /// Make a DELETE request
    pub fn delete(&self, url: &str) -> ureq::Request {
        self.agent().delete(url)
    }
}

impl<'a> Drop for PooledConnectionGuard<'a> {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            self.pool.return_connection(conn);
        }
    }
}

#[derive(Debug, Default)]
struct PoolMetrics {
    hits: usize,
    misses: usize,
    connections_created: usize,
    connections_reused: usize,
    connections_expired: usize,
}

/// Statistics about the connection pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total connections currently pooled
    pub total_connections: usize,
    /// Connections per host
    pub connections_by_host: HashMap<String, usize>,
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Total connections created
    pub connections_created: usize,
    /// Connections successfully reused
    pub connections_reused: usize,
    /// Connections that expired
    pub connections_expired: usize,
    /// Hit rate (0-1)
    pub hit_rate: f64,
}

/// Global connection pool singleton
static GLOBAL_POOL: std::sync::OnceLock<ConnectionPool> = std::sync::OnceLock::new();

/// Get the global connection pool
pub fn global_pool() -> &'static ConnectionPool {
    GLOBAL_POOL.get_or_init(|| ConnectionPool::default())
}

/// Initialize the global pool with custom config
pub fn init_global_pool(config: PoolConfig) {
    let _ = GLOBAL_POOL.set(ConnectionPool::new(config));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_host() {
        assert_eq!(
            ConnectionPool::normalize_host("http://localhost:11434/api/generate"),
            "localhost:11434"
        );
        assert_eq!(
            ConnectionPool::normalize_host("https://api.openai.com/v1/chat"),
            "api.openai.com"
        );
        assert_eq!(
            ConnectionPool::normalize_host("localhost:8080"),
            "localhost:8080"
        );
    }

    #[test]
    fn test_pool_creation() {
        let pool = ConnectionPool::default();
        let stats = pool.stats();

        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_connection_pooling() {
        let pool = ConnectionPool::default();

        // First request creates a connection
        {
            let _conn = pool.get("http://localhost:11434");
            // Connection used here
        }
        // Connection returned to pool

        let stats = pool.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.connections_created, 1);

        // Second request should reuse
        {
            let _conn = pool.get("http://localhost:11434");
        }

        let stats = pool.stats();
        // Should have 1 hit now (reused connection)
        assert!(stats.connections_reused >= 1);
    }

    #[test]
    fn test_multiple_hosts() {
        let pool = ConnectionPool::default();

        {
            let _conn1 = pool.get("http://host1:1234");
            let _conn2 = pool.get("http://host2:5678");
        }

        let stats = pool.stats();
        assert_eq!(stats.connections_created, 2);
    }

    #[test]
    fn test_cleanup() {
        let config = PoolConfig {
            idle_timeout: Duration::from_millis(10),
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);

        {
            let _conn = pool.get("http://localhost:11434");
        }

        // Wait for connection to expire
        std::thread::sleep(Duration::from_millis(20));

        pool.cleanup();

        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_global_pool() {
        let pool = global_pool();
        let _conn = pool.get("http://test:1234");

        let stats = pool.stats();
        assert!(stats.connections_created >= 1);
    }
}
