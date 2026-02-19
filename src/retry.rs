//! Retry mechanisms with exponential backoff
//!
//! This module provides retry functionality for network operations with configurable
//! backoff strategies, jitter, and circuit breaker patterns.

use std::time::{Duration, Instant};
use std::thread;
use anyhow::{Result, anyhow};

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff (e.g., 2.0 doubles delay each retry)
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to delays
    pub add_jitter: bool,
    /// Maximum jitter as a fraction of delay (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Timeout for each individual attempt
    pub attempt_timeout: Option<Duration>,
    /// Errors that should trigger a retry
    pub retryable_errors: Vec<RetryableError>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            add_jitter: true,
            jitter_factor: 0.25,
            attempt_timeout: Some(Duration::from_secs(30)),
            retryable_errors: vec![
                RetryableError::ConnectionRefused,
                RetryableError::Timeout,
                RetryableError::ServerError,
                RetryableError::RateLimited,
            ],
        }
    }
}

impl RetryConfig {
    /// Create a config optimized for fast operations
    pub fn fast() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            add_jitter: true,
            jitter_factor: 0.1,
            attempt_timeout: Some(Duration::from_secs(5)),
            retryable_errors: vec![
                RetryableError::ConnectionRefused,
                RetryableError::Timeout,
            ],
        }
    }

    /// Create a config for aggressive retrying
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 1.5,
            add_jitter: true,
            jitter_factor: 0.3,
            attempt_timeout: Some(Duration::from_secs(60)),
            retryable_errors: vec![
                RetryableError::ConnectionRefused,
                RetryableError::Timeout,
                RetryableError::ServerError,
                RetryableError::RateLimited,
                RetryableError::ServiceUnavailable,
            ],
        }
    }

    /// Create a config with no retries
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Calculate delay for a specific retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.initial_delay.as_secs_f64()
            * self.backoff_multiplier.powi(attempt as i32);

        let capped_delay = base_delay.min(self.max_delay.as_secs_f64());

        let final_delay = if self.add_jitter {
            let jitter_range = capped_delay * self.jitter_factor;
            let jitter = (rand_simple() * 2.0 - 1.0) * jitter_range;
            (capped_delay + jitter).max(0.0)
        } else {
            capped_delay
        };

        Duration::from_secs_f64(final_delay)
    }
}

/// Types of errors that can be retried
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryableError {
    /// Connection was refused
    ConnectionRefused,
    /// Request timed out
    Timeout,
    /// Server returned 5xx error
    ServerError,
    /// Rate limited (429)
    RateLimited,
    /// Service unavailable (503)
    ServiceUnavailable,
    /// DNS resolution failed
    DnsError,
    /// Connection reset
    ConnectionReset,
    /// Network unreachable
    NetworkUnreachable,
}

impl RetryableError {
    /// Check if an error message indicates this type of error
    pub fn matches(&self, error: &str) -> bool {
        let error_lower = error.to_lowercase();
        match self {
            RetryableError::ConnectionRefused => {
                error_lower.contains("connection refused")
                    || error_lower.contains("connrefused")
                    || error_lower.contains("econnrefused")
            }
            RetryableError::Timeout => {
                error_lower.contains("timeout")
                    || error_lower.contains("timed out")
                    || error_lower.contains("etimedout")
            }
            RetryableError::ServerError => {
                error_lower.contains("500")
                    || error_lower.contains("502")
                    || error_lower.contains("504")
                    || error_lower.contains("internal server error")
                    || error_lower.contains("bad gateway")
                    || error_lower.contains("gateway timeout")
            }
            RetryableError::RateLimited => {
                error_lower.contains("429")
                    || error_lower.contains("rate limit")
                    || error_lower.contains("too many requests")
                    || error_lower.contains("throttl")
            }
            RetryableError::ServiceUnavailable => {
                error_lower.contains("503")
                    || error_lower.contains("service unavailable")
                    || error_lower.contains("temporarily unavailable")
            }
            RetryableError::DnsError => {
                error_lower.contains("dns")
                    || error_lower.contains("resolve")
                    || error_lower.contains("getaddrinfo")
                    || error_lower.contains("name resolution")
            }
            RetryableError::ConnectionReset => {
                error_lower.contains("connection reset")
                    || error_lower.contains("econnreset")
                    || error_lower.contains("broken pipe")
            }
            RetryableError::NetworkUnreachable => {
                error_lower.contains("network unreachable")
                    || error_lower.contains("enetunreach")
                    || error_lower.contains("no route to host")
            }
        }
    }
}

/// Result of a retry operation
#[derive(Debug, Clone)]
pub struct RetryResult<T> {
    /// The result value if successful
    pub value: Option<T>,
    /// Total number of attempts made
    pub attempts: u32,
    /// Total time spent retrying
    pub total_duration: Duration,
    /// History of errors encountered
    pub error_history: Vec<RetryAttempt>,
    /// Whether the operation succeeded
    pub success: bool,
}

/// Information about a single retry attempt
#[derive(Debug, Clone)]
pub struct RetryAttempt {
    /// Attempt number (0-indexed)
    pub attempt: u32,
    /// Error message if failed
    pub error: Option<String>,
    /// Duration of this attempt
    pub duration: Duration,
    /// Delay before next attempt (if any)
    pub delay_after: Option<Duration>,
}

/// Retry executor
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create a new retry executor with the given config
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute an operation with retries
    pub fn execute<T, F>(&self, mut operation: F) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
    {
        let start_time = Instant::now();
        let mut error_history = Vec::new();
        let mut attempt = 0;

        loop {
            let attempt_start = Instant::now();

            match operation() {
                Ok(value) => {
                    return RetryResult {
                        value: Some(value),
                        attempts: attempt + 1,
                        total_duration: start_time.elapsed(),
                        error_history,
                        success: true,
                    };
                }
                Err(e) => {
                    let error_str = e.to_string();
                    let attempt_duration = attempt_start.elapsed();

                    // Check if this error is retryable
                    let is_retryable = self.config.retryable_errors.iter()
                        .any(|re| re.matches(&error_str));

                    let can_retry = attempt < self.config.max_retries && is_retryable;
                    let delay_after = if can_retry {
                        Some(self.config.calculate_delay(attempt))
                    } else {
                        None
                    };

                    error_history.push(RetryAttempt {
                        attempt,
                        error: Some(error_str.clone()),
                        duration: attempt_duration,
                        delay_after,
                    });

                    if !can_retry {
                        return RetryResult {
                            value: None,
                            attempts: attempt + 1,
                            total_duration: start_time.elapsed(),
                            error_history,
                            success: false,
                        };
                    }

                    // Wait before next retry
                    if let Some(delay) = delay_after {
                        thread::sleep(delay);
                    }

                    attempt += 1;
                }
            }
        }
    }

    /// Execute with a callback for each attempt
    pub fn execute_with_callback<T, F, C>(&self, mut operation: F, mut on_retry: C) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
        C: FnMut(u32, &str, Duration),
    {
        let start_time = Instant::now();
        let mut error_history = Vec::new();
        let mut attempt = 0;

        loop {
            let attempt_start = Instant::now();

            match operation() {
                Ok(value) => {
                    return RetryResult {
                        value: Some(value),
                        attempts: attempt + 1,
                        total_duration: start_time.elapsed(),
                        error_history,
                        success: true,
                    };
                }
                Err(e) => {
                    let error_str = e.to_string();
                    let attempt_duration = attempt_start.elapsed();

                    let is_retryable = self.config.retryable_errors.iter()
                        .any(|re| re.matches(&error_str));

                    let can_retry = attempt < self.config.max_retries && is_retryable;
                    let delay_after = if can_retry {
                        Some(self.config.calculate_delay(attempt))
                    } else {
                        None
                    };

                    error_history.push(RetryAttempt {
                        attempt,
                        error: Some(error_str.clone()),
                        duration: attempt_duration,
                        delay_after,
                    });

                    if can_retry {
                        // Call the retry callback
                        on_retry(attempt, &error_str, delay_after.unwrap_or_default());
                    }

                    if !can_retry {
                        return RetryResult {
                            value: None,
                            attempts: attempt + 1,
                            total_duration: start_time.elapsed(),
                            error_history,
                            success: false,
                        };
                    }

                    if let Some(delay) = delay_after {
                        thread::sleep(delay);
                    }

                    attempt += 1;
                }
            }
        }
    }
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state of the circuit
    state: CircuitState,
    /// Number of consecutive failures
    failure_count: u32,
    /// Threshold for opening the circuit
    failure_threshold: u32,
    /// Time to wait before attempting recovery
    recovery_timeout: Duration,
    /// Time when the circuit was opened
    opened_at: Option<Instant>,
    /// Success count in half-open state
    half_open_successes: u32,
    /// Required successes to close circuit
    success_threshold: u32,
}

/// State of the circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests pass through normally
    Closed,
    /// Circuit is open, requests are rejected immediately
    Open,
    /// Circuit is testing if service recovered
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            recovery_timeout,
            opened_at: None,
            half_open_successes: 0,
            success_threshold: 2,
        }
    }

    /// Check if a request should be allowed
    pub fn should_allow(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Some(opened_at) = self.opened_at {
                    if opened_at.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.half_open_successes = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.opened_at = None;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                // Failed while testing, go back to open
                self.state = CircuitState::Open;
                self.opened_at = Some(Instant::now());
            }
            CircuitState::Open => {}
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Get failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }

    /// Reset the circuit breaker
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.opened_at = None;
        self.half_open_successes = 0;
    }

    /// Execute an operation with circuit breaker protection
    pub fn execute<T, F>(&mut self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if !self.should_allow() {
            return Err(anyhow!("Circuit breaker is open"));
        }

        match operation() {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }
}

/// Combined retry executor with circuit breaker
pub struct ResilientExecutor {
    retry: RetryExecutor,
    circuit_breaker: CircuitBreaker,
}

impl ResilientExecutor {
    /// Create a new resilient executor
    pub fn new(retry_config: RetryConfig, failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            retry: RetryExecutor::new(retry_config),
            circuit_breaker: CircuitBreaker::new(failure_threshold, recovery_timeout),
        }
    }

    /// Execute an operation with both retry and circuit breaker protection
    pub fn execute<T, F>(&mut self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        if !self.circuit_breaker.should_allow() {
            return Err(anyhow!("Circuit breaker is open, service appears unavailable"));
        }

        let result = self.retry.execute(&mut operation);

        if result.success {
            self.circuit_breaker.record_success();
            Ok(result.value.expect("value must be present on success"))
        } else {
            self.circuit_breaker.record_failure();
            let last_error = result.error_history
                .last()
                .and_then(|a| a.error.clone())
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(anyhow!("Operation failed after {} attempts: {}", result.attempts, last_error))
        }
    }

    /// Get circuit breaker state
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state()
    }

    /// Reset the circuit breaker
    pub fn reset_circuit(&mut self) {
        self.circuit_breaker.reset();
    }
}

/// Simple pseudo-random number generator (no external deps)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos as f64 % 1000.0) / 1000.0
}

/// Convenience function to retry an operation with default config
pub fn retry<T, F>(operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let executor = RetryExecutor::new(RetryConfig::default());
    let result = executor.execute(operation);
    if result.success {
        Ok(result.value.expect("value must be present on success"))
    } else {
        let last_error = result.error_history
            .last()
            .and_then(|a| a.error.clone())
            .unwrap_or_else(|| "Unknown error".to_string());
        Err(anyhow!("{}", last_error))
    }
}

/// Convenience function to retry with custom config
pub fn retry_with_config<T, F>(config: RetryConfig, operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let executor = RetryExecutor::new(config);
    let result = executor.execute(operation);
    if result.success {
        Ok(result.value.expect("value must be present on success"))
    } else {
        let last_error = result.error_history
            .last()
            .and_then(|a| a.error.clone())
            .unwrap_or_else(|| "Unknown error".to_string());
        Err(anyhow!("{}", last_error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_retry_success_first_attempt() {
        let executor = RetryExecutor::new(RetryConfig::default());
        let result = executor.execute(|| Ok(42));

        assert!(result.success);
        assert_eq!(result.value, Some(42));
        assert_eq!(result.attempts, 1);
        assert!(result.error_history.is_empty());
    }

    #[test]
    fn test_retry_success_after_failures() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            initial_delay: Duration::from_millis(1),
            ..RetryConfig::default()
        });

        let result = executor.execute(|| {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(anyhow!("Connection refused"))
            } else {
                Ok(42)
            }
        });

        assert!(result.success);
        assert_eq!(result.value, Some(42));
        assert_eq!(result.attempts, 3);
        assert_eq!(result.error_history.len(), 2);
    }

    #[test]
    fn test_retry_max_attempts_exceeded() {
        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            ..RetryConfig::default()
        });

        let result: RetryResult<i32> = executor.execute(|| {
            Err(anyhow!("Connection refused"))
        });

        assert!(!result.success);
        assert_eq!(result.value, None);
        assert_eq!(result.attempts, 3); // Initial + 2 retries
    }

    #[test]
    fn test_non_retryable_error() {
        let executor = RetryExecutor::new(RetryConfig::default());

        let result: RetryResult<i32> = executor.execute(|| {
            Err(anyhow!("Invalid API key"))
        });

        assert!(!result.success);
        assert_eq!(result.attempts, 1); // No retries for non-retryable errors
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let mut cb = CircuitBreaker::new(3, Duration::from_millis(100));

        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(10));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        thread::sleep(Duration::from_millis(15));

        assert!(cb.should_allow());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_retryable_error_matching() {
        assert!(RetryableError::ConnectionRefused.matches("Connection refused by server"));
        assert!(RetryableError::Timeout.matches("Request timed out"));
        assert!(RetryableError::ServerError.matches("HTTP 500 Internal Server Error"));
        assert!(RetryableError::RateLimited.matches("429 Too Many Requests"));
        assert!(!RetryableError::ConnectionRefused.matches("Invalid API key"));
    }

    #[test]
    fn test_calculate_delay_with_backoff() {
        let config = RetryConfig {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            add_jitter: false,
            ..Default::default()
        };

        let delay0 = config.calculate_delay(0);
        let delay1 = config.calculate_delay(1);
        let delay2 = config.calculate_delay(2);

        assert_eq!(delay0.as_millis(), 100);
        assert_eq!(delay1.as_millis(), 200);
        assert_eq!(delay2.as_millis(), 400);
    }

    #[test]
    fn test_delay_capped_at_max() {
        let config = RetryConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 10.0,
            add_jitter: false,
            ..Default::default()
        };

        let delay = config.calculate_delay(5);
        assert!(delay <= config.max_delay);
    }
}
