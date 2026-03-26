//! Retry mechanisms with exponential backoff
//!
//! This module provides retry functionality for network operations with configurable
//! backoff strategies, jitter, and circuit breaker patterns.

use anyhow::{anyhow, Result};
use std::thread;
use std::time::{Duration, Instant};

/// Retry strategy configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// How to handle rate limit (429) errors specifically.
    /// Default: `RateLimitStrategy::Retry` (treat like any other retryable error).
    pub rate_limit_strategy: RateLimitStrategy,
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
            rate_limit_strategy: RateLimitStrategy::default(),
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
            retryable_errors: vec![RetryableError::ConnectionRefused, RetryableError::Timeout],
            rate_limit_strategy: RateLimitStrategy::ImmediateFallback,
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
            rate_limit_strategy: RateLimitStrategy::WaitForReset {
                max_wait_secs: 120,
                default_wait_secs: 30,
            },
        }
    }

    /// Create a config with no retries
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Create a config that waits for rate limits to clear.
    /// Best for single-provider setups where you prefer waiting over failing.
    pub fn patient() -> Self {
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
            rate_limit_strategy: RateLimitStrategy::WaitForReset {
                max_wait_secs: 300,
                default_wait_secs: 60,
            },
        }
    }

    /// Calculate delay for a specific retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);

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

/// Strategy for handling rate limit (429) errors specifically.
///
/// Rate limits are different from transient errors — the provider tells us
/// exactly when we can retry. This enum lets the caller choose what to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum RateLimitStrategy {
    /// Treat rate limits like any other retryable error — use exponential backoff.
    /// If max retries are exhausted, fail (or fall through to FallbackChain).
    Retry,
    /// Wait for the full duration indicated by `retry-after` header (or a default
    /// wait time if no header is present). Blocks until the provider is available again.
    /// Best for single-provider setups where you'd rather wait than fail.
    WaitForReset {
        /// Maximum time to wait (seconds). If retry-after exceeds this, fall back to `Retry` behavior.
        /// 0 = unlimited (will wait however long the provider says).
        max_wait_secs: u64,
        /// Default wait time (seconds) if no retry-after header is present.
        default_wait_secs: u64,
    },
    /// Ask the user what to do: wait, retry, switch provider, or abort.
    /// Returns `RateLimitDecision` via the callback.
    AskUser,
    /// Immediately give up on this provider and let FallbackChain try the next one.
    /// Useful in multi-provider setups where switching is cheap.
    ImmediateFallback,
}

impl Default for RateLimitStrategy {
    fn default() -> Self {
        Self::Retry
    }
}

/// User's decision when `RateLimitStrategy::AskUser` is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitDecision {
    /// Wait the specified duration, then retry.
    Wait,
    /// Retry immediately (ignore rate limit).
    RetryNow,
    /// Give up on this provider, try the next one.
    SwitchProvider,
    /// Abort the entire operation.
    Abort,
}

/// Information about a rate limit event, passed to the user callback.
#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    /// Provider name (if known).
    pub provider: String,
    /// Suggested wait time from `retry-after` header (if present).
    pub retry_after_secs: Option<u64>,
    /// How many retries have been attempted so far.
    pub attempts_so_far: u32,
    /// Total time spent so far.
    pub elapsed: Duration,
}

/// Types of errors that can be retried
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

    /// Execute an operation with retries.
    ///
    /// Rate limit errors (429) are handled according to `RetryConfig::rate_limit_strategy`:
    /// - `Retry` — normal exponential backoff (default)
    /// - `WaitForReset` — wait for the `retry-after` duration (or default), then retry once
    /// - `AskUser` — calls `on_rate_limit` callback (use `execute_with_rate_limit_handler`)
    /// - `ImmediateFallback` — fail immediately so FallbackChain can try next provider
    pub fn execute<T, F>(&self, mut operation: F) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
    {
        self.execute_inner(&mut operation, None::<fn(RateLimitInfo) -> RateLimitDecision>)
    }

    /// Execute with a callback for rate limit decisions.
    ///
    /// When `rate_limit_strategy` is `AskUser`, this callback is invoked with
    /// `RateLimitInfo` and must return a `RateLimitDecision`.
    pub fn execute_with_rate_limit_handler<T, F, H>(
        &self,
        mut operation: F,
        handler: H,
    ) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
        H: Fn(RateLimitInfo) -> RateLimitDecision,
    {
        self.execute_inner(&mut operation, Some(handler))
    }

    fn execute_inner<T, F, H>(
        &self,
        operation: &mut F,
        rate_limit_handler: Option<H>,
    ) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
        H: Fn(RateLimitInfo) -> RateLimitDecision,
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

                    // Check if this is a rate limit error
                    let is_rate_limited = RetryableError::RateLimited.matches(&error_str);

                    // Check if this error is retryable at all
                    let is_retryable = self
                        .config
                        .retryable_errors
                        .iter()
                        .any(|re| re.matches(&error_str));

                    // Handle rate limit with special strategy
                    if is_rate_limited {
                        let retry_after = parse_retry_after(&error_str);

                        match self.config.rate_limit_strategy {
                            RateLimitStrategy::ImmediateFallback => {
                                error_history.push(RetryAttempt {
                                    attempt,
                                    error: Some(error_str),
                                    duration: attempt_duration,
                                    delay_after: None,
                                });
                                return RetryResult {
                                    value: None,
                                    attempts: attempt + 1,
                                    total_duration: start_time.elapsed(),
                                    error_history,
                                    success: false,
                                };
                            }
                            RateLimitStrategy::WaitForReset { max_wait_secs, default_wait_secs } => {
                                let wait_secs = retry_after.unwrap_or(default_wait_secs);
                                let wait_secs = if max_wait_secs > 0 { wait_secs.min(max_wait_secs) } else { wait_secs };
                                let wait = Duration::from_secs(wait_secs);

                                error_history.push(RetryAttempt {
                                    attempt,
                                    error: Some(format!("{} (waiting {}s for rate limit reset)", error_str, wait_secs)),
                                    duration: attempt_duration,
                                    delay_after: Some(wait),
                                });

                                thread::sleep(wait);
                                attempt += 1;
                                continue;
                            }
                            RateLimitStrategy::AskUser => {
                                let info = RateLimitInfo {
                                    provider: extract_provider(&error_str),
                                    retry_after_secs: retry_after,
                                    attempts_so_far: attempt + 1,
                                    elapsed: start_time.elapsed(),
                                };

                                let decision = if let Some(ref handler) = rate_limit_handler {
                                    handler(info)
                                } else {
                                    // No handler provided — fall back to normal retry
                                    RateLimitDecision::RetryNow
                                };

                                match decision {
                                    RateLimitDecision::Wait => {
                                        let wait_secs = retry_after.unwrap_or(60);
                                        let wait = Duration::from_secs(wait_secs);
                                        error_history.push(RetryAttempt {
                                            attempt,
                                            error: Some(format!("{} (user chose: wait {}s)", error_str, wait_secs)),
                                            duration: attempt_duration,
                                            delay_after: Some(wait),
                                        });
                                        thread::sleep(wait);
                                        attempt += 1;
                                        continue;
                                    }
                                    RateLimitDecision::RetryNow => {
                                        error_history.push(RetryAttempt {
                                            attempt,
                                            error: Some(format!("{} (user chose: retry now)", error_str)),
                                            duration: attempt_duration,
                                            delay_after: None,
                                        });
                                        attempt += 1;
                                        continue;
                                    }
                                    RateLimitDecision::SwitchProvider | RateLimitDecision::Abort => {
                                        error_history.push(RetryAttempt {
                                            attempt,
                                            error: Some(format!("{} (user chose: {})", error_str,
                                                if decision == RateLimitDecision::Abort { "abort" } else { "switch provider" })),
                                            duration: attempt_duration,
                                            delay_after: None,
                                        });
                                        return RetryResult {
                                            value: None,
                                            attempts: attempt + 1,
                                            total_duration: start_time.elapsed(),
                                            error_history,
                                            success: false,
                                        };
                                    }
                                }
                            }
                            RateLimitStrategy::Retry => {
                                // Fall through to normal retry logic below
                            }
                        }
                    }

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

    /// Execute with a callback for each retry attempt.
    /// Rate limit strategy is still respected.
    pub fn execute_with_callback<T, F, C>(
        &self,
        operation: F,
        on_retry: C,
    ) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
        C: FnMut(u32, &str, Duration),
    {
        self.execute_with_callback_and_rate_limit(
            operation,
            on_retry,
            None::<fn(RateLimitInfo) -> RateLimitDecision>,
        )
    }

    /// Execute with both a retry callback and a rate limit handler.
    pub fn execute_with_callback_and_rate_limit<T, F, C, H>(
        &self,
        mut operation: F,
        mut on_retry: C,
        rate_limit_handler: Option<H>,
    ) -> RetryResult<T>
    where
        F: FnMut() -> Result<T>,
        C: FnMut(u32, &str, Duration),
        H: Fn(RateLimitInfo) -> RateLimitDecision,
    {
        let result = self.execute_inner(&mut operation, rate_limit_handler);
        // Replay error history through callback
        for entry in &result.error_history {
            if let Some(ref err) = entry.error {
                on_retry(entry.attempt, err, entry.delay_after.unwrap_or_default());
            }
        }
        result
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
#[non_exhaustive]
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

/// Combined retry executor with circuit breaker.
///
/// Optionally integrates with [`AdaptiveTimeout`](crate::adaptive_timeout::AdaptiveTimeout)
/// for dynamic timeout adjustment and [`DeadLetterQueue`](crate::message_queue::DeadLetterQueue)
/// for capturing permanently failed requests.
pub struct ResilientExecutor {
    retry: RetryExecutor,
    circuit_breaker: CircuitBreaker,
    /// Optional adaptive timeout — when set, overrides the static `attempt_timeout`
    /// in `RetryConfig` with a dynamically calculated timeout based on observed latency.
    pub adaptive_timeout: Option<std::sync::Arc<crate::adaptive_timeout::AdaptiveTimeout>>,
    /// Optional dead letter queue — when set, failed requests that exhaust all retries
    /// are automatically added to this queue with their error history.
    pub dead_letter_queue: Option<std::sync::Arc<crate::message_queue::DeadLetterQueue>>,
}

impl ResilientExecutor {
    /// Create a new resilient executor
    pub fn new(
        retry_config: RetryConfig,
        failure_threshold: u32,
        recovery_timeout: Duration,
    ) -> Self {
        Self {
            retry: RetryExecutor::new(retry_config),
            circuit_breaker: CircuitBreaker::new(failure_threshold, recovery_timeout),
            adaptive_timeout: None,
            dead_letter_queue: None,
        }
    }

    /// Set an adaptive timeout source. When set, the dynamic timeout
    /// is used instead of the static `attempt_timeout` in `RetryConfig`.
    pub fn with_adaptive_timeout(
        mut self,
        timeout: std::sync::Arc<crate::adaptive_timeout::AdaptiveTimeout>,
    ) -> Self {
        self.adaptive_timeout = Some(timeout);
        self
    }

    /// Set a dead letter queue for capturing permanently failed requests.
    pub fn with_dead_letter_queue(
        mut self,
        dlq: std::sync::Arc<crate::message_queue::DeadLetterQueue>,
    ) -> Self {
        self.dead_letter_queue = Some(dlq);
        self
    }

    /// Execute an operation with both retry and circuit breaker protection.
    ///
    /// When an [`AdaptiveTimeout`](crate::adaptive_timeout::AdaptiveTimeout) is attached,
    /// each successful attempt records its latency so future timeouts adapt.
    ///
    /// When a [`DeadLetterQueue`](crate::message_queue::DeadLetterQueue) is attached,
    /// permanently failed operations (all retries exhausted) are captured with their
    /// full error history.
    pub fn execute<T, F>(&mut self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        if !self.circuit_breaker.should_allow() {
            return Err(anyhow!(
                "Circuit breaker is open, service appears unavailable"
            ));
        }

        let result = self.retry.execute(&mut operation);

        if result.success {
            self.circuit_breaker.record_success();
            // Record latency for adaptive timeout
            if let Some(ref at) = self.adaptive_timeout {
                at.record(result.total_duration);
            }
            Ok(result.value.expect("value must be present on success"))
        } else {
            self.circuit_breaker.record_failure();
            let last_error = result
                .error_history
                .last()
                .and_then(|a| a.error.clone())
                .unwrap_or_else(|| "Unknown error".to_string());

            // Dead-letter the failed operation
            if let Some(ref dlq) = self.dead_letter_queue {
                let error_history: Vec<String> = result
                    .error_history
                    .iter()
                    .filter_map(|a| a.error.clone())
                    .collect();

                let category = crate::message_queue::FailureCategory::from_error(&last_error);
                let msg = crate::message_queue::QueueMessage::new(
                    &format!("Failed operation after {} attempts", result.attempts),
                );
                dlq.add_detailed(
                    msg,
                    last_error.clone(),
                    category,
                    result.attempts,
                    error_history,
                );
            }

            Err(anyhow!(
                "Operation failed after {} attempts: {}",
                result.attempts,
                last_error
            ))
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

/// Extract retry-after seconds from an error message.
/// Looks for patterns like "retry after 30", "retry-after: 60", "retry_after=45".
fn parse_retry_after(error: &str) -> Option<u64> {
    let lower = error.to_lowercase();
    // Try "retry.after.*N" pattern
    if let Some(idx) = lower.find("retry") {
        let after_retry = &lower[idx..];
        // Find first number after "retry"
        let mut num_start = None;
        for (i, c) in after_retry.char_indices() {
            if c.is_ascii_digit() {
                if num_start.is_none() {
                    num_start = Some(i);
                }
            } else if num_start.is_some() {
                let n = &after_retry[num_start.unwrap()..i];
                if let Ok(secs) = n.parse::<u64>() {
                    if secs > 0 && secs < 86400 {
                        return Some(secs);
                    }
                }
                num_start = None;
            }
        }
        // Check if number runs to end of string
        if let Some(start) = num_start {
            if let Ok(secs) = after_retry[start..].parse::<u64>() {
                if secs > 0 && secs < 86400 {
                    return Some(secs);
                }
            }
        }
    }
    None
}

/// Try to extract a provider name from an error message.
fn extract_provider(error: &str) -> String {
    let lower = error.to_lowercase();
    let providers = [
        "openai", "anthropic", "claude", "gemini", "google", "bedrock",
        "ollama", "lmstudio", "together", "groq", "huggingface", "cohere",
    ];
    for p in &providers {
        if lower.contains(p) {
            return p.to_string();
        }
    }
    "unknown".to_string()
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
        let last_error = result
            .error_history
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
        let last_error = result
            .error_history
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

        let result: RetryResult<i32> = executor.execute(|| Err(anyhow!("Connection refused")));

        assert!(!result.success);
        assert_eq!(result.value, None);
        assert_eq!(result.attempts, 3); // Initial + 2 retries
    }

    #[test]
    fn test_non_retryable_error() {
        let executor = RetryExecutor::new(RetryConfig::default());

        let result: RetryResult<i32> = executor.execute(|| Err(anyhow!("Invalid API key")));

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

    #[test]
    fn test_circuit_breaker_reset_and_failure_count() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(100));

        // Initially closed with zero failures
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);

        // Record failures until circuit opens
        cb.record_failure();
        assert_eq!(cb.failure_count(), 1);
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);
        assert_eq!(cb.state(), CircuitState::Open);

        // should_allow returns false while open
        assert!(!cb.should_allow());

        // Reset should bring everything back to initial state
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(cb.should_allow());

        // After reset, circuit operates normally again
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 1);
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_resilient_executor_records_adaptive_timeout() {
        use crate::adaptive_timeout::{AdaptiveTimeout, AdaptiveTimeoutConfig};

        let at = Arc::new(AdaptiveTimeout::new(AdaptiveTimeoutConfig::responsive()));
        let mut executor = ResilientExecutor::new(
            RetryConfig::no_retry(),
            5,
            Duration::from_secs(30),
        )
        .with_adaptive_timeout(at.clone());

        // Execute a successful operation
        let result = executor.execute(|| Ok(42));
        assert!(result.is_ok());

        // Adaptive timeout should have recorded the latency
        assert!(at.sample_count() >= 1);
    }

    #[test]
    fn test_resilient_executor_dead_letters_on_failure() {
        use crate::message_queue::DeadLetterQueue;

        let dlq = Arc::new(DeadLetterQueue::new(100));
        let mut executor = ResilientExecutor::new(
            RetryConfig {
                max_retries: 1,
                initial_delay: Duration::from_millis(1),
                ..RetryConfig::default()
            },
            5,
            Duration::from_secs(30),
        )
        .with_dead_letter_queue(dlq.clone());

        // Execute a failing operation (timeout error → retryable)
        let result: Result<i32> = executor.execute(|| Err(anyhow!("Connection refused")));
        assert!(result.is_err());

        // The DLQ should have captured the failed operation
        assert_eq!(dlq.len(), 1);
        let stats = dlq.stats();
        assert_eq!(stats.total, 1);
    }

    #[test]
    fn test_resilient_executor_no_dlq_without_config() {
        let mut executor = ResilientExecutor::new(
            RetryConfig::no_retry(),
            5,
            Duration::from_secs(30),
        );

        // No DLQ attached — failure should not panic
        let result: Result<i32> = executor.execute(|| Err(anyhow!("some error")));
        assert!(result.is_err());
    }

    #[test]
    fn test_rate_limit_strategy_immediate_fallback() {
        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            rate_limit_strategy: RateLimitStrategy::ImmediateFallback,
            ..RetryConfig::default()
        });

        let result: RetryResult<i32> = executor.execute(|| {
            Err(anyhow!("429 Too Many Requests"))
        });

        assert!(!result.success);
        assert_eq!(result.attempts, 1); // No retries — immediate fallback
    }

    #[test]
    fn test_rate_limit_strategy_retry_default() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            rate_limit_strategy: RateLimitStrategy::Retry,
            ..RetryConfig::default()
        });

        let result = executor.execute(|| {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(anyhow!("429 rate limit exceeded"))
            } else {
                Ok(42)
            }
        });

        assert!(result.success);
        assert_eq!(result.attempts, 3);
    }

    #[test]
    fn test_rate_limit_strategy_ask_user_abort() {
        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(1),
            rate_limit_strategy: RateLimitStrategy::AskUser,
            ..RetryConfig::default()
        });

        let result: RetryResult<i32> = executor.execute_with_rate_limit_handler(
            || Err(anyhow!("429 Too Many Requests from openai")),
            |info| {
                assert_eq!(info.provider, "openai");
                RateLimitDecision::Abort
            },
        );

        assert!(!result.success);
        assert_eq!(result.attempts, 1);
    }

    #[test]
    fn test_rate_limit_strategy_ask_user_switch_provider() {
        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(1),
            rate_limit_strategy: RateLimitStrategy::AskUser,
            ..RetryConfig::default()
        });

        let result: RetryResult<i32> = executor.execute_with_rate_limit_handler(
            || Err(anyhow!("429 rate limited by anthropic")),
            |info| {
                assert_eq!(info.provider, "anthropic");
                RateLimitDecision::SwitchProvider
            },
        );

        assert!(!result.success);
        assert_eq!(result.attempts, 1);
    }

    #[test]
    fn test_rate_limit_strategy_ask_user_retry_now() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(1),
            rate_limit_strategy: RateLimitStrategy::AskUser,
            ..RetryConfig::default()
        });

        let result = executor.execute_with_rate_limit_handler(
            || {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                if count < 1 {
                    Err(anyhow!("429 rate limit"))
                } else {
                    Ok(99)
                }
            },
            |_info| RateLimitDecision::RetryNow,
        );

        assert!(result.success);
        assert_eq!(result.value, Some(99));
    }

    #[test]
    fn test_parse_retry_after() {
        assert_eq!(parse_retry_after("429 Too Many Requests, retry after 30 seconds"), Some(30));
        assert_eq!(parse_retry_after("rate limit, retry-after: 60"), Some(60));
        assert_eq!(parse_retry_after("no retry info here"), None);
        assert_eq!(parse_retry_after("retry_after=45"), Some(45));
    }

    #[test]
    fn test_extract_provider() {
        assert_eq!(extract_provider("429 from OpenAI API"), "openai");
        assert_eq!(extract_provider("Anthropic rate limited"), "anthropic");
        assert_eq!(extract_provider("some error"), "unknown");
    }

    #[test]
    fn test_patient_preset() {
        let config = RetryConfig::patient();
        assert_eq!(config.max_retries, 3);
        assert!(matches!(
            config.rate_limit_strategy,
            RateLimitStrategy::WaitForReset { max_wait_secs: 300, default_wait_secs: 60 }
        ));
    }

    #[test]
    fn test_rate_limit_strategy_serialization() {
        let strategy = RateLimitStrategy::WaitForReset {
            max_wait_secs: 120,
            default_wait_secs: 30,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let restored: RateLimitStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, strategy);
    }
}
