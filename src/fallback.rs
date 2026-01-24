//! Automatic fallback between providers
//!
//! This module provides automatic failover between AI providers
//! when one becomes unavailable or returns errors.
//!
//! # Features
//!
//! - **Automatic failover**: Switch to backup provider on failure
//! - **Health monitoring**: Track provider availability
//! - **Circuit breaker**: Prevent hammering failing providers
//! - **Priority ordering**: Prefer certain providers
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::fallback::{FallbackChain, FallbackProvider};
//!
//! let chain = FallbackChain::new()
//!     .add_provider(FallbackProvider::new("ollama", "http://localhost:11434"))
//!     .add_provider(FallbackProvider::new("lmstudio", "http://localhost:1234"));
//!
//! // Chain will try ollama first, then lmstudio if ollama fails
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for a fallback provider
#[derive(Debug, Clone)]
pub struct FallbackProvider {
    /// Provider name/ID
    pub name: String,
    /// Provider URL
    pub url: String,
    /// Priority (higher = preferred)
    pub priority: i32,
    /// Maximum consecutive failures before circuit opens
    pub max_failures: usize,
    /// Time to wait before retrying after circuit opens
    pub recovery_time: Duration,
    /// Custom timeout for this provider
    pub timeout: Option<Duration>,
    /// Whether provider is enabled
    pub enabled: bool,
}

impl FallbackProvider {
    /// Create a new provider config
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            priority: 0,
            max_failures: 3,
            recovery_time: Duration::from_secs(30),
            timeout: None,
            enabled: true,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set max failures before circuit opens
    pub fn with_max_failures(mut self, max: usize) -> Self {
        self.max_failures = max;
        self
    }

    /// Set recovery time
    pub fn with_recovery_time(mut self, duration: Duration) -> Self {
        self.recovery_time = duration;
        self
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// State of a provider in the chain
#[derive(Debug, Clone)]
pub struct ProviderState {
    /// Current status
    pub status: ProviderStatus,
    /// Consecutive failure count
    pub failure_count: usize,
    /// Last failure time
    pub last_failure: Option<Instant>,
    /// Last success time
    pub last_success: Option<Instant>,
    /// Total requests
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Circuit opened time
    pub circuit_opened_at: Option<Instant>,
}

impl Default for ProviderState {
    fn default() -> Self {
        Self {
            status: ProviderStatus::Available,
            failure_count: 0,
            last_failure: None,
            last_success: None,
            total_requests: 0,
            successful_requests: 0,
            circuit_opened_at: None,
        }
    }
}

/// Provider availability status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderStatus {
    /// Provider is available
    Available,
    /// Provider is degraded (some failures)
    Degraded,
    /// Circuit is open (provider unavailable)
    CircuitOpen,
    /// Provider is disabled
    Disabled,
}

/// Result of a fallback attempt
#[derive(Debug, Clone)]
pub struct FallbackResult<T> {
    /// The result value
    pub value: T,
    /// Provider that succeeded
    pub provider: String,
    /// Providers that were tried
    pub tried: Vec<String>,
    /// Providers that failed
    pub failed: Vec<(String, String)>,
    /// Total time taken
    pub duration: Duration,
}

/// Error from fallback chain
#[derive(Debug, Clone)]
pub struct FallbackError {
    /// All errors encountered
    pub errors: Vec<(String, String)>,
    /// Total time taken
    pub duration: Duration,
}

impl std::fmt::Display for FallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "All providers failed: ")?;
        for (i, (provider, error)) in self.errors.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", provider, error)?;
        }
        Ok(())
    }
}

impl std::error::Error for FallbackError {}

/// Fallback chain for trying multiple providers
pub struct FallbackChain {
    providers: Vec<FallbackProvider>,
    states: Arc<RwLock<HashMap<String, ProviderState>>>,
    on_fallback: Option<Box<dyn Fn(&str, &str, &str) + Send + Sync>>,
}

impl FallbackChain {
    /// Create a new fallback chain
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            states: Arc::new(RwLock::new(HashMap::new())),
            on_fallback: None,
        }
    }

    /// Add a provider to the chain
    pub fn add_provider(mut self, provider: FallbackProvider) -> Self {
        let name = provider.name.clone();
        self.providers.push(provider);
        self.providers.sort_by(|a, b| b.priority.cmp(&a.priority));

        self.states.write().unwrap().insert(name, ProviderState::default());

        self
    }

    /// Set fallback callback
    pub fn on_fallback<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, &str) + Send + Sync + 'static,
    {
        self.on_fallback = Some(Box::new(f));
        self
    }

    /// Try to execute with fallback
    pub fn try_with<F, T, E>(&self, mut f: F) -> Result<FallbackResult<T>, FallbackError>
    where
        F: FnMut(&FallbackProvider) -> Result<T, E>,
        E: std::fmt::Display,
    {
        let start = Instant::now();
        let mut tried = Vec::new();
        let mut errors = Vec::new();

        for provider in &self.providers {
            if !provider.enabled {
                continue;
            }

            // Check circuit breaker
            {
                let states = self.states.read().unwrap();
                if let Some(state) = states.get(&provider.name) {
                    if state.status == ProviderStatus::CircuitOpen {
                        if let Some(opened) = state.circuit_opened_at {
                            if opened.elapsed() < provider.recovery_time {
                                continue; // Skip this provider
                            }
                        }
                    }
                }
            }

            tried.push(provider.name.clone());

            // Update request count
            {
                let mut states = self.states.write().unwrap();
                if let Some(state) = states.get_mut(&provider.name) {
                    state.total_requests += 1;
                }
            }

            match f(provider) {
                Ok(value) => {
                    // Record success
                    {
                        let mut states = self.states.write().unwrap();
                        if let Some(state) = states.get_mut(&provider.name) {
                            state.failure_count = 0;
                            state.last_success = Some(Instant::now());
                            state.successful_requests += 1;
                            state.status = ProviderStatus::Available;
                            state.circuit_opened_at = None;
                        }
                    }

                    return Ok(FallbackResult {
                        value,
                        provider: provider.name.clone(),
                        tried,
                        failed: errors,
                        duration: start.elapsed(),
                    });
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    errors.push((provider.name.clone(), error_msg.clone()));

                    // Record failure
                    let should_open_circuit = {
                        let mut states = self.states.write().unwrap();
                        if let Some(state) = states.get_mut(&provider.name) {
                            state.failure_count += 1;
                            state.last_failure = Some(Instant::now());

                            if state.failure_count >= provider.max_failures {
                                state.status = ProviderStatus::CircuitOpen;
                                state.circuit_opened_at = Some(Instant::now());
                                true
                            } else {
                                state.status = ProviderStatus::Degraded;
                                false
                            }
                        } else {
                            false
                        }
                    };

                    // Call fallback callback
                    if let Some(ref callback) = self.on_fallback {
                        callback(&provider.name, &error_msg,
                            tried.get(tried.len().saturating_sub(2))
                                .map(|s| s.as_str())
                                .unwrap_or("none"));
                    }

                    if should_open_circuit {
                        // Log that circuit was opened
                    }
                }
            }
        }

        Err(FallbackError {
            errors,
            duration: start.elapsed(),
        })
    }

    /// Get provider state
    pub fn get_state(&self, name: &str) -> Option<ProviderState> {
        self.states.read().unwrap().get(name).cloned()
    }

    /// Get all provider states
    pub fn all_states(&self) -> HashMap<String, ProviderState> {
        self.states.read().unwrap().clone()
    }

    /// Reset provider state
    pub fn reset_provider(&self, name: &str) {
        if let Some(state) = self.states.write().unwrap().get_mut(name) {
            *state = ProviderState::default();
        }
    }

    /// Reset all providers
    pub fn reset_all(&self) {
        let mut states = self.states.write().unwrap();
        for state in states.values_mut() {
            *state = ProviderState::default();
        }
    }

    /// Get available providers
    pub fn available_providers(&self) -> Vec<&FallbackProvider> {
        let states = self.states.read().unwrap();
        self.providers.iter()
            .filter(|p| {
                p.enabled && states.get(&p.name)
                    .map(|s| s.status != ProviderStatus::CircuitOpen)
                    .unwrap_or(true)
            })
            .collect()
    }

    /// Get primary provider
    pub fn primary(&self) -> Option<&FallbackProvider> {
        self.available_providers().first().copied()
    }

    /// Enable a provider
    pub fn enable(&mut self, name: &str) -> bool {
        if let Some(p) = self.providers.iter_mut().find(|p| p.name == name) {
            p.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a provider
    pub fn disable(&mut self, name: &str) -> bool {
        if let Some(p) = self.providers.iter_mut().find(|p| p.name == name) {
            p.enabled = false;
            true
        } else {
            false
        }
    }

    /// Get provider by name
    pub fn get_provider(&self, name: &str) -> Option<&FallbackProvider> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// List all providers
    pub fn providers(&self) -> &[FallbackProvider] {
        &self.providers
    }
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Health checker for providers
pub struct HealthChecker {
    chain: Arc<Mutex<FallbackChain>>,
    check_interval: Duration,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(chain: FallbackChain, check_interval: Duration) -> Self {
        Self {
            chain: Arc::new(Mutex::new(chain)),
            check_interval,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get the chain
    pub fn chain(&self) -> &Arc<Mutex<FallbackChain>> {
        &self.chain
    }

    /// Run a health check
    pub fn check<F>(&self, checker: F)
    where
        F: Fn(&FallbackProvider) -> bool,
    {
        let chain = self.chain.lock().unwrap();
        for provider in &chain.providers {
            if !provider.enabled {
                continue;
            }

            let healthy = checker(provider);

            if let Some(state) = chain.states.write().unwrap().get_mut(&provider.name) {
                if healthy {
                    state.status = ProviderStatus::Available;
                    state.failure_count = 0;
                } else {
                    state.failure_count += 1;
                    if state.failure_count >= provider.max_failures {
                        state.status = ProviderStatus::CircuitOpen;
                        state.circuit_opened_at = Some(Instant::now());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_chain_success() {
        let chain = FallbackChain::new()
            .add_provider(FallbackProvider::new("primary", "http://localhost:1"))
            .add_provider(FallbackProvider::new("secondary", "http://localhost:2"));

        let result = chain.try_with(|p| -> Result<String, &str> {
            Ok(format!("Response from {}", p.name))
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.provider, "primary");
        assert_eq!(result.tried.len(), 1);
    }

    #[test]
    fn test_fallback_chain_failover() {
        let chain = FallbackChain::new()
            .add_provider(FallbackProvider::new("primary", "http://localhost:1"))
            .add_provider(FallbackProvider::new("secondary", "http://localhost:2"));

        let result = chain.try_with(|p| -> Result<String, &str> {
            if p.name == "primary" {
                Err("Primary failed")
            } else {
                Ok("Success from secondary".to_string())
            }
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.provider, "secondary");
        assert_eq!(result.tried.len(), 2);
        assert_eq!(result.failed.len(), 1);
    }

    #[test]
    fn test_fallback_chain_all_fail() {
        let chain = FallbackChain::new()
            .add_provider(FallbackProvider::new("p1", "http://localhost:1"))
            .add_provider(FallbackProvider::new("p2", "http://localhost:2"));

        let result = chain.try_with(|_| -> Result<String, &str> {
            Err("Failed")
        });

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.errors.len(), 2);
    }

    #[test]
    fn test_circuit_breaker() {
        let chain = FallbackChain::new()
            .add_provider(
                FallbackProvider::new("test", "http://localhost:1")
                    .with_max_failures(2)
            );

        // Fail twice to open circuit
        for _ in 0..2 {
            let _ = chain.try_with(|_| -> Result<(), &str> {
                Err("Failed")
            });
        }

        let state = chain.get_state("test").unwrap();
        assert_eq!(state.status, ProviderStatus::CircuitOpen);
    }

    #[test]
    fn test_priority_ordering() {
        let chain = FallbackChain::new()
            .add_provider(FallbackProvider::new("low", "http://1").with_priority(1))
            .add_provider(FallbackProvider::new("high", "http://2").with_priority(10))
            .add_provider(FallbackProvider::new("medium", "http://3").with_priority(5));

        let primary = chain.primary().unwrap();
        assert_eq!(primary.name, "high");
    }

    #[test]
    fn test_reset_provider() {
        let chain = FallbackChain::new()
            .add_provider(FallbackProvider::new("test", "http://localhost:1"));

        // Fail to change state
        let _ = chain.try_with(|_| -> Result<(), &str> {
            Err("Failed")
        });

        let state = chain.get_state("test").unwrap();
        assert!(state.failure_count > 0);

        // Reset
        chain.reset_provider("test");

        let state = chain.get_state("test").unwrap();
        assert_eq!(state.failure_count, 0);
    }
}
