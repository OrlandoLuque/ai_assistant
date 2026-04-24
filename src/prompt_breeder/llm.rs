//! Minimal `LlmClient` trait for PromptBreeder. We keep this surface small
//! and self-contained so the breeder can run with any backend (including
//! the `MockLlmClient` used across tests). Callers who already have a V95
//! `LlmClient` wire it in via the `LlmClient::adapt` helper or by providing
//! their own impl of this trait.

use std::sync::{Arc, Mutex};

use super::config::RetryPolicy;

/// Wire-level usage returned by a completion. Optional because not every
/// provider reports it; when absent we fall back to a cheap length estimate.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Successful completion response.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub usage: TokenUsage,
    pub latency_ms: u64,
}

#[derive(Debug, Clone)]
pub enum LlmError {
    Transport(String),
    Timeout,
    InvalidResponse(String),
    CapacityExceeded,
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(s) => write!(f, "llm transport error: {s}"),
            Self::Timeout => f.write_str("llm timeout"),
            Self::InvalidResponse(s) => write!(f, "invalid llm response: {s}"),
            Self::CapacityExceeded => f.write_str("llm capacity exceeded"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Synchronous-style LLM client used by the breeder. Implementations SHOULD
/// block the caller thread — the breeder runs a single evaluation loop and
/// manages its own parallelism via `ParallelEvaluation` (V97.1).
pub trait LlmClient: Send + Sync {
    fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError>;

    /// Hint about the caller-provided fingerprint, used only by the
    /// `MockLlmClient` and adapters.
    fn backend_id(&self) -> &str {
        "unknown"
    }
}

/// Retry wrapper — applies `RetryPolicy` to any inner client. On persistent
/// failure the wrapper returns the last error.
pub struct RetryingLlmClient {
    inner: Arc<dyn LlmClient>,
    policy: RetryPolicy,
}

impl RetryingLlmClient {
    pub fn new(inner: Arc<dyn LlmClient>, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }
}

impl LlmClient for RetryingLlmClient {
    fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let mut attempt = 0u32;
        loop {
            match self.inner.complete(prompt) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    if attempt >= self.policy.max_retries {
                        return Err(e);
                    }
                    let delay = match self.policy.backoff {
                        super::config::Backoff::Fixed { ms } => ms,
                        super::config::Backoff::Exponential { base_ms, factor } => {
                            let mul = (factor as f64).powi(attempt as i32);
                            (base_ms as f64 * mul) as u64
                        }
                    };
                    std::thread::sleep(std::time::Duration::from_millis(delay));
                    attempt += 1;
                }
            }
        }
    }

    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }
}

/// Price table for `BudgetLimit::MaxCostUsd`. USD per 1M tokens. If a provider
/// isn't in the map, cost budgets pass through with a warning ledger event
/// (enforced at the breeder level).
#[derive(Debug, Clone)]
pub struct CostEstimator {
    /// `(provider/model) -> (input_usd_per_mtok, output_usd_per_mtok)`.
    pub prices: std::collections::HashMap<String, (f64, f64)>,
}

impl Default for CostEstimator {
    fn default() -> Self {
        let mut prices = std::collections::HashMap::new();
        // Conservative placeholder prices. Callers should override these
        // for accurate budgeting. The breeder does not ship any live cost
        // tracking from the provider APIs themselves.
        prices.insert("anthropic/claude-opus-4-7".into(), (15.0, 75.0));
        prices.insert("openai/gpt-4o".into(), (5.0, 15.0));
        prices.insert("ollama/local".into(), (0.0, 0.0));
        Self { prices }
    }
}

impl CostEstimator {
    pub fn estimate(&self, fingerprint: &str, usage: TokenUsage) -> f64 {
        if let Some((in_p, out_p)) = self.prices.get(fingerprint) {
            let in_cost = in_p * (usage.input_tokens as f64) / 1_000_000.0;
            let out_cost = out_p * (usage.output_tokens as f64) / 1_000_000.0;
            in_cost + out_cost
        } else {
            0.0
        }
    }
}

// =============================================================================
// Mock client — used by tests + by CLI `dry-run` mode so the binary runs
// without any live provider wired.
// =============================================================================

/// Programmable mock. Returns a fixed response or a sequence of scripted
/// responses, and counts calls for assertions.
pub struct MockLlmClient {
    responses: Mutex<MockState>,
}

struct MockState {
    script: Vec<String>,
    cursor: usize,
    default: String,
    calls: u64,
    fail_mode: FailMode,
    fail_after: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    Never,
    Always(u64),
    AfterCalls(u64),
}

impl MockLlmClient {
    pub fn returning(s: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(MockState {
                script: vec![],
                cursor: 0,
                default: s.into(),
                calls: 0,
                fail_mode: FailMode::Never,
                fail_after: 0,
            }),
        }
    }

    pub fn scripted(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(MockState {
                script: responses,
                cursor: 0,
                default: String::new(),
                calls: 0,
                fail_mode: FailMode::Never,
                fail_after: 0,
            }),
        }
    }

    pub fn with_failure(self, mode: FailMode) -> Self {
        if let Ok(mut s) = self.responses.lock() {
            s.fail_mode = mode;
        }
        self
    }

    pub fn call_count(&self) -> u64 {
        self.responses.lock().map(|s| s.calls).unwrap_or(0)
    }
}

impl LlmClient for MockLlmClient {
    fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let mut state = self
            .responses
            .lock()
            .map_err(|_| LlmError::InvalidResponse("mock poisoned".into()))?;
        state.calls += 1;
        let calls = state.calls;
        state.fail_after = state.fail_after.saturating_add(1);
        match state.fail_mode {
            FailMode::Always(_) => return Err(LlmError::Transport("mock always fails".into())),
            FailMode::AfterCalls(n) if calls > n => {
                return Err(LlmError::Transport("mock failed after calls".into()));
            }
            _ => {}
        }
        let text = if state.cursor < state.script.len() {
            let t = state.script[state.cursor].clone();
            state.cursor += 1;
            t
        } else {
            state.default.clone()
        };
        let usage = TokenUsage {
            input_tokens: (prompt.len() as u64 / 4).max(1),
            output_tokens: (text.len() as u64 / 4).max(1),
        };
        Ok(LlmResponse {
            text,
            usage,
            latency_ms: 1,
        })
    }

    fn backend_id(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{Backoff, RetryPolicy};
    use super::*;

    #[test]
    fn mock_returning_is_constant() {
        let m = MockLlmClient::returning("hi");
        assert_eq!(m.complete("x").unwrap().text, "hi");
        assert_eq!(m.complete("y").unwrap().text, "hi");
        assert_eq!(m.call_count(), 2);
    }

    #[test]
    fn mock_scripted_advances_cursor() {
        let m = MockLlmClient::scripted(vec!["one".into(), "two".into()]);
        assert_eq!(m.complete("p").unwrap().text, "one");
        assert_eq!(m.complete("p").unwrap().text, "two");
        // After script end, fallback is empty default.
        assert_eq!(m.complete("p").unwrap().text, "");
    }

    #[test]
    fn retry_wrapper_succeeds_after_transient_failure() {
        let inner =
            Arc::new(MockLlmClient::returning("ok").with_failure(FailMode::AfterCalls(u64::MAX)));
        let policy = RetryPolicy {
            max_retries: 2,
            backoff: Backoff::Fixed { ms: 0 },
        };
        let client = RetryingLlmClient::new(inner, policy);
        assert_eq!(client.complete("x").unwrap().text, "ok");
    }

    #[test]
    fn retry_wrapper_propagates_persistent_failure() {
        let inner = Arc::new(MockLlmClient::returning("ok").with_failure(FailMode::Always(0)));
        let policy = RetryPolicy {
            max_retries: 1,
            backoff: Backoff::Fixed { ms: 0 },
        };
        let client = RetryingLlmClient::new(inner, policy);
        assert!(client.complete("x").is_err());
    }

    #[test]
    fn cost_estimator_known_model() {
        let e = CostEstimator::default();
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let cost = e.estimate("anthropic/claude-opus-4-7", usage);
        assert!((cost - 15.0).abs() < 1e-9);
    }

    #[test]
    fn cost_estimator_unknown_returns_zero() {
        let e = CostEstimator::default();
        let cost = e.estimate("unknown/x", TokenUsage::default());
        assert_eq!(cost, 0.0);
    }
}
