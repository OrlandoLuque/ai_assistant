//! Online evaluation and feedback hooks
//!
//! Real-time quality monitoring for AI model outputs with configurable hooks,
//! sampling, alerting, and execution fingerprinting.

#[cfg(feature = "eval")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "eval")]
use std::collections::HashMap;
#[cfg(feature = "eval")]
use std::fmt;
#[cfg(feature = "eval")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "eval")]
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

#[cfg(feature = "eval")]
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// FeedbackScore
// ---------------------------------------------------------------------------

/// A single feedback score produced by a hook.
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackScore {
    pub hook_name: String,
    pub score: f64,
    pub details: Option<String>,
    pub timestamp: u64,
}

#[cfg(feature = "eval")]
impl FeedbackScore {
    pub fn new(hook_name: &str, score: f64) -> Self {
        Self {
            hook_name: hook_name.to_string(),
            score: score.clamp(0.0, 1.0),
            details: None,
            timestamp: now_unix_secs(),
        }
    }

    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// EvalContext
// ---------------------------------------------------------------------------

/// Context passed to every hook evaluation.
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    pub agent_id: Option<String>,
    pub task_type: Option<String>,
    pub metadata: HashMap<String, String>,
    pub latency_ms: Option<u64>,
    pub token_count: Option<usize>,
}

#[cfg(feature = "eval")]
impl EvalContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_agent_id(mut self, id: &str) -> Self {
        self.agent_id = Some(id.to_string());
        self
    }

    pub fn with_task_type(mut self, task: &str) -> Self {
        self.task_type = Some(task.to_string());
        self
    }

    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.latency_ms = Some(ms);
        self
    }

    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// FeedbackHook trait
// ---------------------------------------------------------------------------

/// Trait for pluggable feedback hooks that score model outputs.
#[cfg(feature = "eval")]
pub trait FeedbackHook: Send + Sync {
    /// Human-readable name of the hook.
    fn name(&self) -> &str;
    /// Evaluate an input/output pair and return a score.
    fn evaluate(&self, input: &str, output: &str, context: &EvalContext) -> FeedbackScore;
}

// ---------------------------------------------------------------------------
// AlertConfig / AlertEvent
// ---------------------------------------------------------------------------

/// Defines when an alert should fire for a specific hook.
#[cfg(feature = "eval")]
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub hook_name: String,
    /// Alert if score falls below this value.
    pub threshold: f64,
    /// Number of consecutive low scores before firing.
    pub consecutive_failures: usize,
}

#[cfg(feature = "eval")]
impl AlertConfig {
    pub fn new(hook_name: &str, threshold: f64, consecutive_failures: usize) -> Self {
        Self {
            hook_name: hook_name.to_string(),
            threshold,
            consecutive_failures,
        }
    }
}

/// An alert event fired when thresholds are breached.
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub hook_name: String,
    pub score: f64,
    pub threshold: f64,
    pub consecutive_count: usize,
    pub timestamp: u64,
    pub message: String,
}

// ---------------------------------------------------------------------------
// EvalSamplingConfig
// ---------------------------------------------------------------------------

/// Controls how many evaluations are actually run.
#[cfg(feature = "eval")]
#[derive(Debug, Clone)]
pub struct EvalSamplingConfig {
    /// Fraction of requests to evaluate (0.0 - 1.0).
    pub sample_rate: f64,
    pub min_samples_per_hour: usize,
    pub max_samples_per_hour: usize,
}

#[cfg(feature = "eval")]
impl Default for EvalSamplingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1.0,
            min_samples_per_hour: 0,
            max_samples_per_hour: usize::MAX,
        }
    }
}

#[cfg(feature = "eval")]
impl EvalSamplingConfig {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate: sample_rate.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    pub fn with_hourly_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_samples_per_hour = min;
        self.max_samples_per_hour = max;
        self
    }
}

// ---------------------------------------------------------------------------
// ExecutionFingerprint
// ---------------------------------------------------------------------------

/// Unique fingerprint for an execution, useful for deduplication and audit.
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionFingerprint {
    pub agent_id: String,
    pub task_hash: String,
    pub timestamp: u64,
    pub random_id: String,
}

#[cfg(feature = "eval")]
impl ExecutionFingerprint {
    /// Create a new fingerprint. The `task` string is hashed into a hex digest
    /// and a pseudo-random id is generated from the current timestamp.
    pub fn new(agent_id: &str, task: &str) -> Self {
        let task_hash = Self::simple_hash(task);
        let ts = now_unix_secs();
        // Pseudo-random id from XOR of timestamp, task hash bytes, and a counter
        let random_id = Self::pseudo_random_id(ts, &task_hash);
        Self {
            agent_id: agent_id.to_string(),
            task_hash,
            timestamp: ts,
            random_id,
        }
    }

    /// Simple deterministic hash (DJB2 variant) returning a hex string.
    fn simple_hash(input: &str) -> String {
        let mut hash: u64 = 5381;
        for byte in input.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        format!("{:016x}", hash)
    }

    /// Generate a pseudo-random hex id from timestamp and hash.
    fn pseudo_random_id(ts: u64, hash: &str) -> String {
        let hash_bytes: u64 = hash
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(u64::from(b)));
        let mixed = ts
            .wrapping_mul(6364136223846793005)
            .wrapping_add(hash_bytes);
        format!("{:016x}", mixed)
    }
}

// ---------------------------------------------------------------------------
// Built-in hooks
// ---------------------------------------------------------------------------

/// Scores responses based on response latency.
#[cfg(feature = "eval")]
#[derive(Debug)]
pub struct LatencyHook {
    pub max_acceptable_ms: u64,
}

#[cfg(feature = "eval")]
impl LatencyHook {
    pub fn new(max_acceptable_ms: u64) -> Self {
        Self { max_acceptable_ms }
    }
}

#[cfg(feature = "eval")]
impl FeedbackHook for LatencyHook {
    fn name(&self) -> &str {
        "latency"
    }

    fn evaluate(&self, _input: &str, _output: &str, context: &EvalContext) -> FeedbackScore {
        let latency = context.latency_ms.unwrap_or(0);
        let score = if latency <= self.max_acceptable_ms {
            1.0
        } else {
            // Proportional decrease: at 2x max it's 0.5, at 3x it's 0.33, etc.
            let ratio = self.max_acceptable_ms as f64 / latency as f64;
            ratio.clamp(0.0, 1.0)
        };
        FeedbackScore::new("latency", score)
            .with_details(&format!("latency={}ms, max={}ms", latency, self.max_acceptable_ms))
    }
}

/// Scores based on estimated token cost.
#[cfg(feature = "eval")]
#[derive(Debug)]
pub struct CostHook {
    pub cost_per_token: f64,
    pub max_cost: f64,
}

#[cfg(feature = "eval")]
impl CostHook {
    pub fn new(cost_per_token: f64, max_cost: f64) -> Self {
        Self {
            cost_per_token,
            max_cost,
        }
    }
}

#[cfg(feature = "eval")]
impl FeedbackHook for CostHook {
    fn name(&self) -> &str {
        "cost"
    }

    fn evaluate(&self, _input: &str, _output: &str, context: &EvalContext) -> FeedbackScore {
        let tokens = context.token_count.unwrap_or(0);
        let cost = tokens as f64 * self.cost_per_token;
        let score = if cost <= 0.0 || self.max_cost <= 0.0 {
            1.0
        } else if cost >= self.max_cost {
            0.0
        } else {
            1.0 - (cost / self.max_cost)
        };
        FeedbackScore::new("cost", score)
            .with_details(&format!("tokens={}, cost={:.6}, max={:.6}", tokens, cost, self.max_cost))
    }
}

/// Scores relevance via TF-IDF cosine similarity between input and output.
#[cfg(feature = "eval")]
#[derive(Debug)]
pub struct RelevanceHook;

#[cfg(feature = "eval")]
impl RelevanceHook {
    pub fn new() -> Self {
        Self
    }

    /// Tokenise a string into lowercase words.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(String::from)
            .collect()
    }

    /// Build term-frequency map.
    fn term_freq(tokens: &[String]) -> HashMap<String, f64> {
        let mut tf: HashMap<String, f64> = HashMap::new();
        let len = tokens.len().max(1) as f64;
        for t in tokens {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        for v in tf.values_mut() {
            *v /= len;
        }
        tf
    }

    /// TF-IDF cosine similarity between two texts.
    ///
    /// Uses a two-document corpus for IDF computation.
    fn tfidf_cosine(text_a: &str, text_b: &str) -> f64 {
        let tokens_a = Self::tokenize(text_a);
        let tokens_b = Self::tokenize(text_b);
        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        let tf_a = Self::term_freq(&tokens_a);
        let tf_b = Self::term_freq(&tokens_b);

        // Build vocabulary
        let mut vocab: HashMap<String, (bool, bool)> = HashMap::new();
        for key in tf_a.keys() {
            vocab.entry(key.clone()).or_insert((false, false)).0 = true;
        }
        for key in tf_b.keys() {
            vocab.entry(key.clone()).or_insert((false, false)).1 = true;
        }

        let num_docs: f64 = 2.0;

        // Compute TF-IDF vectors and cosine similarity in a single pass
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (term, (in_a, in_b)) in &vocab {
            let df = (*in_a as u32 + *in_b as u32) as f64;
            let idf = (num_docs / df).ln() + 1.0;

            let tfidf_a = tf_a.get(term).copied().unwrap_or(0.0) * idf;
            let tfidf_b = tf_b.get(term).copied().unwrap_or(0.0) * idf;

            dot += tfidf_a * tfidf_b;
            norm_a += tfidf_a * tfidf_a;
            norm_b += tfidf_b * tfidf_b;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }
}

#[cfg(feature = "eval")]
impl Default for RelevanceHook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "eval")]
impl FeedbackHook for RelevanceHook {
    fn name(&self) -> &str {
        "relevance"
    }

    fn evaluate(&self, input: &str, output: &str, _context: &EvalContext) -> FeedbackScore {
        let similarity = Self::tfidf_cosine(input, output);
        FeedbackScore::new("relevance", similarity)
            .with_details(&format!("tfidf_cosine={:.4}", similarity))
    }
}

/// Scores based on presence of toxic / blocked words.
#[cfg(feature = "eval")]
#[derive(Debug)]
pub struct ToxicityHook {
    pub blocked_words: Vec<String>,
}

#[cfg(feature = "eval")]
impl ToxicityHook {
    pub fn new(blocked_words: Vec<String>) -> Self {
        Self { blocked_words }
    }
}

#[cfg(feature = "eval")]
impl FeedbackHook for ToxicityHook {
    fn name(&self) -> &str {
        "toxicity"
    }

    fn evaluate(&self, _input: &str, output: &str, _context: &EvalContext) -> FeedbackScore {
        let lower = output.to_lowercase();
        let found: Vec<&String> = self
            .blocked_words
            .iter()
            .filter(|w| lower.contains(&w.to_lowercase()))
            .collect();
        if found.is_empty() {
            FeedbackScore::new("toxicity", 1.0)
                .with_details("no toxic words detected")
        } else {
            FeedbackScore::new("toxicity", 0.0).with_details(&format!(
                "blocked words found: {}",
                found.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// OnlineEvaluator
// ---------------------------------------------------------------------------

/// Orchestrates feedback hooks, sampling, alerting, and score collection.
#[cfg(feature = "eval")]
pub struct OnlineEvaluator {
    hooks: Vec<Box<dyn FeedbackHook>>,
    alerts: Vec<AlertConfig>,
    sampling: EvalSamplingConfig,
    scores: Arc<Mutex<Vec<FeedbackScore>>>,
    alert_log: Arc<Mutex<Vec<AlertEvent>>>,
    alert_callback: Option<Box<dyn Fn(&AlertEvent) + Send + Sync>>,
    consecutive_failures: HashMap<String, usize>,
    sample_count: usize,
    /// Simple pseudo-random state for sampling decisions.
    rng_state: u64,
}

#[cfg(feature = "eval")]
impl fmt::Debug for OnlineEvaluator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnlineEvaluator")
            .field("hooks_count", &self.hooks.len())
            .field("alerts", &self.alerts)
            .field("sampling", &self.sampling)
            .field("sample_count", &self.sample_count)
            .finish()
    }
}

#[cfg(feature = "eval")]
impl OnlineEvaluator {
    /// Create a new evaluator with 100% sampling (evaluate everything).
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            alerts: Vec::new(),
            sampling: EvalSamplingConfig::default(),
            scores: Arc::new(Mutex::new(Vec::new())),
            alert_log: Arc::new(Mutex::new(Vec::new())),
            alert_callback: None,
            consecutive_failures: HashMap::new(),
            sample_count: 0,
            rng_state: now_unix_secs().wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    /// Set a custom sampling configuration.
    pub fn with_sampling(mut self, config: EvalSamplingConfig) -> Self {
        self.sampling = config;
        self
    }

    /// Add a feedback hook.
    pub fn add_hook(&mut self, hook: Box<dyn FeedbackHook>) {
        self.hooks.push(hook);
    }

    /// Add an alert configuration.
    pub fn add_alert(&mut self, config: AlertConfig) {
        self.alerts.push(config);
    }

    /// Register a callback that fires when an alert threshold is breached.
    pub fn on_alert<F>(&mut self, callback: F)
    where
        F: Fn(&AlertEvent) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Box::new(callback));
    }

    /// Evaluate an input/output pair through all registered hooks.
    ///
    /// Returns the collected scores, or an empty vec if sampling skips this call.
    pub fn evaluate(
        &mut self,
        input: &str,
        output: &str,
        context: &EvalContext,
    ) -> Vec<FeedbackScore> {
        // Sampling gate
        if !self.should_sample() {
            return Vec::new();
        }
        self.sample_count += 1;

        let mut results = Vec::with_capacity(self.hooks.len());
        for hook in &self.hooks {
            let score = hook.evaluate(input, output, context);
            results.push(score);
        }

        // Store scores
        if let Ok(mut stored) = self.scores.lock() {
            stored.extend(results.clone());
        }

        // Check alerts
        self.check_alerts(&results);

        results
    }

    /// Return all collected scores.
    pub fn get_scores(&self) -> Vec<FeedbackScore> {
        self.scores
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Compute the average score for a given hook.
    pub fn get_average_score(&self, hook_name: &str) -> Option<f64> {
        let scores = self.scores.lock().ok()?;
        let matching: Vec<f64> = scores
            .iter()
            .filter(|s| s.hook_name == hook_name)
            .map(|s| s.score)
            .collect();
        if matching.is_empty() {
            None
        } else {
            Some(matching.iter().sum::<f64>() / matching.len() as f64)
        }
    }

    /// Clear all collected scores and reset consecutive failure counters.
    pub fn reset_scores(&mut self) {
        if let Ok(mut s) = self.scores.lock() {
            s.clear();
        }
        self.consecutive_failures.clear();
        self.sample_count = 0;
    }

    /// Return the number of evaluations performed.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Return all fired alert events.
    pub fn get_alert_log(&self) -> Vec<AlertEvent> {
        self.alert_log
            .lock()
            .map(|l| l.clone())
            .unwrap_or_default()
    }

    /// Return number of registered hooks.
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    // -- private helpers ---------------------------------------------------

    /// Determine if this invocation should be sampled.
    fn should_sample(&mut self) -> bool {
        // Check max_samples_per_hour ceiling first (applies regardless of sample_rate)
        if self.sample_count >= self.sampling.max_samples_per_hour {
            return false;
        }
        if self.sampling.sample_rate >= 1.0 {
            return true;
        }
        if self.sampling.sample_rate <= 0.0 {
            return false;
        }
        // Simple xorshift-based PRNG for sampling decision
        let rand_val = self.next_rand_f64();
        rand_val < self.sampling.sample_rate
    }

    /// Advance the internal PRNG and return a f64 in [0, 1).
    fn next_rand_f64(&mut self) -> f64 {
        // Xorshift64
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    /// Check alert thresholds against new scores and fire callbacks.
    fn check_alerts(&mut self, scores: &[FeedbackScore]) {
        for score in scores {
            // Find matching alert configs
            let matching_alerts: Vec<AlertConfig> = self
                .alerts
                .iter()
                .filter(|a| a.hook_name == score.hook_name)
                .cloned()
                .collect();

            for alert_cfg in &matching_alerts {
                let counter = self
                    .consecutive_failures
                    .entry(alert_cfg.hook_name.clone())
                    .or_insert(0);

                if score.score < alert_cfg.threshold {
                    *counter += 1;
                } else {
                    *counter = 0;
                }

                if *counter >= alert_cfg.consecutive_failures {
                    let event = AlertEvent {
                        hook_name: alert_cfg.hook_name.clone(),
                        score: score.score,
                        threshold: alert_cfg.threshold,
                        consecutive_count: *counter,
                        timestamp: now_unix_secs(),
                        message: format!(
                            "Hook '{}' scored {:.4} (threshold {:.4}) for {} consecutive evaluations",
                            alert_cfg.hook_name, score.score, alert_cfg.threshold, *counter
                        ),
                    };
                    if let Ok(mut log) = self.alert_log.lock() {
                        log.push(event.clone());
                    }
                    if let Some(ref cb) = self.alert_callback {
                        cb(&event);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "eval")]
impl Default for OnlineEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg(feature = "eval")]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- FeedbackScore tests -----------------------------------------------

    #[test]
    fn test_feedback_score_creation() {
        let score = FeedbackScore::new("test_hook", 0.85);
        assert_eq!(score.hook_name, "test_hook");
        assert!((score.score - 0.85).abs() < f64::EPSILON);
        assert!(score.details.is_none());
        assert!(score.timestamp > 0);
    }

    #[test]
    fn test_feedback_score_with_details() {
        let score = FeedbackScore::new("hook", 0.5).with_details("some detail");
        assert_eq!(score.details.as_deref(), Some("some detail"));
    }

    #[test]
    fn test_feedback_score_clamped_high() {
        let score = FeedbackScore::new("hook", 1.5);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_score_clamped_low() {
        let score = FeedbackScore::new("hook", -0.3);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    // -- EvalContext tests -------------------------------------------------

    #[test]
    fn test_eval_context_default() {
        let ctx = EvalContext::default();
        assert!(ctx.agent_id.is_none());
        assert!(ctx.task_type.is_none());
        assert!(ctx.metadata.is_empty());
        assert!(ctx.latency_ms.is_none());
        assert!(ctx.token_count.is_none());
    }

    #[test]
    fn test_eval_context_builder() {
        let ctx = EvalContext::new()
            .with_agent_id("agent-1")
            .with_task_type("summarize")
            .with_latency_ms(120)
            .with_token_count(500)
            .with_metadata("model", "gpt-4");
        assert_eq!(ctx.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(ctx.task_type.as_deref(), Some("summarize"));
        assert_eq!(ctx.latency_ms, Some(120));
        assert_eq!(ctx.token_count, Some(500));
        assert_eq!(ctx.metadata.get("model").map(|s| s.as_str()), Some("gpt-4"));
    }

    // -- LatencyHook tests -------------------------------------------------

    #[test]
    fn test_latency_hook_fast() {
        let hook = LatencyHook::new(200);
        let ctx = EvalContext::new().with_latency_ms(50);
        let score = hook.evaluate("hi", "hello", &ctx);
        assert_eq!(score.hook_name, "latency");
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_hook_slow() {
        let hook = LatencyHook::new(100);
        let ctx = EvalContext::new().with_latency_ms(400);
        let score = hook.evaluate("hi", "hello", &ctx);
        // 100/400 = 0.25
        assert!((score.score - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_latency_hook_boundary() {
        let hook = LatencyHook::new(100);
        let ctx = EvalContext::new().with_latency_ms(100);
        let score = hook.evaluate("hi", "hello", &ctx);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_hook_no_latency() {
        let hook = LatencyHook::new(100);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "hello", &ctx);
        // latency defaults to 0 => perfect score
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_hook_double() {
        let hook = LatencyHook::new(100);
        let ctx = EvalContext::new().with_latency_ms(200);
        let score = hook.evaluate("hi", "hello", &ctx);
        // 100/200 = 0.5
        assert!((score.score - 0.5).abs() < 0.01);
    }

    // -- CostHook tests ----------------------------------------------------

    #[test]
    fn test_cost_hook_cheap() {
        let hook = CostHook::new(0.001, 1.0);
        let ctx = EvalContext::new().with_token_count(100);
        let score = hook.evaluate("hi", "hello", &ctx);
        // cost = 0.1, max = 1.0 => score = 0.9
        assert!((score.score - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_cost_hook_expensive() {
        let hook = CostHook::new(0.01, 1.0);
        let ctx = EvalContext::new().with_token_count(200);
        let score = hook.evaluate("hi", "hello", &ctx);
        // cost = 2.0 >= max 1.0 => score = 0.0
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_hook_zero_tokens() {
        let hook = CostHook::new(0.01, 1.0);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "hello", &ctx);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_hook_zero_max() {
        let hook = CostHook::new(0.01, 0.0);
        let ctx = EvalContext::new().with_token_count(10);
        let score = hook.evaluate("hi", "hello", &ctx);
        // max_cost=0 => score 1.0 (guard)
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    // -- RelevanceHook tests -----------------------------------------------

    #[test]
    fn test_relevance_hook_relevant() {
        let hook = RelevanceHook::new();
        let ctx = EvalContext::default();
        let score = hook.evaluate(
            "What is machine learning and deep learning?",
            "Machine learning is a subset of AI. Deep learning uses neural networks for machine learning tasks.",
            &ctx,
        );
        // Shared terms should yield high similarity
        assert!(score.score > 0.3, "Expected relevant score > 0.3, got {}", score.score);
    }

    #[test]
    fn test_relevance_hook_irrelevant() {
        let hook = RelevanceHook::new();
        let ctx = EvalContext::default();
        let score = hook.evaluate(
            "What is quantum computing?",
            "The recipe for chocolate cake requires flour, sugar, and cocoa powder.",
            &ctx,
        );
        assert!(score.score < 0.3, "Expected irrelevant score < 0.3, got {}", score.score);
    }

    #[test]
    fn test_relevance_hook_identical() {
        let hook = RelevanceHook::new();
        let ctx = EvalContext::default();
        let score = hook.evaluate("hello world test", "hello world test", &ctx);
        assert!((score.score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_relevance_hook_empty_input() {
        let hook = RelevanceHook::new();
        let ctx = EvalContext::default();
        let score = hook.evaluate("", "some output", &ctx);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_relevance_hook_empty_output() {
        let hook = RelevanceHook::new();
        let ctx = EvalContext::default();
        let score = hook.evaluate("some input", "", &ctx);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    // -- ToxicityHook tests ------------------------------------------------

    #[test]
    fn test_toxicity_hook_clean() {
        let hook = ToxicityHook::new(vec!["badword".to_string(), "terrible".to_string()]);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "This is a perfectly fine response.", &ctx);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_toxicity_hook_toxic() {
        let hook = ToxicityHook::new(vec!["badword".to_string(), "terrible".to_string()]);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "This contains a badword in it.", &ctx);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_toxicity_hook_case_insensitive() {
        let hook = ToxicityHook::new(vec!["BadWord".to_string()]);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "There is a BADWORD here.", &ctx);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_toxicity_hook_empty_blocked() {
        let hook = ToxicityHook::new(vec![]);
        let ctx = EvalContext::default();
        let score = hook.evaluate("hi", "Anything goes here.", &ctx);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    // -- OnlineEvaluator tests ---------------------------------------------

    #[test]
    fn test_evaluator_basic() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::new().with_latency_ms(100);
        let scores = eval.evaluate("hi", "hello", &ctx);
        assert_eq!(scores.len(), 1);
        assert!((scores[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluator_all_hooks() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(200)));
        eval.add_hook(Box::new(CostHook::new(0.001, 1.0)));
        eval.add_hook(Box::new(RelevanceHook::new()));
        eval.add_hook(Box::new(ToxicityHook::new(vec!["bad".to_string()])));
        assert_eq!(eval.hook_count(), 4);

        let ctx = EvalContext::new().with_latency_ms(100).with_token_count(50);
        let scores = eval.evaluate("machine learning", "machine learning is great", &ctx);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_evaluator_sampling_rate_zero() {
        let mut eval = OnlineEvaluator::new()
            .with_sampling(EvalSamplingConfig::new(0.0));
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::default();
        // With 0% sampling, nothing should be evaluated
        for _ in 0..10 {
            let scores = eval.evaluate("hi", "hello", &ctx);
            assert!(scores.is_empty());
        }
        assert_eq!(eval.sample_count(), 0);
    }

    #[test]
    fn test_evaluator_sampling_rate_full() {
        let mut eval = OnlineEvaluator::new()
            .with_sampling(EvalSamplingConfig::new(1.0));
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::new().with_latency_ms(50);
        for _ in 0..5 {
            let scores = eval.evaluate("hi", "hello", &ctx);
            assert_eq!(scores.len(), 1);
        }
        assert_eq!(eval.sample_count(), 5);
    }

    #[test]
    fn test_evaluator_sampling_rate_partial() {
        let mut eval = OnlineEvaluator::new()
            .with_sampling(EvalSamplingConfig::new(0.5));
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::new().with_latency_ms(50);
        let mut sampled = 0;
        for _ in 0..100 {
            let scores = eval.evaluate("hi", "hello", &ctx);
            if !scores.is_empty() {
                sampled += 1;
            }
        }
        // With 50% sampling, expect roughly 30-70 out of 100
        assert!(sampled > 10, "Expected at least some samples, got {}", sampled);
        assert!(sampled < 90, "Expected some skips, got {}", sampled);
    }

    #[test]
    fn test_evaluator_alert_fires() {
        let alert_count = Arc::new(AtomicUsize::new(0));
        let alert_count_clone = alert_count.clone();

        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(100)));
        eval.add_alert(AlertConfig::new("latency", 0.5, 1));
        eval.on_alert(move |_event| {
            alert_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Trigger a low-score evaluation (latency 400ms, max 100ms => score 0.25)
        let ctx = EvalContext::new().with_latency_ms(400);
        eval.evaluate("hi", "hello", &ctx);

        assert!(alert_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_evaluator_consecutive_failures() {
        let alert_count = Arc::new(AtomicUsize::new(0));
        let alert_count_clone = alert_count.clone();

        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(100)));
        eval.add_alert(AlertConfig::new("latency", 0.5, 3));
        eval.on_alert(move |_event| {
            alert_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let slow_ctx = EvalContext::new().with_latency_ms(400);

        // First two failures: no alert yet
        eval.evaluate("hi", "hello", &slow_ctx);
        eval.evaluate("hi", "hello", &slow_ctx);
        assert_eq!(alert_count.load(Ordering::SeqCst), 0);

        // Third failure: alert fires
        eval.evaluate("hi", "hello", &slow_ctx);
        assert!(alert_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_evaluator_consecutive_reset_on_good_score() {
        let alert_count = Arc::new(AtomicUsize::new(0));
        let alert_count_clone = alert_count.clone();

        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(100)));
        eval.add_alert(AlertConfig::new("latency", 0.5, 3));
        eval.on_alert(move |_event| {
            alert_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let slow_ctx = EvalContext::new().with_latency_ms(400);
        let fast_ctx = EvalContext::new().with_latency_ms(50);

        // Two failures
        eval.evaluate("hi", "hello", &slow_ctx);
        eval.evaluate("hi", "hello", &slow_ctx);
        // One good score resets counter
        eval.evaluate("hi", "hello", &fast_ctx);
        // Two more failures — still below threshold of 3
        eval.evaluate("hi", "hello", &slow_ctx);
        eval.evaluate("hi", "hello", &slow_ctx);
        assert_eq!(alert_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_evaluator_get_average_score() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(100)));

        let ctx1 = EvalContext::new().with_latency_ms(50);
        let ctx2 = EvalContext::new().with_latency_ms(200);

        eval.evaluate("a", "b", &ctx1); // score 1.0
        eval.evaluate("a", "b", &ctx2); // score 0.5

        let avg = eval.get_average_score("latency");
        assert!(avg.is_some());
        assert!((avg.unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_get_average_score_missing_hook() {
        let eval = OnlineEvaluator::new();
        assert!(eval.get_average_score("nonexistent").is_none());
    }

    #[test]
    fn test_evaluator_reset() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::new().with_latency_ms(100);
        eval.evaluate("hi", "hello", &ctx);
        assert!(!eval.get_scores().is_empty());
        assert_eq!(eval.sample_count(), 1);

        eval.reset_scores();
        assert!(eval.get_scores().is_empty());
        assert_eq!(eval.sample_count(), 0);
    }

    #[test]
    fn test_evaluator_get_scores() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(200)));
        eval.add_hook(Box::new(ToxicityHook::new(vec![])));

        let ctx = EvalContext::new().with_latency_ms(50);
        eval.evaluate("hi", "hello", &ctx);
        eval.evaluate("hi", "hello", &ctx);

        let all_scores = eval.get_scores();
        assert_eq!(all_scores.len(), 4); // 2 hooks × 2 evaluations
    }

    #[test]
    fn test_evaluator_alert_log() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(LatencyHook::new(100)));
        eval.add_alert(AlertConfig::new("latency", 0.5, 1));

        let ctx = EvalContext::new().with_latency_ms(400);
        eval.evaluate("hi", "hello", &ctx);

        let log = eval.get_alert_log();
        assert!(!log.is_empty());
        assert_eq!(log[0].hook_name, "latency");
    }

    #[test]
    fn test_evaluator_no_hooks() {
        let mut eval = OnlineEvaluator::new();
        let ctx = EvalContext::default();
        let scores = eval.evaluate("hi", "hello", &ctx);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_evaluator_default() {
        let eval = OnlineEvaluator::default();
        assert_eq!(eval.hook_count(), 0);
        assert_eq!(eval.sample_count(), 0);
    }

    // -- ExecutionFingerprint tests ----------------------------------------

    #[test]
    fn test_execution_fingerprint_unique() {
        let fp1 = ExecutionFingerprint::new("agent-1", "task A");
        let fp2 = ExecutionFingerprint::new("agent-1", "task B");
        assert_ne!(fp1.task_hash, fp2.task_hash);
    }

    #[test]
    fn test_execution_fingerprint_fields() {
        let fp = ExecutionFingerprint::new("agent-x", "summarize document");
        assert_eq!(fp.agent_id, "agent-x");
        assert!(!fp.task_hash.is_empty());
        assert!(!fp.random_id.is_empty());
        assert!(fp.timestamp > 0);
    }

    #[test]
    fn test_execution_fingerprint_deterministic_hash() {
        let fp1 = ExecutionFingerprint::new("a", "same task");
        let fp2 = ExecutionFingerprint::new("b", "same task");
        // Same task => same hash
        assert_eq!(fp1.task_hash, fp2.task_hash);
    }

    #[test]
    fn test_execution_fingerprint_different_random_ids() {
        // Two fingerprints created at the same second may still differ because
        // the random_id mixes in the hash. Different agents or tasks yield different ids.
        let fp1 = ExecutionFingerprint::new("agent-1", "task X");
        let fp2 = ExecutionFingerprint::new("agent-1", "task Y");
        assert_ne!(fp1.random_id, fp2.random_id);
    }

    // -- AlertConfig tests -------------------------------------------------

    #[test]
    fn test_alert_config_creation() {
        let cfg = AlertConfig::new("latency", 0.5, 3);
        assert_eq!(cfg.hook_name, "latency");
        assert!((cfg.threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.consecutive_failures, 3);
    }

    // -- AlertEvent tests --------------------------------------------------

    #[test]
    fn test_alert_event_serialization() {
        let event = AlertEvent {
            hook_name: "latency".to_string(),
            score: 0.2,
            threshold: 0.5,
            consecutive_count: 3,
            timestamp: 1000,
            message: "test alert".to_string(),
        };
        let json = serde_json::to_string(&event);
        assert!(json.is_ok());
        let deserialized: Result<AlertEvent, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
        let de = deserialized.unwrap();
        assert_eq!(de.hook_name, "latency");
        assert_eq!(de.consecutive_count, 3);
    }

    #[test]
    fn test_feedback_score_serialization() {
        let score = FeedbackScore::new("test", 0.75).with_details("info");
        let json = serde_json::to_string(&score);
        assert!(json.is_ok());
        let de: FeedbackScore = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(de.hook_name, "test");
        assert!((de.score - 0.75).abs() < f64::EPSILON);
    }

    // -- EvalSamplingConfig tests ------------------------------------------

    #[test]
    fn test_sampling_config_defaults() {
        let cfg = EvalSamplingConfig::default();
        assert!((cfg.sample_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.min_samples_per_hour, 0);
        assert_eq!(cfg.max_samples_per_hour, usize::MAX);
    }

    #[test]
    fn test_sampling_config_custom() {
        let cfg = EvalSamplingConfig::new(0.5).with_hourly_bounds(10, 100);
        assert!((cfg.sample_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.min_samples_per_hour, 10);
        assert_eq!(cfg.max_samples_per_hour, 100);
    }

    #[test]
    fn test_sampling_config_clamped() {
        let cfg = EvalSamplingConfig::new(2.0);
        assert!((cfg.sample_rate - 1.0).abs() < f64::EPSILON);
        let cfg2 = EvalSamplingConfig::new(-0.5);
        assert!((cfg2.sample_rate - 0.0).abs() < f64::EPSILON);
    }

    // -- Max samples per hour test -----------------------------------------

    #[test]
    fn test_evaluator_max_samples_per_hour() {
        let mut eval = OnlineEvaluator::new()
            .with_sampling(EvalSamplingConfig {
                sample_rate: 1.0,
                min_samples_per_hour: 0,
                max_samples_per_hour: 3,
            });
        eval.add_hook(Box::new(LatencyHook::new(200)));
        let ctx = EvalContext::new().with_latency_ms(50);

        for _ in 0..10 {
            eval.evaluate("hi", "hello", &ctx);
        }
        // Only 3 should have been sampled
        assert_eq!(eval.sample_count(), 3);
    }

    // -- Custom hook test --------------------------------------------------

    struct CustomHook;
    impl FeedbackHook for CustomHook {
        fn name(&self) -> &str {
            "custom"
        }
        fn evaluate(&self, _input: &str, _output: &str, _context: &EvalContext) -> FeedbackScore {
            FeedbackScore::new("custom", 0.42)
        }
    }

    #[test]
    fn test_custom_hook() {
        let mut eval = OnlineEvaluator::new();
        eval.add_hook(Box::new(CustomHook));
        let ctx = EvalContext::default();
        let scores = eval.evaluate("hi", "hello", &ctx);
        assert_eq!(scores.len(), 1);
        assert!((scores[0].score - 0.42).abs() < f64::EPSILON);
    }

    // -- RelevanceHook internal tests --------------------------------------

    #[test]
    fn test_relevance_tokenize() {
        let tokens = RelevanceHook::tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char words filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_relevance_tfidf_empty() {
        let sim = RelevanceHook::tfidf_cosine("", "");
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }
}
