//! Unified Guardrails Pipeline
//!
//! Orchestrates multiple guardrail components through a single pipeline interface.
//! Each component implements the [`Guard`] trait and is run at the appropriate stage
//! (pre-send, post-receive, or both). The pipeline collects results, applies a
//! configurable block threshold, and optionally logs violations.
//!
//! # Built-in guards
//!
//! - [`ContentLengthGuard`] — enforces a maximum character limit
//! - [`RateLimitGuard`] — sliding-window rate limiting
//! - [`PatternGuard`] — substring blocklist matching
//! - [`ToxicityGuard`] — wraps [`crate::advanced_guardrails::ToxicityDetector`]
//! - [`PiiGuard`] — wraps [`crate::pii_detection::PiiDetector`]
//! - [`AttackGuard`] — wraps [`crate::advanced_guardrails::AttackDetector`]

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::advanced_guardrails::{AttackDetector, ToxicityDetector};
use crate::pii_detection::PiiDetector;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Stage where a guard runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardStage {
    /// Before sending input to the model.
    PreSend,
    /// After receiving output from the model.
    PostReceive,
    /// Both pre-send and post-receive.
    Both,
}

/// Action to take when a guard triggers.
#[derive(Debug, Clone)]
pub enum GuardAction {
    /// Content is acceptable.
    Pass,
    /// Content is suspicious but not blocked.
    Warn(String),
    /// Content must be blocked.
    Block(String),
}

/// Result of a single guard check.
#[derive(Debug, Clone)]
pub struct GuardCheckResult {
    /// Name of the guard that produced this result.
    pub guard_name: String,
    /// The action recommended by the guard.
    pub action: GuardAction,
    /// Severity score in the range 0.0 to 1.0.
    pub score: f64,
    /// Human-readable details about the check.
    pub details: String,
}

/// Unified guard trait. All guardrail components implement this.
pub trait Guard: Send + Sync {
    /// A short, unique name for this guard.
    fn name(&self) -> &str;
    /// The pipeline stage(s) where this guard should run.
    fn stage(&self) -> GuardStage;
    /// Run the guard against `text` and return a result.
    fn check(&self, text: &str) -> GuardCheckResult;
}

/// Aggregated result from running all applicable guards in a pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    /// Whether all guards passed (no score >= block threshold).
    pub passed: bool,
    /// Individual results from every guard that ran.
    pub results: Vec<GuardCheckResult>,
    /// If blocked, the name of the first guard that caused the block.
    pub blocked_by: Option<String>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Main guardrail pipeline orchestrator.
///
/// Holds an ordered list of [`Guard`] trait objects and runs the appropriate
/// subset for input (pre-send) or output (post-receive) checks.
pub struct GuardrailPipeline {
    guards: Vec<Box<dyn Guard>>,
    block_threshold: f64,
    log_violations: bool,
    violations: Vec<GuardCheckResult>,
}

impl GuardrailPipeline {
    /// Create a new pipeline with default settings.
    ///
    /// Defaults: `block_threshold = 0.8`, `log_violations = false`.
    pub fn new() -> Self {
        Self {
            guards: Vec::new(),
            block_threshold: 0.8,
            log_violations: false,
            violations: Vec::new(),
        }
    }

    /// Set the score threshold at or above which a guard result triggers a block.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.block_threshold = threshold;
        self
    }

    /// Enable or disable violation logging.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.log_violations = enabled;
        self
    }

    /// Add a guard to the pipeline.
    pub fn add_guard(&mut self, guard: Box<dyn Guard>) {
        self.guards.push(guard);
    }

    /// Run all guards whose stage is [`GuardStage::PreSend`] or [`GuardStage::Both`].
    pub fn check_input(&mut self, text: &str) -> PipelineResult {
        self.run_stage(text, |stage| stage == GuardStage::PreSend || stage == GuardStage::Both)
    }

    /// Run all guards whose stage is [`GuardStage::PostReceive`] or [`GuardStage::Both`].
    pub fn check_output(&mut self, text: &str) -> PipelineResult {
        self.run_stage(text, |stage| stage == GuardStage::PostReceive || stage == GuardStage::Both)
    }

    /// Return all recorded violations (requires logging to be enabled).
    pub fn violations(&self) -> &[GuardCheckResult] {
        &self.violations
    }

    /// Clear the recorded violations log.
    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }

    /// Return the number of guards registered in the pipeline.
    pub fn guard_count(&self) -> usize {
        self.guards.len()
    }

    // Internal: run every guard whose stage satisfies the predicate.
    fn run_stage<F>(&mut self, text: &str, stage_filter: F) -> PipelineResult
    where
        F: Fn(GuardStage) -> bool,
    {
        let mut results = Vec::new();
        let mut blocked_by: Option<String> = None;

        for guard in &self.guards {
            if !stage_filter(guard.stage()) {
                continue;
            }

            let result = guard.check(text);

            // Record violation if logging is enabled and score meets threshold.
            if self.log_violations && result.score >= self.block_threshold {
                self.violations.push(result.clone());
            }

            // Track the first guard that triggers a block.
            if result.score >= self.block_threshold && blocked_by.is_none() {
                blocked_by = Some(result.guard_name.clone());
            }

            results.push(result);
        }

        let passed = blocked_by.is_none();

        PipelineResult {
            passed,
            results,
            blocked_by,
        }
    }
}

impl Default for GuardrailPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in guards
// ---------------------------------------------------------------------------

/// Blocks text that exceeds a maximum character count.
pub struct ContentLengthGuard {
    max_chars: usize,
}

impl ContentLengthGuard {
    /// Create a new guard with the given character limit.
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl Guard for ContentLengthGuard {
    fn name(&self) -> &str {
        "content_length"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::Both
    }

    fn check(&self, text: &str) -> GuardCheckResult {
        let len = text.len();
        if len > self.max_chars {
            let ratio = (len as f64 / self.max_chars as f64).min(2.0) / 2.0;
            GuardCheckResult {
                guard_name: self.name().to_string(),
                action: GuardAction::Block(format!(
                    "Content length {} exceeds limit {}",
                    len, self.max_chars
                )),
                score: 0.5 + ratio * 0.5, // 0.5 – 1.0 depending on how far over
                details: format!("len={}, max={}", len, self.max_chars),
            }
        } else {
            GuardCheckResult {
                guard_name: self.name().to_string(),
                action: GuardAction::Pass,
                score: 0.0,
                details: format!("len={}, max={}", len, self.max_chars),
            }
        }
    }
}

/// Sliding-window rate limiter. Uses interior mutability to track timestamps.
pub struct RateLimitGuard {
    max_requests: usize,
    window_secs: u64,
    timestamps: Mutex<VecDeque<Instant>>,
}

impl RateLimitGuard {
    /// Create a new rate limit guard.
    ///
    /// * `max_requests` — maximum number of requests allowed in the window.
    /// * `window_secs` — size of the sliding window in seconds.
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            timestamps: Mutex::new(VecDeque::new()),
        }
    }
}

impl Guard for RateLimitGuard {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::PreSend
    }

    fn check(&self, _text: &str) -> GuardCheckResult {
        let mut ts = self.timestamps.lock().expect("lock poisoned");
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.window_secs);

        // Remove timestamps outside the window.
        while let Some(front) = ts.front() {
            if now.duration_since(*front) > window {
                ts.pop_front();
            } else {
                break;
            }
        }

        // Record current request.
        ts.push_back(now);

        let count = ts.len();
        if count > self.max_requests {
            GuardCheckResult {
                guard_name: self.name().to_string(),
                action: GuardAction::Block(format!(
                    "Rate limit exceeded: {} requests in {}s window (max {})",
                    count, self.window_secs, self.max_requests
                )),
                score: 1.0,
                details: format!(
                    "count={}, max={}, window={}s",
                    count, self.max_requests, self.window_secs
                ),
            }
        } else {
            GuardCheckResult {
                guard_name: self.name().to_string(),
                action: GuardAction::Pass,
                score: 0.0,
                details: format!(
                    "count={}, max={}, window={}s",
                    count, self.max_requests, self.window_secs
                ),
            }
        }
    }
}

/// Blocks text containing any of a set of forbidden substrings (case-insensitive).
pub struct PatternGuard {
    patterns: Vec<String>,
}

impl PatternGuard {
    /// Create a new pattern guard from a list of blocked substrings.
    pub fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }
}

impl Guard for PatternGuard {
    fn name(&self) -> &str {
        "pattern"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::Both
    }

    fn check(&self, text: &str) -> GuardCheckResult {
        let lower = text.to_lowercase();
        for pattern in &self.patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return GuardCheckResult {
                    guard_name: self.name().to_string(),
                    action: GuardAction::Block(format!("Blocked pattern matched: {}", pattern)),
                    score: 1.0,
                    details: format!("matched pattern: {}", pattern),
                };
            }
        }
        GuardCheckResult {
            guard_name: self.name().to_string(),
            action: GuardAction::Pass,
            score: 0.0,
            details: "no patterns matched".to_string(),
        }
    }
}

/// Wraps [`ToxicityDetector`] as a post-receive guard.
pub struct ToxicityGuard {
    detector: ToxicityDetector,
}

impl ToxicityGuard {
    /// Create a new toxicity guard with default configuration.
    pub fn new() -> Self {
        Self {
            detector: ToxicityDetector::default(),
        }
    }

    /// Create a toxicity guard wrapping the given detector.
    pub fn with_detector(detector: ToxicityDetector) -> Self {
        Self { detector }
    }
}

impl Default for ToxicityGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Guard for ToxicityGuard {
    fn name(&self) -> &str {
        "toxicity"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::PostReceive
    }

    fn check(&self, text: &str) -> GuardCheckResult {
        let result = self.detector.detect(text);
        let score = result.overall_score as f64;
        let action = if result.is_toxic {
            GuardAction::Block("Toxic content detected".to_string())
        } else if score > 0.0 {
            GuardAction::Warn(format!("Toxicity score: {:.2}", score))
        } else {
            GuardAction::Pass
        };
        GuardCheckResult {
            guard_name: self.name().to_string(),
            action,
            score,
            details: format!(
                "toxic={}, score={:.2}, matches={}",
                result.is_toxic,
                result.overall_score,
                result.matches.len()
            ),
        }
    }
}

/// Wraps [`PiiDetector`] as a guard that runs on both stages.
pub struct PiiGuard {
    detector: PiiDetector,
}

impl PiiGuard {
    /// Create a new PII guard with default configuration.
    pub fn new() -> Self {
        Self {
            detector: PiiDetector::default(),
        }
    }

    /// Create a PII guard wrapping the given detector.
    pub fn with_detector(detector: PiiDetector) -> Self {
        Self { detector }
    }
}

impl Default for PiiGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Guard for PiiGuard {
    fn name(&self) -> &str {
        "pii"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::Both
    }

    fn check(&self, text: &str) -> GuardCheckResult {
        let result = self.detector.detect(text);
        let count = result.detections.len();
        let score = if count == 0 {
            0.0
        } else {
            // Average confidence of all detections, capped at 1.0
            let total: f64 = result.detections.iter().map(|d| d.confidence).sum();
            (total / count as f64).min(1.0)
        };
        let action = if result.has_pii {
            GuardAction::Block(format!("PII detected: {} item(s)", count))
        } else {
            GuardAction::Pass
        };
        GuardCheckResult {
            guard_name: self.name().to_string(),
            action,
            score,
            details: format!("pii_found={}, count={}", result.has_pii, count),
        }
    }
}

/// Wraps [`AttackDetector`] as a pre-send guard.
pub struct AttackGuard {
    detector: AttackDetector,
}

impl AttackGuard {
    /// Create a new attack guard with default configuration.
    pub fn new() -> Self {
        Self {
            detector: AttackDetector::default(),
        }
    }

    /// Create an attack guard wrapping the given detector.
    pub fn with_detector(detector: AttackDetector) -> Self {
        Self { detector }
    }
}

impl Default for AttackGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Guard for AttackGuard {
    fn name(&self) -> &str {
        "attack"
    }

    fn stage(&self) -> GuardStage {
        GuardStage::PreSend
    }

    fn check(&self, text: &str) -> GuardCheckResult {
        let result = self.detector.detect(text);
        let score = result.risk_score as f64;
        let action = if result.is_high_risk() {
            GuardAction::Block("High-risk attack pattern detected".to_string())
        } else if result.is_suspicious() {
            GuardAction::Warn(format!("Suspicious input (risk={:.2})", score))
        } else {
            GuardAction::Pass
        };
        GuardCheckResult {
            guard_name: self.name().to_string(),
            action,
            score,
            details: format!(
                "risk={:.2}, attacks={}",
                result.risk_score,
                result.detected_attacks.len()
            ),
        }
    }
}

// =============================================================================
// STREAMING GUARDRAILS (v4 - item 5.2)
// =============================================================================

/// Action for streaming guard evaluation.
#[derive(Debug, Clone)]
pub enum StreamGuardAction {
    /// Chunk is acceptable.
    Pass,
    /// Chunk is suspicious — flag for review but do not stop.
    Flag(String),
    /// Temporarily pause streaming for further analysis.
    Pause,
    /// Block the stream entirely.
    Block(String),
}

/// Trait for guards that operate on streaming content chunk-by-chunk.
pub trait StreamingGuard: Send + Sync {
    /// A short, unique name for this streaming guard.
    fn name(&self) -> &str;
    /// Evaluate the current `chunk` against the full `accumulated` text so far.
    fn check_chunk(&self, chunk: &str, accumulated: &str) -> StreamGuardAction;
    /// Minimum number of whitespace-delimited tokens to accumulate before evaluating.
    fn min_buffer_tokens(&self) -> usize {
        10
    }
}

/// Configuration for the streaming guardrail pipeline.
#[derive(Debug, Clone)]
pub struct StreamingGuardrailConfig {
    /// Minimum number of whitespace-delimited tokens before the first evaluation.
    pub min_buffer_size: usize,
    /// Number of new chunks between consecutive evaluations.
    pub eval_interval: usize,
    /// Maximum accumulated buffer size (in characters) before forcing a trim.
    pub max_buffer_size: usize,
}

impl Default for StreamingGuardrailConfig {
    fn default() -> Self {
        Self {
            min_buffer_size: 10,
            eval_interval: 5,
            max_buffer_size: 500,
        }
    }
}

/// Metrics collected during streaming guardrail evaluation.
#[derive(Debug, Clone, Default)]
pub struct StreamingGuardrailMetrics {
    /// Total chunks received so far.
    pub chunks_received: usize,
    /// Number of chunks that triggered a full evaluation pass.
    pub chunks_evaluated: usize,
    /// Number of Flag actions returned.
    pub flags: usize,
    /// Number of Block actions returned.
    pub blocks: usize,
    /// Number of Pause actions returned.
    pub pauses: usize,
    /// Cumulative evaluation time in milliseconds.
    pub total_eval_time_ms: u64,
}

/// Result from processing a single streaming chunk.
#[derive(Debug, Clone)]
pub struct StreamChunkResult {
    /// The worst (most severe) action across all guards for this chunk.
    pub action: StreamGuardAction,
    /// Per-guard results for this chunk (only populated when evaluation ran).
    pub guard_results: Vec<(String, StreamGuardAction)>,
    /// Whether a full evaluation pass actually ran for this chunk.
    pub was_evaluated: bool,
}

/// Pipeline that runs multiple [`StreamingGuard`]s over a stream of text chunks.
pub struct StreamingGuardrailPipeline {
    guards: Vec<Box<dyn StreamingGuard>>,
    config: StreamingGuardrailConfig,
    accumulated: String,
    tokens_since_eval: usize,
    metrics: StreamingGuardrailMetrics,
}

impl StreamingGuardrailPipeline {
    /// Create a new pipeline with default configuration and no guards.
    pub fn new() -> Self {
        Self {
            guards: Vec::new(),
            config: StreamingGuardrailConfig::default(),
            accumulated: String::new(),
            tokens_since_eval: 0,
            metrics: StreamingGuardrailMetrics::default(),
        }
    }

    /// Create a new pipeline with the given configuration.
    pub fn with_config(config: StreamingGuardrailConfig) -> Self {
        Self {
            guards: Vec::new(),
            config,
            accumulated: String::new(),
            tokens_since_eval: 0,
            metrics: StreamingGuardrailMetrics::default(),
        }
    }

    /// Add a streaming guard to the pipeline (builder pattern).
    pub fn add_guard(mut self, guard: Box<dyn StreamingGuard>) -> Self {
        self.guards.push(guard);
        self
    }

    /// Process a new chunk of streaming text.
    ///
    /// The chunk is appended to the internal buffer. If the buffer has reached
    /// the minimum token count **and** enough chunks have arrived since the last
    /// evaluation, all registered guards are run against the accumulated text.
    pub fn process_chunk(&mut self, chunk: &str) -> StreamChunkResult {
        self.accumulated.push_str(chunk);
        self.metrics.chunks_received += 1;
        self.tokens_since_eval += 1;

        // Trim to max_buffer_size (keep the tail).
        if self.accumulated.len() > self.config.max_buffer_size {
            let start = self.accumulated.len() - self.config.max_buffer_size;
            self.accumulated = self.accumulated[start..].to_string();
        }

        // Count whitespace-delimited tokens in accumulated buffer.
        let token_count = self.accumulated.split_whitespace().count();

        // Decide whether to evaluate.
        let should_eval =
            token_count >= self.config.min_buffer_size && self.tokens_since_eval >= self.config.eval_interval;

        if !should_eval {
            return StreamChunkResult {
                action: StreamGuardAction::Pass,
                guard_results: Vec::new(),
                was_evaluated: false,
            };
        }

        // Run all guards and collect results.
        self.tokens_since_eval = 0;
        self.metrics.chunks_evaluated += 1;

        let start_time = Instant::now();
        let mut guard_results: Vec<(String, StreamGuardAction)> = Vec::new();
        let mut worst = StreamGuardAction::Pass;

        for guard in &self.guards {
            let action = guard.check_chunk(chunk, &self.accumulated);
            worst = Self::worse_action(&worst, &action);
            guard_results.push((guard.name().to_string(), action));
        }

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.total_eval_time_ms += elapsed_ms;

        // Update metric counters based on worst action.
        match &worst {
            StreamGuardAction::Pass => {}
            StreamGuardAction::Flag(_) => self.metrics.flags += 1,
            StreamGuardAction::Pause => self.metrics.pauses += 1,
            StreamGuardAction::Block(_) => self.metrics.blocks += 1,
        }

        StreamChunkResult {
            action: worst,
            guard_results,
            was_evaluated: true,
        }
    }

    /// Reset the pipeline state (buffer, counters, metrics).
    pub fn reset(&mut self) {
        self.accumulated.clear();
        self.tokens_since_eval = 0;
        self.metrics = StreamingGuardrailMetrics::default();
    }

    /// Return a reference to the current metrics.
    pub fn get_metrics(&self) -> &StreamingGuardrailMetrics {
        &self.metrics
    }

    /// Return the accumulated text buffer.
    pub fn get_accumulated(&self) -> &str {
        &self.accumulated
    }

    /// Public test-accessible version of [`worse_action`].
    #[cfg(test)]
    pub(crate) fn worse_action_pub(a: &StreamGuardAction, b: &StreamGuardAction) -> StreamGuardAction {
        Self::worse_action(a, b)
    }

    /// Return the more severe of two [`StreamGuardAction`]s.
    ///
    /// Severity ordering: Pass < Pause < Flag < Block.
    fn worse_action(a: &StreamGuardAction, b: &StreamGuardAction) -> StreamGuardAction {
        fn severity(action: &StreamGuardAction) -> u8 {
            match action {
                StreamGuardAction::Pass => 0,
                StreamGuardAction::Pause => 1,
                StreamGuardAction::Flag(_) => 2,
                StreamGuardAction::Block(_) => 3,
            }
        }

        if severity(b) > severity(a) {
            b.clone()
        } else {
            a.clone()
        }
    }
}

impl Default for StreamingGuardrailPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in streaming guards
// ---------------------------------------------------------------------------

/// Streaming PII guard — detects email addresses, phone numbers, and SSN-like
/// patterns in the accumulated text using lightweight heuristics.
pub struct StreamingPiiGuard;

impl StreamingGuard for StreamingPiiGuard {
    fn name(&self) -> &str {
        "streaming_pii"
    }

    fn check_chunk(&self, _chunk: &str, accumulated: &str) -> StreamGuardAction {
        // Email heuristic: word@word.word
        if accumulated
            .split_whitespace()
            .any(|token| {
                let parts: Vec<&str> = token.split('@').collect();
                parts.len() == 2
                    && !parts[0].is_empty()
                    && parts[1].contains('.')
                    && parts[1].len() > 2
            })
        {
            return StreamGuardAction::Flag("Possible email address detected".to_string());
        }

        // Phone heuristic: sequences of 10+ digits (ignoring separators).
        let digits: String = accumulated
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        // Look for runs of 9+ consecutive digits in the digit-only string.
        if digits.len() >= 9 {
            // SSN pattern: exactly NNN-NN-NNNN
            if accumulated.contains(|c: char| c == '-') {
                for word in accumulated.split_whitespace() {
                    let trimmed = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '-');
                    if trimmed.len() == 11 {
                        let parts: Vec<&str> = trimmed.split('-').collect();
                        if parts.len() == 3
                            && parts[0].len() == 3
                            && parts[1].len() == 2
                            && parts[2].len() == 4
                            && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
                        {
                            return StreamGuardAction::Block(
                                "Possible SSN pattern detected".to_string(),
                            );
                        }
                    }
                }
            }

            // Generic phone heuristic: token with 10+ digits.
            for word in accumulated.split_whitespace() {
                let word_digits: usize = word.chars().filter(|c| c.is_ascii_digit()).count();
                if word_digits >= 10 {
                    return StreamGuardAction::Flag(
                        "Possible phone number detected".to_string(),
                    );
                }
            }
        }

        StreamGuardAction::Pass
    }
}

/// Streaming toxicity guard — flags content matching a configurable blocklist
/// of toxic words/phrases (case-insensitive).
pub struct StreamingToxicityGuard {
    /// List of blocked words/phrases (stored in lowercase).
    pub blocklist: Vec<String>,
}

impl StreamingToxicityGuard {
    /// Create a new toxicity guard with the given blocklist.
    pub fn new(blocklist: Vec<String>) -> Self {
        Self {
            blocklist: blocklist.into_iter().map(|w| w.to_lowercase()).collect(),
        }
    }

    /// Create a toxicity guard with a small default blocklist.
    pub fn with_defaults() -> Self {
        Self::new(vec![
            "hate".to_string(),
            "violence".to_string(),
            "kill".to_string(),
            "murder".to_string(),
            "terrorist".to_string(),
        ])
    }
}

impl StreamingGuard for StreamingToxicityGuard {
    fn name(&self) -> &str {
        "streaming_toxicity"
    }

    fn check_chunk(&self, _chunk: &str, accumulated: &str) -> StreamGuardAction {
        let lower = accumulated.to_lowercase();
        for word in &self.blocklist {
            if lower.contains(word.as_str()) {
                return StreamGuardAction::Block(format!("Toxic content matched: {}", word));
            }
        }
        StreamGuardAction::Pass
    }
}

/// Streaming pattern guard — blocks content that matches any of a set of
/// forbidden substrings (case-insensitive).
pub struct StreamingPatternGuard {
    /// Blocked patterns (stored in lowercase).
    pub blocked_patterns: Vec<String>,
}

impl StreamingPatternGuard {
    /// Create a new pattern guard from a list of blocked substrings.
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            blocked_patterns: patterns.into_iter().map(|p| p.to_lowercase()).collect(),
        }
    }
}

impl StreamingGuard for StreamingPatternGuard {
    fn name(&self) -> &str {
        "streaming_pattern"
    }

    fn check_chunk(&self, _chunk: &str, accumulated: &str) -> StreamGuardAction {
        let lower = accumulated.to_lowercase();
        for pattern in &self.blocked_patterns {
            if lower.contains(pattern.as_str()) {
                return StreamGuardAction::Block(format!("Blocked pattern matched: {}", pattern));
            }
        }
        StreamGuardAction::Pass
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    struct PassGuard;
    impl Guard for PassGuard {
        fn name(&self) -> &str {
            "pass"
        }
        fn stage(&self) -> GuardStage {
            GuardStage::Both
        }
        fn check(&self, _text: &str) -> GuardCheckResult {
            GuardCheckResult {
                guard_name: "pass".into(),
                action: GuardAction::Pass,
                score: 0.0,
                details: String::new(),
            }
        }
    }

    struct BlockGuard {
        score: f64,
    }
    impl Guard for BlockGuard {
        fn name(&self) -> &str {
            "block"
        }
        fn stage(&self) -> GuardStage {
            GuardStage::Both
        }
        fn check(&self, _text: &str) -> GuardCheckResult {
            GuardCheckResult {
                guard_name: "block".into(),
                action: GuardAction::Block("blocked".into()),
                score: self.score,
                details: "always blocks".into(),
            }
        }
    }

    struct PreSendOnlyGuard;
    impl Guard for PreSendOnlyGuard {
        fn name(&self) -> &str {
            "presend_only"
        }
        fn stage(&self) -> GuardStage {
            GuardStage::PreSend
        }
        fn check(&self, _text: &str) -> GuardCheckResult {
            GuardCheckResult {
                guard_name: "presend_only".into(),
                action: GuardAction::Block("presend block".into()),
                score: 1.0,
                details: "presend only guard".into(),
            }
        }
    }

    // -- tests --

    #[test]
    fn test_trait_object_safety() {
        // Ensure Guard can be used as a trait object.
        let guard: Box<dyn Guard> = Box::new(PassGuard);
        assert_eq!(guard.name(), "pass");
        let result = guard.check("hello");
        assert!(matches!(result.action, GuardAction::Pass));
    }

    #[test]
    fn test_empty_pipeline_passes() {
        let mut pipeline = GuardrailPipeline::new();
        let result = pipeline.check_input("anything");
        assert!(result.passed);
        assert!(result.results.is_empty());
        assert!(result.blocked_by.is_none());
    }

    #[test]
    fn test_pipeline_with_pass_guard() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add_guard(Box::new(PassGuard));
        let result = pipeline.check_input("hello");
        assert!(result.passed);
        assert_eq!(result.results.len(), 1);
        assert!(result.blocked_by.is_none());
    }

    #[test]
    fn test_pipeline_with_block_guard() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add_guard(Box::new(BlockGuard { score: 1.0 }));
        let result = pipeline.check_input("hello");
        assert!(!result.passed);
        assert_eq!(result.blocked_by, Some("block".to_string()));
    }

    #[test]
    fn test_pipeline_multiple_guards_ordering() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add_guard(Box::new(PassGuard));
        pipeline.add_guard(Box::new(BlockGuard { score: 1.0 }));
        let result = pipeline.check_input("hello");

        // Both guards should have run.
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].guard_name, "pass");
        assert_eq!(result.results[1].guard_name, "block");
        // Pipeline is blocked by the second guard.
        assert!(!result.passed);
        assert_eq!(result.blocked_by, Some("block".to_string()));
    }

    #[test]
    fn test_pipeline_threshold() {
        // Score below threshold should not trigger a block.
        let mut pipeline = GuardrailPipeline::new().with_threshold(0.9);
        pipeline.add_guard(Box::new(BlockGuard { score: 0.5 }));
        let result = pipeline.check_input("hello");
        assert!(result.passed);

        // Score at threshold should trigger a block.
        let mut pipeline2 = GuardrailPipeline::new().with_threshold(0.5);
        pipeline2.add_guard(Box::new(BlockGuard { score: 0.5 }));
        let result2 = pipeline2.check_input("hello");
        assert!(!result2.passed);
    }

    #[test]
    fn test_content_length_guard_pass() {
        let guard = ContentLengthGuard::new(100);
        let result = guard.check("short text");
        assert!(matches!(result.action, GuardAction::Pass));
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_content_length_guard_block() {
        let guard = ContentLengthGuard::new(5);
        let result = guard.check("this text is way too long");
        assert!(matches!(result.action, GuardAction::Block(_)));
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_rate_limit_guard_pass() {
        let guard = RateLimitGuard::new(5, 60);
        // First call should pass.
        let result = guard.check("request");
        assert!(matches!(result.action, GuardAction::Pass));
    }

    #[test]
    fn test_rate_limit_guard_block() {
        let guard = RateLimitGuard::new(2, 60);
        // Fire 3 requests — the third should exceed the limit.
        let _ = guard.check("request 1");
        let _ = guard.check("request 2");
        let result = guard.check("request 3");
        assert!(matches!(result.action, GuardAction::Block(_)));
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_pattern_guard_match() {
        let guard = PatternGuard::new(vec!["forbidden".to_string(), "banned".to_string()]);
        let result = guard.check("This contains a FORBIDDEN word");
        assert!(matches!(result.action, GuardAction::Block(_)));
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_pattern_guard_no_match() {
        let guard = PatternGuard::new(vec!["forbidden".to_string()]);
        let result = guard.check("This is perfectly fine");
        assert!(matches!(result.action, GuardAction::Pass));
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_violation_logging() {
        let mut pipeline = GuardrailPipeline::new()
            .with_threshold(0.8)
            .with_logging(true);
        pipeline.add_guard(Box::new(BlockGuard { score: 0.9 }));

        let _ = pipeline.check_input("test");
        assert_eq!(pipeline.violations().len(), 1);
        assert_eq!(pipeline.violations()[0].guard_name, "block");

        // Running again should accumulate.
        let _ = pipeline.check_output("test");
        assert_eq!(pipeline.violations().len(), 2);
    }

    #[test]
    fn test_clear_violations() {
        let mut pipeline = GuardrailPipeline::new()
            .with_threshold(0.8)
            .with_logging(true);
        pipeline.add_guard(Box::new(BlockGuard { score: 0.9 }));

        let _ = pipeline.check_input("test");
        assert!(!pipeline.violations().is_empty());

        pipeline.clear_violations();
        assert!(pipeline.violations().is_empty());
    }

    #[test]
    fn test_guard_stage_filtering() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add_guard(Box::new(PreSendOnlyGuard));

        // check_output should skip PreSend-only guards.
        let result = pipeline.check_output("hello");
        assert!(result.passed);
        assert!(result.results.is_empty());

        // check_input should include PreSend guards.
        let result = pipeline.check_input("hello");
        assert!(!result.passed);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].guard_name, "presend_only");
    }

    // ========================================================================
    // Streaming guardrail tests (v4 - item 5.2)
    // ========================================================================

    #[test]
    fn test_stream_guard_action_variants() {
        let pass = StreamGuardAction::Pass;
        assert!(matches!(pass, StreamGuardAction::Pass));

        let flag = StreamGuardAction::Flag("reason".into());
        assert!(matches!(flag, StreamGuardAction::Flag(_)));

        let pause = StreamGuardAction::Pause;
        assert!(matches!(pause, StreamGuardAction::Pause));

        let block = StreamGuardAction::Block("blocked".into());
        assert!(matches!(block, StreamGuardAction::Block(_)));

        // Verify Debug is implemented.
        let _ = format!("{:?}", StreamGuardAction::Pass);
        let _ = format!("{:?}", StreamGuardAction::Flag("x".into()));
        let _ = format!("{:?}", StreamGuardAction::Pause);
        let _ = format!("{:?}", StreamGuardAction::Block("x".into()));
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingGuardrailConfig::default();
        assert_eq!(config.min_buffer_size, 10);
        assert_eq!(config.eval_interval, 5);
        assert_eq!(config.max_buffer_size, 500);
    }

    #[test]
    fn test_streaming_pipeline_new() {
        let pipeline = StreamingGuardrailPipeline::new();
        assert!(pipeline.get_accumulated().is_empty());
        assert_eq!(pipeline.get_metrics().chunks_received, 0);
        assert_eq!(pipeline.get_metrics().chunks_evaluated, 0);
    }

    #[test]
    fn test_pipeline_with_config() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 3,
            eval_interval: 2,
            max_buffer_size: 100,
        };
        let pipeline = StreamingGuardrailPipeline::with_config(config);
        assert!(pipeline.get_accumulated().is_empty());
        assert_eq!(pipeline.get_metrics().chunks_received, 0);
    }

    #[test]
    fn test_pipeline_add_guard() {
        let pipeline = StreamingGuardrailPipeline::new()
            .add_guard(Box::new(StreamingPiiGuard))
            .add_guard(Box::new(StreamingToxicityGuard::with_defaults()));
        // Pipeline should have 2 guards — verify by processing a chunk and
        // checking the guard_results count when an evaluation happens.
        // (No public guard_count exposed on streaming pipeline — indirect test.)
        assert!(pipeline.get_accumulated().is_empty());
    }

    #[test]
    fn test_pipeline_process_chunk_not_enough_tokens() {
        // Default min_buffer_size=10, eval_interval=5.
        // A few short chunks should not trigger evaluation.
        let mut pipeline = StreamingGuardrailPipeline::new()
            .add_guard(Box::new(StreamingPiiGuard));
        let result = pipeline.process_chunk("hello ");
        assert!(!result.was_evaluated);
        assert!(matches!(result.action, StreamGuardAction::Pass));
        assert!(result.guard_results.is_empty());
        assert_eq!(pipeline.get_metrics().chunks_received, 1);
        assert_eq!(pipeline.get_metrics().chunks_evaluated, 0);
    }

    #[test]
    fn test_pipeline_process_chunk_evaluates() {
        // Use small config so evaluation triggers quickly.
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingPiiGuard));

        // First chunk won't have enough tokens yet (only 1 token).
        let r1 = pipeline.process_chunk("hello ");
        assert!(!r1.was_evaluated);

        // Second chunk gives 2 tokens and eval_interval=1 is met.
        let r2 = pipeline.process_chunk("world ");
        assert!(r2.was_evaluated);
        assert!(matches!(r2.action, StreamGuardAction::Pass));
        assert_eq!(r2.guard_results.len(), 1);
        assert_eq!(r2.guard_results[0].0, "streaming_pii");
    }

    #[test]
    fn test_pipeline_metrics_tracking() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingPiiGuard));

        pipeline.process_chunk("hello ");
        pipeline.process_chunk("world ");
        pipeline.process_chunk("foo ");

        let m = pipeline.get_metrics();
        assert_eq!(m.chunks_received, 3);
        // Second and third chunks should have triggered evaluation.
        assert!(m.chunks_evaluated >= 2);
    }

    #[test]
    fn test_pipeline_block_action() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingToxicityGuard::with_defaults()));

        pipeline.process_chunk("I ");
        let result = pipeline.process_chunk("hate everything ");
        assert!(result.was_evaluated);
        assert!(matches!(result.action, StreamGuardAction::Block(_)));
        assert_eq!(pipeline.get_metrics().blocks, 1);
    }

    #[test]
    fn test_pipeline_flag_action() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingPiiGuard));

        pipeline.process_chunk("contact ");
        let result = pipeline.process_chunk("user@example.com ");
        assert!(result.was_evaluated);
        assert!(matches!(result.action, StreamGuardAction::Flag(_)));
        assert_eq!(pipeline.get_metrics().flags, 1);
    }

    #[test]
    fn test_pipeline_reset() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingPiiGuard));

        pipeline.process_chunk("hello ");
        pipeline.process_chunk("world ");
        assert!(!pipeline.get_accumulated().is_empty());
        assert!(pipeline.get_metrics().chunks_received > 0);

        pipeline.reset();
        assert!(pipeline.get_accumulated().is_empty());
        assert_eq!(pipeline.get_metrics().chunks_received, 0);
        assert_eq!(pipeline.get_metrics().chunks_evaluated, 0);
    }

    #[test]
    fn test_streaming_pii_guard_email() {
        let guard = StreamingPiiGuard;
        let action = guard.check_chunk("chunk", "please contact user@example.com for info");
        assert!(matches!(action, StreamGuardAction::Flag(_)));
        if let StreamGuardAction::Flag(msg) = action {
            assert!(msg.contains("email"));
        }
    }

    #[test]
    fn test_streaming_pii_guard_phone() {
        let guard = StreamingPiiGuard;
        let action = guard.check_chunk("chunk", "call me at 1234567890 today");
        assert!(matches!(action, StreamGuardAction::Flag(_)));
        if let StreamGuardAction::Flag(msg) = action {
            assert!(msg.contains("phone"));
        }
    }

    #[test]
    fn test_streaming_pii_guard_ssn() {
        let guard = StreamingPiiGuard;
        let action = guard.check_chunk("chunk", "my ssn is 123-45-6789 please keep secret");
        assert!(matches!(action, StreamGuardAction::Block(_)));
        if let StreamGuardAction::Block(msg) = action {
            assert!(msg.contains("SSN"));
        }
    }

    #[test]
    fn test_streaming_pii_guard_clean() {
        let guard = StreamingPiiGuard;
        let action = guard.check_chunk("chunk", "this is perfectly clean text");
        assert!(matches!(action, StreamGuardAction::Pass));
    }

    #[test]
    fn test_streaming_toxicity_guard_blocks() {
        let guard = StreamingToxicityGuard::with_defaults();
        let action = guard.check_chunk("chunk", "I hate this thing");
        assert!(matches!(action, StreamGuardAction::Block(_)));
        if let StreamGuardAction::Block(msg) = action {
            assert!(msg.contains("hate"));
        }
    }

    #[test]
    fn test_streaming_toxicity_guard_custom_blocklist() {
        let guard =
            StreamingToxicityGuard::new(vec!["badword".to_string(), "evilphrase".to_string()]);
        let action = guard.check_chunk("chunk", "this contains badword in it");
        assert!(matches!(action, StreamGuardAction::Block(_)));
    }

    #[test]
    fn test_streaming_toxicity_guard_clean() {
        let guard = StreamingToxicityGuard::with_defaults();
        let action = guard.check_chunk("chunk", "the weather is nice today");
        assert!(matches!(action, StreamGuardAction::Pass));
    }

    #[test]
    fn test_streaming_pattern_guard_blocks() {
        let guard = StreamingPatternGuard::new(vec![
            "secret_key".to_string(),
            "password123".to_string(),
        ]);
        let action = guard.check_chunk("chunk", "the secret_key is leaked");
        assert!(matches!(action, StreamGuardAction::Block(_)));
        if let StreamGuardAction::Block(msg) = action {
            assert!(msg.contains("secret_key"));
        }
    }

    #[test]
    fn test_streaming_pattern_guard_case_insensitive() {
        let guard = StreamingPatternGuard::new(vec!["forbidden".to_string()]);
        let action = guard.check_chunk("chunk", "this is FORBIDDEN content");
        assert!(matches!(action, StreamGuardAction::Block(_)));
    }

    #[test]
    fn test_streaming_pattern_guard_clean() {
        let guard = StreamingPatternGuard::new(vec!["secret".to_string()]);
        let action = guard.check_chunk("chunk", "nothing wrong here");
        assert!(matches!(action, StreamGuardAction::Pass));
    }

    #[test]
    fn test_worse_action_ordering() {
        // Pass < Pause < Flag < Block
        let pass = StreamGuardAction::Pass;
        let pause = StreamGuardAction::Pause;
        let flag = StreamGuardAction::Flag("f".into());
        let block = StreamGuardAction::Block("b".into());

        // Pass vs Pause -> Pause
        let r = StreamingGuardrailPipeline::worse_action_pub(&pass, &pause);
        assert!(matches!(r, StreamGuardAction::Pause));

        // Pause vs Flag -> Flag
        let r = StreamingGuardrailPipeline::worse_action_pub(&pause, &flag);
        assert!(matches!(r, StreamGuardAction::Flag(_)));

        // Flag vs Block -> Block
        let r = StreamingGuardrailPipeline::worse_action_pub(&flag, &block);
        assert!(matches!(r, StreamGuardAction::Block(_)));

        // Pass vs Block -> Block
        let r = StreamingGuardrailPipeline::worse_action_pub(&pass, &block);
        assert!(matches!(r, StreamGuardAction::Block(_)));

        // Block vs Pass -> Block (order should not matter)
        let r = StreamingGuardrailPipeline::worse_action_pub(&block, &pass);
        assert!(matches!(r, StreamGuardAction::Block(_)));

        // Same severity: Pass vs Pass -> Pass
        let r = StreamingGuardrailPipeline::worse_action_pub(&pass, &StreamGuardAction::Pass);
        assert!(matches!(r, StreamGuardAction::Pass));
    }

    #[test]
    fn test_streaming_pipeline_default_trait() {
        let pipeline = StreamingGuardrailPipeline::default();
        assert!(pipeline.get_accumulated().is_empty());
        assert_eq!(pipeline.get_metrics().chunks_received, 0);
    }

    #[test]
    fn test_streaming_pipeline_max_buffer_trim() {
        let config = StreamingGuardrailConfig {
            min_buffer_size: 1,
            eval_interval: 1,
            max_buffer_size: 20,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config);

        // Feed a chunk that exceeds max_buffer_size.
        pipeline.process_chunk("this is a long chunk that exceeds the buffer limit");
        assert!(pipeline.get_accumulated().len() <= 20);
    }

    #[test]
    fn test_streaming_guard_min_buffer_tokens_default() {
        let guard = StreamingPiiGuard;
        assert_eq!(guard.min_buffer_tokens(), 10);
    }

    #[test]
    fn test_streaming_pipeline_multiple_guards_worst_wins() {
        // PII guard will Flag, toxicity guard will Block -> Block should win.
        let config = StreamingGuardrailConfig {
            min_buffer_size: 2,
            eval_interval: 1,
            max_buffer_size: 500,
        };
        let mut pipeline = StreamingGuardrailPipeline::with_config(config)
            .add_guard(Box::new(StreamingPiiGuard))
            .add_guard(Box::new(StreamingToxicityGuard::with_defaults()));

        pipeline.process_chunk("contact user@example.com ");
        let result = pipeline.process_chunk("I hate everything ");
        assert!(result.was_evaluated);
        // Toxicity block is worse than PII flag, so Block wins.
        assert!(matches!(result.action, StreamGuardAction::Block(_)));
    }
}
