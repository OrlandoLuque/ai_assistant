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
}
