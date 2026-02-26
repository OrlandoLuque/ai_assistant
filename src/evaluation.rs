//! Evaluation and benchmarking system
//!
//! Automated quality evaluation, performance benchmarking, and A/B testing
//! for AI model responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Evaluation metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    // Quality metrics
    Relevance,
    Coherence,
    Fluency,
    Factuality,
    Helpfulness,
    Safety,

    // Performance metrics
    Latency,
    TokensPerSecond,
    TimeToFirstToken,

    // Cost metrics
    PromptTokens,
    CompletionTokens,
    TotalCost,

    // Custom
    Custom,
}

/// Evaluation result for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub metric_type: MetricType,
    pub name: String,
    pub value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub threshold: Option<f64>,
    pub passed: Option<bool>,
    pub details: Option<String>,
}

impl MetricResult {
    pub fn new(metric_type: MetricType, name: &str, value: f64) -> Self {
        Self {
            metric_type,
            name: name.to_string(),
            value,
            min_value: 0.0,
            max_value: 1.0,
            threshold: None,
            passed: None,
            details: None,
        }
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self.passed = Some(self.value >= threshold);
        self
    }

    pub fn normalized(&self) -> f64 {
        (self.value - self.min_value) / (self.max_value - self.min_value)
    }
}

/// Evaluation sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSample {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub reference: Option<String>,
    pub context: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl EvalSample {
    pub fn new(id: &str, prompt: &str, response: &str) -> Self {
        Self {
            id: id.to_string(),
            prompt: prompt.to_string(),
            response: response.to_string(),
            reference: None,
            context: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_reference(mut self, reference: &str) -> Self {
        self.reference = Some(reference.to_string());
        self
    }

    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Evaluation result for a sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub sample_id: String,
    pub metrics: Vec<MetricResult>,
    pub overall_score: f64,
    pub passed: bool,
    pub duration: Duration,
    pub timestamp: u64,
}

impl EvalResult {
    pub fn get_metric(&self, metric_type: MetricType) -> Option<&MetricResult> {
        self.metrics.iter().find(|m| m.metric_type == metric_type)
    }

    pub fn get_metric_value(&self, metric_type: MetricType) -> Option<f64> {
        self.get_metric(metric_type).map(|m| m.value)
    }
}

/// Evaluator trait for custom evaluators
pub trait Evaluator: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, sample: &EvalSample) -> Vec<MetricResult>;
}

/// Basic text quality evaluator
#[derive(Debug)]
pub struct TextQualityEvaluator {
    min_length: usize,
    max_length: usize,
}

impl TextQualityEvaluator {
    pub fn new() -> Self {
        Self {
            min_length: 10,
            max_length: 10000,
        }
    }

    pub fn with_length_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_length = min;
        self.max_length = max;
        self
    }

    fn evaluate_fluency(&self, text: &str) -> f64 {
        // Simple fluency heuristics
        let sentences: Vec<&str> = text
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();
        if sentences.is_empty() {
            return 0.0;
        }

        let mut score: f64 = 0.5;

        // Check sentence length variety
        let avg_len: f64 = sentences
            .iter()
            .map(|s| s.split_whitespace().count() as f64)
            .sum::<f64>()
            / sentences.len() as f64;
        if (5.0..25.0).contains(&avg_len) {
            score += 0.2;
        }

        // Check for proper capitalization
        let proper_caps = sentences
            .iter()
            .filter(|s| {
                s.trim()
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            })
            .count();
        score += (proper_caps as f64 / sentences.len() as f64) * 0.2;

        // Check for repeated words (sign of poor fluency)
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().copied().collect();
        let repetition_ratio = unique_words.len() as f64 / words.len().max(1) as f64;
        score += repetition_ratio * 0.1;

        score.min(1.0)
    }

    fn evaluate_coherence(&self, text: &str) -> f64 {
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        // Single paragraph is considered coherent
        if paragraphs.len() <= 1 {
            return 0.8;
        }

        let mut score: f64 = 0.5;

        // Check for transition words
        let transitions = [
            "however",
            "therefore",
            "additionally",
            "furthermore",
            "moreover",
            "consequently",
            "thus",
            "hence",
        ];
        let text_lower = text.to_lowercase();
        let transition_count = transitions
            .iter()
            .filter(|t| text_lower.contains(*t))
            .count();
        score += (transition_count as f64 / 3.0).min(0.3);

        // Check for pronouns (indicate connected discourse)
        let pronouns = ["it", "this", "that", "these", "they", "them"];
        let pronoun_count = pronouns
            .iter()
            .map(|p| text_lower.matches(&format!(" {} ", p)).count())
            .sum::<usize>();
        score += (pronoun_count as f64 / 10.0).min(0.2);

        score.min(1.0)
    }
}

impl Default for TextQualityEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for TextQualityEvaluator {
    fn name(&self) -> &str {
        "text_quality"
    }

    fn evaluate(&self, sample: &EvalSample) -> Vec<MetricResult> {
        let mut results = Vec::new();
        let response = &sample.response;

        // Length check
        let length_score = if response.len() < self.min_length {
            0.3
        } else if response.len() > self.max_length {
            0.5
        } else {
            1.0
        };
        results.push(MetricResult::new(
            MetricType::Custom,
            "length",
            length_score,
        ));

        // Fluency
        let fluency = self.evaluate_fluency(response);
        results.push(MetricResult::new(MetricType::Fluency, "fluency", fluency));

        // Coherence
        let coherence = self.evaluate_coherence(response);
        results.push(MetricResult::new(
            MetricType::Coherence,
            "coherence",
            coherence,
        ));

        results
    }
}

/// Relevance evaluator (compares response to prompt/reference)
#[derive(Debug)]
pub struct RelevanceEvaluator;

impl RelevanceEvaluator {
    pub fn new() -> Self {
        Self
    }

    fn word_overlap(&self, text1: &str, text2: &str) -> f64 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();

        let words1: std::collections::HashSet<&str> = text1_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        let words2: std::collections::HashSet<&str> = text2_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f64 / union as f64
    }
}

impl Default for RelevanceEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for RelevanceEvaluator {
    fn name(&self) -> &str {
        "relevance"
    }

    fn evaluate(&self, sample: &EvalSample) -> Vec<MetricResult> {
        let mut results = Vec::new();

        // Relevance to prompt
        let prompt_relevance = self.word_overlap(&sample.prompt, &sample.response);
        results.push(MetricResult::new(
            MetricType::Relevance,
            "prompt_relevance",
            prompt_relevance,
        ));

        // Relevance to reference (if available)
        if let Some(ref reference) = sample.reference {
            let ref_relevance = self.word_overlap(reference, &sample.response);
            results.push(MetricResult::new(
                MetricType::Relevance,
                "reference_relevance",
                ref_relevance,
            ));
        }

        // Context relevance (if available)
        if let Some(ref context) = sample.context {
            let ctx_relevance = self.word_overlap(context, &sample.response);
            results.push(MetricResult::new(
                MetricType::Relevance,
                "context_relevance",
                ctx_relevance,
            ));
        }

        results
    }
}

/// Safety evaluator
#[derive(Debug)]
pub struct SafetyEvaluator {
    blocked_patterns: Vec<regex::Regex>,
    warning_patterns: Vec<regex::Regex>,
}

impl SafetyEvaluator {
    pub fn new() -> Self {
        Self {
            blocked_patterns: Vec::new(),
            warning_patterns: Vec::new(),
        }
    }

    pub fn add_blocked_pattern(mut self, pattern: &str) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.blocked_patterns.push(re);
        }
        self
    }

    pub fn add_warning_pattern(mut self, pattern: &str) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.warning_patterns.push(re);
        }
        self
    }

    pub fn with_default_patterns(mut self) -> Self {
        // Add common safety patterns (simplified)
        let blocked = ["(?i)password.*is.*:", "(?i)api.*key.*:", "(?i)secret.*:"];
        let warnings = ["(?i)disclaimer", "(?i)not.*financial.*advice"];

        for pattern in blocked {
            if let Ok(re) = regex::Regex::new(pattern) {
                self.blocked_patterns.push(re);
            }
        }

        for pattern in warnings {
            if let Ok(re) = regex::Regex::new(pattern) {
                self.warning_patterns.push(re);
            }
        }

        self
    }
}

impl Default for SafetyEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for SafetyEvaluator {
    fn name(&self) -> &str {
        "safety"
    }

    fn evaluate(&self, sample: &EvalSample) -> Vec<MetricResult> {
        let mut results = Vec::new();
        let response = &sample.response;

        let mut safety_score: f64 = 1.0;
        let mut details = Vec::new();

        // Check blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(response) {
                safety_score = 0.0;
                details.push(format!("Blocked pattern matched: {}", pattern.as_str()));
            }
        }

        // Check warning patterns
        for pattern in &self.warning_patterns {
            if pattern.is_match(response) {
                safety_score = safety_score.min(0.7);
                details.push(format!("Warning pattern matched: {}", pattern.as_str()));
            }
        }

        let mut result =
            MetricResult::new(MetricType::Safety, "safety", safety_score).with_threshold(0.5);

        if !details.is_empty() {
            result.details = Some(details.join("; "));
        }

        results.push(result);
        results
    }
}

/// Performance benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub p50_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub tokens_per_second: Option<f64>,
    pub time_to_first_token: Option<Duration>,
}

impl BenchmarkResult {
    pub fn from_durations(name: &str, durations: &[Duration]) -> Self {
        let mut sorted: Vec<Duration> = durations.to_vec();
        sorted.sort();

        let total: Duration = sorted.iter().sum();
        let count = sorted.len();

        let p50_idx = (count as f64 * 0.5) as usize;
        let p95_idx = (count as f64 * 0.95) as usize;
        let p99_idx = (count as f64 * 0.99) as usize;

        Self {
            name: name.to_string(),
            iterations: count,
            total_duration: total,
            avg_duration: total / count as u32,
            min_duration: sorted.first().copied().unwrap_or_default(),
            max_duration: sorted.last().copied().unwrap_or_default(),
            p50_duration: sorted.get(p50_idx).copied().unwrap_or_default(),
            p95_duration: sorted
                .get(p95_idx.min(count - 1))
                .copied()
                .unwrap_or_default(),
            p99_duration: sorted
                .get(p99_idx.min(count - 1))
                .copied()
                .unwrap_or_default(),
            tokens_per_second: None,
            time_to_first_token: None,
        }
    }
}

/// Benchmark runner
#[derive(Debug)]
pub struct Benchmarker {
    warmup_iterations: usize,
    test_iterations: usize,
}

impl Benchmarker {
    pub fn new(warmup: usize, iterations: usize) -> Self {
        Self {
            warmup_iterations: warmup,
            test_iterations: iterations,
        }
    }

    pub fn run<F>(&self, name: &str, mut f: F) -> BenchmarkResult
    where
        F: FnMut() -> (),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Benchmark
        let mut durations = Vec::with_capacity(self.test_iterations);
        for _ in 0..self.test_iterations {
            let start = Instant::now();
            f();
            durations.push(start.elapsed());
        }

        BenchmarkResult::from_durations(name, &durations)
    }
}

impl Default for Benchmarker {
    fn default() -> Self {
        Self::new(3, 10)
    }
}

/// A/B Test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestConfig {
    pub name: String,
    pub variant_a: String,
    pub variant_b: String,
    pub traffic_split: f64, // 0.0-1.0, percentage for variant A
    pub min_samples: usize,
    pub confidence_level: f64,
}

impl AbTestConfig {
    pub fn new(name: &str, variant_a: &str, variant_b: &str) -> Self {
        Self {
            name: name.to_string(),
            variant_a: variant_a.to_string(),
            variant_b: variant_b.to_string(),
            traffic_split: 0.5,
            min_samples: 100,
            confidence_level: 0.95,
        }
    }

    pub fn with_split(mut self, split: f64) -> Self {
        self.traffic_split = split.clamp(0.0, 1.0);
        self
    }
}

/// A/B Test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestResult {
    pub config: AbTestConfig,
    pub variant_a_samples: usize,
    pub variant_b_samples: usize,
    pub variant_a_score: f64,
    pub variant_b_score: f64,
    pub difference: f64,
    pub p_value: f64,
    pub significant: bool,
    pub winner: Option<String>,
}

/// A/B Test manager
pub struct AbTestManager {
    tests: HashMap<String, AbTestConfig>,
    results_a: HashMap<String, Vec<f64>>,
    results_b: HashMap<String, Vec<f64>>,
}

impl AbTestManager {
    pub fn new() -> Self {
        Self {
            tests: HashMap::new(),
            results_a: HashMap::new(),
            results_b: HashMap::new(),
        }
    }

    pub fn register_test(&mut self, config: AbTestConfig) {
        self.results_a.insert(config.name.clone(), Vec::new());
        self.results_b.insert(config.name.clone(), Vec::new());
        self.tests.insert(config.name.clone(), config);
    }

    pub fn assign_variant(&self, test_name: &str, user_id: &str) -> Option<&str> {
        let config = self.tests.get(test_name)?;

        // Deterministic assignment based on user ID hash
        let hash = user_id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let normalized = (hash % 10000) as f64 / 10000.0;

        if normalized < config.traffic_split {
            Some(&config.variant_a)
        } else {
            Some(&config.variant_b)
        }
    }

    pub fn record_result(&mut self, test_name: &str, variant: &str, score: f64) {
        if let Some(config) = self.tests.get(test_name) {
            if variant == config.variant_a {
                if let Some(results) = self.results_a.get_mut(test_name) {
                    results.push(score);
                }
            } else if variant == config.variant_b {
                if let Some(results) = self.results_b.get_mut(test_name) {
                    results.push(score);
                }
            }
        }
    }

    pub fn get_results(&self, test_name: &str) -> Option<AbTestResult> {
        let config = self.tests.get(test_name)?;
        let results_a = self.results_a.get(test_name)?;
        let results_b = self.results_b.get(test_name)?;

        let mean_a = if results_a.is_empty() {
            0.0
        } else {
            results_a.iter().sum::<f64>() / results_a.len() as f64
        };
        let mean_b = if results_b.is_empty() {
            0.0
        } else {
            results_b.iter().sum::<f64>() / results_b.len() as f64
        };

        let difference = mean_a - mean_b;

        // Simplified p-value calculation (in real implementation, use proper statistical test)
        let p_value = self.calculate_p_value(results_a, results_b);
        let significant = p_value < (1.0 - config.confidence_level);

        let winner = if significant {
            if mean_a > mean_b {
                Some(config.variant_a.clone())
            } else {
                Some(config.variant_b.clone())
            }
        } else {
            None
        };

        Some(AbTestResult {
            config: config.clone(),
            variant_a_samples: results_a.len(),
            variant_b_samples: results_b.len(),
            variant_a_score: mean_a,
            variant_b_score: mean_b,
            difference,
            p_value,
            significant,
            winner,
        })
    }

    /// Welch's two-sample t-test with proper Student's t-distribution CDF.
    fn calculate_p_value(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() < 2 || b.len() < 2 {
            return 1.0;
        }

        let n_a = a.len() as f64;
        let n_b = b.len() as f64;

        let mean_a = a.iter().sum::<f64>() / n_a;
        let mean_b = b.iter().sum::<f64>() / n_b;

        let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
        let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

        let se_sq = var_a / n_a + var_b / n_b;
        if se_sq == 0.0 {
            return 1.0;
        }
        let se = se_sq.sqrt();
        let t = (mean_a - mean_b).abs() / se;

        // Welch–Satterthwaite degrees of freedom
        let num = se_sq * se_sq;
        let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
        if denom == 0.0 {
            return 1.0;
        }
        let df = num / denom;

        // Two-tailed p-value from Student's t-distribution
        let p = 2.0 * (1.0 - student_t_cdf(t, df));
        p.clamp(0.0, 1.0)
    }
}

impl Default for AbTestManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Log-gamma function via Lanczos approximation (g=7, n=9 coefficients).
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let ln_pi = std::f64::consts::PI.ln();
        return ln_pi - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}

/// Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation when x > (a+1)/(a+b+2) for faster convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    // Compute the log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    let ln_prefix =
        a * x.ln() + b * (1.0 - x).ln() - ln_gamma(a) - ln_gamma(b) + ln_gamma(a + b) - a.ln();

    // Lentz's continued fraction for I_x(a,b)
    let max_iter = 200;
    let epsilon = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step: d_{2m}
        let numerator_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + numerator_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        result *= d * c;

        // Odd step: d_{2m+1}
        let numerator_odd =
            -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + numerator_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    (ln_prefix + result.ln()).exp()
}

/// CDF of Student's t-distribution with `df` degrees of freedom, evaluated at `t`.
fn student_t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    let ibeta = regularized_incomplete_beta(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

/// Evaluation suite
pub struct EvalSuite {
    evaluators: Vec<Box<dyn Evaluator>>,
    weights: HashMap<MetricType, f64>,
    pass_threshold: f64,
}

impl EvalSuite {
    pub fn new() -> Self {
        Self {
            evaluators: Vec::new(),
            weights: HashMap::new(),
            pass_threshold: 0.7,
        }
    }

    pub fn add_evaluator<E: Evaluator + 'static>(&mut self, evaluator: E) {
        self.evaluators.push(Box::new(evaluator));
    }

    pub fn set_weight(&mut self, metric_type: MetricType, weight: f64) {
        self.weights.insert(metric_type, weight);
    }

    pub fn set_pass_threshold(&mut self, threshold: f64) {
        self.pass_threshold = threshold;
    }

    pub fn evaluate(&self, sample: &EvalSample) -> EvalResult {
        let start = Instant::now();
        let mut all_metrics = Vec::new();

        // Run all evaluators
        for evaluator in &self.evaluators {
            let metrics = evaluator.evaluate(sample);
            all_metrics.extend(metrics);
        }

        // Calculate weighted overall score
        let mut total_weight: f64 = 0.0;
        let mut weighted_sum: f64 = 0.0;

        for metric in &all_metrics {
            let weight = self
                .weights
                .get(&metric.metric_type)
                .copied()
                .unwrap_or(1.0);
            weighted_sum += metric.normalized() * weight;
            total_weight += weight;
        }

        let overall_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let passed = overall_score >= self.pass_threshold;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        EvalResult {
            sample_id: sample.id.clone(),
            metrics: all_metrics,
            overall_score,
            passed,
            duration: start.elapsed(),
            timestamp,
        }
    }

    pub fn evaluate_batch(&self, samples: &[EvalSample]) -> Vec<EvalResult> {
        samples.iter().map(|s| self.evaluate(s)).collect()
    }

    pub fn summary(&self, results: &[EvalResult]) -> EvalSummary {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();

        let avg_score = if total > 0 {
            results.iter().map(|r| r.overall_score).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let avg_duration = if total > 0 {
            Duration::from_nanos(
                (results.iter().map(|r| r.duration.as_nanos()).sum::<u128>() / total as u128)
                    as u64,
            )
        } else {
            Duration::ZERO
        };

        // Aggregate metrics
        let mut metric_scores: HashMap<MetricType, Vec<f64>> = HashMap::new();
        for result in results {
            for metric in &result.metrics {
                metric_scores
                    .entry(metric.metric_type)
                    .or_default()
                    .push(metric.value);
            }
        }

        let metric_averages: HashMap<MetricType, f64> = metric_scores
            .into_iter()
            .map(|(k, v)| {
                let avg = v.iter().sum::<f64>() / v.len() as f64;
                (k, avg)
            })
            .collect();

        EvalSummary {
            total_samples: total,
            passed_samples: passed,
            pass_rate: passed as f64 / total.max(1) as f64,
            avg_score,
            avg_duration,
            metric_averages,
        }
    }
}

impl Default for EvalSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    pub total_samples: usize,
    pub passed_samples: usize,
    pub pass_rate: f64,
    pub avg_score: f64,
    #[serde(with = "duration_serde")]
    pub avg_duration: Duration,
    pub metric_averages: HashMap<MetricType, f64>,
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, s: S) -> Result<S::Ok, S::Error> {
        duration.as_millis().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let millis = u64::deserialize(d)?;
        Ok(Duration::from_millis(millis))
    }
}

// ============================================================================
// LLM-as-Judge Evaluator
// ============================================================================

use std::sync::Arc;

/// Configuration for LLM-as-judge evaluation.
#[derive(Debug, Clone)]
pub struct LlmJudgeConfig {
    /// Metrics to evaluate (default: Relevance, Coherence, Helpfulness, Safety)
    pub metrics: Vec<MetricType>,
    /// Custom rubric/grading criteria (optional, uses default if None)
    pub rubric: Option<String>,
    /// Number of judge calls for majority voting (1 = single judge, 3+ = voting)
    pub num_judges: usize,
}

impl Default for LlmJudgeConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                MetricType::Relevance,
                MetricType::Coherence,
                MetricType::Helpfulness,
                MetricType::Safety,
            ],
            rubric: None,
            num_judges: 1,
        }
    }
}

/// LLM-as-Judge evaluator that uses a language model to evaluate responses.
///
/// Takes a generator callback that produces LLM responses given a prompt.
/// The generator is the same pattern used in `agentic_loop.rs`.
pub struct LlmJudgeEvaluator {
    /// Callback that generates an LLM response given a prompt string
    generator: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
    config: LlmJudgeConfig,
}

impl LlmJudgeEvaluator {
    /// Create a new LLM-as-Judge evaluator with the given generator callback.
    pub fn new<F>(generator: F) -> Self
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        Self {
            generator: Arc::new(generator),
            config: LlmJudgeConfig::default(),
        }
    }

    pub fn with_config(mut self, config: LlmJudgeConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the grading prompt for the judge.
    fn build_grading_prompt(&self, sample: &EvalSample) -> String {
        let metrics_str = self
            .config
            .metrics
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
            .join(", ");

        let rubric = self.config.rubric.as_deref().unwrap_or(
            "Rate each metric on a scale of 1-10 where:\n\
             1-3: Poor quality\n\
             4-6: Acceptable quality\n\
             7-9: Good quality\n\
             10: Excellent quality",
        );

        let mut prompt = format!(
            "You are an expert evaluator. Grade the following AI response on these metrics: {}\n\n\
             {}\n\n\
             ## Prompt\n{}\n\n\
             ## Response\n{}",
            metrics_str, rubric, sample.prompt, sample.response
        );

        if let Some(ref reference) = sample.reference {
            prompt.push_str(&format!("\n\n## Reference Answer\n{}", reference));
        }
        if let Some(ref context) = sample.context {
            prompt.push_str(&format!("\n\n## Context\n{}", context));
        }

        prompt.push_str(&format!(
            "\n\nRespond ONLY with a JSON object mapping each metric to its score (1-10). Example:\n\
             {{{}}}",
            self.config.metrics.iter()
                .map(|m| format!("\"{:?}\": 7", m))
                .collect::<Vec<_>>()
                .join(", ")
        ));

        prompt
    }

    /// Parse scores from LLM response. Supports:
    /// - JSON: {"Relevance": 8, "Coherence": 7}
    /// - Text: "Score: 8/10" or "Relevance: 8"
    fn parse_scores(&self, response: &str) -> HashMap<MetricType, f64> {
        let mut scores = HashMap::new();

        // Try JSON parsing first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(obj) = json.as_object() {
                for metric in &self.config.metrics {
                    let key = format!("{:?}", metric);
                    if let Some(val) = obj.get(&key).and_then(|v| v.as_f64()) {
                        scores.insert(*metric, val);
                    }
                }
                if !scores.is_empty() {
                    return scores;
                }
            }
        }

        // Try extracting JSON from markdown code block
        let json_re = regex::Regex::new(r"(?s)\{[^}]+\}").ok();
        if let Some(re) = json_re {
            if let Some(m) = re.find(response) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(m.as_str()) {
                    if let Some(obj) = json.as_object() {
                        for metric in &self.config.metrics {
                            let key = format!("{:?}", metric);
                            if let Some(val) = obj.get(&key).and_then(|v| v.as_f64()) {
                                scores.insert(*metric, val);
                            }
                        }
                        if !scores.is_empty() {
                            return scores;
                        }
                    }
                }
            }
        }

        // Fallback: look for "Score: X/10" or "Metric: X" patterns
        let score_re = regex::Regex::new(r"(\w+)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?").ok();
        if let Some(re) = score_re {
            for cap in re.captures_iter(response) {
                let name = &cap[1];
                if let Ok(val) = cap[2].parse::<f64>() {
                    for metric in &self.config.metrics {
                        if format!("{:?}", metric).to_lowercase() == name.to_lowercase() {
                            scores.insert(*metric, val);
                        }
                    }
                }
            }
        }

        scores
    }
}

impl Evaluator for LlmJudgeEvaluator {
    fn name(&self) -> &str {
        "llm_judge"
    }

    fn evaluate(&self, sample: &EvalSample) -> Vec<MetricResult> {
        let prompt = self.build_grading_prompt(sample);

        // Collect scores from multiple judges for majority voting
        let mut all_scores: Vec<HashMap<MetricType, f64>> = Vec::new();

        for _ in 0..self.config.num_judges {
            match (self.generator)(&prompt) {
                Ok(response) => {
                    let scores = self.parse_scores(&response);
                    if !scores.is_empty() {
                        all_scores.push(scores);
                    }
                }
                Err(_) => continue,
            }
        }

        if all_scores.is_empty() {
            // All judges failed — return zero scores
            return self
                .config
                .metrics
                .iter()
                .map(|m| MetricResult::new(*m, &format!("{:?}", m), 0.0).with_range(0.0, 10.0))
                .collect();
        }

        // Average scores across judges
        let mut final_scores: HashMap<MetricType, f64> = HashMap::new();
        for metric in &self.config.metrics {
            let values: Vec<f64> = all_scores
                .iter()
                .filter_map(|s| s.get(metric).copied())
                .collect();
            if !values.is_empty() {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                final_scores.insert(*metric, avg);
            }
        }

        // Build results
        self.config
            .metrics
            .iter()
            .map(|m| {
                let value = final_scores.get(m).copied().unwrap_or(0.0);
                MetricResult::new(*m, &format!("{:?}", m), value).with_range(0.0, 10.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_result() {
        let result = MetricResult::new(MetricType::Fluency, "fluency", 0.8).with_threshold(0.7);

        assert!(result.passed.unwrap());
        assert_eq!(result.normalized(), 0.8);
    }

    #[test]
    fn test_text_quality_evaluator() {
        let evaluator = TextQualityEvaluator::new();
        let sample = EvalSample::new("1", "What is AI?", "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines.");

        let results = evaluator.evaluate(&sample);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_relevance_evaluator() {
        let evaluator = RelevanceEvaluator::new();
        let sample = EvalSample::new(
            "1",
            "Tell me about cats",
            "Cats are domesticated animals known for their independence.",
        )
        .with_reference("Cats are popular pets worldwide.");

        let results = evaluator.evaluate(&sample);
        assert!(results.len() >= 1);
    }

    #[test]
    fn test_benchmark() {
        let benchmarker = Benchmarker::new(1, 5);
        let result = benchmarker.run("test", || {
            std::thread::sleep(std::time::Duration::from_micros(100));
        });

        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn test_ab_test() {
        let mut manager = AbTestManager::new();
        manager.register_test(AbTestConfig::new("test", "model_a", "model_b"));

        // Record some results
        for i in 0..20 {
            manager.record_result("test", "model_a", 0.8 + (i as f64 * 0.01));
            manager.record_result("test", "model_b", 0.7 + (i as f64 * 0.01));
        }

        let result = manager.get_results("test").unwrap();
        assert_eq!(result.variant_a_samples, 20);
        assert_eq!(result.variant_b_samples, 20);
    }

    #[test]
    fn test_eval_suite() {
        let mut suite = EvalSuite::new();
        suite.add_evaluator(TextQualityEvaluator::new());
        suite.add_evaluator(RelevanceEvaluator::new());
        suite.set_pass_threshold(0.3);

        let sample = EvalSample::new("1", "Hello", "Hi there, how can I help you today?");
        let result = suite.evaluate(&sample);

        assert!(!result.metrics.is_empty());
    }

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1 → ln(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Gamma(2) = 1 → ln(1) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Gamma(5) = 24 → ln(24) ≈ 3.1781
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-8);
        // Gamma(0.5) = sqrt(pi) → ln(sqrt(pi)) ≈ 0.5724
        assert!((ln_gamma(0.5) - (std::f64::consts::PI.sqrt().ln())).abs() < 1e-8);
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        // CDF at t=0 should be 0.5 for any df
        assert!((student_t_cdf(0.0, 10.0) - 0.5).abs() < 1e-10);
        assert!((student_t_cdf(0.0, 1.0) - 0.5).abs() < 1e-10);

        // CDF(-t) = 1 - CDF(t) (symmetry)
        let cdf_pos = student_t_cdf(2.0, 10.0);
        let cdf_neg = student_t_cdf(-2.0, 10.0);
        assert!((cdf_pos + cdf_neg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_student_t_cdf_known_values() {
        // For df=1 (Cauchy): CDF(1) = 0.75
        assert!((student_t_cdf(1.0, 1.0) - 0.75).abs() < 1e-4);

        // For large df, should approach standard normal
        // Normal CDF(1.96) ≈ 0.975
        let cdf_large_df = student_t_cdf(1.96, 1000.0);
        assert!((cdf_large_df - 0.975).abs() < 0.002);
    }

    #[test]
    fn test_welch_t_test_significant() {
        let mut manager = AbTestManager::new();
        manager.register_test(AbTestConfig::new("sig_test", "A", "B"));

        // Clearly different distributions: A ~ 0.9, B ~ 0.1
        for _ in 0..50 {
            manager.record_result("sig_test", "A", 0.9);
            manager.record_result("sig_test", "B", 0.1);
        }
        // Add slight variance
        for i in 0..10 {
            manager.record_result("sig_test", "A", 0.85 + (i as f64 * 0.01));
            manager.record_result("sig_test", "B", 0.05 + (i as f64 * 0.01));
        }

        let result = manager.get_results("sig_test").unwrap();
        // Should have very low p-value (highly significant)
        assert!(result.p_value < 0.001, "p_value was {}", result.p_value);
        assert!(result.significant);
        assert_eq!(result.winner, Some("A".to_string()));
    }

    #[test]
    fn test_llm_judge_config_defaults() {
        let config = LlmJudgeConfig {
            metrics: vec![MetricType::Relevance, MetricType::Fluency],
            rubric: None,
            num_judges: 3,
        };
        assert_eq!(config.metrics.len(), 2);
        assert_eq!(config.num_judges, 3);
        assert!(config.rubric.is_none());
    }

    #[test]
    fn test_llm_judge_build_grading_prompt() {
        let judge =
            LlmJudgeEvaluator::new(|_: &str| Ok("{}".to_string())).with_config(LlmJudgeConfig {
                metrics: vec![MetricType::Relevance],
                rubric: Some("Be strict".to_string()),
                num_judges: 1,
            });
        let sample = EvalSample::new("1", "What is Rust?", "Rust is a programming language.");
        let prompt = judge.build_grading_prompt(&sample);
        assert!(prompt.contains("Relevance"));
        assert!(prompt.contains("Be strict"));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rust is a programming language"));
    }

    #[test]
    fn test_llm_judge_parse_scores_json() {
        let judge =
            LlmJudgeEvaluator::new(|_: &str| Ok("{}".to_string())).with_config(LlmJudgeConfig {
                metrics: vec![MetricType::Relevance, MetricType::Fluency],
                rubric: None,
                num_judges: 1,
            });
        let response = r#"{"Relevance": 8.5, "Fluency": 7.0}"#;
        let scores = judge.parse_scores(response);
        assert!((scores[&MetricType::Relevance] - 8.5).abs() < 0.01);
        assert!((scores[&MetricType::Fluency] - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_parse_scores_markdown_json() {
        let judge =
            LlmJudgeEvaluator::new(|_: &str| Ok("{}".to_string())).with_config(LlmJudgeConfig {
                metrics: vec![MetricType::Relevance],
                rubric: None,
                num_judges: 1,
            });
        let response = "Here are the scores:\n```json\n{\"Relevance\": 9.0}\n```\n";
        let scores = judge.parse_scores(response);
        assert!((scores[&MetricType::Relevance] - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_parse_scores_regex_fallback() {
        let judge =
            LlmJudgeEvaluator::new(|_: &str| Ok("{}".to_string())).with_config(LlmJudgeConfig {
                metrics: vec![MetricType::Relevance],
                rubric: None,
                num_judges: 1,
            });
        let response = "I would rate this response:\nRelevance: 7/10\nOverall good quality.";
        let scores = judge.parse_scores(response);
        assert!((scores[&MetricType::Relevance] - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_evaluate_with_mock() {
        let judge = LlmJudgeEvaluator::new(|_: &str| {
            Ok(r#"{"Relevance": 8.0, "Fluency": 7.5}"#.to_string())
        })
        .with_config(LlmJudgeConfig {
            metrics: vec![MetricType::Relevance, MetricType::Fluency],
            rubric: None,
            num_judges: 3,
        });
        let sample = EvalSample::new("1", "What is AI?", "AI is artificial intelligence.");
        let results = judge.evaluate(&sample);
        assert_eq!(results.len(), 2);
        assert!((results[0].value - 8.0).abs() < 0.01);
        assert!((results[1].value - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_evaluate_all_failures() {
        let judge = LlmJudgeEvaluator::new(|_: &str| Err("LLM unavailable".to_string()))
            .with_config(LlmJudgeConfig {
                metrics: vec![MetricType::Relevance],
                rubric: None,
                num_judges: 2,
            });
        let sample = EvalSample::new("1", "test", "test response");
        let results = judge.evaluate(&sample);
        assert_eq!(results.len(), 1);
        assert!((results[0].value - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_majority_voting() {
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();
        let judge = LlmJudgeEvaluator::new(move |_: &str| {
            let n = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match n {
                0 => Ok(r#"{"Relevance": 8.0}"#.to_string()),
                1 => Ok(r#"{"Relevance": 9.0}"#.to_string()),
                _ => Ok(r#"{"Relevance": 8.5}"#.to_string()),
            }
        })
        .with_config(LlmJudgeConfig {
            metrics: vec![MetricType::Relevance],
            rubric: None,
            num_judges: 3,
        });
        let sample = EvalSample::new("1", "test", "test response");
        let results = judge.evaluate(&sample);
        // Average of 8.0, 9.0, 8.5 = 8.5
        assert!((results[0].value - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_llm_judge_custom_rubric() {
        let judge = LlmJudgeEvaluator::new(|prompt: &str| {
            // Verify rubric is included in prompt
            assert!(prompt.contains("Custom rubric for evaluation"));
            Ok(r#"{"Relevance": 6.0}"#.to_string())
        })
        .with_config(LlmJudgeConfig {
            metrics: vec![MetricType::Relevance],
            rubric: Some("Custom rubric for evaluation".to_string()),
            num_judges: 1,
        });
        let sample = EvalSample::new("1", "test", "test response");
        let results = judge.evaluate(&sample);
        assert!((results[0].value - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_welch_t_test_not_significant() {
        let mut manager = AbTestManager::new();
        manager.register_test(AbTestConfig::new("ns_test", "A", "B"));

        // Same distribution: both around 0.5 with variance
        for i in 0..30 {
            let v = 0.4 + (i as f64 * 0.007);
            manager.record_result("ns_test", "A", v);
            manager.record_result("ns_test", "B", v + 0.001);
        }

        let result = manager.get_results("ns_test").unwrap();
        // p-value should be high (not significant)
        assert!(result.p_value > 0.05, "p_value was {}", result.p_value);
        assert!(!result.significant);
        assert!(result.winner.is_none());
    }
}
