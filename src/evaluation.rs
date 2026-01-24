//! Evaluation and benchmarking system
//!
//! Automated quality evaluation, performance benchmarking, and A/B testing
//! for AI model responses.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

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
        let sentences: Vec<&str> = text.split(['.', '!', '?']).filter(|s| !s.trim().is_empty()).collect();
        if sentences.is_empty() {
            return 0.0;
        }

        let mut score: f64 = 0.5;

        // Check sentence length variety
        let avg_len: f64 = sentences.iter().map(|s| s.split_whitespace().count() as f64).sum::<f64>() / sentences.len() as f64;
        if (5.0..25.0).contains(&avg_len) {
            score += 0.2;
        }

        // Check for proper capitalization
        let proper_caps = sentences.iter().filter(|s| {
            s.trim().chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        }).count();
        score += (proper_caps as f64 / sentences.len() as f64) * 0.2;

        // Check for repeated words (sign of poor fluency)
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().copied().collect();
        let repetition_ratio = unique_words.len() as f64 / words.len().max(1) as f64;
        score += repetition_ratio * 0.1;

        score.min(1.0)
    }

    fn evaluate_coherence(&self, text: &str) -> f64 {
        let paragraphs: Vec<&str> = text.split("\n\n").filter(|p| !p.trim().is_empty()).collect();

        // Single paragraph is considered coherent
        if paragraphs.len() <= 1 {
            return 0.8;
        }

        let mut score: f64 = 0.5;

        // Check for transition words
        let transitions = ["however", "therefore", "additionally", "furthermore", "moreover", "consequently", "thus", "hence"];
        let text_lower = text.to_lowercase();
        let transition_count = transitions.iter().filter(|t| text_lower.contains(*t)).count();
        score += (transition_count as f64 / 3.0).min(0.3);

        // Check for pronouns (indicate connected discourse)
        let pronouns = ["it", "this", "that", "these", "they", "them"];
        let pronoun_count = pronouns.iter().map(|p| {
            text_lower.matches(&format!(" {} ", p)).count()
        }).sum::<usize>();
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
        results.push(MetricResult::new(MetricType::Custom, "length", length_score));

        // Fluency
        let fluency = self.evaluate_fluency(response);
        results.push(MetricResult::new(MetricType::Fluency, "fluency", fluency));

        // Coherence
        let coherence = self.evaluate_coherence(response);
        results.push(MetricResult::new(MetricType::Coherence, "coherence", coherence));

        results
    }
}

/// Relevance evaluator (compares response to prompt/reference)
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
        results.push(MetricResult::new(MetricType::Relevance, "prompt_relevance", prompt_relevance));

        // Relevance to reference (if available)
        if let Some(ref reference) = sample.reference {
            let ref_relevance = self.word_overlap(reference, &sample.response);
            results.push(MetricResult::new(MetricType::Relevance, "reference_relevance", ref_relevance));
        }

        // Context relevance (if available)
        if let Some(ref context) = sample.context {
            let ctx_relevance = self.word_overlap(context, &sample.response);
            results.push(MetricResult::new(MetricType::Relevance, "context_relevance", ctx_relevance));
        }

        results
    }
}

/// Safety evaluator
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

        let mut result = MetricResult::new(MetricType::Safety, "safety", safety_score)
            .with_threshold(0.5);

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
            p95_duration: sorted.get(p95_idx.min(count - 1)).copied().unwrap_or_default(),
            p99_duration: sorted.get(p99_idx.min(count - 1)).copied().unwrap_or_default(),
            tokens_per_second: None,
            time_to_first_token: None,
        }
    }
}

/// Benchmark runner
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
        let hash = user_id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
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

        let mean_a = if results_a.is_empty() { 0.0 } else {
            results_a.iter().sum::<f64>() / results_a.len() as f64
        };
        let mean_b = if results_b.is_empty() { 0.0 } else {
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

    fn calculate_p_value(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() < 2 || b.len() < 2 {
            return 1.0;
        }

        let mean_a = a.iter().sum::<f64>() / a.len() as f64;
        let mean_b = b.iter().sum::<f64>() / b.len() as f64;

        let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
        let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;

        let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();

        if se == 0.0 {
            return 1.0;
        }

        let t = (mean_a - mean_b).abs() / se;

        // Simplified: convert t-score to approximate p-value
        // In real implementation, use proper t-distribution
        let p = (-0.5 * t * t).exp();

        p.min(1.0)
    }
}

impl Default for AbTestManager {
    fn default() -> Self {
        Self::new()
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
            let weight = self.weights.get(&metric.metric_type).copied().unwrap_or(1.0);
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
                (results.iter().map(|r| r.duration.as_nanos()).sum::<u128>() / total as u128) as u64
            )
        } else {
            Duration::ZERO
        };

        // Aggregate metrics
        let mut metric_scores: HashMap<MetricType, Vec<f64>> = HashMap::new();
        for result in results {
            for metric in &result.metrics {
                metric_scores.entry(metric.metric_type)
                    .or_default()
                    .push(metric.value);
            }
        }

        let metric_averages: HashMap<MetricType, f64> = metric_scores.into_iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_result() {
        let result = MetricResult::new(MetricType::Fluency, "fluency", 0.8)
            .with_threshold(0.7);

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
        let sample = EvalSample::new("1", "Tell me about cats", "Cats are domesticated animals known for their independence.")
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
}
