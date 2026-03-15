//! Benchmarking suite for AI assistant components
//!
//! This module provides tools to measure and track performance of various
//! components including response generation, RAG search, and embedding operations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BenchmarkConfig {
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations (not counted)
    pub warmup_iterations: usize,
    /// Timeout per operation
    pub timeout_ms: u64,
    /// Record individual timings
    pub record_individual: bool,
    /// Include memory measurements
    pub measure_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 2,
            timeout_ms: 30000,
            record_individual: true,
            measure_memory: false,
        }
    }
}

// ============================================================================
// Benchmark Results
// ============================================================================

/// Statistics for a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    /// Minimum duration
    pub min_ms: f64,
    /// Maximum duration
    pub max_ms: f64,
    /// Mean duration
    pub mean_ms: f64,
    /// Median duration
    pub median_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// Percentile 95
    pub p95_ms: f64,
    /// Percentile 99
    pub p99_ms: f64,
    /// Total iterations
    pub iterations: usize,
    /// Success rate
    pub success_rate: f64,
}

impl BenchmarkStats {
    /// Calculate stats from durations
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self::default();
        }

        let mut times_ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = times_ms.len();
        let sum: f64 = times_ms.iter().sum();
        let mean = sum / n as f64;

        let variance: f64 = times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;

        let std_dev = variance.sqrt();

        Self {
            min_ms: times_ms[0],
            max_ms: times_ms[n - 1],
            mean_ms: mean,
            median_ms: times_ms[n / 2],
            std_dev_ms: std_dev,
            p95_ms: times_ms[(n as f64 * 0.95) as usize],
            p99_ms: times_ms[(n as f64 * 0.99).min(n as f64 - 1.0) as usize],
            iterations: n,
            success_rate: 1.0,
        }
    }
}

impl Default for BenchmarkStats {
    fn default() -> Self {
        Self {
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            std_dev_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            iterations: 0,
            success_rate: 0.0,
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Description
    pub description: String,
    /// Statistics
    pub stats: BenchmarkStats,
    /// Individual timings (if recorded)
    pub individual_times_ms: Vec<f64>,
    /// Number of errors
    pub errors: usize,
    /// Error messages
    pub error_messages: Vec<String>,
    /// When the benchmark was run
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl BenchmarkResult {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            stats: BenchmarkStats::default(),
            individual_times_ms: Vec::new(),
            errors: 0,
            error_messages: Vec::new(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Calculate throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.stats.mean_ms > 0.0 {
            1000.0 / self.stats.mean_ms
        } else {
            0.0
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "{}: mean={:.2}ms, median={:.2}ms, p95={:.2}ms, throughput={:.1}/s",
            self.name,
            self.stats.mean_ms,
            self.stats.median_ms,
            self.stats.p95_ms,
            self.throughput()
        )
    }
}

// ============================================================================
// Benchmark Suite
// ============================================================================

/// Collection of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Suite name
    pub name: String,
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
    /// When the suite was started
    pub started_at: DateTime<Utc>,
    /// When the suite completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Suite-level metadata
    pub metadata: HashMap<String, String>,
}

impl BenchmarkSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Mark suite as complete
    pub fn complete(&mut self) {
        self.completed_at = Some(Utc::now());
    }

    /// Get total runtime
    pub fn total_runtime(&self) -> Duration {
        self.completed_at
            .map(|end| (end - self.started_at).to_std().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Get suite summary
    pub fn summary(&self) -> String {
        let mut summary = format!("Benchmark Suite: {}\n", self.name);
        summary.push_str(&format!("Total benchmarks: {}\n", self.results.len()));

        if let Some(end) = self.completed_at {
            let runtime = (end - self.started_at).num_milliseconds();
            summary.push_str(&format!("Total runtime: {}ms\n", runtime));
        }

        summary.push_str("\nResults:\n");
        for result in &self.results {
            summary.push_str(&format!("  - {}\n", result.summary()));
        }

        summary
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Runner for benchmarks
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run a benchmark
    pub fn run<F>(&self, name: &str, description: &str, mut f: F) -> BenchmarkResult
    where
        F: FnMut() -> Result<(), String>,
    {
        let mut result = BenchmarkResult::new(name, description);
        let mut durations = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = f();
        }

        // Actual runs
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            match f() {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    durations.push(elapsed);

                    if self.config.record_individual {
                        result
                            .individual_times_ms
                            .push(elapsed.as_secs_f64() * 1000.0);
                    }
                }
                Err(e) => {
                    result.errors += 1;
                    result.error_messages.push(e);
                }
            }
        }

        // Calculate stats
        result.stats = BenchmarkStats::from_durations(&durations);
        result.stats.success_rate =
            (self.config.iterations - result.errors) as f64 / self.config.iterations as f64;

        result
    }

    /// Run a benchmark with setup
    pub fn run_with_setup<S, F, T>(
        &self,
        name: &str,
        description: &str,
        setup: S,
        mut f: F,
    ) -> BenchmarkResult
    where
        S: Fn() -> T,
        F: FnMut(T) -> Result<(), String>,
    {
        let mut result = BenchmarkResult::new(name, description);
        let mut durations = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let data = setup();
            let _ = f(data);
        }

        // Actual runs
        for _ in 0..self.config.iterations {
            let data = setup();
            let start = Instant::now();

            match f(data) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    durations.push(elapsed);

                    if self.config.record_individual {
                        result
                            .individual_times_ms
                            .push(elapsed.as_secs_f64() * 1000.0);
                    }
                }
                Err(e) => {
                    result.errors += 1;
                    result.error_messages.push(e);
                }
            }
        }

        result.stats = BenchmarkStats::from_durations(&durations);
        result.stats.success_rate =
            (self.config.iterations - result.errors) as f64 / self.config.iterations as f64;

        result
    }
}

// ============================================================================
// Built-in Benchmarks
// ============================================================================

/// Built-in benchmark for token estimation
pub fn benchmark_token_estimation(runner: &BenchmarkRunner) -> BenchmarkResult {
    runner.run("token_estimation", "Estimate tokens for text", || {
        let text = "This is a sample text for benchmarking token estimation. ".repeat(100);
        let _tokens = crate::context::estimate_tokens(&text);
        Ok(())
    })
}

/// Built-in benchmark for entity extraction
pub fn benchmark_entity_extraction(runner: &BenchmarkRunner) -> BenchmarkResult {
    let extractor =
        crate::entities::EntityExtractor::new(crate::entities::EntityExtractorConfig::default());

    runner.run(
        "entity_extraction",
        "Extract entities from text",
        || {
            let text = "Contact John Smith at john@example.com or visit https://example.com for more info. The project uses Rust v1.75.0.";
            let _entities = extractor.extract(text);
            Ok(())
        },
    )
}

/// Built-in benchmark for quality analysis
pub fn benchmark_quality_analysis(runner: &BenchmarkRunner) -> BenchmarkResult {
    let analyzer = crate::quality::QualityAnalyzer::new(crate::quality::QualityConfig::default());

    runner.run(
        "quality_analysis",
        "Analyze response quality",
        || {
            let query = "How do I create a function in Rust?";
            let response = "To create a function in Rust, you use the `fn` keyword followed by the function name and parameters. Here's an example:\n\n```rust\nfn greet(name: &str) {\n    println!(\"Hello, {}!\", name);\n}\n```\n\nThe function can return values using the -> syntax.";
            let _score = analyzer.analyze(query, response, None);
            Ok(())
        },
    )
}

/// Built-in benchmark for language detection
pub fn benchmark_language_detection(runner: &BenchmarkRunner) -> BenchmarkResult {
    let detector = crate::i18n::LanguageDetector::new();

    runner.run("language_detection", "Detect language of text", || {
        let texts = [
            "The quick brown fox jumps over the lazy dog.",
            "El rápido zorro marrón salta sobre el perro perezoso.",
            "Der schnelle braune Fuchs springt über den faulen Hund.",
            "Le rapide renard brun saute par-dessus le chien paresseux.",
        ];

        for text in texts {
            let _detected = detector.detect(text);
        }
        Ok(())
    })
}

/// Built-in benchmark for sentiment analysis
pub fn benchmark_sentiment_analysis(runner: &BenchmarkRunner) -> BenchmarkResult {
    let analyzer = crate::analysis::SentimentAnalyzer::new();

    runner.run("sentiment_analysis", "Analyze sentiment of text", || {
        let texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. I'm very disappointed with the quality.",
            "The item arrived on time. It works as expected.",
        ];

        for text in texts {
            let _sentiment = analyzer.analyze_message(text);
        }
        Ok(())
    })
}

/// Built-in benchmark for topic detection
pub fn benchmark_topic_detection(runner: &BenchmarkRunner) -> BenchmarkResult {
    let detector = crate::analysis::TopicDetector::new();

    // Create sample messages for topic detection
    let messages = vec![
        crate::ChatMessage::user("We need to fix the authentication bug in the login system.".to_string()),
        crate::ChatMessage::assistant("I'll help you debug that. Can you share the error message?".to_string()),
        crate::ChatMessage::user("The database queries are also slow and need optimization. Can you review the Python code in the API module?".to_string()),
    ];

    runner.run("topic_detection", "Detect topics in messages", || {
        let _topics = detector.detect_topics(&messages);
        Ok(())
    })
}

/// Run all built-in benchmarks
pub fn run_all_benchmarks(config: BenchmarkConfig) -> BenchmarkSuite {
    let runner = BenchmarkRunner::new(config);
    let mut suite = BenchmarkSuite::new("AI Assistant Benchmarks");

    suite.add_result(benchmark_token_estimation(&runner));
    suite.add_result(benchmark_entity_extraction(&runner));
    suite.add_result(benchmark_quality_analysis(&runner));
    suite.add_result(benchmark_language_detection(&runner));
    suite.add_result(benchmark_sentiment_analysis(&runner));
    suite.add_result(benchmark_topic_detection(&runner));

    suite.complete();
    suite
}

// ============================================================================
// Comparison Tools
// ============================================================================

/// Compare two benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Baseline result
    pub baseline_name: String,
    /// New result
    pub new_name: String,
    /// Mean speedup (positive = faster)
    pub mean_speedup_percent: f64,
    /// Median speedup
    pub median_speedup_percent: f64,
    /// P95 change
    pub p95_change_percent: f64,
    /// Summary
    pub summary: String,
}

/// Compare two benchmark results
pub fn compare_results(baseline: &BenchmarkResult, new: &BenchmarkResult) -> BenchmarkComparison {
    let mean_speedup = if baseline.stats.mean_ms > 0.0 {
        ((baseline.stats.mean_ms - new.stats.mean_ms) / baseline.stats.mean_ms) * 100.0
    } else {
        0.0
    };

    let median_speedup = if baseline.stats.median_ms > 0.0 {
        ((baseline.stats.median_ms - new.stats.median_ms) / baseline.stats.median_ms) * 100.0
    } else {
        0.0
    };

    let p95_change = if baseline.stats.p95_ms > 0.0 {
        ((baseline.stats.p95_ms - new.stats.p95_ms) / baseline.stats.p95_ms) * 100.0
    } else {
        0.0
    };

    let direction = if mean_speedup > 0.0 {
        "faster"
    } else {
        "slower"
    };

    let summary = format!(
        "{} vs {}: {:.1}% {} (mean: {:.2}ms -> {:.2}ms)",
        new.name,
        baseline.name,
        mean_speedup.abs(),
        direction,
        baseline.stats.mean_ms,
        new.stats.mean_ms
    );

    BenchmarkComparison {
        baseline_name: baseline.name.clone(),
        new_name: new.name.clone(),
        mean_speedup_percent: mean_speedup,
        median_speedup_percent: median_speedup,
        p95_change_percent: p95_change,
        summary,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_stats() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(14),
        ];

        let stats = BenchmarkStats::from_durations(&durations);

        assert_eq!(stats.iterations, 5);
        assert!(stats.min_ms >= 10.0);
        assert!(stats.max_ms <= 15.0);
        assert!(stats.mean_ms > 10.0 && stats.mean_ms < 15.0);
    }

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new(BenchmarkConfig {
            iterations: 5,
            warmup_iterations: 1,
            ..Default::default()
        });

        let result = runner.run("test", "Test benchmark", || {
            std::thread::sleep(Duration::from_micros(100));
            Ok(())
        });

        assert_eq!(result.stats.iterations, 5);
        assert!(result.stats.mean_ms > 0.0);
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("Test Suite");

        let mut result = BenchmarkResult::new("test", "Test");
        result.stats.mean_ms = 10.0;

        suite.add_result(result);
        suite.complete();

        assert_eq!(suite.results.len(), 1);
        assert!(suite.completed_at.is_some());
    }

    #[test]
    fn test_comparison() {
        let mut baseline = BenchmarkResult::new("baseline", "Baseline");
        baseline.stats.mean_ms = 100.0;
        baseline.stats.median_ms = 95.0;
        baseline.stats.p95_ms = 150.0;

        let mut new = BenchmarkResult::new("new", "New");
        new.stats.mean_ms = 80.0;
        new.stats.median_ms = 75.0;
        new.stats.p95_ms = 120.0;

        let comparison = compare_results(&baseline, &new);

        assert!(comparison.mean_speedup_percent > 0.0);
        assert!(comparison.summary.contains("faster"));
    }

    #[test]
    fn test_run_all_benchmarks() {
        let config = BenchmarkConfig {
            iterations: 3,
            warmup_iterations: 1,
            ..Default::default()
        };

        let suite = run_all_benchmarks(config);

        assert!(suite.results.len() > 0);
        assert!(suite.completed_at.is_some());
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.warmup_iterations, 2);
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.record_individual);
        assert!(!config.measure_memory);
    }

    #[test]
    fn test_benchmark_result_metadata() {
        let mut result = BenchmarkResult::new("test_bench", "A test benchmark");
        result.metadata.insert("version".to_string(), "1.0".to_string());
        assert_eq!(result.name, "test_bench");
        assert_eq!(result.description, "A test benchmark");
        assert_eq!(result.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_benchmark_result_throughput() {
        let mut result = BenchmarkResult::new("throughput_test", "Test");
        result.stats.mean_ms = 10.0;
        let throughput = result.throughput();
        assert!((throughput - 100.0).abs() < 0.01);

        // Zero mean should yield 0 throughput
        result.stats.mean_ms = 0.0;
        assert_eq!(result.throughput(), 0.0);
    }

    #[test]
    fn test_suite_to_json() {
        let mut suite = BenchmarkSuite::new("JSON Suite");
        let result = BenchmarkResult::new("bench1", "First benchmark");
        suite.add_result(result);

        let json = suite.to_json();
        assert!(json.contains("JSON Suite"));
        assert!(json.contains("bench1"));
    }

    #[test]
    fn test_stats_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let stats = BenchmarkStats::from_durations(&durations);
        assert_eq!(stats.iterations, 5);
        assert!((stats.mean_ms - 30.0).abs() < 0.1);
        assert!((stats.min_ms - 10.0).abs() < 0.1);
        assert!((stats.max_ms - 50.0).abs() < 0.1);
        assert_eq!(stats.success_rate, 1.0);
    }
}
