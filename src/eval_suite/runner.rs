//! Benchmark execution engine.
//!
//! Orchestrates LLM calls via a generator callback, collects results with timing and cost
//! tracking, and produces `BenchmarkRunResult` for analysis.

use super::dataset::{BenchmarkDataset, BenchmarkProblem};
use super::scoring::{DefaultScorer, ProblemScorer};
use crate::error::EvalSuiteError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifies a model for cost tracking and comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentifier {
    /// Model name (e.g., "llama3.1:8b", "gpt-4o", "claude-3.5-sonnet")
    pub name: String,
    /// Provider name (e.g., "ollama", "openai", "anthropic")
    pub provider: String,
    /// Optional variant/size (e.g., "8b", "70b")
    pub variant: Option<String>,
}

impl std::fmt::Display for ModelIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.name)?;
        if let Some(ref v) = self.variant {
            write!(f, " ({})", v)?;
        }
        Ok(())
    }
}

/// Token usage for a single LLM call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input (prompt) tokens
    pub input_tokens: usize,
    /// Number of output (completion) tokens
    pub output_tokens: usize,
}

impl TokenUsage {
    /// Total tokens (input + output).
    pub fn total(&self) -> usize {
        self.input_tokens + self.output_tokens
    }
}

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunConfig {
    /// Number of samples per problem for Pass@k (k=1 for accuracy benchmarks)
    pub samples_per_problem: usize,
    /// Temperature for generation
    pub temperature: f32,
    /// Max tokens to generate per response
    pub max_tokens: Option<usize>,
    /// Timeout per problem (seconds)
    pub timeout_secs: u64,
    /// Max retries on error
    pub max_retries: usize,
    /// Model identifier for this run
    pub model_id: ModelIdentifier,
    /// Whether to include chain-of-thought instruction
    pub chain_of_thought: bool,
    /// Custom prompt template (use {prompt} and optionally {system} as placeholders)
    pub prompt_template: Option<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            samples_per_problem: 1,
            temperature: 0.0,
            max_tokens: None,
            timeout_secs: 120,
            max_retries: 1,
            model_id: ModelIdentifier {
                name: "unknown".to_string(),
                provider: "unknown".to_string(),
                variant: None,
            },
            chain_of_thought: false,
            prompt_template: None,
        }
    }
}

/// The result of running one problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemResult {
    /// Problem identifier
    pub problem_id: String,
    /// Model that produced this result
    pub model_id: ModelIdentifier,
    /// All k sample responses
    pub responses: Vec<String>,
    /// Score per sample (0.0-1.0)
    pub scores: Vec<f64>,
    /// Pass/fail per sample
    pub passed: Vec<bool>,
    /// Latency per sample in milliseconds
    pub latencies_ms: Vec<u64>,
    /// Token usage per sample
    pub token_counts: Vec<TokenUsage>,
    /// Cost estimate per sample (USD)
    pub cost_estimates: Vec<f64>,
    /// If the problem failed entirely
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Result of an entire benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResult {
    /// Unique run identifier
    pub run_id: String,
    /// Model used for this run
    pub model_id: ModelIdentifier,
    /// Name of the dataset evaluated
    pub dataset_name: String,
    /// Suite type
    pub suite_type: super::dataset::BenchmarkSuiteType,
    /// Per-problem results
    pub results: Vec<ProblemResult>,
    /// Unix timestamp when the run started
    pub started_at: u64,
    /// Unix timestamp when the run completed
    pub completed_at: u64,
    /// Total cost across all problems
    pub total_cost: f64,
    /// Total token usage across all problems
    pub total_tokens: TokenUsage,
}

impl BenchmarkRunResult {
    /// Calculate accuracy: fraction of problems with at least one passing sample.
    pub fn accuracy(&self) -> f64 {
        super::scoring::accuracy(&self.results)
    }

    /// Calculate mean score across all problems.
    pub fn mean_score(&self) -> f64 {
        super::scoring::mean_score(&self.results)
    }

    /// Number of problems evaluated.
    pub fn problem_count(&self) -> usize {
        self.results.len()
    }

    /// Number of problems that had errors.
    pub fn error_count(&self) -> usize {
        self.results.iter().filter(|r| r.error.is_some()).count()
    }

    /// Mean latency across all samples.
    pub fn mean_latency_ms(&self) -> f64 {
        let all: Vec<u64> = self.results.iter().flat_map(|r| r.latencies_ms.iter().copied()).collect();
        if all.is_empty() {
            return 0.0;
        }
        all.iter().sum::<u64>() as f64 / all.len() as f64
    }
}

// ---------------------------------------------------------------------------
// BenchmarkSuiteRunner
// ---------------------------------------------------------------------------

/// Benchmark execution engine that runs problems against an LLM via a generator callback.
pub struct BenchmarkSuiteRunner {
    /// Generator callback: prompt → LLM response
    generator: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
    /// Scorer for evaluating responses
    scorer: Box<dyn ProblemScorer>,
}

impl BenchmarkSuiteRunner {
    /// Create a new runner with the given generator callback.
    ///
    /// The generator receives a prompt string and should return the LLM's response.
    /// This is the same pattern used by `LlmJudgeEvaluator` in evaluation.rs.
    pub fn new<F>(generator: F) -> Self
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        Self {
            generator: Arc::new(generator),
            scorer: Box::new(DefaultScorer),
        }
    }

    /// Use a custom scorer instead of the default.
    pub fn with_scorer(mut self, scorer: Box<dyn ProblemScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// Run a single problem, producing `samples_per_problem` responses.
    pub fn run_problem(
        &self,
        problem: &BenchmarkProblem,
        config: &RunConfig,
    ) -> ProblemResult {
        let mut responses = Vec::new();
        let mut scores = Vec::new();
        let mut passed = Vec::new();
        let mut latencies_ms = Vec::new();
        let mut token_counts = Vec::new();
        let mut cost_estimates = Vec::new();
        let mut last_error = None;

        let prompt = self.build_prompt(problem, config);

        for _ in 0..config.samples_per_problem {
            let start = std::time::Instant::now();

            let mut result = None;
            for attempt in 0..=config.max_retries {
                match (self.generator)(&prompt) {
                    Ok(response) => {
                        result = Some(Ok(response));
                        break;
                    }
                    Err(e) => {
                        if attempt == config.max_retries {
                            result = Some(Err(e));
                        }
                    }
                }
            }

            let elapsed = start.elapsed().as_millis() as u64;
            latencies_ms.push(elapsed);

            match result {
                Some(Ok(response)) => {
                    let score = self.scorer.score(problem, &response);
                    let pass = self.scorer.passed(problem, &response);

                    // Approximate token count (len/4 heuristic, same as providers.rs)
                    let input_tokens = prompt.len() / 4;
                    let output_tokens = response.len() / 4;
                    token_counts.push(TokenUsage { input_tokens, output_tokens });

                    // Approximate cost (placeholder — real cost would come from CostEstimator)
                    cost_estimates.push(0.0);

                    scores.push(score);
                    passed.push(pass);
                    responses.push(response);
                }
                Some(Err(e)) => {
                    last_error = Some(e.clone());
                    responses.push(String::new());
                    scores.push(0.0);
                    passed.push(false);
                    token_counts.push(TokenUsage::default());
                    cost_estimates.push(0.0);
                }
                None => {
                    last_error = Some("No result from generator".to_string());
                    responses.push(String::new());
                    scores.push(0.0);
                    passed.push(false);
                    token_counts.push(TokenUsage::default());
                    cost_estimates.push(0.0);
                }
            }
        }

        ProblemResult {
            problem_id: problem.id.clone(),
            model_id: config.model_id.clone(),
            responses,
            scores,
            passed,
            latencies_ms,
            token_counts,
            cost_estimates,
            error: last_error,
            metadata: HashMap::new(),
        }
    }

    /// Run an entire dataset, returning aggregated results.
    pub fn run_dataset(
        &self,
        dataset: &BenchmarkDataset,
        config: &RunConfig,
    ) -> Result<BenchmarkRunResult, EvalSuiteError> {
        if dataset.is_empty() {
            return Err(EvalSuiteError::NoResults {
                reason: "Dataset contains no problems".to_string(),
            });
        }

        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut results = Vec::with_capacity(dataset.len());
        for problem in &dataset.problems {
            results.push(self.run_problem(problem, config));
        }

        let completed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let total_cost: f64 = results.iter().flat_map(|r| r.cost_estimates.iter()).sum();
        let total_input: usize = results.iter().flat_map(|r| r.token_counts.iter()).map(|t| t.input_tokens).sum();
        let total_output: usize = results.iter().flat_map(|r| r.token_counts.iter()).map(|t| t.output_tokens).sum();

        Ok(BenchmarkRunResult {
            run_id: uuid::Uuid::new_v4().to_string(),
            model_id: config.model_id.clone(),
            dataset_name: dataset.name.clone(),
            suite_type: dataset.suite_type.clone(),
            results,
            started_at,
            completed_at,
            total_cost,
            total_tokens: TokenUsage {
                input_tokens: total_input,
                output_tokens: total_output,
            },
        })
    }

    /// Build the prompt to send to the LLM, applying templates and options.
    fn build_prompt(&self, problem: &BenchmarkProblem, config: &RunConfig) -> String {
        let base_prompt = &problem.prompt;

        // Apply custom template if provided
        let prompt = if let Some(ref template) = config.prompt_template {
            template
                .replace("{prompt}", base_prompt)
                .replace("{system}", problem.system_prompt.as_deref().unwrap_or(""))
        } else if config.chain_of_thought {
            format!(
                "{}\n\nThink step by step before giving your final answer.",
                base_prompt
            )
        } else {
            base_prompt.clone()
        };

        // Enrich prompt with format-specific context
        self.enrich_prompt(&prompt, &problem.answer_format)
    }

    /// Enrich a prompt with format-specific context for better LLM responses.
    fn enrich_prompt(
        &self,
        prompt: &str,
        answer_format: &super::dataset::AnswerFormat,
    ) -> String {
        use super::dataset::AnswerFormat;
        match answer_format {
            AnswerFormat::CompetitiveProgrammingCode {
                language,
                test_cases,
                time_limit_ms,
                memory_limit_mb,
                ..
            } => {
                let mut enriched = prompt.to_string();
                if let Some(tl) = time_limit_ms {
                    enriched.push_str(&format!("\n\nTime limit: {} ms", tl));
                }
                if let Some(ml) = memory_limit_mb {
                    enriched.push_str(&format!("\nMemory limit: {} MB", ml));
                }
                enriched.push_str(&format!("\nLanguage: {}", language));
                // Show up to 2 sample test cases
                for (i, (input, output)) in test_cases.iter().take(2).enumerate() {
                    enriched.push_str(&format!(
                        "\n\nSample Input {}:\n{}\nSample Output {}:\n{}",
                        i + 1,
                        input,
                        i + 1,
                        output
                    ));
                }
                enriched
            }
            AnswerFormat::CodeEdit {
                language,
                original_code,
                ..
            } => {
                format!(
                    "## Original Code ({})\n```{}\n{}\n```\n\n## Instruction\n{}",
                    language, language, original_code, prompt
                )
            }
            AnswerFormat::TerminalSequence { environment, .. } => {
                format!("## Environment\n{}\n\n## Task\n{}", environment, prompt)
            }
            _ => prompt.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dataset::*;

    fn mock_generator(responses: Vec<&str>) -> Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> {
        let responses: Vec<String> = responses.into_iter().map(|s| s.to_string()).collect();
        let idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        Arc::new(move |_prompt: &str| {
            let i = idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if i < responses.len() {
                Ok(responses[i].clone())
            } else {
                Ok(responses.last().cloned().unwrap_or_default())
            }
        })
    }

    fn failing_generator() -> Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> {
        Arc::new(|_| Err("LLM unavailable".to_string()))
    }

    #[test]
    fn test_run_single_mc_problem() {
        let gen = mock_generator(vec!["B"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_mc_problem("t/1", "What is 2+2? A)3 B)4 C)5 D)6", vec!["A","B","C","D"], "B");
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let result = runner.run_problem(&problem, &config);
        assert_eq!(result.problem_id, "t/1");
        assert_eq!(result.scores.len(), 1);
        assert_eq!(result.scores[0], 1.0);
        assert!(result.passed[0]);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_run_single_numeric_problem() {
        let gen = mock_generator(vec!["The answer is 42"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_numeric_problem("t/1", "What is 6*7?", 42.0, 0.01);
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let result = runner.run_problem(&problem, &config);
        assert_eq!(result.scores[0], 1.0);
    }

    #[test]
    fn test_run_problem_generator_error() {
        let gen = failing_generator();
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_mc_problem("t/1", "Q?", vec!["A","B"], "A");
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let result = runner.run_problem(&problem, &config);
        assert!(result.error.is_some());
        assert_eq!(result.scores[0], 0.0);
        assert!(!result.passed[0]);
    }

    #[test]
    fn test_run_problem_with_retries() {
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let gen: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> = Arc::new(move |_| {
            let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 { Err("transient".into()) } else { Ok("B".into()) }
        });
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_mc_problem("t/1", "Q?", vec!["A","B"], "B");
        let config = RunConfig {
            max_retries: 2,
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };

        let result = runner.run_problem(&problem, &config);
        assert!(result.error.is_none());
        assert_eq!(result.scores[0], 1.0);
    }

    #[test]
    fn test_run_dataset_all_pass() {
        let gen = mock_generator(vec!["B", "B"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, vec![
            make_mc_problem("t/1", "Q1", vec!["A","B"], "B"),
            make_mc_problem("t/2", "Q2", vec!["A","B"], "B"),
        ]);
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let run = runner.run_dataset(&ds, &config).expect("should succeed");
        assert_eq!(run.results.len(), 2);
        assert!((run.accuracy() - 1.0).abs() < 0.01);
        assert_eq!(run.problem_count(), 2);
        assert_eq!(run.error_count(), 0);
    }

    #[test]
    fn test_run_dataset_mixed_results() {
        let gen = mock_generator(vec!["B", "A"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, vec![
            make_mc_problem("t/1", "Q1", vec!["A","B"], "B"),
            make_mc_problem("t/2", "Q2", vec!["A","B"], "B"),
        ]);
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let run = runner.run_dataset(&ds, &config).expect("should succeed");
        assert!((run.accuracy() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_run_dataset_empty() {
        let gen = mock_generator(vec![]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let ds = BenchmarkDataset::from_problems("empty", BenchmarkSuiteType::Mmlu, vec![]);
        let config = RunConfig { model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None }, ..Default::default() };

        let result = runner.run_dataset(&ds, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_config_defaults() {
        let config = RunConfig::default();
        assert_eq!(config.samples_per_problem, 1);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 1);
        assert!(!config.chain_of_thought);
        assert!(config.prompt_template.is_none());
    }

    #[test]
    fn test_model_identifier_display() {
        let m = ModelIdentifier { name: "gpt-4".into(), provider: "openai".into(), variant: None };
        assert_eq!(m.to_string(), "openai/gpt-4");

        let m2 = ModelIdentifier { name: "llama3.1".into(), provider: "ollama".into(), variant: Some("8b".into()) };
        assert_eq!(m2.to_string(), "ollama/llama3.1 (8b)");
    }

    #[test]
    fn test_token_usage() {
        let t = TokenUsage { input_tokens: 100, output_tokens: 50 };
        assert_eq!(t.total(), 150);
    }

    #[test]
    fn test_run_with_chain_of_thought() {
        let prompt_capture = Arc::new(std::sync::Mutex::new(String::new()));
        let pc = prompt_capture.clone();
        let gen: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> = Arc::new(move |prompt: &str| {
            *pc.lock().unwrap() = prompt.to_string();
            Ok("B".into())
        });
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_mc_problem("t/1", "Q?", vec!["A","B"], "B");
        let config = RunConfig {
            chain_of_thought: true,
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };

        runner.run_problem(&problem, &config);
        let captured = prompt_capture.lock().unwrap().clone();
        assert!(captured.contains("step by step"));
    }

    #[test]
    fn test_run_with_custom_template() {
        let prompt_capture = Arc::new(std::sync::Mutex::new(String::new()));
        let pc = prompt_capture.clone();
        let gen: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> = Arc::new(move |prompt: &str| {
            *pc.lock().unwrap() = prompt.to_string();
            Ok("B".into())
        });
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_mc_problem("t/1", "What is 2+2?", vec!["A","B"], "B");
        let config = RunConfig {
            prompt_template: Some("SYSTEM: {system}\nUSER: {prompt}\nANSWER:".to_string()),
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };

        runner.run_problem(&problem, &config);
        let captured = prompt_capture.lock().unwrap().clone();
        assert!(captured.contains("USER: What is 2+2?"));
        assert!(captured.contains("ANSWER:"));
    }

    #[test]
    fn test_benchmark_run_result_stats() {
        let model = ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None };
        let run = BenchmarkRunResult {
            run_id: "r1".into(),
            model_id: model.clone(),
            dataset_name: "test".into(),
            suite_type: BenchmarkSuiteType::Mmlu,
            results: vec![
                ProblemResult {
                    problem_id: "1".into(), model_id: model.clone(),
                    responses: vec!["A".into()], scores: vec![1.0], passed: vec![true],
                    latencies_ms: vec![100], token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 10 }],
                    cost_estimates: vec![0.001], error: None, metadata: HashMap::new(),
                },
                ProblemResult {
                    problem_id: "2".into(), model_id: model.clone(),
                    responses: vec!["wrong".into()], scores: vec![0.0], passed: vec![false],
                    latencies_ms: vec![200], token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 20 }],
                    cost_estimates: vec![0.002], error: Some("err".into()), metadata: HashMap::new(),
                },
            ],
            started_at: 1000,
            completed_at: 1010,
            total_cost: 0.003,
            total_tokens: TokenUsage { input_tokens: 100, output_tokens: 30 },
        };

        assert!((run.accuracy() - 0.5).abs() < 0.01);
        assert!((run.mean_score() - 0.5).abs() < 0.01);
        assert_eq!(run.problem_count(), 2);
        assert_eq!(run.error_count(), 1);
        assert!((run.mean_latency_ms() - 150.0).abs() < 0.01);
    }

    // ── Tests for new benchmark format prompt enrichment ──

    #[test]
    fn test_run_competitive_programming_problem() {
        let gen = mock_generator(vec!["n = int(input())\nprint(sum(range(n)))"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_competitive_problem(
            "cp/1", "Read N and print sum of 0..N-1",
            "n = int(input())\nprint(sum(range(n)))", "python",
            vec![("5", "10"), ("3", "3")], "easy",
        );
        let config = RunConfig {
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };
        let result = runner.run_problem(&problem, &config);
        assert_eq!(result.problem_id, "cp/1");
        assert!(!result.scores.is_empty());
        // The prompt should have been enriched with sample I/O
        let prompt = runner.build_prompt(&problem, &config);
        assert!(prompt.contains("Sample Input 1:"));
        assert!(prompt.contains("Sample Output 1:"));
        assert!(prompt.contains("Language: python"));
    }

    #[test]
    fn test_run_code_edit_problem() {
        let gen = mock_generator(vec!["def divide(a, b):\n    if b == 0: return None\n    return a / b"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_code_edit_problem(
            "ce/1", "Add zero division guard",
            "def divide(a, b):\n    return a / b",
            "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
            "python",
        );
        let config = RunConfig {
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };
        let result = runner.run_problem(&problem, &config);
        assert_eq!(result.problem_id, "ce/1");
        // The prompt should have original code
        let prompt = runner.build_prompt(&problem, &config);
        assert!(prompt.contains("## Original Code (python)"));
        assert!(prompt.contains("def divide(a, b):"));
        assert!(prompt.contains("## Instruction"));
    }

    #[test]
    fn test_run_terminal_problem() {
        let gen = mock_generator(vec!["```bash\n$ find . -name '*.py'\n$ wc -l\n```"]);
        let runner = BenchmarkSuiteRunner { generator: gen, scorer: Box::new(DefaultScorer) };
        let problem = make_terminal_problem(
            "tb/1", "Count lines of Python code",
            "Ubuntu 22.04, bash, ~/project",
            vec!["find . -name '*.py'", "wc -l"],
            "find . -name '*.py' | xargs wc -l",
        );
        let config = RunConfig {
            model_id: ModelIdentifier { name: "test".into(), provider: "mock".into(), variant: None },
            ..Default::default()
        };
        let result = runner.run_problem(&problem, &config);
        assert_eq!(result.problem_id, "tb/1");
        // The prompt should have environment context
        let prompt = runner.build_prompt(&problem, &config);
        assert!(prompt.contains("## Environment"));
        assert!(prompt.contains("Ubuntu 22.04"));
        assert!(prompt.contains("## Task"));
    }
}
