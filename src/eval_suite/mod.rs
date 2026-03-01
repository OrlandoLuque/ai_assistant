//! Evaluation benchmark suite for running standard AI benchmarks against real LLMs.
//!
//! This module provides a comprehensive evaluation framework that supports:
//! - Standard benchmark suites (HumanEval, MMLU, GSM8K, SWE-bench, etc.)
//! - Multi-model comparison with statistical significance testing
//! - Per-subtask performance analysis with optimal model routing
//! - Ablation studies for measuring technique/concept impact
//! - Report generation with cost breakdowns
//!
//! # Architecture
//!
//! The system uses a callback-based LLM integration pattern (same as `LlmJudgeEvaluator`):
//! ```ignore
//! let generator = |prompt: &str| -> Result<String, String> {
//!     // Call any LLM provider here
//!     Ok("response".to_string())
//! };
//! let runner = BenchmarkSuiteRunner::new(generator);
//! ```
//!
//! This decouples the evaluation logic from provider internals and enables
//! testing with mock generators.

mod dataset;
mod scoring;
mod runner;
mod comparison;
mod ablation;
mod subtask;
mod report;
mod agent_config;
mod config_search;
mod feature_combos;

pub use dataset::{
    make_code_problem, make_mc_problem, make_numeric_problem, AnswerFormat, BenchmarkDataset,
    BenchmarkProblem, BenchmarkSuiteType, ProblemCategory,
};

pub use scoring::{
    accuracy, mean_score, pass_at_k, DefaultScorer, EloCalculator, ProblemScorer,
};

pub use runner::{
    BenchmarkRunResult, BenchmarkSuiteRunner, ModelIdentifier, ProblemResult, RunConfig,
    TokenUsage,
};

pub use comparison::{ComparisonConfig, ComparisonMatrix};

pub use ablation::{AblationEngine, AblationRecommendation, AblationResult, AblationStudy, RunSummary};

pub use subtask::{Subtask, SubtaskAnalysis, SubtaskAnalyzer, SubtaskPerformance};

pub use report::{CostBreakdown, EvalSuiteReport, ReportBuilder, ReportSummary};

pub use agent_config::{
    ConfigMeasurement, EvalAgentConfig, MultiModelGenerator, SearchDimension,
};

pub use config_search::{
    ConfigSearchConfig, ConfigSearchEngine, ConfigSearchResult, EvolutionSnapshot, SearchCost,
    SearchIteration, SearchObjective,
};
