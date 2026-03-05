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
    filter_by_contamination_cutoff, filter_by_language, make_code_edit_problem, make_code_problem,
    make_competitive_problem, make_livecode_problem, make_mc_problem, make_numeric_problem,
    make_terminal_problem, AnswerFormat, BenchmarkDataset, BenchmarkProblem, BenchmarkSuiteType,
    ProblemCategory,
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Register MCP tools for benchmark evaluation.
///
/// Provides 6 MCP tools for running benchmarks, comparing model results,
/// and generating reports — all through the MCP protocol.
///
/// # Tools registered
///
/// - `eval.list_suites` — list available benchmark suite types
/// - `eval.create_dataset` — create an inline benchmark dataset
/// - `eval.run` — run a benchmark against the configured LLM
/// - `eval.get_result` — retrieve a stored run result
/// - `eval.compare` — compare multiple run results
/// - `eval.report` — generate a summary report
pub fn register_eval_tools(
    server: &mut crate::mcp_protocol::McpServer,
    generator: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
) {
    use crate::mcp_protocol::McpTool;

    // Shared state for storing run results and datasets
    let results_store: Arc<Mutex<HashMap<String, BenchmarkRunResult>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let datasets_store: Arc<Mutex<HashMap<String, BenchmarkDataset>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // --- eval.list_suites ---
    server.register_tool(
        McpTool::new("eval.list_suites", "List all available benchmark suite types"),
        move |_args| {
            let suites = vec![
                serde_json::json!({"name": "HumanEval", "description": "Function completion from docstring (164 problems, Pass@k)"}),
                serde_json::json!({"name": "MBPP", "description": "Mostly basic programming problems (974 problems)"}),
                serde_json::json!({"name": "SWE-bench", "description": "Real-world bug fixing from issue descriptions"}),
                serde_json::json!({"name": "MMLU", "description": "Massive multitask language understanding (multiple-choice)"}),
                serde_json::json!({"name": "GSM8K", "description": "Grade school math word problems (chain-of-thought)"}),
                serde_json::json!({"name": "ARC", "description": "AI2 reasoning challenge (abstract reasoning)"}),
                serde_json::json!({"name": "AgentBench", "description": "Multi-step agent task completion"}),
                serde_json::json!({"name": "TaskBench", "description": "Tool usage and orchestration evaluation"}),
                serde_json::json!({"name": "GAIA", "description": "General AI assistant tasks requiring multiple tools"}),
                serde_json::json!({"name": "LiveCodeBench", "description": "Contamination-aware competitive programming"}),
                serde_json::json!({"name": "Aider-Polyglot", "description": "Multi-language code editing evaluation"}),
                serde_json::json!({"name": "Terminal-Bench", "description": "Complex SWE tasks in real terminal environments"}),
                serde_json::json!({"name": "APPS", "description": "Introductory-to-competition level programming"}),
                serde_json::json!({"name": "CodeContests", "description": "Competitive programming from Google DeepMind"}),
            ];

            Ok(serde_json::json!({
                "suites": suites,
                "count": suites.len(),
            }))
        },
    );

    // --- eval.create_dataset ---
    let ds_store = datasets_store.clone();
    server.register_tool(
        McpTool::new("eval.create_dataset", "Create an inline benchmark dataset with custom problems")
            .with_property("name", "string", "Dataset name", true)
            .with_property("suite_type", "string", "Suite type: humaneval, mmlu, gsm8k, custom, etc.", false)
            .with_property("num_problems", "number", "Number of sample multiple-choice problems to generate (default 5)", false),
        move |args| {
            let name = args.get("name").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: name")?;
            let num = args.get("num_problems").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

            let suite_type = match args.get("suite_type").and_then(|v| v.as_str()).unwrap_or("mmlu") {
                "humaneval" => BenchmarkSuiteType::HumanEval,
                "mbpp" => BenchmarkSuiteType::Mbpp,
                "mmlu" => BenchmarkSuiteType::Mmlu,
                "gsm8k" => BenchmarkSuiteType::Gsm8k,
                "arc" => BenchmarkSuiteType::Arc,
                other => BenchmarkSuiteType::Custom(other.to_string()),
            };

            // Generate sample problems
            let mut problems = Vec::new();
            for i in 0..num {
                problems.push(make_mc_problem(
                    &format!("{}-{}", name, i + 1),
                    &format!("Sample question {} for {}", i + 1, name),
                    vec!["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                    "A",
                ));
            }

            let dataset = BenchmarkDataset::from_problems(name, suite_type, problems);
            let problem_count = dataset.len();

            ds_store.lock().map_err(|e| e.to_string())?
                .insert(name.to_string(), dataset);

            Ok(serde_json::json!({
                "status": "created",
                "name": name,
                "problem_count": problem_count,
            }))
        },
    );

    // --- eval.run ---
    let gen = generator.clone();
    let rs = results_store.clone();
    let ds = datasets_store.clone();
    server.register_tool(
        McpTool::new("eval.run", "Run a benchmark evaluation against the configured LLM")
            .with_property("dataset", "string", "Dataset name (must be created first with eval.create_dataset)", true)
            .with_property("model_name", "string", "Model name for tracking (default: 'default')", false)
            .with_property("provider", "string", "Provider name for tracking (default: 'local')", false)
            .with_property("samples_per_problem", "number", "Samples per problem for Pass@k (default 1)", false)
            .with_property("temperature", "number", "Generation temperature (default 0.0)", false),
        move |args| {
            let dataset_name = args.get("dataset").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: dataset")?;

            let datasets = ds.lock().map_err(|e| e.to_string())?;
            let dataset = datasets.get(dataset_name)
                .ok_or_else(|| format!("Dataset '{}' not found. Create it first with eval.create_dataset", dataset_name))?;

            let model_name = args.get("model_name").and_then(|v| v.as_str()).unwrap_or("default");
            let provider = args.get("provider").and_then(|v| v.as_str()).unwrap_or("local");
            let samples = args.get("samples_per_problem").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let temp = args.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

            let config = RunConfig {
                samples_per_problem: samples,
                temperature: temp,
                model_id: ModelIdentifier {
                    name: model_name.to_string(),
                    provider: provider.to_string(),
                    variant: None,
                },
                ..Default::default()
            };

            let runner = BenchmarkSuiteRunner::new({
                let g = gen.clone();
                move |prompt: &str| g(prompt)
            });

            let run_result = runner.run_dataset(dataset, &config)
                .map_err(|e| format!("Benchmark run failed: {}", e))?;

            let run_id = run_result.run_id.clone();
            let accuracy = run_result.accuracy();
            let mean_score = run_result.mean_score();
            let problem_count = run_result.problem_count();
            let error_count = run_result.error_count();
            let mean_latency = run_result.mean_latency_ms();
            let total_cost = run_result.total_cost;

            rs.lock().map_err(|e| e.to_string())?
                .insert(run_id.clone(), run_result);

            Ok(serde_json::json!({
                "run_id": run_id,
                "accuracy": accuracy,
                "mean_score": mean_score,
                "problem_count": problem_count,
                "error_count": error_count,
                "mean_latency_ms": mean_latency,
                "total_cost": total_cost,
            }))
        },
    );

    // --- eval.get_result ---
    let rs = results_store.clone();
    server.register_tool(
        McpTool::new("eval.get_result", "Get detailed results from a specific benchmark run")
            .with_property("run_id", "string", "The run ID returned by eval.run", true),
        move |args| {
            let run_id = args.get("run_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: run_id")?;

            let store = rs.lock().map_err(|e| e.to_string())?;
            let result = store.get(run_id)
                .ok_or_else(|| format!("Run '{}' not found", run_id))?;

            Ok(serde_json::json!({
                "run_id": result.run_id,
                "model": result.model_id.name,
                "provider": result.model_id.provider,
                "dataset": result.dataset_name,
                "accuracy": result.accuracy(),
                "mean_score": result.mean_score(),
                "problem_count": result.problem_count(),
                "error_count": result.error_count(),
                "mean_latency_ms": result.mean_latency_ms(),
                "total_cost": result.total_cost,
                "total_tokens": result.total_tokens.total(),
            }))
        },
    );

    // --- eval.compare ---
    let rs = results_store.clone();
    server.register_tool(
        McpTool::new("eval.compare", "Compare results from multiple benchmark runs")
            .with_property("run_ids", "string", "Comma-separated run IDs to compare", true),
        move |args| {
            let run_ids_str = args.get("run_ids").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: run_ids")?;

            let ids: Vec<&str> = run_ids_str.split(',').map(|s| s.trim()).collect();
            if ids.len() < 2 {
                return Err("Need at least 2 run IDs to compare".to_string());
            }

            let store = rs.lock().map_err(|e| e.to_string())?;
            let mut runs = Vec::new();
            for id in &ids {
                let result = store.get(*id)
                    .ok_or_else(|| format!("Run '{}' not found", id))?;
                runs.push(result.clone());
            }

            let config = ComparisonConfig::default();
            let matrix = ComparisonMatrix::from_runs(&runs, &config);

            let best = matrix.best_per_metric();
            let best_json: serde_json::Value = best.iter()
                .map(|(metric, model)| (metric.clone(), serde_json::json!(model.to_string())))
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();

            Ok(serde_json::json!({
                "models": matrix.models.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
                "metrics": matrix.metrics,
                "scores": matrix.scores,
                "elo_ratings": matrix.elo_ratings,
                "best_per_metric": best_json,
            }))
        },
    );

    // --- eval.report ---
    let rs = results_store.clone();
    server.register_tool(
        McpTool::new("eval.report", "Generate a summary evaluation report")
            .with_property("title", "string", "Report title", true)
            .with_property("run_ids", "string", "Comma-separated run IDs to include", true),
        move |args| {
            let title = args.get("title").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: title")?;
            let run_ids_str = args.get("run_ids").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: run_ids")?;

            let ids: Vec<&str> = run_ids_str.split(',').map(|s| s.trim()).collect();
            let store = rs.lock().map_err(|e| e.to_string())?;

            let mut builder = ReportBuilder::new(title);
            for id in &ids {
                let result = store.get(*id)
                    .ok_or_else(|| format!("Run '{}' not found", id))?;
                builder = builder.add_run(result.clone());
            }

            // Add comparison if multiple runs
            if ids.len() >= 2 {
                let runs: Vec<BenchmarkRunResult> = ids.iter()
                    .filter_map(|id| store.get(*id).cloned())
                    .collect();
                let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());
                builder = builder.with_comparison(matrix);
            }

            let report = builder.build();
            let json = report.to_json();

            Ok(serde_json::json!({
                "report_id": report.report_id,
                "title": report.title,
                "runs_count": report.runs.len(),
                "has_comparison": report.comparison.is_some(),
                "report_json": json,
            }))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_eval_server() -> crate::mcp_protocol::McpServer {
        let mut server = crate::mcp_protocol::McpServer::new("test-eval", "1.0.0");
        // Mock generator that returns "A" for all prompts (correct for our MC problems)
        let generator: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> =
            Arc::new(|_prompt| Ok("A".to_string()));
        register_eval_tools(&mut server, generator);
        server
    }

    fn call_eval_tool(server: &crate::mcp_protocol::McpServer, name: &str, args: serde_json::Value) -> serde_json::Value {
        use crate::mcp_protocol::McpRequest;
        let init = McpRequest::new("initialize")
            .with_id(0u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }));
        server.handle_request(init);

        let req = McpRequest::new("tools/call")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "name": name,
                "arguments": args,
            }));
        let resp = server.handle_request(req);
        resp.result.unwrap_or_default()
    }

    fn extract_eval_text(result: &serde_json::Value) -> serde_json::Value {
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            if let Some(first) = content.first() {
                if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                    return serde_json::from_str(text).unwrap_or_default();
                }
            }
        }
        serde_json::Value::Null
    }

    #[test]
    fn test_eval_mcp_list_suites() {
        let server = make_test_eval_server();
        let result = call_eval_tool(&server, "eval.list_suites", serde_json::json!({}));
        let data = extract_eval_text(&result);
        let suites = data["suites"].as_array().unwrap();
        assert!(suites.len() >= 14);
        assert!(suites.iter().any(|s| s["name"] == "HumanEval"));
        assert!(suites.iter().any(|s| s["name"] == "MMLU"));
    }

    #[test]
    fn test_eval_mcp_create_dataset() {
        let server = make_test_eval_server();
        let result = call_eval_tool(&server, "eval.create_dataset", serde_json::json!({
            "name": "test-ds",
            "num_problems": 3,
        }));
        let data = extract_eval_text(&result);
        assert_eq!(data["status"], "created");
        assert_eq!(data["name"], "test-ds");
        assert_eq!(data["problem_count"], 3);
    }

    #[test]
    fn test_eval_mcp_run_basic() {
        let server = make_test_eval_server();
        // Create dataset first
        call_eval_tool(&server, "eval.create_dataset", serde_json::json!({
            "name": "bench-1",
            "num_problems": 3,
        }));
        // Run benchmark
        let result = call_eval_tool(&server, "eval.run", serde_json::json!({
            "dataset": "bench-1",
            "model_name": "test-model",
            "provider": "mock",
        }));
        let data = extract_eval_text(&result);
        assert!(data.get("run_id").is_some());
        assert_eq!(data["problem_count"], 3);
    }

    #[test]
    fn test_eval_mcp_get_result() {
        let server = make_test_eval_server();
        call_eval_tool(&server, "eval.create_dataset", serde_json::json!({"name": "ds-get", "num_problems": 2}));
        let run_result = call_eval_tool(&server, "eval.run", serde_json::json!({"dataset": "ds-get", "model_name": "m1"}));
        let run_data = extract_eval_text(&run_result);
        let run_id = run_data["run_id"].as_str().unwrap();

        let result = call_eval_tool(&server, "eval.get_result", serde_json::json!({"run_id": run_id}));
        let data = extract_eval_text(&result);
        assert_eq!(data["model"], "m1");
        assert_eq!(data["problem_count"], 2);
    }

    #[test]
    fn test_eval_mcp_run_not_found() {
        let server = make_test_eval_server();
        let result = call_eval_tool(&server, "eval.get_result", serde_json::json!({"run_id": "nonexistent-id"}));
        // Should be an error response — extract_eval_text returns Null on error
        let data = extract_eval_text(&result);
        // When a tool returns Err, the server wraps it as an error response
        // So data should be Null or the error info should be in the response
        assert!(data.is_null() || result.get("error").is_some() || data.get("run_id").is_none());
    }

    #[test]
    fn test_eval_mcp_compare_runs() {
        let server = make_test_eval_server();
        // Create and run two benchmarks
        call_eval_tool(&server, "eval.create_dataset", serde_json::json!({"name": "ds-cmp", "num_problems": 3}));
        let r1 = extract_eval_text(&call_eval_tool(&server, "eval.run", serde_json::json!({"dataset": "ds-cmp", "model_name": "model-a", "provider": "p1"})));
        let r2 = extract_eval_text(&call_eval_tool(&server, "eval.run", serde_json::json!({"dataset": "ds-cmp", "model_name": "model-b", "provider": "p2"})));

        let id1 = r1["run_id"].as_str().unwrap();
        let id2 = r2["run_id"].as_str().unwrap();

        let result = call_eval_tool(&server, "eval.compare", serde_json::json!({
            "run_ids": format!("{},{}", id1, id2),
        }));
        let data = extract_eval_text(&result);
        let models = data["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert!(data.get("elo_ratings").is_some());
        assert!(data.get("best_per_metric").is_some());
    }

    #[test]
    fn test_eval_mcp_report() {
        let server = make_test_eval_server();
        call_eval_tool(&server, "eval.create_dataset", serde_json::json!({"name": "ds-rpt", "num_problems": 2}));
        let r1 = extract_eval_text(&call_eval_tool(&server, "eval.run", serde_json::json!({"dataset": "ds-rpt", "model_name": "m1"})));
        let run_id = r1["run_id"].as_str().unwrap();

        let result = call_eval_tool(&server, "eval.report", serde_json::json!({
            "title": "Test Report",
            "run_ids": run_id,
        }));
        let data = extract_eval_text(&result);
        assert_eq!(data["title"], "Test Report");
        assert_eq!(data["runs_count"], 1);
        assert!(data.get("report_json").is_some());
    }

    #[test]
    fn test_eval_mcp_run_with_config_change() {
        // Run benchmark, change model name (simulating AI changing config mid-way), run again
        let server = make_test_eval_server();
        call_eval_tool(&server, "eval.create_dataset", serde_json::json!({"name": "ds-cfg", "num_problems": 2}));

        // First run with model-a
        let r1 = extract_eval_text(&call_eval_tool(&server, "eval.run", serde_json::json!({
            "dataset": "ds-cfg",
            "model_name": "model-a",
            "provider": "ollama",
        })));
        assert_eq!(r1["problem_count"], 2);

        // Second run with model-b (simulates switching model mid-session)
        let r2 = extract_eval_text(&call_eval_tool(&server, "eval.run", serde_json::json!({
            "dataset": "ds-cfg",
            "model_name": "model-b",
            "provider": "openai",
            "temperature": 0.5,
        })));
        assert_eq!(r2["problem_count"], 2);

        // Verify both results are stored and have different model names
        let id1 = r1["run_id"].as_str().unwrap();
        let id2 = r2["run_id"].as_str().unwrap();
        assert_ne!(id1, id2);

        let det1 = extract_eval_text(&call_eval_tool(&server, "eval.get_result", serde_json::json!({"run_id": id1})));
        let det2 = extract_eval_text(&call_eval_tool(&server, "eval.get_result", serde_json::json!({"run_id": id2})));
        assert_eq!(det1["model"], "model-a");
        assert_eq!(det1["provider"], "ollama");
        assert_eq!(det2["model"], "model-b");
        assert_eq!(det2["provider"], "openai");
    }
}
