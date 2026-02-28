//! Benchmark dataset loading and problem definitions.
//!
//! Supports loading benchmark problems from JSONL and JSON files in a standardized format
//! compatible with common AI benchmarks (HumanEval, MMLU, GSM8K, etc.).

use crate::error::EvalSuiteError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Standard benchmark suite types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkSuiteType {
    /// HumanEval: function completion from docstring (164 problems, Pass@k)
    HumanEval,
    /// MBPP: mostly basic programming problems (974 problems)
    Mbpp,
    /// SWE-bench: real-world bug fixing from issue descriptions
    SweBench,
    /// MMLU: massive multitask language understanding (multiple-choice)
    Mmlu,
    /// GSM8K: grade school math word problems (chain-of-thought)
    Gsm8k,
    /// ARC: AI2 reasoning challenge (abstract reasoning)
    Arc,
    /// AgentBench: multi-step agent task completion
    AgentBench,
    /// TaskBench: tool usage and orchestration evaluation
    TaskBench,
    /// GAIA: general AI assistant tasks requiring multiple tools
    Gaia,
    /// Custom user-defined benchmark suite
    Custom(String),
}

impl std::fmt::Display for BenchmarkSuiteType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HumanEval => write!(f, "HumanEval"),
            Self::Mbpp => write!(f, "MBPP"),
            Self::SweBench => write!(f, "SWE-bench"),
            Self::Mmlu => write!(f, "MMLU"),
            Self::Gsm8k => write!(f, "GSM8K"),
            Self::Arc => write!(f, "ARC"),
            Self::AgentBench => write!(f, "AgentBench"),
            Self::TaskBench => write!(f, "TaskBench"),
            Self::Gaia => write!(f, "GAIA"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Problem category for cross-suite grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemCategory {
    /// Code generation, completion, bug fixing
    Coding,
    /// Mathematical problem solving
    Math,
    /// Logical and abstract reasoning
    Reasoning,
    /// Factual knowledge questions
    Knowledge,
    /// Multi-step agent task completion
    AgentTask,
    /// Tool selection and usage
    ToolUse,
    /// Custom user-defined category
    Custom(String),
}

impl std::fmt::Display for ProblemCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coding => write!(f, "Coding"),
            Self::Math => write!(f, "Math"),
            Self::Reasoning => write!(f, "Reasoning"),
            Self::Knowledge => write!(f, "Knowledge"),
            Self::AgentTask => write!(f, "AgentTask"),
            Self::ToolUse => write!(f, "ToolUse"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Answer format expected from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnswerFormat {
    /// Free-form code generation (evaluated by reference matching or test execution)
    Code {
        /// Programming language (e.g., "python", "rust")
        language: String,
    },
    /// Multiple choice (evaluated against correct answer letter)
    MultipleChoice {
        /// Available options (e.g., ["A", "B", "C", "D"])
        options: Vec<String>,
        /// Correct answer (e.g., "B")
        correct: String,
    },
    /// Numeric answer (evaluated with tolerance)
    Numeric {
        /// Correct numeric answer
        correct: f64,
        /// Acceptable tolerance (absolute)
        tolerance: f64,
    },
    /// Free text (evaluated by exact/fuzzy match or LLM judge)
    FreeText,
    /// Agent trajectory (evaluated by task completion criteria)
    AgentTrajectory {
        /// Description of what constitutes success
        success_criteria: String,
    },
}

// ---------------------------------------------------------------------------
// Core structs
// ---------------------------------------------------------------------------

/// A single benchmark problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProblem {
    /// Unique identifier (e.g., "humaneval/0", "mmlu/abstract_algebra/5")
    pub id: String,
    /// Which benchmark suite this problem belongs to
    pub suite: BenchmarkSuiteType,
    /// Problem category for cross-suite analysis
    pub category: ProblemCategory,
    /// The prompt to send to the LLM
    pub prompt: String,
    /// Optional system prompt override
    pub system_prompt: Option<String>,
    /// Expected answer format for scoring
    pub answer_format: AnswerFormat,
    /// Reference solution (if available) for comparison scoring
    pub reference_solution: Option<String>,
    /// Test cases (for code problems) to validate the solution
    pub test_cases: Option<String>,
    /// Arbitrary metadata (e.g., source, domain, subtopic)
    pub metadata: HashMap<String, String>,
    /// Difficulty level (e.g., "easy", "medium", "hard")
    pub difficulty: Option<String>,
    /// Tags for filtering (e.g., ["loops", "recursion", "string"])
    pub tags: Vec<String>,
}

/// A loaded benchmark dataset containing multiple problems.
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    /// Dataset name (e.g., "HumanEval", "MMLU-Abstract-Algebra")
    pub name: String,
    /// Suite type
    pub suite_type: BenchmarkSuiteType,
    /// All problems in the dataset
    pub problems: Vec<BenchmarkProblem>,
    /// Dataset-level metadata
    pub metadata: HashMap<String, String>,
}

impl BenchmarkDataset {
    /// Load a benchmark dataset from a JSONL file (one problem per line).
    pub fn from_jsonl(path: &str, suite_type: BenchmarkSuiteType) -> Result<Self, EvalSuiteError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            EvalSuiteError::DatasetLoadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            }
        })?;

        let mut problems = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let problem: BenchmarkProblem =
                serde_json::from_str(trimmed).map_err(|e| EvalSuiteError::InvalidProblem {
                    problem_id: format!("line_{}", line_num + 1),
                    reason: e.to_string(),
                })?;
            problems.push(problem);
        }

        if problems.is_empty() {
            return Err(EvalSuiteError::DatasetLoadFailed {
                path: path.to_string(),
                reason: "Dataset contains no valid problems".to_string(),
            });
        }

        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            name,
            suite_type,
            problems,
            metadata: HashMap::new(),
        })
    }

    /// Load a benchmark dataset from a JSON file (array of problems).
    pub fn from_json(path: &str, suite_type: BenchmarkSuiteType) -> Result<Self, EvalSuiteError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            EvalSuiteError::DatasetLoadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            }
        })?;

        let problems: Vec<BenchmarkProblem> =
            serde_json::from_str(&content).map_err(|e| EvalSuiteError::DatasetLoadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })?;

        if problems.is_empty() {
            return Err(EvalSuiteError::DatasetLoadFailed {
                path: path.to_string(),
                reason: "Dataset contains no problems".to_string(),
            });
        }

        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            name,
            suite_type,
            problems,
            metadata: HashMap::new(),
        })
    }

    /// Create a dataset from an inline vector of problems (useful for tests).
    pub fn from_problems(
        name: &str,
        suite_type: BenchmarkSuiteType,
        problems: Vec<BenchmarkProblem>,
    ) -> Self {
        Self {
            name: name.to_string(),
            suite_type,
            problems,
            metadata: HashMap::new(),
        }
    }

    /// Filter problems by a predicate, returning a new dataset.
    pub fn filter(&self, predicate: impl Fn(&BenchmarkProblem) -> bool) -> Self {
        Self {
            name: self.name.clone(),
            suite_type: self.suite_type.clone(),
            problems: self.problems.iter().filter(|p| predicate(p)).cloned().collect(),
            metadata: self.metadata.clone(),
        }
    }

    /// Take a deterministic random sample of N problems.
    ///
    /// Uses a simple seeded shuffle (Fisher-Yates with LCG) for reproducibility.
    pub fn sample(&self, n: usize, seed: u64) -> Self {
        let count = n.min(self.problems.len());
        let mut indices: Vec<usize> = (0..self.problems.len()).collect();

        // Simple LCG PRNG for deterministic shuffling
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let sampled: Vec<BenchmarkProblem> = indices[..count]
            .iter()
            .map(|&i| self.problems[i].clone())
            .collect();

        Self {
            name: format!("{}_sample_{}", self.name, count),
            suite_type: self.suite_type.clone(),
            problems: sampled,
            metadata: self.metadata.clone(),
        }
    }

    /// Number of problems in this dataset.
    pub fn len(&self) -> usize {
        self.problems.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    /// Get unique categories present in this dataset.
    pub fn categories(&self) -> Vec<ProblemCategory> {
        let mut cats: Vec<ProblemCategory> = self
            .problems
            .iter()
            .map(|p| p.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cats.sort_by_key(|c| c.to_string());
        cats
    }

    /// Get unique difficulty levels present in this dataset.
    pub fn difficulties(&self) -> Vec<String> {
        let mut diffs: Vec<String> = self
            .problems
            .iter()
            .filter_map(|p| p.difficulty.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        diffs.sort();
        diffs
    }
}

// ---------------------------------------------------------------------------
// Helper: create a simple problem for testing
// ---------------------------------------------------------------------------

/// Create a simple multiple-choice problem (convenience for tests and examples).
pub fn make_mc_problem(
    id: &str,
    prompt: &str,
    options: Vec<&str>,
    correct: &str,
) -> BenchmarkProblem {
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::Mmlu,
        category: ProblemCategory::Knowledge,
        prompt: prompt.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::MultipleChoice {
            options: options.into_iter().map(|s| s.to_string()).collect(),
            correct: correct.to_string(),
        },
        reference_solution: None,
        test_cases: None,
        metadata: HashMap::new(),
        difficulty: None,
        tags: Vec::new(),
    }
}

/// Create a simple numeric problem (convenience for tests and examples).
pub fn make_numeric_problem(
    id: &str,
    prompt: &str,
    correct: f64,
    tolerance: f64,
) -> BenchmarkProblem {
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::Gsm8k,
        category: ProblemCategory::Math,
        prompt: prompt.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::Numeric { correct, tolerance },
        reference_solution: Some(correct.to_string()),
        test_cases: None,
        metadata: HashMap::new(),
        difficulty: None,
        tags: Vec::new(),
    }
}

/// Create a simple code problem (convenience for tests and examples).
pub fn make_code_problem(
    id: &str,
    prompt: &str,
    reference: &str,
    language: &str,
) -> BenchmarkProblem {
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::HumanEval,
        category: ProblemCategory::Coding,
        prompt: prompt.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::Code {
            language: language.to_string(),
        },
        reference_solution: Some(reference.to_string()),
        test_cases: None,
        metadata: HashMap::new(),
        difficulty: None,
        tags: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_problems() -> Vec<BenchmarkProblem> {
        vec![
            make_mc_problem("mmlu/1", "What is 2+2?", vec!["3", "4", "5", "6"], "B"),
            make_mc_problem("mmlu/2", "Capital of France?", vec!["London", "Paris", "Berlin", "Rome"], "B"),
            make_numeric_problem("gsm8k/1", "If Alice has 5 apples and gives 2 away, how many?", 3.0, 0.01),
            make_code_problem("humaneval/0", "Write a function that adds two numbers", "def add(a, b): return a + b", "python"),
            BenchmarkProblem {
                id: "custom/1".to_string(),
                suite: BenchmarkSuiteType::Custom("MyBench".to_string()),
                category: ProblemCategory::Reasoning,
                prompt: "What comes next: 1, 1, 2, 3, 5, ?".to_string(),
                system_prompt: Some("You are a math tutor.".to_string()),
                answer_format: AnswerFormat::Numeric { correct: 8.0, tolerance: 0.0 },
                reference_solution: Some("8".to_string()),
                test_cases: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), "fibonacci".to_string());
                    m
                },
                difficulty: Some("easy".to_string()),
                tags: vec!["sequences".to_string(), "fibonacci".to_string()],
            },
        ]
    }

    #[test]
    fn test_from_problems_inline() {
        let problems = sample_problems();
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, problems.clone());
        assert_eq!(ds.name, "test");
        assert_eq!(ds.len(), 5);
        assert!(!ds.is_empty());
        assert_eq!(ds.suite_type, BenchmarkSuiteType::Mmlu);
    }

    #[test]
    fn test_filter_by_category() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let math_only = ds.filter(|p| p.category == ProblemCategory::Math);
        assert_eq!(math_only.len(), 1);
        assert_eq!(math_only.problems[0].id, "gsm8k/1");
    }

    #[test]
    fn test_filter_by_difficulty() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let easy = ds.filter(|p| p.difficulty.as_deref() == Some("easy"));
        assert_eq!(easy.len(), 1);
        assert_eq!(easy.problems[0].id, "custom/1");
    }

    #[test]
    fn test_filter_empty_result() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let empty = ds.filter(|_| false);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_sample_deterministic() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let s1 = ds.sample(3, 42);
        let s2 = ds.sample(3, 42);
        assert_eq!(s1.len(), 3);
        assert_eq!(s2.len(), 3);
        // Same seed → same results
        for i in 0..3 {
            assert_eq!(s1.problems[i].id, s2.problems[i].id);
        }
    }

    #[test]
    fn test_sample_different_seeds() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let s1 = ds.sample(3, 42);
        let s2 = ds.sample(3, 99);
        // Different seeds should (very likely) produce different orderings
        let ids1: Vec<&str> = s1.problems.iter().map(|p| p.id.as_str()).collect();
        let ids2: Vec<&str> = s2.problems.iter().map(|p| p.id.as_str()).collect();
        // At least one should differ (probabilistic, but extremely likely with different seeds)
        assert!(ids1 != ids2 || ds.len() <= 1);
    }

    #[test]
    fn test_sample_larger_than_dataset() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let s = ds.sample(100, 42);
        assert_eq!(s.len(), ds.len()); // Capped at dataset size
    }

    #[test]
    fn test_categories() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let cats = ds.categories();
        assert!(cats.len() >= 3); // Knowledge, Math, Coding, Reasoning
    }

    #[test]
    fn test_difficulties() {
        let ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        let diffs = ds.difficulties();
        assert!(diffs.contains(&"easy".to_string()));
    }

    #[test]
    fn test_suite_type_display() {
        assert_eq!(BenchmarkSuiteType::HumanEval.to_string(), "HumanEval");
        assert_eq!(BenchmarkSuiteType::Mmlu.to_string(), "MMLU");
        assert_eq!(BenchmarkSuiteType::Gsm8k.to_string(), "GSM8K");
        assert_eq!(BenchmarkSuiteType::SweBench.to_string(), "SWE-bench");
        assert_eq!(BenchmarkSuiteType::Custom("MyBench".into()).to_string(), "MyBench");
    }

    #[test]
    fn test_problem_category_display() {
        assert_eq!(ProblemCategory::Coding.to_string(), "Coding");
        assert_eq!(ProblemCategory::Math.to_string(), "Math");
        assert_eq!(ProblemCategory::Custom("Special".into()).to_string(), "Special");
    }

    #[test]
    fn test_answer_format_variants() {
        let code = AnswerFormat::Code { language: "python".to_string() };
        let mc = AnswerFormat::MultipleChoice { options: vec!["A".into()], correct: "A".into() };
        let num = AnswerFormat::Numeric { correct: 42.0, tolerance: 0.1 };
        let free = AnswerFormat::FreeText;
        let agent = AnswerFormat::AgentTrajectory { success_criteria: "done".to_string() };
        // Just verify they can be serialized
        assert!(serde_json::to_string(&code).is_ok());
        assert!(serde_json::to_string(&mc).is_ok());
        assert!(serde_json::to_string(&num).is_ok());
        assert!(serde_json::to_string(&free).is_ok());
        assert!(serde_json::to_string(&agent).is_ok());
    }

    #[test]
    fn test_problem_serialization_roundtrip() {
        let problem = make_mc_problem("test/1", "What is 1+1?", vec!["1", "2", "3"], "B");
        let json = serde_json::to_string(&problem).expect("serialize");
        let back: BenchmarkProblem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "test/1");
        assert_eq!(back.prompt, "What is 1+1?");
    }

    #[test]
    fn test_dataset_metadata() {
        let mut ds = BenchmarkDataset::from_problems("test", BenchmarkSuiteType::Mmlu, sample_problems());
        ds.metadata.insert("version".to_string(), "1.0".to_string());
        assert_eq!(ds.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_from_jsonl_file_not_found() {
        let result = BenchmarkDataset::from_jsonl("/nonexistent/path.jsonl", BenchmarkSuiteType::Mmlu);
        assert!(result.is_err());
        if let Err(EvalSuiteError::DatasetLoadFailed { path, .. }) = result {
            assert_eq!(path, "/nonexistent/path.jsonl");
        }
    }

    #[test]
    fn test_from_json_file_not_found() {
        let result = BenchmarkDataset::from_json("/nonexistent/path.json", BenchmarkSuiteType::Mmlu);
        assert!(result.is_err());
        if let Err(EvalSuiteError::DatasetLoadFailed { path, .. }) = result {
            assert_eq!(path, "/nonexistent/path.json");
        }
    }

    #[test]
    fn test_make_helpers() {
        let mc = make_mc_problem("t/1", "Q?", vec!["A", "B"], "A");
        assert!(matches!(mc.answer_format, AnswerFormat::MultipleChoice { .. }));

        let num = make_numeric_problem("t/2", "Q?", 42.0, 0.1);
        assert!(matches!(num.answer_format, AnswerFormat::Numeric { .. }));

        let code = make_code_problem("t/3", "Q?", "ref", "python");
        assert!(matches!(code.answer_format, AnswerFormat::Code { .. }));
    }
}
