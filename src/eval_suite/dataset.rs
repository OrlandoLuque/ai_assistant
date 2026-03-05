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
    /// LiveCodeBench: contamination-aware competitive programming (LeetCode, Codeforces, AtCoder)
    LiveCodeBench,
    /// Aider Polyglot: multi-language code editing evaluation
    AiderPolyglot,
    /// Terminal-Bench: complex SWE tasks in real terminal environments
    TerminalBench,
    /// APPS: introductory-to-competition level programming problems
    Apps,
    /// CodeContests: competitive programming from Google DeepMind
    CodeContests,
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
            Self::LiveCodeBench => write!(f, "LiveCodeBench"),
            Self::AiderPolyglot => write!(f, "Aider-Polyglot"),
            Self::TerminalBench => write!(f, "Terminal-Bench"),
            Self::Apps => write!(f, "APPS"),
            Self::CodeContests => write!(f, "CodeContests"),
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
    /// Competitive programming (algorithmic, stdin/stdout based)
    CompetitiveProgramming,
    /// Code editing and refactoring (given existing code + instructions)
    CodeEditing,
    /// Terminal/shell command sequence tasks
    TerminalTask,
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
            Self::CompetitiveProgramming => write!(f, "CompetitiveProgramming"),
            Self::CodeEditing => write!(f, "CodeEditing"),
            Self::TerminalTask => write!(f, "TerminalTask"),
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
    /// Competitive programming code with stdin/stdout test cases
    CompetitiveProgrammingCode {
        /// Programming language (e.g., "python", "cpp", "rust")
        language: String,
        /// Test case pairs: (input_stdin, expected_stdout)
        test_cases: Vec<(String, String)>,
        /// Optional time limit in milliseconds
        time_limit_ms: Option<u64>,
        /// Optional memory limit in megabytes
        memory_limit_mb: Option<u64>,
    },
    /// Code editing: given original code, produce edited version
    CodeEdit {
        /// Programming language
        language: String,
        /// The original source code to be edited
        original_code: String,
        /// Test commands or expected outcomes after editing
        verification: Option<String>,
    },
    /// Terminal command sequence: evaluated by matching expected commands or final state
    TerminalSequence {
        /// Initial environment description
        environment: String,
        /// Expected commands or actions (for trajectory matching)
        expected_commands: Vec<String>,
        /// Verification command and expected output (final state check)
        verification: Option<(String, String)>,
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

/// Create a LiveCodeBench competitive programming problem with contamination timestamp.
pub fn make_livecode_problem(
    id: &str,
    prompt: &str,
    reference: &str,
    language: &str,
    test_cases: Vec<(&str, &str)>,
    timestamp_unix: u64,
    difficulty: &str,
    source_platform: &str,
) -> BenchmarkProblem {
    let mut metadata = HashMap::new();
    metadata.insert("timestamp".to_string(), timestamp_unix.to_string());
    metadata.insert("source_platform".to_string(), source_platform.to_string());
    metadata.insert("language".to_string(), language.to_string());
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::LiveCodeBench,
        category: ProblemCategory::CompetitiveProgramming,
        prompt: prompt.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::CompetitiveProgrammingCode {
            language: language.to_string(),
            test_cases: test_cases
                .into_iter()
                .map(|(i, o)| (i.to_string(), o.to_string()))
                .collect(),
            time_limit_ms: None,
            memory_limit_mb: None,
        },
        reference_solution: Some(reference.to_string()),
        test_cases: None,
        metadata,
        difficulty: Some(difficulty.to_string()),
        tags: Vec::new(),
    }
}

/// Create an APPS / CodeContests competitive programming problem.
pub fn make_competitive_problem(
    id: &str,
    prompt: &str,
    reference: &str,
    language: &str,
    test_cases: Vec<(&str, &str)>,
    difficulty: &str,
) -> BenchmarkProblem {
    let mut metadata = HashMap::new();
    metadata.insert("language".to_string(), language.to_string());
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::Apps,
        category: ProblemCategory::CompetitiveProgramming,
        prompt: prompt.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::CompetitiveProgrammingCode {
            language: language.to_string(),
            test_cases: test_cases
                .into_iter()
                .map(|(i, o)| (i.to_string(), o.to_string()))
                .collect(),
            time_limit_ms: None,
            memory_limit_mb: None,
        },
        reference_solution: Some(reference.to_string()),
        test_cases: None,
        metadata,
        difficulty: Some(difficulty.to_string()),
        tags: Vec::new(),
    }
}

/// Create an Aider Polyglot code editing problem.
pub fn make_code_edit_problem(
    id: &str,
    instruction: &str,
    original_code: &str,
    reference_edit: &str,
    language: &str,
) -> BenchmarkProblem {
    let mut metadata = HashMap::new();
    metadata.insert("language".to_string(), language.to_string());
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::AiderPolyglot,
        category: ProblemCategory::CodeEditing,
        prompt: instruction.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::CodeEdit {
            language: language.to_string(),
            original_code: original_code.to_string(),
            verification: None,
        },
        reference_solution: Some(reference_edit.to_string()),
        test_cases: None,
        metadata,
        difficulty: None,
        tags: Vec::new(),
    }
}

/// Create a Terminal-Bench terminal task problem.
pub fn make_terminal_problem(
    id: &str,
    task_description: &str,
    environment: &str,
    expected_commands: Vec<&str>,
    reference_solution: &str,
) -> BenchmarkProblem {
    BenchmarkProblem {
        id: id.to_string(),
        suite: BenchmarkSuiteType::TerminalBench,
        category: ProblemCategory::TerminalTask,
        prompt: task_description.to_string(),
        system_prompt: None,
        answer_format: AnswerFormat::TerminalSequence {
            environment: environment.to_string(),
            expected_commands: expected_commands
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            verification: None,
        },
        reference_solution: Some(reference_solution.to_string()),
        test_cases: None,
        metadata: HashMap::new(),
        difficulty: None,
        tags: Vec::new(),
    }
}

/// Filter a LiveCodeBench dataset by contamination cutoff date.
///
/// Returns only problems with timestamps strictly after the given Unix timestamp,
/// ensuring no data contamination from training data before the cutoff.
pub fn filter_by_contamination_cutoff(
    dataset: &BenchmarkDataset,
    cutoff_unix: u64,
) -> BenchmarkDataset {
    dataset.filter(|p| {
        p.metadata
            .get("timestamp")
            .and_then(|ts| ts.parse::<u64>().ok())
            .map_or(false, |ts| ts > cutoff_unix)
    })
}

/// Filter a dataset by programming language.
///
/// Checks both `metadata["language"]` and the language field in answer format variants.
pub fn filter_by_language(dataset: &BenchmarkDataset, language: &str) -> BenchmarkDataset {
    let lang_lower = language.to_lowercase();
    dataset.filter(|p| {
        // Check metadata first
        if let Some(meta_lang) = p.metadata.get("language") {
            if meta_lang.to_lowercase() == lang_lower {
                return true;
            }
        }
        // Check answer format language field
        match &p.answer_format {
            AnswerFormat::Code { language: l }
            | AnswerFormat::CompetitiveProgrammingCode { language: l, .. }
            | AnswerFormat::CodeEdit { language: l, .. } => l.to_lowercase() == lang_lower,
            _ => false,
        }
    })
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

    // ── New benchmark suite type display tests ──

    #[test]
    fn test_new_suite_type_display() {
        assert_eq!(BenchmarkSuiteType::LiveCodeBench.to_string(), "LiveCodeBench");
        assert_eq!(BenchmarkSuiteType::AiderPolyglot.to_string(), "Aider-Polyglot");
        assert_eq!(BenchmarkSuiteType::TerminalBench.to_string(), "Terminal-Bench");
        assert_eq!(BenchmarkSuiteType::Apps.to_string(), "APPS");
        assert_eq!(BenchmarkSuiteType::CodeContests.to_string(), "CodeContests");
    }

    #[test]
    fn test_new_category_display() {
        assert_eq!(ProblemCategory::CompetitiveProgramming.to_string(), "CompetitiveProgramming");
        assert_eq!(ProblemCategory::CodeEditing.to_string(), "CodeEditing");
        assert_eq!(ProblemCategory::TerminalTask.to_string(), "TerminalTask");
    }

    // ── New helper constructor tests ──

    #[test]
    fn test_make_livecode_problem() {
        let p = make_livecode_problem(
            "lcb/1",
            "Read N numbers and print their sum",
            "n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums))",
            "python",
            vec![("3\n1 2 3", "6"), ("1\n42", "42")],
            1700000000,
            "easy",
            "codeforces",
        );
        assert_eq!(p.id, "lcb/1");
        assert_eq!(p.suite, BenchmarkSuiteType::LiveCodeBench);
        assert_eq!(p.category, ProblemCategory::CompetitiveProgramming);
        assert_eq!(p.metadata.get("timestamp").unwrap(), "1700000000");
        assert_eq!(p.metadata.get("source_platform").unwrap(), "codeforces");
        assert_eq!(p.difficulty, Some("easy".to_string()));
        if let AnswerFormat::CompetitiveProgrammingCode { language, test_cases, .. } = &p.answer_format {
            assert_eq!(language, "python");
            assert_eq!(test_cases.len(), 2);
            assert_eq!(test_cases[0], ("3\n1 2 3".to_string(), "6".to_string()));
        } else {
            panic!("Expected CompetitiveProgrammingCode format");
        }
    }

    #[test]
    fn test_make_competitive_problem() {
        let p = make_competitive_problem(
            "apps/42",
            "Given an array, find two numbers that add up to target.",
            "def two_sum(nums, target): ...",
            "python",
            vec![("4 9\n2 7 11 15", "0 1")],
            "interview",
        );
        assert_eq!(p.suite, BenchmarkSuiteType::Apps);
        assert_eq!(p.category, ProblemCategory::CompetitiveProgramming);
        assert_eq!(p.difficulty, Some("interview".to_string()));
        assert!(matches!(p.answer_format, AnswerFormat::CompetitiveProgrammingCode { .. }));
    }

    #[test]
    fn test_make_code_edit_problem() {
        let p = make_code_edit_problem(
            "aider/py/1",
            "Add error handling for division by zero",
            "def divide(a, b):\n    return a / b",
            "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
            "python",
        );
        assert_eq!(p.suite, BenchmarkSuiteType::AiderPolyglot);
        assert_eq!(p.category, ProblemCategory::CodeEditing);
        assert_eq!(p.metadata.get("language").unwrap(), "python");
        if let AnswerFormat::CodeEdit { language, original_code, .. } = &p.answer_format {
            assert_eq!(language, "python");
            assert!(original_code.contains("def divide"));
        } else {
            panic!("Expected CodeEdit format");
        }
        assert!(p.reference_solution.as_ref().unwrap().contains("if b == 0"));
    }

    #[test]
    fn test_make_terminal_problem() {
        let p = make_terminal_problem(
            "tb/1",
            "Find all Python files larger than 1MB and compress them",
            "Ubuntu 22.04, bash, /home/user/project with 50 .py files",
            vec![
                "find /home/user/project -name '*.py' -size +1M",
                "tar czf large_py.tar.gz $(find /home/user/project -name '*.py' -size +1M)",
            ],
            "find . -name '*.py' -size +1M -exec tar czf large.tar.gz {} +",
        );
        assert_eq!(p.suite, BenchmarkSuiteType::TerminalBench);
        assert_eq!(p.category, ProblemCategory::TerminalTask);
        if let AnswerFormat::TerminalSequence { environment, expected_commands, .. } = &p.answer_format {
            assert!(environment.contains("Ubuntu"));
            assert_eq!(expected_commands.len(), 2);
        } else {
            panic!("Expected TerminalSequence format");
        }
    }

    // ── Serde round-trip tests for new AnswerFormat variants ──

    #[test]
    fn test_competitive_programming_code_serde() {
        let fmt = AnswerFormat::CompetitiveProgrammingCode {
            language: "cpp".to_string(),
            test_cases: vec![
                ("5\n1 2 3 4 5".to_string(), "15".to_string()),
                ("1\n42".to_string(), "42".to_string()),
            ],
            time_limit_ms: Some(2000),
            memory_limit_mb: Some(256),
        };
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: AnswerFormat = serde_json::from_str(&json).expect("deserialize");
        if let AnswerFormat::CompetitiveProgrammingCode { language, test_cases, time_limit_ms, memory_limit_mb } = back {
            assert_eq!(language, "cpp");
            assert_eq!(test_cases.len(), 2);
            assert_eq!(time_limit_ms, Some(2000));
            assert_eq!(memory_limit_mb, Some(256));
        } else {
            panic!("Wrong variant after deserialization");
        }
    }

    #[test]
    fn test_code_edit_serde() {
        let fmt = AnswerFormat::CodeEdit {
            language: "rust".to_string(),
            original_code: "fn main() {}".to_string(),
            verification: Some("cargo test".to_string()),
        };
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: AnswerFormat = serde_json::from_str(&json).expect("deserialize");
        if let AnswerFormat::CodeEdit { language, original_code, verification } = back {
            assert_eq!(language, "rust");
            assert_eq!(original_code, "fn main() {}");
            assert_eq!(verification, Some("cargo test".to_string()));
        } else {
            panic!("Wrong variant after deserialization");
        }
    }

    #[test]
    fn test_terminal_sequence_serde() {
        let fmt = AnswerFormat::TerminalSequence {
            environment: "debian:latest".to_string(),
            expected_commands: vec!["ls -la".to_string(), "grep foo bar.txt".to_string()],
            verification: Some(("cat result.txt".to_string(), "success".to_string())),
        };
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: AnswerFormat = serde_json::from_str(&json).expect("deserialize");
        if let AnswerFormat::TerminalSequence { environment, expected_commands, verification } = back {
            assert_eq!(environment, "debian:latest");
            assert_eq!(expected_commands.len(), 2);
            assert_eq!(verification, Some(("cat result.txt".to_string(), "success".to_string())));
        } else {
            panic!("Wrong variant after deserialization");
        }
    }

    // ── Filter function tests ──

    #[test]
    fn test_filter_by_contamination_cutoff() {
        let problems = vec![
            make_livecode_problem("lcb/1", "P1", "ref1", "python", vec![("1", "1")], 1690000000, "easy", "leetcode"),
            make_livecode_problem("lcb/2", "P2", "ref2", "python", vec![("2", "2")], 1700000000, "medium", "codeforces"),
            make_livecode_problem("lcb/3", "P3", "ref3", "python", vec![("3", "3")], 1710000000, "hard", "atcoder"),
        ];
        let ds = BenchmarkDataset::from_problems("lcb", BenchmarkSuiteType::LiveCodeBench, problems);

        // Cutoff at 1700000000 → only problems after that timestamp
        let filtered = filter_by_contamination_cutoff(&ds, 1700000000);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.problems[0].id, "lcb/3");

        // Cutoff at 0 → all problems
        let all = filter_by_contamination_cutoff(&ds, 0);
        assert_eq!(all.len(), 3);

        // Cutoff far in the future → no problems
        let none = filter_by_contamination_cutoff(&ds, u64::MAX);
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_filter_by_language() {
        let problems = vec![
            make_code_problem("he/1", "P1", "ref", "python"),
            make_code_problem("he/2", "P2", "ref", "rust"),
            make_competitive_problem("apps/1", "P3", "ref", "cpp", vec![("1", "1")], "easy"),
            make_code_edit_problem("aider/1", "Edit", "code", "edited", "python"),
        ];
        let ds = BenchmarkDataset::from_problems("mixed", BenchmarkSuiteType::Custom("mixed".into()), problems);

        let python = filter_by_language(&ds, "python");
        assert_eq!(python.len(), 2); // he/1 + aider/1

        let rust = filter_by_language(&ds, "rust");
        assert_eq!(rust.len(), 1);
        assert_eq!(rust.problems[0].id, "he/2");

        let cpp = filter_by_language(&ds, "cpp");
        assert_eq!(cpp.len(), 1);
        assert_eq!(cpp.problems[0].id, "apps/1");

        let java = filter_by_language(&ds, "java");
        assert_eq!(java.len(), 0);
    }
}
