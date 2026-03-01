//! Scoring, metrics, and ELO calculation for benchmark evaluation.
//!
//! Provides the `ProblemScorer` trait, `DefaultScorer` implementation that dispatches
//! by answer format, and utility functions for Pass@k, accuracy, mean score, and ELO.

use super::dataset::{AnswerFormat, BenchmarkProblem};
use super::runner::{ModelIdentifier, ProblemResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ProblemScorer trait
// ---------------------------------------------------------------------------

/// Trait for scoring LLM responses against benchmark problems.
pub trait ProblemScorer: Send + Sync {
    /// Score a response (0.0 = completely wrong, 1.0 = perfectly correct).
    fn score(&self, problem: &BenchmarkProblem, response: &str) -> f64;

    /// Whether the response passes (default: score >= 0.99).
    fn passed(&self, problem: &BenchmarkProblem, response: &str) -> bool {
        self.score(problem, response) >= 0.99
    }
}

// ---------------------------------------------------------------------------
// DefaultScorer
// ---------------------------------------------------------------------------

/// Default scorer that dispatches scoring by the problem's `AnswerFormat`.
pub struct DefaultScorer;

impl ProblemScorer for DefaultScorer {
    fn score(&self, problem: &BenchmarkProblem, response: &str) -> f64 {
        match &problem.answer_format {
            AnswerFormat::MultipleChoice { correct, .. } => {
                score_multiple_choice(response, correct)
            }
            AnswerFormat::Numeric { correct, tolerance } => {
                score_numeric(response, *correct, *tolerance)
            }
            AnswerFormat::Code { .. } => {
                score_code_heuristic(response, problem.reference_solution.as_deref())
            }
            AnswerFormat::FreeText => {
                score_free_text(response, problem.reference_solution.as_deref())
            }
            AnswerFormat::AgentTrajectory { .. } => {
                // Agent trajectory scoring requires custom implementation
                0.0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Score a multiple-choice response by extracting the answer letter.
fn score_multiple_choice(response: &str, correct: &str) -> f64 {
    if let Some(extracted) = extract_answer_letter(response) {
        if extracted.eq_ignore_ascii_case(correct) {
            return 1.0;
        }
    }
    // Fallback: check if the response contains just the correct answer
    let trimmed = response.trim();
    if trimmed.eq_ignore_ascii_case(correct) {
        return 1.0;
    }
    0.0
}

/// Score a numeric response by parsing the number and checking tolerance.
fn score_numeric(response: &str, correct: f64, tolerance: f64) -> f64 {
    if let Some(extracted) = extract_numeric_answer(response) {
        if (extracted - correct).abs() <= tolerance {
            return 1.0;
        }
        // Partial credit based on distance
        let distance = (extracted - correct).abs();
        let max_distance = correct.abs().max(1.0) * 10.0; // 10x as max reasonable distance
        if distance < max_distance {
            return (1.0 - distance / max_distance).max(0.0);
        }
    }
    0.0
}

/// Heuristic scoring for code problems (reference matching).
///
/// Without a test runner, we use normalized text similarity against the reference.
fn score_code_heuristic(response: &str, reference: Option<&str>) -> f64 {
    let reference = match reference {
        Some(r) if !r.is_empty() => r,
        _ => return 0.0,
    };

    let norm_response = normalize_code(response);
    let norm_reference = normalize_code(reference);

    if norm_response == norm_reference {
        return 1.0;
    }

    // Jaccard similarity on tokens
    let resp_tokens: std::collections::HashSet<&str> = norm_response.split_whitespace().collect();
    let ref_tokens: std::collections::HashSet<&str> = norm_reference.split_whitespace().collect();

    if ref_tokens.is_empty() {
        return 0.0;
    }

    let intersection = resp_tokens.intersection(&ref_tokens).count() as f64;
    let union = resp_tokens.union(&ref_tokens).count() as f64;

    if union == 0.0 {
        return 0.0;
    }

    intersection / union
}

/// Score free-text responses by exact/fuzzy matching against reference.
fn score_free_text(response: &str, reference: Option<&str>) -> f64 {
    let reference = match reference {
        Some(r) if !r.is_empty() => r,
        _ => return 0.0,
    };

    let norm_resp = response.trim().to_lowercase();
    let norm_ref = reference.trim().to_lowercase();

    if norm_resp == norm_ref {
        return 1.0;
    }

    // Check if the response contains the reference answer
    if norm_resp.contains(&norm_ref) || norm_ref.contains(&norm_resp) {
        return 0.8;
    }

    // Word overlap
    let resp_words: std::collections::HashSet<&str> = norm_resp.split_whitespace().collect();
    let ref_words: std::collections::HashSet<&str> = norm_ref.split_whitespace().collect();

    if ref_words.is_empty() {
        return 0.0;
    }

    let overlap = resp_words.intersection(&ref_words).count() as f64;
    overlap / ref_words.len() as f64
}

/// Extract a single answer letter (A, B, C, D, etc.) from an LLM response.
///
/// Searches the entire response text for common answer patterns. Prioritizes
/// explicit answer declarations ("the answer is X") over implicit patterns.
fn extract_answer_letter(response: &str) -> Option<String> {
    let trimmed = response.trim();
    let lower = trimmed.to_lowercase();

    // Pattern 1: single letter response (highest confidence)
    if trimmed.len() == 1 && trimmed.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
        return Some(trimmed.to_uppercase());
    }

    // Pattern 2: "(X)" or "X)" or "X." at start of response
    for pattern in &["(", ""] {
        let search = trimmed.strip_prefix(pattern).unwrap_or(trimmed);
        let first_char = search.chars().next()?;
        if first_char.is_ascii_alphabetic() {
            let second = search.chars().nth(1);
            if second == Some(')') || second == Some('.') || second == Some(':') || second.is_none()
            {
                return Some(first_char.to_uppercase().to_string());
            }
        }
    }

    // Pattern 3: Scan ENTIRE text for answer phrases (not just prefix)
    // "the answer is X", "answer: X", "correct answer is X", "answer is X"
    let answer_markers = [
        "the correct answer is ",
        "the answer is ",
        "correct answer is ",
        "my answer is ",
        "answer is ",
        "answer: ",
    ];
    for marker in &answer_markers {
        if let Some(pos) = lower.find(marker) {
            let after = &trimmed[pos + marker.len()..];
            let after = after.trim();
            // Extract letter: could be "B", "B)", "(B)", "**B**", "B."
            let cleaned = after
                .trim_start_matches('(')
                .trim_start_matches('*')
                .trim_start_matches('*');
            if let Some(ch) = cleaned.chars().next() {
                if ch.is_ascii_alphabetic() {
                    // Validate it's a single option letter (A-F range)
                    let upper = ch.to_ascii_uppercase();
                    if upper >= 'A' && upper <= 'F' {
                        return Some(upper.to_string());
                    }
                }
            }
        }
    }

    // Pattern 4: "therefore, X" or "so the answer is X" — look for conclusion markers
    let conclusion_markers = ["therefore, ", "therefore ", "thus, ", "so, ", "hence, "];
    for marker in &conclusion_markers {
        if let Some(pos) = lower.rfind(marker) {
            let after = &trimmed[pos + marker.len()..];
            let after = after
                .trim()
                .trim_start_matches('(')
                .trim_start_matches('*')
                .trim_start_matches('*');
            if let Some(ch) = after.chars().next() {
                if ch.is_ascii_alphabetic() {
                    let upper = ch.to_ascii_uppercase();
                    if upper >= 'A' && upper <= 'F' {
                        let next = after.chars().nth(1);
                        if next == Some(')') || next == Some('.') || next == Some(' ')
                            || next == Some('*') || next == Some(',') || next.is_none()
                        {
                            return Some(upper.to_string());
                        }
                    }
                }
            }
        }
    }

    // Pattern 5: Scan for "X)" or "**X**" on its own line (common in multi-line responses)
    // Search from the END of the response (last answer is usually the final one)
    for line in trimmed.lines().rev() {
        let line = line.trim();
        // Skip empty or very long lines (explanations)
        if line.is_empty() || line.len() > 40 {
            continue;
        }
        // "X)" or "X."
        if line.len() >= 2 {
            let first = line.chars().next().unwrap_or(' ');
            let second = line.chars().nth(1).unwrap_or(' ');
            if first.is_ascii_alphabetic()
                && first.to_ascii_uppercase() >= 'A'
                && first.to_ascii_uppercase() <= 'F'
                && (second == ')' || second == '.')
            {
                return Some(first.to_uppercase().to_string());
            }
        }
        // "**X**" (markdown bold)
        let stripped = line.trim_start_matches('*').trim_end_matches('*').trim();
        if stripped.len() == 1 {
            let ch = stripped.chars().next().unwrap_or(' ');
            if ch.is_ascii_alphabetic()
                && ch.to_ascii_uppercase() >= 'A'
                && ch.to_ascii_uppercase() <= 'F'
            {
                return Some(ch.to_uppercase().to_string());
            }
        }
    }

    // Pattern 6: Last resort — scan for isolated "X)" pattern anywhere in text
    let bytes = lower.as_bytes();
    for i in (0..bytes.len().saturating_sub(1)).rev() {
        let ch = bytes[i] as char;
        let next = bytes[i + 1] as char;
        if ch.is_ascii_alphabetic()
            && ch.to_ascii_uppercase() >= 'A'
            && ch.to_ascii_uppercase() <= 'F'
            && next == ')'
        {
            // Make sure it's not inside a word (check previous char)
            let prev_is_separator =
                i == 0 || !bytes[i - 1].is_ascii_alphabetic();
            if prev_is_separator {
                return Some(ch.to_uppercase().to_string());
            }
        }
    }

    None
}

/// Extract a numeric answer from an LLM response.
fn extract_numeric_answer(response: &str) -> Option<f64> {
    let trimmed = response.trim();

    // Try direct parse first
    if let Ok(n) = trimmed.parse::<f64>() {
        return Some(n);
    }

    // Pattern: "The answer is X" or "= X"
    for prefix in &["the answer is ", "answer: ", "answer is ", "= "] {
        if let Some(rest) = trimmed.to_lowercase().find(prefix) {
            let after = &trimmed[rest + prefix.len()..];
            let num_str: String = after
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',')
                .filter(|c| *c != ',')
                .collect();
            if let Ok(n) = num_str.parse::<f64>() {
                return Some(n);
            }
        }
    }

    // Scan for the last number in the response (common in chain-of-thought)
    let mut last_number = None;
    let mut chars = trimmed.chars().peekable();
    let mut pos = 0;
    while pos < trimmed.len() {
        let c = trimmed.as_bytes()[pos] as char;
        if c.is_ascii_digit() || (c == '-' && pos + 1 < trimmed.len() && (trimmed.as_bytes()[pos + 1] as char).is_ascii_digit()) {
            let start = pos;
            pos += 1;
            while pos < trimmed.len() {
                let nc = trimmed.as_bytes()[pos] as char;
                if nc.is_ascii_digit() || nc == '.' {
                    pos += 1;
                } else {
                    break;
                }
            }
            if let Ok(n) = trimmed[start..pos].parse::<f64>() {
                last_number = Some(n);
            }
        } else {
            pos += 1;
        }
        let _ = chars.next();
    }

    last_number
}

/// Normalize code for comparison (strip whitespace, comments, blank lines).
fn normalize_code(code: &str) -> String {
    code.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Aggregate metrics
// ---------------------------------------------------------------------------

/// Calculate Pass@k using the unbiased estimator from the Codex paper.
///
/// Given `n` total samples and `c` correct samples, estimates the probability
/// that at least one of `k` samples is correct.
///
/// Formula: 1 - C(n-c, k) / C(n, k)
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n == 0 || k == 0 {
        return 0.0;
    }
    if c >= n {
        return 1.0;
    }
    if k > n {
        return if c > 0 { 1.0 } else { 0.0 };
    }

    // Use log-space to avoid overflow: log(C(n,k)) = sum(log(n-i) - log(k-i)) for i in 0..k
    let log_comb = |total: usize, choose: usize| -> f64 {
        if choose > total {
            return f64::NEG_INFINITY;
        }
        let mut result = 0.0_f64;
        for i in 0..choose {
            result += ((total - i) as f64).ln() - ((choose - i) as f64).ln();
        }
        result
    };

    let log_num = log_comb(n - c, k);
    let log_den = log_comb(n, k);

    if log_num == f64::NEG_INFINITY {
        return 1.0; // C(n-c, k) = 0, so all k must include a correct one
    }

    1.0 - (log_num - log_den).exp()
}

/// Calculate accuracy: fraction of problems where at least one sample passed.
pub fn accuracy(results: &[ProblemResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let passed = results.iter().filter(|r| r.passed.iter().any(|&p| p)).count();
    passed as f64 / results.len() as f64
}

/// Calculate mean score across all problems (averaging each problem's mean score).
pub fn mean_score(results: &[ProblemResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum: f64 = results
        .iter()
        .map(|r| {
            if r.scores.is_empty() {
                0.0
            } else {
                r.scores.iter().sum::<f64>() / r.scores.len() as f64
            }
        })
        .sum();
    sum / results.len() as f64
}

// ---------------------------------------------------------------------------
// ELO calculator
// ---------------------------------------------------------------------------

/// ELO rating calculator for ranking models by pairwise comparison.
pub struct EloCalculator {
    ratings: HashMap<ModelIdentifier, f64>,
    k_factor: f64,
}

impl EloCalculator {
    /// Create a new ELO calculator with the given K-factor.
    ///
    /// Higher K-factor means ratings change more per match.
    /// Default: 32.0 (standard for most applications).
    pub fn new(k_factor: f64) -> Self {
        Self {
            ratings: HashMap::new(),
            k_factor,
        }
    }

    /// Update ratings based on a pairwise comparison.
    ///
    /// `score_a` and `score_b` are the actual scores (e.g., accuracy) for each model.
    /// The model with the higher score "wins" the match.
    pub fn update_from_pairwise(
        &mut self,
        model_a: &ModelIdentifier,
        model_b: &ModelIdentifier,
        score_a: f64,
        score_b: f64,
    ) {
        let ra = *self.ratings.get(model_a).unwrap_or(&1500.0);
        let rb = *self.ratings.get(model_b).unwrap_or(&1500.0);

        // Expected scores
        let ea = 1.0 / (1.0 + 10.0_f64.powf((rb - ra) / 400.0));
        let eb = 1.0 / (1.0 + 10.0_f64.powf((ra - rb) / 400.0));

        // Actual outcome (0.0 = loss, 0.5 = draw, 1.0 = win)
        let (sa, sb) = if (score_a - score_b).abs() < 1e-10 {
            (0.5, 0.5)
        } else if score_a > score_b {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };

        let new_ra = ra + self.k_factor * (sa - ea);
        let new_rb = rb + self.k_factor * (sb - eb);

        self.ratings.insert(model_a.clone(), new_ra);
        self.ratings.insert(model_b.clone(), new_rb);
    }

    /// Get all current ratings.
    pub fn ratings(&self) -> &HashMap<ModelIdentifier, f64> {
        &self.ratings
    }

    /// Get models ranked by ELO rating (highest first).
    pub fn ranked(&self) -> Vec<(ModelIdentifier, f64)> {
        let mut ranked: Vec<(ModelIdentifier, f64)> = self
            .ratings
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dataset::{make_mc_problem, make_numeric_problem, make_code_problem};

    #[test]
    fn test_score_multiple_choice_correct() {
        let problem = make_mc_problem("t/1", "Q?", vec!["A", "B", "C", "D"], "B");
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "B"), 1.0);
        assert_eq!(scorer.score(&problem, "The answer is B"), 1.0);
        assert_eq!(scorer.score(&problem, "b"), 1.0);
    }

    #[test]
    fn test_score_multiple_choice_wrong() {
        let problem = make_mc_problem("t/1", "Q?", vec!["A", "B", "C", "D"], "B");
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "A"), 0.0);
        assert_eq!(scorer.score(&problem, "The answer is C"), 0.0);
    }

    #[test]
    fn test_score_multiple_choice_extraction_patterns() {
        let problem = make_mc_problem("t/1", "Q?", vec!["A", "B", "C", "D"], "C");
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "C)"), 1.0);
        assert_eq!(scorer.score(&problem, "(C)"), 1.0);
        assert_eq!(scorer.score(&problem, "Answer: C"), 1.0);
    }

    #[test]
    fn test_score_numeric_exact() {
        let problem = make_numeric_problem("t/1", "Q?", 42.0, 0.01);
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "42"), 1.0);
        assert_eq!(scorer.score(&problem, "42.0"), 1.0);
    }

    #[test]
    fn test_score_numeric_within_tolerance() {
        let problem = make_numeric_problem("t/1", "Q?", 42.0, 0.5);
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "42.3"), 1.0);
        assert_eq!(scorer.score(&problem, "41.6"), 1.0);
    }

    #[test]
    fn test_score_numeric_outside_tolerance() {
        let problem = make_numeric_problem("t/1", "Q?", 42.0, 0.01);
        let scorer = DefaultScorer;
        let score = scorer.score(&problem, "50");
        assert!(score < 1.0);
    }

    #[test]
    fn test_score_numeric_extraction() {
        let problem = make_numeric_problem("t/1", "Q?", 3.0, 0.01);
        let scorer = DefaultScorer;
        let score = scorer.score(&problem, "Let me think... The answer is 3");
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_score_free_text_exact_match() {
        let mut problem = make_mc_problem("t/1", "Q?", vec![], "");
        problem.answer_format = AnswerFormat::FreeText;
        problem.reference_solution = Some("Paris".to_string());
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "Paris"), 1.0);
        assert_eq!(scorer.score(&problem, "paris"), 1.0);
    }

    #[test]
    fn test_score_free_text_no_reference() {
        let mut problem = make_mc_problem("t/1", "Q?", vec![], "");
        problem.answer_format = AnswerFormat::FreeText;
        problem.reference_solution = None;
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "anything"), 0.0);
    }

    #[test]
    fn test_score_code_heuristic_exact() {
        let problem = make_code_problem("t/1", "Q?", "def add(a, b): return a + b", "python");
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "def add(a, b): return a + b"), 1.0);
    }

    #[test]
    fn test_score_code_heuristic_partial() {
        let problem = make_code_problem("t/1", "Q?", "def add(a, b): return a + b", "python");
        let scorer = DefaultScorer;
        let score = scorer.score(&problem, "def add(x, y): return x + y");
        assert!(score > 0.0 && score < 1.0); // Partial match
    }

    #[test]
    fn test_pass_at_k_formula() {
        // n=10, c=5, k=1 → ~50%
        let p = pass_at_k(10, 5, 1);
        assert!((p - 0.5).abs() < 0.01);

        // n=10, c=10, k=1 → 100%
        assert_eq!(pass_at_k(10, 10, 1), 1.0);

        // n=10, c=0, k=1 → 0%
        assert_eq!(pass_at_k(10, 0, 1), 0.0);

        // Edge cases
        assert_eq!(pass_at_k(0, 0, 1), 0.0);
        assert_eq!(pass_at_k(5, 3, 0), 0.0);
    }

    #[test]
    fn test_pass_at_k_higher_k() {
        // More samples → higher pass rate
        let p1 = pass_at_k(10, 3, 1);
        let p5 = pass_at_k(10, 3, 5);
        assert!(p5 > p1);
    }

    #[test]
    fn test_accuracy_calculation() {
        let model = ModelIdentifier { name: "test".into(), provider: "test".into(), variant: None };
        let results = vec![
            ProblemResult {
                problem_id: "1".into(),
                model_id: model.clone(),
                responses: vec!["A".into()],
                scores: vec![1.0],
                passed: vec![true],
                latencies_ms: vec![100],
                token_counts: vec![],
                cost_estimates: vec![0.001],
                error: None,
                metadata: HashMap::new(),
            },
            ProblemResult {
                problem_id: "2".into(),
                model_id: model.clone(),
                responses: vec!["wrong".into()],
                scores: vec![0.0],
                passed: vec![false],
                latencies_ms: vec![100],
                token_counts: vec![],
                cost_estimates: vec![0.001],
                error: None,
                metadata: HashMap::new(),
            },
        ];
        assert!((accuracy(&results) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mean_score_calculation() {
        let model = ModelIdentifier { name: "test".into(), provider: "test".into(), variant: None };
        let results = vec![
            ProblemResult {
                problem_id: "1".into(),
                model_id: model.clone(),
                responses: vec!["A".into()],
                scores: vec![1.0],
                passed: vec![true],
                latencies_ms: vec![100],
                token_counts: vec![],
                cost_estimates: vec![0.001],
                error: None,
                metadata: HashMap::new(),
            },
            ProblemResult {
                problem_id: "2".into(),
                model_id: model.clone(),
                responses: vec!["B".into()],
                scores: vec![0.5],
                passed: vec![false],
                latencies_ms: vec![100],
                token_counts: vec![],
                cost_estimates: vec![0.001],
                error: None,
                metadata: HashMap::new(),
            },
        ];
        assert!((mean_score(&results) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_accuracy_empty() {
        assert_eq!(accuracy(&[]), 0.0);
    }

    #[test]
    fn test_mean_score_empty() {
        assert_eq!(mean_score(&[]), 0.0);
    }

    #[test]
    fn test_elo_calculator() {
        let mut elo = EloCalculator::new(32.0);
        let model_a = ModelIdentifier { name: "gpt-4".into(), provider: "openai".into(), variant: None };
        let model_b = ModelIdentifier { name: "llama3".into(), provider: "ollama".into(), variant: Some("8b".into()) };

        // Model A wins
        elo.update_from_pairwise(&model_a, &model_b, 0.9, 0.6);

        let ratings = elo.ratings();
        assert!(ratings[&model_a] > 1500.0);
        assert!(ratings[&model_b] < 1500.0);
    }

    #[test]
    fn test_elo_draw() {
        let mut elo = EloCalculator::new(32.0);
        let model_a = ModelIdentifier { name: "a".into(), provider: "p".into(), variant: None };
        let model_b = ModelIdentifier { name: "b".into(), provider: "p".into(), variant: None };

        elo.update_from_pairwise(&model_a, &model_b, 0.5, 0.5);

        // Draw at equal ratings → no change
        let ratings = elo.ratings();
        assert!((ratings[&model_a] - 1500.0).abs() < 0.01);
        assert!((ratings[&model_b] - 1500.0).abs() < 0.01);
    }

    #[test]
    fn test_elo_ranked() {
        let mut elo = EloCalculator::new(32.0);
        let model_a = ModelIdentifier { name: "a".into(), provider: "p".into(), variant: None };
        let model_b = ModelIdentifier { name: "b".into(), provider: "p".into(), variant: None };
        let model_c = ModelIdentifier { name: "c".into(), provider: "p".into(), variant: None };

        elo.update_from_pairwise(&model_a, &model_b, 0.9, 0.1);
        elo.update_from_pairwise(&model_b, &model_c, 0.9, 0.1);
        elo.update_from_pairwise(&model_a, &model_c, 0.9, 0.1);

        let ranked = elo.ranked();
        assert_eq!(ranked[0].0.name, "a"); // Best
        assert_eq!(ranked[2].0.name, "c"); // Worst
    }

    #[test]
    fn test_extract_answer_letter_patterns() {
        // Basic patterns
        assert_eq!(extract_answer_letter("B"), Some("B".to_string()));
        assert_eq!(extract_answer_letter("The answer is C"), Some("C".to_string()));
        assert_eq!(extract_answer_letter("Answer: A"), Some("A".to_string()));
        assert_eq!(extract_answer_letter("(D)"), Some("D".to_string()));
        assert_eq!(extract_answer_letter("B)"), Some("B".to_string()));

        // Embedded in longer text (the real-world LLM case)
        assert_eq!(
            extract_answer_letter("Let me calculate 15 * 17 = 255. The answer is B."),
            Some("B".to_string()),
        );
        assert_eq!(
            extract_answer_letter("After analyzing the options, the correct answer is D) Some flowers are roses."),
            Some("D".to_string()),
        );
        assert_eq!(
            extract_answer_letter("Mercury is the closest planet to the Sun.\n\nTherefore, B"),
            Some("B".to_string()),
        );
        assert_eq!(
            extract_answer_letter("The symbol for gold is Au.\n\n**C**"),
            Some("C".to_string()),
        );
    }

    #[test]
    fn test_extract_numeric_answer_patterns() {
        assert_eq!(extract_numeric_answer("42"), Some(42.0));
        assert_eq!(extract_numeric_answer("42.5"), Some(42.5));
        assert_eq!(extract_numeric_answer("The answer is 3"), Some(3.0));
        assert!(extract_numeric_answer("Let me calculate... 2 + 3 = 5").is_some());
    }

    #[test]
    fn test_default_scorer_agent_trajectory() {
        let problem = BenchmarkProblem {
            id: "t/1".into(),
            suite: super::super::dataset::BenchmarkSuiteType::AgentBench,
            category: super::super::dataset::ProblemCategory::AgentTask,
            prompt: "Do something".into(),
            system_prompt: None,
            answer_format: AnswerFormat::AgentTrajectory { success_criteria: "done".into() },
            reference_solution: None,
            test_cases: None,
            metadata: HashMap::new(),
            difficulty: None,
            tags: vec![],
        };
        let scorer = DefaultScorer;
        assert_eq!(scorer.score(&problem, "I did it"), 0.0);
    }

    #[test]
    fn test_passed_threshold() {
        let problem = make_mc_problem("t/1", "Q?", vec!["A", "B"], "A");
        let scorer = DefaultScorer;
        assert!(scorer.passed(&problem, "A"));
        assert!(!scorer.passed(&problem, "B"));
    }
}
