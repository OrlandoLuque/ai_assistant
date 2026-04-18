//! Benchmark runner: query the model for every sample, score, aggregate.
//!
//! Orthogonal concerns kept out of here:
//!  * *Loading* samples — that's [`BenchmarkLoader`](super::types::BenchmarkLoader).
//!  * *Provider-specific LLM calls* — caller supplies a `generate` closure.
//!  * *Calibration* — sweeping thresholds lives in [`calibration`](super::calibration).
//!
//! Scoring is per [`SampleType`](super::types::SampleType); the runner picks
//! the right scorer by dispatch. Scorers are intentionally lightweight —
//! Jaccard word-overlap and keyword matching — so the whole pipeline runs
//! without requiring a second LLM on hand.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::{BenchmarkSample, GroundTruth, Label};

/// Per-sample runner outcome.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// The sample's id (stable across runs).
    pub id: String,
    /// The prompt shown to the model (post-formatting).
    pub prompt: String,
    /// The raw model response.
    pub response: String,
    /// Time spent generating.
    pub duration: Duration,
    /// Score in [0.0, 1.0] — typically correctness, or a per-task proxy.
    pub score: f64,
    /// Whether the sample was counted as "correct" under the default threshold.
    pub correct: bool,
    /// Auxiliary fields (faithfulness, jaccard, predicted-label, ...).
    pub details: HashMap<String, String>,
}

/// Aggregated benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Name of the benchmark (loader.name()).
    pub benchmark: String,
    /// Number of samples run.
    pub total: usize,
    /// Number scored as correct.
    pub correct: usize,
    /// Accuracy in [0.0, 1.0].
    pub accuracy: f64,
    /// Mean of `SampleResult::score`.
    pub mean_score: f64,
    /// Wall time for the whole run.
    pub total_duration: Duration,
    /// Per-category accuracy (if `sample.category` was set).
    pub per_category: HashMap<String, (usize, usize)>,
    /// All per-sample results (callers keep or trim as they like).
    pub samples: Vec<SampleResult>,
}

/// Options for a benchmark run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Limit the number of samples run (per-loader; smoke-tests use small N).
    pub limit: Option<usize>,
    /// Threshold for the default correctness cutoff applied to `score`.
    /// Must be in [0.0, 1.0]. Default 0.5.
    pub correctness_threshold: f64,
    /// Stop early if this many consecutive samples error. `0` = never.
    pub max_consecutive_errors: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            limit: None,
            correctness_threshold: 0.5,
            max_consecutive_errors: 10,
        }
    }
}

/// Run the benchmark. `generate(prompt) -> model_response` is the only side
/// effect. Errors in generation become empty-string responses scoring 0.0.
pub fn run<F>(
    benchmark_name: &str,
    samples: &[BenchmarkSample],
    opts: &RunOptions,
    mut generate: F,
) -> BenchmarkReport
where
    F: FnMut(&str) -> Result<String, String>,
{
    let start = Instant::now();
    let limit = opts.limit.unwrap_or(samples.len());
    let take = samples.iter().take(limit);

    let mut results: Vec<SampleResult> = Vec::with_capacity(limit);
    let mut consecutive_errors = 0usize;

    for sample in take {
        let t0 = Instant::now();
        let response = match generate(&sample.prompt) {
            Ok(r) => {
                consecutive_errors = 0;
                r
            }
            Err(e) => {
                consecutive_errors += 1;
                if opts.max_consecutive_errors > 0
                    && consecutive_errors >= opts.max_consecutive_errors
                {
                    results.push(make_error_result(sample, t0.elapsed(), &e));
                    break;
                }
                results.push(make_error_result(sample, t0.elapsed(), &e));
                continue;
            }
        };
        let duration = t0.elapsed();
        results.push(score_sample(sample, &response, duration, opts));
    }

    aggregate(benchmark_name, results, start.elapsed())
}

fn make_error_result(sample: &BenchmarkSample, duration: Duration, err: &str) -> SampleResult {
    let mut details = HashMap::new();
    details.insert("error".into(), err.to_string());
    SampleResult {
        id: sample.id.clone(),
        prompt: sample.prompt.clone(),
        response: String::new(),
        duration,
        score: 0.0,
        correct: false,
        details,
    }
}

fn score_sample(
    sample: &BenchmarkSample,
    response: &str,
    duration: Duration,
    opts: &RunOptions,
) -> SampleResult {
    let (score, details) = match &sample.ground_truth {
        GroundTruth::Answer { correct, incorrect } => score_qa(response, correct, incorrect),
        GroundTruth::HallucinationPair {
            right,
            hallucinated,
        } => score_pair(response, right, hallucinated),
        GroundTruth::AtomicClaims(facts) => score_atomic(response, facts),
        GroundTruth::SupportsRefutes { label, evidence } => {
            score_labelled(response, *label, evidence)
        }
        GroundTruth::ContextualReference { context, reference } => {
            score_contextual(response, context, reference)
        }
    };
    SampleResult {
        id: sample.id.clone(),
        prompt: sample.prompt.clone(),
        response: response.to_string(),
        duration,
        score,
        correct: score >= opts.correctness_threshold,
        details,
    }
}

fn score_qa(
    response: &str,
    correct: &[String],
    incorrect: &[String],
) -> (f64, HashMap<String, String>) {
    let r = normalize(response);
    let best_correct = correct
        .iter()
        .map(|c| jaccard(&r, &normalize(c)))
        .fold(0.0_f64, f64::max);
    let best_incorrect = incorrect
        .iter()
        .map(|c| jaccard(&r, &normalize(c)))
        .fold(0.0_f64, f64::max);
    let score = if best_correct > best_incorrect {
        best_correct
    } else if best_correct == best_incorrect {
        // Tie: credit half.
        best_correct * 0.5
    } else {
        // The response is closer to an incorrect reference than any correct one.
        (best_correct - best_incorrect).max(0.0)
    };
    let mut d = HashMap::new();
    d.insert("jaccard_correct".into(), format!("{best_correct:.3}"));
    d.insert("jaccard_incorrect".into(), format!("{best_incorrect:.3}"));
    (score.clamp(0.0, 1.0), d)
}

fn score_pair(response: &str, right: &str, hallucinated: &str) -> (f64, HashMap<String, String>) {
    let r = normalize(response);
    let jr = jaccard(&r, &normalize(right));
    let jh = jaccard(&r, &normalize(hallucinated));
    let mut d = HashMap::new();
    d.insert("jaccard_right".into(), format!("{jr:.3}"));
    d.insert("jaccard_hallucinated".into(), format!("{jh:.3}"));
    let score = if jr > jh {
        // Scale by margin so strongly-right answers score higher than tied ones.
        0.5 + (jr - jh) * 0.5
    } else if jr == jh {
        0.5
    } else {
        0.5 - (jh - jr) * 0.5
    };
    (score.clamp(0.0, 1.0), d)
}

fn score_atomic(response: &str, facts: &[(String, bool)]) -> (f64, HashMap<String, String>) {
    // Proto-FActScore without a claim-decomposition LLM: check which supported
    // facts are echoed (high word overlap) in the model response, and which
    // unsupported facts are NOT echoed. Precision = (echoed supported) /
    // (echoed supported + echoed unsupported).
    let r = normalize(response);
    let mut echoed_supported = 0usize;
    let mut echoed_unsupported = 0usize;
    for (fact, is_true) in facts {
        let overlap = jaccard(&r, &normalize(fact));
        if overlap >= 0.3 {
            if *is_true {
                echoed_supported += 1;
            } else {
                echoed_unsupported += 1;
            }
        }
    }
    let mut d = HashMap::new();
    d.insert("echoed_supported".into(), echoed_supported.to_string());
    d.insert("echoed_unsupported".into(), echoed_unsupported.to_string());
    let total_echoed = echoed_supported + echoed_unsupported;
    let score = if total_echoed == 0 {
        // Model said nothing factually matchable — count as 0.
        0.0
    } else {
        echoed_supported as f64 / total_echoed as f64
    };
    (score, d)
}

fn score_labelled(
    response: &str,
    gold: Label,
    _evidence: &[String],
) -> (f64, HashMap<String, String>) {
    let pred = predict_label(response);
    let mut d = HashMap::new();
    d.insert("predicted".into(), format!("{pred:?}"));
    d.insert("gold".into(), format!("{gold:?}"));
    let correct = pred == gold;
    (if correct { 1.0 } else { 0.0 }, d)
}

fn predict_label(s: &str) -> Label {
    let l = s.to_lowercase();
    let supports = l.contains("supports") || l.contains("supported") || l.contains("true");
    let refutes = l.contains("refutes") || l.contains("refuted") || l.contains("false");
    let nei = l.contains("not enough info")
        || l.contains("insufficient")
        || l.contains("unknown")
        || l.contains("nei");
    // Prefer the most specific label — NEI beats both if present.
    if nei {
        Label::NotEnoughInfo
    } else if supports && !refutes {
        Label::Supports
    } else if refutes && !supports {
        Label::Refutes
    } else {
        Label::NotEnoughInfo
    }
}

fn score_contextual(
    response: &str,
    context: &[String],
    reference: &str,
) -> (f64, HashMap<String, String>) {
    // Two-factor: faithfulness (is the answer grounded in context?) + similarity
    // (does it match the reference?). Both via Jaccard. Final score is min() —
    // any answer that fails either check gets a low score.
    let r = normalize(response);
    let ctx_combined: String = context.join(" ");
    let ctx_norm = normalize(&ctx_combined);
    let ref_norm = normalize(reference);

    let faithful = jaccard(&r, &ctx_norm);
    let similar = jaccard(&r, &ref_norm);

    let mut d = HashMap::new();
    d.insert("faithfulness".into(), format!("{faithful:.3}"));
    d.insert("similarity".into(), format!("{similar:.3}"));
    let score = faithful.min(similar);
    (score, d)
}

fn aggregate(name: &str, samples: Vec<SampleResult>, total_duration: Duration) -> BenchmarkReport {
    let total = samples.len();
    let correct = samples.iter().filter(|s| s.correct).count();
    let mean_score = if total == 0 {
        0.0
    } else {
        samples.iter().map(|s| s.score).sum::<f64>() / total as f64
    };
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };

    let mut per_category: HashMap<String, (usize, usize)> = HashMap::new();
    for s in &samples {
        let cat = s
            .details
            .get("category")
            .cloned()
            .unwrap_or_else(|| "uncategorized".to_string());
        let entry = per_category.entry(cat).or_insert((0, 0));
        entry.0 += 1;
        if s.correct {
            entry.1 += 1;
        }
    }

    BenchmarkReport {
        benchmark: name.to_string(),
        total,
        correct,
        accuracy,
        mean_score,
        total_duration,
        per_category,
        samples,
    }
}

// ---------------------------------------------------------------------------
// Low-level helpers (also used by calibration).

/// Lowercase + alpha-numeric-only whitespace split.
pub(crate) fn normalize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Jaccard over whitespace tokens. Tokens of length 1 are kept (digits matter
/// for factual answers) but stop-words are not filtered — keeps the scorer
/// transparent and language-agnostic.
pub(crate) fn jaccard(a: &str, b: &str) -> f64 {
    let sa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let sb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count();
    let uni = sa.union(&sb).count();
    if uni == 0 {
        0.0
    } else {
        inter as f64 / uni as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_benchmarks::types::{BenchmarkSample, SampleType};

    fn qa_sample(prompt: &str, correct: &[&str], incorrect: &[&str]) -> BenchmarkSample {
        BenchmarkSample::qa(
            "t1",
            prompt,
            correct.iter().map(|s| s.to_string()).collect(),
            incorrect.iter().map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn jaccard_basic() {
        assert!((jaccard("a b c", "b c d") - (2.0 / 4.0)).abs() < 1e-9);
        assert_eq!(jaccard("", ""), 1.0);
        assert_eq!(jaccard("a", "b"), 0.0);
    }

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize("Hello, World!"), "hello world");
    }

    #[test]
    fn qa_scoring_prefers_correct() {
        let (score, _) = score_qa("the answer is paris", &["paris".into()], &["london".into()]);
        assert!(score > 0.0);
        let (low, _) = score_qa(
            "the answer is london",
            &["paris".into()],
            &["london".into()],
        );
        assert!(low == 0.0);
    }

    #[test]
    fn pair_scoring_prefers_right() {
        let (s_right, _) = score_pair("paris is the capital", "paris", "london");
        let (s_wrong, _) = score_pair("london is the capital", "paris", "london");
        assert!(s_right > 0.5);
        assert!(s_wrong < 0.5);
    }

    #[test]
    fn atomic_score_balances_echoed_facts() {
        let facts = vec![
            ("Marie Curie was a physicist.".into(), true),
            ("Marie Curie was born in 1867.".into(), true),
            ("Marie Curie won three Nobel Prizes.".into(), false),
        ];
        let (s, _) = score_atomic("Marie Curie was a physicist born in 1867.", &facts);
        assert!(s > 0.5);
        let (s_bad, _) = score_atomic("Marie Curie won three Nobel Prizes.", &facts);
        assert!(s_bad < 0.5);
    }

    #[test]
    fn label_prediction_supports() {
        assert_eq!(predict_label("The claim is supported."), Label::Supports);
        assert_eq!(predict_label("This is refuted."), Label::Refutes);
        assert_eq!(
            predict_label("Not enough info in the evidence."),
            Label::NotEnoughInfo
        );
    }

    #[test]
    fn run_end_to_end_stub_model() {
        let samples = vec![
            qa_sample("What is 2+2?", &["4"], &["5"]),
            qa_sample("Capital of France?", &["paris"], &["london"]),
        ];
        let report = run("stub", &samples, &RunOptions::default(), |p| {
            if p.contains("France") {
                Ok("paris".into())
            } else {
                Ok("4".into())
            }
        });
        assert_eq!(report.total, 2);
        assert_eq!(report.correct, 2);
        assert!((report.accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn run_tolerates_generator_errors() {
        let samples = vec![qa_sample("q", &["a"], &["b"])];
        let report = run("stub", &samples, &RunOptions::default(), |_| {
            Err("timeout".into())
        });
        assert_eq!(report.total, 1);
        assert_eq!(report.correct, 0);
        assert!(report.samples[0].details.contains_key("error"));
    }

    #[test]
    fn contextual_scoring_uses_min_of_faithful_and_similar() {
        let (s, d) = score_contextual(
            "Paris is the capital",
            &["Paris is the capital of France.".into()],
            "Paris",
        );
        assert!(s > 0.0 && s <= 1.0);
        assert!(d.contains_key("faithfulness"));
        assert!(d.contains_key("similarity"));
    }

    #[test]
    fn sample_type_dispatch_smoke() {
        // Touch every enum variant (compile-time check via exhaustive match).
        let t = SampleType::QA;
        let s = match t {
            SampleType::QA => "qa",
            SampleType::HallucinationPair => "pair",
            SampleType::AtomicClaims => "atomic",
            SampleType::ClaimVsEvidence => "label",
            SampleType::ContextualQA => "ctx",
        };
        assert_eq!(s, "qa");
    }
}
