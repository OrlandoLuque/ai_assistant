//! Fitness evaluation — scores a `Unit`'s output against an `EvalExample`.
//! Built-in evaluators cover exact-match, substring containment, regex,
//! JSON-schema validation, and a wrapping composite. `LlmJudgeEvaluator`
//! takes any `LlmClient` and defers the judgement.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::config::{FitnessObjective, Metric, ProviderFingerprint};
use super::eval::EvalExample;
use super::llm::LlmClient;

/// Numeric score across several metrics. `aggregate` is the single scalar
/// used by selection / replacement; it is computed from `per_metric` via the
/// configured `FitnessObjective`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessScore {
    pub per_metric: HashMap<String, f64>,
    pub aggregate: f64,
    pub sample_count: u32,
    pub fingerprint: ProviderFingerprint,
}

impl FitnessScore {
    pub fn new(fingerprint: ProviderFingerprint) -> Self {
        Self {
            per_metric: HashMap::new(),
            aggregate: 0.0,
            sample_count: 0,
            fingerprint,
        }
    }

    pub fn set(&mut self, metric: Metric, value: f64) {
        self.per_metric.insert(metric.to_string(), value);
    }

    pub fn get(&self, metric: &Metric) -> Option<f64> {
        self.per_metric.get(&metric.to_string()).copied()
    }

    /// Compute `aggregate` using the given objective. When metrics are
    /// missing from `per_metric`, they count as 0.
    pub fn recompute_aggregate(&mut self, objective: &FitnessObjective) {
        self.aggregate = match objective {
            FitnessObjective::Single => self.per_metric.values().next().copied().unwrap_or(0.0),
            FitnessObjective::WeightedSum { weights } => {
                let total_w: f32 = weights.iter().map(|(_, w)| *w).sum();
                if total_w <= 0.0 {
                    return;
                }
                let mut acc = 0.0;
                for (m, w) in weights {
                    if let Some(v) = self.per_metric.get(&m.to_string()) {
                        acc += (*v) * (*w as f64);
                    }
                }
                acc / total_w as f64
            }
            FitnessObjective::Pareto { objectives } => {
                // Pareto ranking happens outside per-unit; as a scalar proxy
                // we use the mean of the listed metrics, which keeps
                // aggregate-based comparisons sane when a non-Pareto code
                // path needs it.
                if objectives.is_empty() {
                    return;
                }
                let sum: f64 = objectives
                    .iter()
                    .map(|m| self.per_metric.get(&m.to_string()).copied().unwrap_or(0.0))
                    .sum();
                sum / (objectives.len() as f64)
            }
        };
    }
}

/// Evaluate a unit's (LLM-parsed) output against an example.
pub trait FitnessEvaluator: Send + Sync {
    fn metrics(&self) -> Vec<Metric>;

    fn evaluate(
        &self,
        example: &EvalExample,
        parsed_output: &str,
        fingerprint: &ProviderFingerprint,
    ) -> FitnessScore;
}

// =============================================================================
// Built-in evaluators
// =============================================================================

pub struct ExactMatchEvaluator;

impl FitnessEvaluator for ExactMatchEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        vec![Metric::ExactMatch]
    }

    fn evaluate(
        &self,
        example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let expected = example.expected.as_deref().unwrap_or("");
        let hit = parsed.trim() == expected.trim();
        let mut score = FitnessScore::new(fp.clone());
        score.set(Metric::ExactMatch, if hit { 1.0 } else { 0.0 });
        score.sample_count = 1;
        score.aggregate = if hit { 1.0 } else { 0.0 };
        score
    }
}

pub struct ContainsEvaluator {
    pub case_insensitive: bool,
}

impl Default for ContainsEvaluator {
    fn default() -> Self {
        Self {
            case_insensitive: true,
        }
    }
}

impl FitnessEvaluator for ContainsEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        vec![Metric::Contains]
    }

    fn evaluate(
        &self,
        example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let expected = example.expected.as_deref().unwrap_or("");
        let hit = if self.case_insensitive {
            parsed.to_lowercase().contains(&expected.to_lowercase())
        } else {
            parsed.contains(expected)
        };
        let mut score = FitnessScore::new(fp.clone());
        score.set(Metric::Contains, if hit { 1.0 } else { 0.0 });
        score.sample_count = 1;
        score.aggregate = if hit { 1.0 } else { 0.0 };
        score
    }
}

pub struct RegexEvaluator {
    pub pattern: String,
}

impl RegexEvaluator {
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
        }
    }
}

impl FitnessEvaluator for RegexEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        vec![Metric::RegexMatch]
    }

    fn evaluate(
        &self,
        _example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let re = regex::Regex::new(&self.pattern);
        let hit = match re {
            Ok(r) => r.is_match(parsed),
            Err(_) => false,
        };
        let mut score = FitnessScore::new(fp.clone());
        score.set(Metric::RegexMatch, if hit { 1.0 } else { 0.0 });
        score.sample_count = 1;
        score.aggregate = if hit { 1.0 } else { 0.0 };
        score
    }
}

pub struct JsonSchemaEvaluator {
    /// Required top-level keys that must be present (keys only; no value types).
    pub required_keys: Vec<String>,
}

impl JsonSchemaEvaluator {
    pub fn new(required_keys: Vec<String>) -> Self {
        Self { required_keys }
    }
}

impl FitnessEvaluator for JsonSchemaEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        vec![Metric::JsonSchemaValid]
    }

    fn evaluate(
        &self,
        _example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let parsed_json: serde_json::Result<serde_json::Value> = serde_json::from_str(parsed);
        let hit = match parsed_json {
            Ok(serde_json::Value::Object(m)) => {
                self.required_keys.iter().all(|k| m.contains_key(k))
            }
            _ => false,
        };
        let mut score = FitnessScore::new(fp.clone());
        score.set(Metric::JsonSchemaValid, if hit { 1.0 } else { 0.0 });
        score.sample_count = 1;
        score.aggregate = if hit { 1.0 } else { 0.0 };
        score
    }
}

/// LLM-judge evaluator. Wraps any `LlmClient` and asks it to grade the
/// parsed output in `[0, 1]`.
pub struct LlmJudgeEvaluator {
    client: Arc<dyn LlmClient>,
    judge_template: String,
}

impl LlmJudgeEvaluator {
    pub fn new(client: Arc<dyn LlmClient>, judge_template: impl Into<String>) -> Self {
        Self {
            client,
            judge_template: judge_template.into(),
        }
    }

    /// Convenience — use a built-in judge prompt that just asks for a score.
    pub fn default(client: Arc<dyn LlmClient>) -> Self {
        Self::new(
            client,
            "You are grading a model answer. Output ONLY a decimal score in [0,1].\n\n\
             Expected: {expected}\n\nActual: {actual}\n\nScore:"
                .to_string(),
        )
    }
}

impl FitnessEvaluator for LlmJudgeEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        vec![Metric::LlmJudgeScore]
    }

    fn evaluate(
        &self,
        example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let filled = self
            .judge_template
            .replace("{expected}", example.expected.as_deref().unwrap_or(""))
            .replace("{actual}", parsed);
        let value = match self.client.complete(&filled) {
            Ok(resp) => parse_decimal(&resp.text).unwrap_or(0.0),
            Err(_) => 0.0,
        };
        let mut score = FitnessScore::new(fp.clone());
        score.set(Metric::LlmJudgeScore, value);
        score.sample_count = 1;
        score.aggregate = value;
        score
    }
}

fn parse_decimal(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    // Accept the whole string first.
    if let Ok(v) = trimmed.parse::<f64>() {
        return Some(v.clamp(0.0, 1.0));
    }
    // Otherwise pick the first numeric run.
    let mut buf = String::new();
    for c in trimmed.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            buf.push(c);
        } else if !buf.is_empty() {
            break;
        }
    }
    buf.parse::<f64>().ok().map(|v| v.clamp(0.0, 1.0))
}

/// Composite evaluator — runs children and merges their per-metric scores.
/// The aggregate is the mean of each child's aggregate.
pub struct CompositeEvaluator {
    pub children: Vec<Arc<dyn FitnessEvaluator>>,
}

impl CompositeEvaluator {
    pub fn new(children: Vec<Arc<dyn FitnessEvaluator>>) -> Self {
        Self { children }
    }
}

impl FitnessEvaluator for CompositeEvaluator {
    fn metrics(&self) -> Vec<Metric> {
        let mut out = Vec::new();
        for c in &self.children {
            for m in c.metrics() {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
        out
    }

    fn evaluate(
        &self,
        example: &EvalExample,
        parsed: &str,
        fp: &ProviderFingerprint,
    ) -> FitnessScore {
        let mut merged = FitnessScore::new(fp.clone());
        let mut sum_agg = 0.0;
        for c in &self.children {
            let s = c.evaluate(example, parsed, fp);
            for (k, v) in s.per_metric {
                merged.per_metric.insert(k, v);
            }
            sum_agg += s.aggregate;
        }
        merged.sample_count = 1;
        merged.aggregate = if self.children.is_empty() {
            0.0
        } else {
            sum_agg / self.children.len() as f64
        };
        merged
    }
}

// =============================================================================
// Pareto (NSGA-II-style fast non-dominated sort + crowding distance)
// =============================================================================

/// Non-dominated sort on a slice of scores. Returns parallel vec: for each
/// input score, the Pareto front rank (0 = best front). Metrics in
/// `objectives` are treated as "higher is better".
pub fn pareto_ranks(scores: &[&FitnessScore], objectives: &[Metric]) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut ranks = vec![0usize; n];
    let mut dominated_counts = vec![0usize; n];
    let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];
    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if dominates_all(scores[p], scores[q], objectives) {
                dominates[p].push(q);
            } else if dominates_all(scores[q], scores[p], objectives) {
                dominated_counts[p] += 1;
            }
        }
    }
    let mut front: Vec<usize> = (0..n).filter(|&i| dominated_counts[i] == 0).collect();
    let mut rank = 0;
    while !front.is_empty() {
        for &p in &front {
            ranks[p] = rank;
        }
        let mut next = Vec::new();
        for &p in &front {
            for &q in &dominates[p] {
                dominated_counts[q] -= 1;
                if dominated_counts[q] == 0 {
                    next.push(q);
                }
            }
        }
        rank += 1;
        front = next;
    }
    ranks
}

fn dominates_all(a: &FitnessScore, b: &FitnessScore, metrics: &[Metric]) -> bool {
    let mut strictly_better = false;
    for m in metrics {
        let av = a.per_metric.get(&m.to_string()).copied().unwrap_or(0.0);
        let bv = b.per_metric.get(&m.to_string()).copied().unwrap_or(0.0);
        if av < bv {
            return false;
        }
        if av > bv {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Crowding distance for each score (parallel vec). Larger = less crowded.
pub fn crowding_distance(scores: &[&FitnessScore], objectives: &[Metric]) -> Vec<f64> {
    let n = scores.len();
    let mut dist = vec![0.0f64; n];
    if n < 3 {
        return dist.into_iter().map(|_| f64::INFINITY).collect();
    }
    for m in objectives {
        let mut indexed: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.per_metric.get(&m.to_string()).copied().unwrap_or(0.0)))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let min = indexed.first().map(|t| t.1).unwrap_or(0.0);
        let max = indexed.last().map(|t| t.1).unwrap_or(0.0);
        let range = (max - min).max(f64::EPSILON);
        dist[indexed[0].0] = f64::INFINITY;
        dist[indexed[n - 1].0] = f64::INFINITY;
        for k in 1..n - 1 {
            dist[indexed[k].0] += (indexed[k + 1].1 - indexed[k - 1].1) / range;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::super::llm::MockLlmClient;
    use super::*;

    fn fp() -> ProviderFingerprint {
        ProviderFingerprint::new("test", "mock")
    }

    #[test]
    fn exact_match_hits() {
        let ev = ExactMatchEvaluator;
        let ex = EvalExample::new("1", "q").with_expected("hello");
        let s = ev.evaluate(&ex, "hello", &fp());
        assert_eq!(s.aggregate, 1.0);
    }

    #[test]
    fn exact_match_misses() {
        let ev = ExactMatchEvaluator;
        let ex = EvalExample::new("1", "q").with_expected("hello");
        let s = ev.evaluate(&ex, "goodbye", &fp());
        assert_eq!(s.aggregate, 0.0);
    }

    #[test]
    fn contains_case_insensitive() {
        let ev = ContainsEvaluator::default();
        let ex = EvalExample::new("1", "q").with_expected("Hello");
        let s = ev.evaluate(&ex, "oh hello world", &fp());
        assert_eq!(s.aggregate, 1.0);
    }

    #[test]
    fn regex_matches() {
        let ev = RegexEvaluator::new(r"\d+");
        let ex = EvalExample::new("1", "q");
        let s = ev.evaluate(&ex, "count is 42", &fp());
        assert_eq!(s.aggregate, 1.0);
    }

    #[test]
    fn json_schema_valid() {
        let ev = JsonSchemaEvaluator::new(vec!["name".into(), "age".into()]);
        let ex = EvalExample::new("1", "q");
        let s = ev.evaluate(&ex, r#"{"name":"a","age":1}"#, &fp());
        assert_eq!(s.aggregate, 1.0);
        let s2 = ev.evaluate(&ex, r#"{"name":"a"}"#, &fp());
        assert_eq!(s2.aggregate, 0.0);
    }

    #[test]
    fn composite_merges_metrics() {
        let ev = CompositeEvaluator::new(vec![
            Arc::new(ExactMatchEvaluator),
            Arc::new(ContainsEvaluator::default()),
        ]);
        let ex = EvalExample::new("1", "q").with_expected("hello");
        let s = ev.evaluate(&ex, "hello", &fp());
        assert!(s.per_metric.contains_key("exact_match"));
        assert!(s.per_metric.contains_key("contains"));
        assert_eq!(s.aggregate, 1.0);
    }

    #[test]
    fn llm_judge_uses_client() {
        let client = Arc::new(MockLlmClient::returning("0.75"));
        let ev = LlmJudgeEvaluator::default(client);
        let ex = EvalExample::new("1", "q").with_expected("hi");
        let s = ev.evaluate(&ex, "hello", &fp());
        assert!((s.aggregate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn pareto_front_finds_dominators() {
        let fp = fp();
        let mut a = FitnessScore::new(fp.clone());
        a.set(Metric::Accuracy, 0.9);
        a.set(Metric::LatencyMs, 100.0);
        let mut b = FitnessScore::new(fp.clone());
        b.set(Metric::Accuracy, 0.8);
        b.set(Metric::LatencyMs, 50.0);
        let mut c = FitnessScore::new(fp);
        c.set(Metric::Accuracy, 0.5);
        c.set(Metric::LatencyMs, 200.0);
        let scores = vec![&a, &b, &c];
        let ranks = pareto_ranks(
            &scores,
            &[Metric::Accuracy, Metric::Custom("neg_latency".into())],
        );
        // Custom metric missing → treated as 0; so only Accuracy matters.
        // a dominates b and c; b dominates c.
        assert_eq!(ranks[0], 0);
        assert_eq!(ranks[2], 2);
    }

    #[test]
    fn recompute_aggregate_weighted() {
        let fp = fp();
        let mut s = FitnessScore::new(fp);
        s.set(Metric::Accuracy, 0.8);
        s.set(Metric::LatencyMs, 0.2);
        s.recompute_aggregate(&FitnessObjective::WeightedSum {
            weights: vec![(Metric::Accuracy, 3.0), (Metric::LatencyMs, 1.0)],
        });
        assert!((s.aggregate - ((0.8 * 3.0 + 0.2 * 1.0) / 4.0)).abs() < 1e-9);
    }
}
