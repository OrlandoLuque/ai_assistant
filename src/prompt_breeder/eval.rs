//! Evaluation dataset + output parsing. The breeder feeds each `Unit` the
//! dataset examples one at a time, gets an LLM response, runs the parser,
//! and passes the parsed output to whichever `FitnessEvaluator` is wired.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::OutputParser;

/// A single evaluation record. `expected` is the authoritative answer for
/// metrics that need one (ExactMatch, Contains, Regex). `metadata` is opaque
/// to the framework and is passed through to custom evaluators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    pub id: String,
    pub input: String,
    pub expected: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl EvalExample {
    pub fn new(id: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Collection of eval examples with optional weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalDataset {
    pub examples: Vec<EvalExample>,
}

impl EvalDataset {
    pub fn new(examples: Vec<EvalExample>) -> Self {
        Self { examples }
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EvalExample> {
        self.examples.iter()
    }

    pub fn push(&mut self, ex: EvalExample) {
        self.examples.push(ex);
    }
}

/// Apply the configured parser to an LLM response before it reaches the
/// fitness evaluator. Returns the slice of the response that callers should
/// use for scoring. Never allocates unless the parser transforms the text.
pub fn parse_output(parser: &OutputParser, raw: &str) -> String {
    match parser {
        OutputParser::Raw => raw.to_string(),
        OutputParser::StripMarkdown => strip_markdown_fences(raw),
        OutputParser::JsonFirst => extract_first_json(raw).unwrap_or_else(|| raw.to_string()),
        OutputParser::AfterMarker { marker } => raw
            .rfind(marker.as_str())
            .map(|i| raw[i + marker.len()..].trim().to_string())
            .unwrap_or_else(|| raw.to_string()),
        OutputParser::RegexCapture { pattern, group } => {
            regex_capture(pattern, raw, *group).unwrap_or_else(|| raw.to_string())
        }
    }
}

fn strip_markdown_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return raw.to_string();
    }
    // Drop first fence line.
    let after_open = match trimmed.find('\n') {
        Some(i) => &trimmed[i + 1..],
        None => return raw.to_string(),
    };
    // Drop trailing fence.
    if let Some(end) = after_open.rfind("```") {
        after_open[..end].trim().to_string()
    } else {
        after_open.trim().to_string()
    }
}

fn extract_first_json(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    let mut start = None;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut open_char = b'{';
    for (i, &b) in bytes.iter().enumerate() {
        if start.is_none() && (b == b'{' || b == b'[') {
            start = Some(i);
            open_char = b;
            depth = 1;
            in_string = false;
            escape = false;
            continue;
        }
        if start.is_some() {
            if in_string {
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                c if c == open_char => depth += 1,
                b'}' if open_char == b'{' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(raw[start.unwrap()..=i].to_string());
                    }
                }
                b']' if open_char == b'[' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(raw[start.unwrap()..=i].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn regex_capture(pattern: &str, raw: &str, group: usize) -> Option<String> {
    // Minimal, opt-in regex via the `regex` crate which is already in the tree.
    let re = regex::Regex::new(pattern).ok()?;
    let caps = re.captures(raw)?;
    caps.get(group).map(|m| m.as_str().to_string())
}

// =============================================================================
// Dataset augmentation (deterministic variants; LLM variant lives in llm.rs)
// =============================================================================

use super::config::{EvalAugmenter, Perturbation};
use super::rng::BreederRng;

/// Apply a deterministic augmenter to the dataset. LLM-based augmentation
/// happens at the breeder level because it needs an `LlmClient`. Returns
/// the number of examples added.
pub fn augment_deterministic(
    dataset: &mut EvalDataset,
    augmenter: &EvalAugmenter,
    rng: &mut BreederRng,
) -> usize {
    match augmenter {
        EvalAugmenter::Bootstrap { factor } => bootstrap(dataset, *factor, rng),
        EvalAugmenter::Adversarial { perturbation } => adversarial(dataset, perturbation, rng),
        // LLM-based augmenters are skipped here; breeder handles them.
        EvalAugmenter::LlmSynthesized { .. } => 0,
    }
}

fn bootstrap(dataset: &mut EvalDataset, factor: f32, rng: &mut BreederRng) -> usize {
    let base = dataset.examples.clone();
    if base.is_empty() || factor <= 0.0 {
        return 0;
    }
    let target = (base.len() as f32 * factor).round() as usize;
    let mut added = 0;
    for i in 0..target {
        let src = rng.gen_range_usize(base.len());
        let mut ex = base[src].clone();
        ex.id = format!("{}#boot{}", ex.id, i);
        dataset.push(ex);
        added += 1;
    }
    added
}

fn adversarial(dataset: &mut EvalDataset, pert: &Perturbation, rng: &mut BreederRng) -> usize {
    let base = dataset.examples.clone();
    let mut added = 0;
    for (i, ex) in base.iter().enumerate() {
        let perturbed_input = apply_perturbation(&ex.input, pert, rng);
        if perturbed_input == ex.input {
            continue;
        }
        let mut new = ex.clone();
        new.id = format!("{}#adv{}", ex.id, i);
        new.input = perturbed_input;
        dataset.push(new);
        added += 1;
    }
    added
}

fn apply_perturbation(text: &str, pert: &Perturbation, rng: &mut BreederRng) -> String {
    match pert {
        Perturbation::TypoInjection { rate } => {
            let mut s: Vec<char> = text.chars().collect();
            if s.is_empty() {
                return text.to_string();
            }
            let n_typos = ((s.len() as f32) * *rate).max(1.0) as usize;
            for _ in 0..n_typos {
                let i = rng.gen_range_usize(s.len());
                let j = rng.gen_range_usize(s.len());
                s.swap(i, j);
            }
            s.into_iter().collect()
        }
        Perturbation::CaseFlip => text
            .chars()
            .map(|c| {
                if c.is_ascii_uppercase() {
                    c.to_ascii_lowercase()
                } else if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c
                }
            })
            .collect(),
        Perturbation::PunctuationStrip => {
            text.chars().filter(|c| !c.is_ascii_punctuation()).collect()
        }
        Perturbation::TokenShuffle { window } => {
            let mut tokens: Vec<&str> = text.split_whitespace().collect();
            if tokens.len() < 2 || *window < 2 {
                return text.to_string();
            }
            let w = (*window).min(tokens.len());
            let start = rng.gen_range_usize(tokens.len().saturating_sub(w) + 1);
            let end = (start + w).min(tokens.len());
            let slice = &mut tokens[start..end];
            rng.shuffle(slice);
            tokens.join(" ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_raw_is_identity() {
        assert_eq!(parse_output(&OutputParser::Raw, "hello"), "hello");
    }

    #[test]
    fn parser_strips_markdown_fences() {
        let raw = "```rust\nlet x = 1;\n```";
        let parsed = parse_output(&OutputParser::StripMarkdown, raw);
        assert_eq!(parsed, "let x = 1;");
    }

    #[test]
    fn parser_extracts_first_json() {
        let raw = "thinking...\n{\"answer\": 42} trailing";
        let parsed = parse_output(&OutputParser::JsonFirst, raw);
        assert_eq!(parsed, "{\"answer\": 42}");
    }

    #[test]
    fn parser_after_marker_takes_last() {
        let raw = "Answer: hello\nAnswer: world";
        let parsed = parse_output(
            &OutputParser::AfterMarker {
                marker: "Answer:".into(),
            },
            raw,
        );
        assert_eq!(parsed, "world");
    }

    #[test]
    fn parser_regex_capture() {
        let raw = "score = 0.87 pct";
        let parsed = parse_output(
            &OutputParser::RegexCapture {
                pattern: r"score\s*=\s*([0-9.]+)".into(),
                group: 1,
            },
            raw,
        );
        assert_eq!(parsed, "0.87");
    }

    #[test]
    fn bootstrap_augments_dataset() {
        let mut d = EvalDataset::new(vec![
            EvalExample::new("a", "hello"),
            EvalExample::new("b", "world"),
        ]);
        let mut r = BreederRng::from_seed(1);
        let added =
            augment_deterministic(&mut d, &EvalAugmenter::Bootstrap { factor: 2.0 }, &mut r);
        assert_eq!(added, 4);
        assert_eq!(d.len(), 6);
    }

    #[test]
    fn adversarial_caseflip_augments() {
        let mut d = EvalDataset::new(vec![EvalExample::new("a", "Hello")]);
        let mut r = BreederRng::from_seed(1);
        let added = augment_deterministic(
            &mut d,
            &EvalAugmenter::Adversarial {
                perturbation: Perturbation::CaseFlip,
            },
            &mut r,
        );
        assert_eq!(added, 1);
        assert_eq!(d.examples[1].input, "hELLO");
    }

    #[test]
    fn llm_augmenter_is_noop_here() {
        let mut d = EvalDataset::new(vec![EvalExample::new("a", "hi")]);
        let mut r = BreederRng::from_seed(1);
        let added = augment_deterministic(
            &mut d,
            &EvalAugmenter::LlmSynthesized {
                n: 5,
                style: "adv".into(),
            },
            &mut r,
        );
        assert_eq!(added, 0);
    }
}
