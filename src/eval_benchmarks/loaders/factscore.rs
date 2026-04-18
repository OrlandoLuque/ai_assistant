//! FActScore loader.
//!
//! Dataset: <https://github.com/shmsw25/FActScore>
//! License: MIT
//! Paper: Min et al. 2023, "FActScore: Fine-grained Atomic Evaluation of
//! Factual Precision in Long-Form Text Generation".
//!
//! Schema: JSONL, one object per line:
//!   {"topic": ..., "prompt": ..., "atomic_facts": [{"fact": ..., "label": bool}, ...]}
//!
//! The upstream release ships annotations under `data/` with minor format
//! variants; we normalize to the shape above.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::http::{download_file, DownloadOptions};
use crate::eval_benchmarks::types::{
    BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, SampleType,
};

/// FActScore bios loader.
#[derive(Debug, Default, Clone)]
pub struct FactScoreLoader;

static FACTSCORE_URLS: &[&str] =
    &["https://raw.githubusercontent.com/shmsw25/FActScore/main/data/bio_ChatGPT_unlabeled.jsonl"];

impl BenchmarkLoader for FactScoreLoader {
    fn name(&self) -> &'static str {
        "factscore"
    }
    fn description(&self) -> &'static str {
        "FActScore: atomic-fact decomposition with supported/unsupported labels (bios)."
    }
    fn license(&self) -> &'static str {
        "MIT"
    }
    fn citation(&self) -> &'static str {
        "Min et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long-Form Text Generation. EMNLP 2023."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        FACTSCORE_URLS
    }
    fn sample_type(&self) -> SampleType {
        SampleType::AtomicClaims
    }
    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        let dest = cache_dir.join("factscore.jsonl");
        download_file(self.download_urls(), &dest, &DownloadOptions::default())
    }
    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BenchmarkError::Io(format!("read {}: {e}", path.display())))?;
        parse_factscore_jsonl(&content, limit)
    }
}

fn parse_factscore_jsonl(
    content: &str,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
    let mut out = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| BenchmarkError::Parse(format!("line {}: {e}", idx + 1)))?;

        let topic = v
            .get("topic")
            .and_then(|x| x.as_str())
            .ok_or_else(|| BenchmarkError::Schema(format!("line {}: missing 'topic'", idx + 1)))?
            .to_string();
        let prompt = v
            .get("prompt")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Tell me a bio of {topic}."));

        let facts_value = v.get("atomic_facts").ok_or_else(|| {
            BenchmarkError::Schema(format!("line {}: missing 'atomic_facts'", idx + 1))
        })?;
        let facts_arr = facts_value.as_array().ok_or_else(|| {
            BenchmarkError::Schema(format!("line {}: 'atomic_facts' must be an array", idx + 1))
        })?;

        let mut atomic: Vec<(String, bool)> = Vec::with_capacity(facts_arr.len());
        for (fidx, f) in facts_arr.iter().enumerate() {
            let fact = f
                .get("fact")
                .and_then(|x| x.as_str())
                .ok_or_else(|| {
                    BenchmarkError::Schema(format!(
                        "line {}.facts[{}]: missing 'fact'",
                        idx + 1,
                        fidx
                    ))
                })?
                .to_string();
            // Accept bool, or string "S"/"NS"/"Supported"/"Unsupported" as labels.
            let label = match f.get("label") {
                Some(serde_json::Value::Bool(b)) => *b,
                Some(serde_json::Value::String(s)) => {
                    let u = s.to_uppercase();
                    u == "S" || u == "SUPPORTED" || u == "TRUE"
                }
                _ => {
                    return Err(BenchmarkError::Schema(format!(
                        "line {}.facts[{}]: missing or invalid 'label'",
                        idx + 1,
                        fidx
                    )))
                }
            };
            atomic.push((fact, label));
        }

        let mut sample = BenchmarkSample {
            id: format!("factscore_{idx}"),
            prompt,
            category: Some(topic.clone()),
            ground_truth: GroundTruth::AtomicClaims(atomic),
            sample_type: SampleType::AtomicClaims,
            metadata: Default::default(),
        };
        sample.metadata.insert("topic".into(), topic);
        out.push(sample);

        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/factscore_sample.jsonl");

    #[test]
    fn loader_metadata() {
        let l = FactScoreLoader;
        assert_eq!(l.name(), "factscore");
        assert_eq!(l.sample_type(), SampleType::AtomicClaims);
    }

    #[test]
    fn parse_two_samples() {
        let samples = parse_factscore_jsonl(FIXTURE, None).unwrap();
        assert_eq!(samples.len(), 2);
        match &samples[0].ground_truth {
            GroundTruth::AtomicClaims(facts) => {
                assert_eq!(facts.len(), 4);
                assert!(facts[0].1); // Marie Curie was a physicist -> true
                assert!(!facts[2].1); // Three Nobel prizes -> false
            }
            _ => panic!("expected AtomicClaims"),
        }
        assert_eq!(samples[0].category.as_deref(), Some("Marie Curie"));
    }

    #[test]
    fn string_labels_accepted() {
        let line = r#"{"topic": "X", "atomic_facts": [{"fact": "F1", "label": "S"}, {"fact": "F2", "label": "NS"}]}"#;
        let samples = parse_factscore_jsonl(line, None).unwrap();
        match &samples[0].ground_truth {
            GroundTruth::AtomicClaims(facts) => {
                assert!(facts[0].1);
                assert!(!facts[1].1);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn missing_atomic_facts_errors() {
        let bad = r#"{"topic": "X"}"#;
        assert!(matches!(
            parse_factscore_jsonl(bad, None),
            Err(BenchmarkError::Schema(_))
        ));
    }

    #[test]
    fn limit_respected() {
        let samples = parse_factscore_jsonl(FIXTURE, Some(1)).unwrap();
        assert_eq!(samples.len(), 1);
    }
}
