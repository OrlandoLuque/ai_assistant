//! HaluEval QA loader.
//!
//! Dataset: <https://github.com/RUCAIBox/HaluEval>
//! License: MIT
//! Paper: Li et al. 2023, "HaluEval: A Large-Scale Hallucination Evaluation
//! Benchmark for Large Language Models".
//!
//! Schema: JSONL, one object per line:
//!   {"knowledge": ..., "question": ..., "right_answer": ..., "hallucinated_answer": ...}
//! We load the `qa_data.json` split (10k pairs).

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::http::{download_file, DownloadOptions};
use crate::eval_benchmarks::types::{
    BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, SampleType,
};

/// HaluEval QA subset loader (10k right/hallucinated answer pairs).
#[derive(Debug, Default, Clone)]
pub struct HaluEvalLoader;

static HALUEVAL_URLS: &[&str] =
    &["https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"];

impl BenchmarkLoader for HaluEvalLoader {
    fn name(&self) -> &'static str {
        "halueval_qa"
    }
    fn description(&self) -> &'static str {
        "HaluEval QA: 10k pairs of (right_answer, hallucinated_answer) with knowledge context."
    }
    fn license(&self) -> &'static str {
        "MIT"
    }
    fn citation(&self) -> &'static str {
        "Li et al. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. EMNLP 2023."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        HALUEVAL_URLS
    }
    fn sample_type(&self) -> SampleType {
        SampleType::HallucinationPair
    }
    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        let dest = cache_dir.join("qa_data.json");
        download_file(self.download_urls(), &dest, &DownloadOptions::default())
    }
    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BenchmarkError::Io(format!("read {}: {e}", path.display())))?;
        parse_halueval_jsonl(&content, limit)
    }
}

fn parse_halueval_jsonl(
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

        let question = v
            .get("question")
            .and_then(|x| x.as_str())
            .ok_or_else(|| BenchmarkError::Schema(format!("line {}: missing 'question'", idx + 1)))?
            .to_string();
        let right = v
            .get("right_answer")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                BenchmarkError::Schema(format!("line {}: missing 'right_answer'", idx + 1))
            })?
            .to_string();
        let hallucinated = v
            .get("hallucinated_answer")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                BenchmarkError::Schema(format!("line {}: missing 'hallucinated_answer'", idx + 1))
            })?
            .to_string();
        let knowledge = v
            .get("knowledge")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();

        let prompt = if knowledge.is_empty() {
            question.clone()
        } else {
            format!("Knowledge: {knowledge}\n\nQuestion: {question}")
        };

        let mut sample = BenchmarkSample {
            id: format!("halueval_qa_{idx}"),
            prompt,
            category: None,
            ground_truth: GroundTruth::HallucinationPair {
                right,
                hallucinated,
            },
            sample_type: SampleType::HallucinationPair,
            metadata: Default::default(),
        };
        if !knowledge.is_empty() {
            sample.metadata.insert("knowledge".into(), knowledge);
        }
        sample.metadata.insert("question".into(), question);
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

    const FIXTURE: &str = include_str!("../fixtures/halueval_qa_sample.jsonl");

    #[test]
    fn loader_metadata() {
        let l = HaluEvalLoader;
        assert_eq!(l.name(), "halueval_qa");
        assert_eq!(l.sample_type(), SampleType::HallucinationPair);
        assert!(!l.download_urls().is_empty());
    }

    #[test]
    fn parse_three_samples() {
        let samples = parse_halueval_jsonl(FIXTURE, None).unwrap();
        assert_eq!(samples.len(), 3);
        let s0 = &samples[0];
        assert!(s0.prompt.contains("Eiffel"));
        match &s0.ground_truth {
            GroundTruth::HallucinationPair {
                right,
                hallucinated,
            } => {
                assert!(right.contains("Paris"));
                assert!(hallucinated.contains("London"));
            }
            _ => panic!("expected HallucinationPair"),
        }
        assert!(s0.metadata.contains_key("knowledge"));
    }

    #[test]
    fn limit_respected() {
        let samples = parse_halueval_jsonl(FIXTURE, Some(1)).unwrap();
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn missing_field_errors() {
        let bad = r#"{"question": "q"}"#;
        let err = parse_halueval_jsonl(bad, None).unwrap_err();
        assert!(matches!(err, BenchmarkError::Schema(_)));
    }

    #[test]
    fn malformed_json_errors() {
        let bad = "{not json";
        let err = parse_halueval_jsonl(bad, None).unwrap_err();
        assert!(matches!(err, BenchmarkError::Parse(_)));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let content = "\n\n".to_string() + FIXTURE;
        let samples = parse_halueval_jsonl(&content, None).unwrap();
        assert_eq!(samples.len(), 3);
    }
}
