//! RAGAS WikiQA-style loader.
//!
//! Fetches from the HuggingFace datasets-server API to avoid introducing a
//! parquet dependency. The API returns JSON with a uniform shape:
//!
//! ```json
//! {"rows": [{"row_idx": 0, "row": {"question": ..., "contexts": [...], "ground_truth": ..., "answer": ...}}, ...]}
//! ```
//!
//! License: Apache-2.0 (RAGAS paper + WikiEval corpus used for tutorials).
//! Paper: Es et al. 2024, "RAGAS: Automated Evaluation of Retrieval Augmented
//! Generation". EACL.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::http::{download_file, DownloadOptions};
use crate::eval_benchmarks::types::{
    BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, SampleType,
};

/// RAGAS WikiQA loader.
#[derive(Debug, Default, Clone)]
pub struct RagasLoader;

// HuggingFace datasets-server endpoint. We request 100 rows; that is the API's
// per-request cap and is enough for smoke-tests and calibration.
static RAGAS_URLS: &[&str] = &[
    "https://datasets-server.huggingface.co/rows?dataset=explodinggradients%2FWikiEval&config=default&split=train&offset=0&length=100",
];

impl BenchmarkLoader for RagasLoader {
    fn name(&self) -> &'static str {
        "ragas_wikiqa"
    }
    fn description(&self) -> &'static str {
        "RAGAS WikiEval: question + retrieved contexts + ground-truth answer for faithfulness scoring."
    }
    fn license(&self) -> &'static str {
        "Apache-2.0"
    }
    fn citation(&self) -> &'static str {
        "Es et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        RAGAS_URLS
    }
    fn sample_type(&self) -> SampleType {
        SampleType::ContextualQA
    }
    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        let dest = cache_dir.join("wikieval.json");
        download_file(self.download_urls(), &dest, &DownloadOptions::default())
    }
    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BenchmarkError::Io(format!("read {}: {e}", path.display())))?;
        parse_ragas_json(&content, limit)
    }
}

fn parse_ragas_json(
    content: &str,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
    let v: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| BenchmarkError::Parse(format!("RAGAS JSON: {e}")))?;

    let rows = v
        .get("rows")
        .and_then(|x| x.as_array())
        .ok_or_else(|| BenchmarkError::Schema("RAGAS: missing 'rows' array".into()))?;

    let mut out = Vec::new();
    for (idx, row_wrap) in rows.iter().enumerate() {
        let row = row_wrap
            .get("row")
            .ok_or_else(|| BenchmarkError::Schema(format!("row {idx}: missing 'row' field")))?;

        let question = row
            .get("question")
            .and_then(|x| x.as_str())
            .ok_or_else(|| BenchmarkError::Schema(format!("row {idx}: missing 'question'")))?
            .to_string();

        let contexts: Vec<String> = row
            .get("contexts")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // ground_truth can appear as scalar string, array of strings, or "answer" fallback.
        let reference = extract_reference(row)
            .ok_or_else(|| BenchmarkError::Schema(format!("row {idx}: missing ground truth")))?;

        let mut sample = BenchmarkSample {
            id: format!("ragas_{idx}"),
            prompt: question,
            category: None,
            ground_truth: GroundTruth::ContextualReference {
                context: contexts.clone(),
                reference: reference.clone(),
            },
            sample_type: SampleType::ContextualQA,
            metadata: Default::default(),
        };
        if let Some(a) = row.get("answer").and_then(|x| x.as_str()) {
            sample.metadata.insert("model_answer".into(), a.to_string());
        }
        sample
            .metadata
            .insert("num_contexts".into(), contexts.len().to_string());
        out.push(sample);

        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

fn extract_reference(row: &serde_json::Value) -> Option<String> {
    if let Some(s) = row.get("ground_truth").and_then(|x| x.as_str()) {
        return Some(s.to_string());
    }
    if let Some(arr) = row.get("ground_truth").and_then(|x| x.as_array()) {
        let joined: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        if !joined.is_empty() {
            return Some(joined.join("; "));
        }
    }
    if let Some(s) = row.get("grounded_answer").and_then(|x| x.as_str()) {
        return Some(s.to_string());
    }
    row.get("answer")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/ragas_sample.json");

    #[test]
    fn loader_metadata() {
        let l = RagasLoader;
        assert_eq!(l.name(), "ragas_wikiqa");
        assert_eq!(l.sample_type(), SampleType::ContextualQA);
    }

    #[test]
    fn parse_three_samples() {
        let samples = parse_ragas_json(FIXTURE, None).unwrap();
        assert_eq!(samples.len(), 3);
        let s0 = &samples[0];
        assert!(s0.prompt.contains("capital of France"));
        match &s0.ground_truth {
            GroundTruth::ContextualReference { context, reference } => {
                assert_eq!(context.len(), 2);
                assert!(context.iter().any(|c| c.contains("Paris")));
                assert_eq!(reference, "Paris");
            }
            _ => panic!("expected ContextualReference"),
        }
        assert_eq!(
            s0.metadata.get("model_answer").map(|s| s.as_str()),
            Some("Paris is the capital of France.")
        );
    }

    #[test]
    fn array_ground_truth_is_joined() {
        let json =
            r#"{"rows": [{"row_idx": 0, "row": {"question": "q", "ground_truth": ["a", "b"]}}]}"#;
        let samples = parse_ragas_json(json, None).unwrap();
        match &samples[0].ground_truth {
            GroundTruth::ContextualReference { reference, .. } => {
                assert_eq!(reference, "a; b");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn missing_rows_errors() {
        let bad = r#"{"somethingelse": []}"#;
        assert!(matches!(
            parse_ragas_json(bad, None),
            Err(BenchmarkError::Schema(_))
        ));
    }

    #[test]
    fn falls_back_to_answer_if_no_ground_truth() {
        let json =
            r#"{"rows": [{"row_idx": 0, "row": {"question": "q", "answer": "only_answer"}}]}"#;
        let samples = parse_ragas_json(json, None).unwrap();
        match &samples[0].ground_truth {
            GroundTruth::ContextualReference { reference, .. } => {
                assert_eq!(reference, "only_answer");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn limit_respected() {
        let samples = parse_ragas_json(FIXTURE, Some(2)).unwrap();
        assert_eq!(samples.len(), 2);
    }
}
