//! FEVER loader (claim-only mode — no wiki pages fetched).
//!
//! Dataset: <https://fever.ai/dataset/fever.html>
//! License: CC-BY-SA 3.0. We only *fetch* on explicit user action; we do not
//! redistribute the data. `requires_opt_in()` returns `true` so the CLI
//! gates the download behind `--accept-license`.
//!
//! Schema: JSONL, one claim per line:
//!   {"id": ..., "label": "SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO",
//!    "claim": ..., "evidence": [[[annotation_id, wiki_id, sentence_id, sentence_text]]]}
//! The nested evidence shape has changed across versions; we accept both the
//! `[[[...]]]` form (with inline sentence text in position 3) and flattened
//! strings.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::http::{download_file, DownloadOptions};
use crate::eval_benchmarks::types::{
    BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, Label, SampleType,
};

/// FEVER claim-only loader.
#[derive(Debug, Default, Clone)]
pub struct FeverLoader;

// Official FEVER dev split via the project's S3 mirror. Fetch-only, no
// redistribution. Upstream maintains this URL per fever.ai/dataset/fever.html.
static FEVER_URLS: &[&str] = &["https://fever.ai/download/fever/shared_task_dev.jsonl"];

impl BenchmarkLoader for FeverLoader {
    fn name(&self) -> &'static str {
        "fever"
    }
    fn description(&self) -> &'static str {
        "FEVER: claim vs. evidence with Supports/Refutes/NotEnoughInfo labels (dev split, claim-only)."
    }
    fn license(&self) -> &'static str {
        "CC-BY-SA-3.0 (fetch-only, not redistributed)"
    }
    fn citation(&self) -> &'static str {
        "Thorne et al. FEVER: A Large-Scale Dataset for Fact Extraction and VERification. NAACL 2018."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        FEVER_URLS
    }
    fn sample_type(&self) -> SampleType {
        SampleType::ClaimVsEvidence
    }
    fn requires_opt_in(&self) -> bool {
        true
    }
    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        let dest = cache_dir.join("fever_dev.jsonl");
        download_file(self.download_urls(), &dest, &DownloadOptions::default())
    }
    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BenchmarkError::Io(format!("read {}: {e}", path.display())))?;
        parse_fever_jsonl(&content, limit)
    }
}

fn parse_fever_jsonl(
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

        let id = match v.get("id") {
            Some(serde_json::Value::Number(n)) => n.to_string(),
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => format!("{}", idx),
        };

        let claim = v
            .get("claim")
            .and_then(|x| x.as_str())
            .ok_or_else(|| BenchmarkError::Schema(format!("line {}: missing 'claim'", idx + 1)))?
            .to_string();

        let label_str = v
            .get("label")
            .and_then(|x| x.as_str())
            .ok_or_else(|| BenchmarkError::Schema(format!("line {}: missing 'label'", idx + 1)))?;
        let label = match label_str {
            "SUPPORTS" | "supports" | "Supports" => Label::Supports,
            "REFUTES" | "refutes" | "Refutes" => Label::Refutes,
            "NOT ENOUGH INFO" | "not enough info" | "NotEnoughInfo" | "nei" | "NEI" => {
                Label::NotEnoughInfo
            }
            other => {
                return Err(BenchmarkError::Schema(format!(
                    "line {}: unknown label {other:?}",
                    idx + 1
                )))
            }
        };

        let evidence = extract_evidence(v.get("evidence"));

        let sample = BenchmarkSample {
            id: format!("fever_{id}"),
            prompt: claim,
            category: Some(format!("{label:?}")),
            ground_truth: GroundTruth::SupportsRefutes { label, evidence },
            sample_type: SampleType::ClaimVsEvidence,
            metadata: Default::default(),
        };
        out.push(sample);

        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

/// Walk FEVER's nested evidence structure and collect any string snippets we
/// find. In the v1.0 format, evidence is `[[[annotation, wiki, sent_id, text]]]`
/// and only position 3 is a free-text sentence. Newer exports flatten it to
/// plain strings; we accept both.
fn extract_evidence(v: Option<&serde_json::Value>) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => {
                let t = s.trim();
                if t.len() > 3 {
                    out.push(t.to_string());
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    walk(item, out);
                }
            }
            _ => {}
        }
    }
    if let Some(v) = v {
        walk(v, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/fever_sample.jsonl");

    #[test]
    fn loader_metadata() {
        let l = FeverLoader;
        assert_eq!(l.name(), "fever");
        assert_eq!(l.sample_type(), SampleType::ClaimVsEvidence);
        assert!(l.requires_opt_in());
        assert!(l.license().starts_with("CC-BY-SA-3.0"));
    }

    #[test]
    fn parse_three_labels() {
        let samples = parse_fever_jsonl(FIXTURE, None).unwrap();
        assert_eq!(samples.len(), 3);
        let labels: Vec<_> = samples
            .iter()
            .map(|s| match &s.ground_truth {
                GroundTruth::SupportsRefutes { label, .. } => *label,
                _ => panic!("expected SupportsRefutes"),
            })
            .collect();
        assert_eq!(labels[0], Label::Supports);
        assert_eq!(labels[1], Label::Refutes);
        assert_eq!(labels[2], Label::NotEnoughInfo);
    }

    #[test]
    fn evidence_text_is_extracted() {
        let samples = parse_fever_jsonl(FIXTURE, None).unwrap();
        match &samples[0].ground_truth {
            GroundTruth::SupportsRefutes { evidence, .. } => {
                assert!(!evidence.is_empty());
                assert!(evidence.iter().any(|e| e.contains("Nikola Tesla")));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn not_enough_info_has_empty_evidence() {
        let samples = parse_fever_jsonl(FIXTURE, None).unwrap();
        match &samples[2].ground_truth {
            GroundTruth::SupportsRefutes { evidence, label } => {
                assert_eq!(*label, Label::NotEnoughInfo);
                assert!(evidence.is_empty());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_label_errors() {
        let bad = r#"{"id": 1, "label": "MAYBE", "claim": "x", "evidence": []}"#;
        assert!(matches!(
            parse_fever_jsonl(bad, None),
            Err(BenchmarkError::Schema(_))
        ));
    }

    #[test]
    fn limit_respected() {
        let samples = parse_fever_jsonl(FIXTURE, Some(2)).unwrap();
        assert_eq!(samples.len(), 2);
    }
}
