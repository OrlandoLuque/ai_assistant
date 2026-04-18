//! TruthfulQA loader.
//!
//! Dataset: <https://github.com/sylinrl/TruthfulQA>
//! License: Apache 2.0 (datasets/TruthfulQA.csv)
//! Paper: Lin et al. 2022, "TruthfulQA: Measuring How Models Mimic Human Falsehoods".
//!
//! Schema (CSV, 7 columns):
//!   Type, Category, Question, Best Answer, Correct Answers, Incorrect Answers, Source
//! where "Correct Answers" and "Incorrect Answers" are semicolon-separated lists.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::http::{download_file, DownloadOptions};
use crate::eval_benchmarks::types::{
    BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, SampleType,
};

/// TruthfulQA generation-variant loader (828 Qs).
#[derive(Debug, Default, Clone)]
pub struct TruthfulQaLoader;

static TRUTHFULQA_URLS: &[&str] =
    &["https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"];

impl BenchmarkLoader for TruthfulQaLoader {
    fn name(&self) -> &'static str {
        "truthfulqa"
    }

    fn description(&self) -> &'static str {
        "TruthfulQA: 817 questions spanning 38 categories, scored against reference correct/incorrect answers."
    }

    fn license(&self) -> &'static str {
        "Apache-2.0"
    }

    fn citation(&self) -> &'static str {
        "Lin, Hilton, Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022."
    }

    fn download_urls(&self) -> &'static [&'static str] {
        TRUTHFULQA_URLS
    }

    fn sample_type(&self) -> SampleType {
        SampleType::QA
    }

    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        let dest = cache_dir.join("TruthfulQA.csv");
        download_file(self.download_urls(), &dest, &DownloadOptions::default())
    }

    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BenchmarkError::Io(format!("read {}: {e}", path.display())))?;
        parse_truthfulqa_csv(&content, limit)
    }
}

/// Parse TruthfulQA CSV content into `BenchmarkSample`s.
/// Public-ish for testing from outside the impl.
fn parse_truthfulqa_csv(
    content: &str,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
    let rows = parse_csv(content)?;
    let mut iter = rows.into_iter();
    let header = iter
        .next()
        .ok_or_else(|| BenchmarkError::Schema("TruthfulQA: empty file, no header row".into()))?;

    let col_type = find_col(&header, "Type")?;
    let col_cat = find_col(&header, "Category")?;
    let col_q = find_col(&header, "Question")?;
    let col_correct = find_col(&header, "Correct Answers")?;
    let col_incorrect = find_col(&header, "Incorrect Answers")?;

    let max_idx = [col_type, col_cat, col_q, col_correct, col_incorrect]
        .iter()
        .copied()
        .max()
        .unwrap_or(0);

    let mut out = Vec::new();
    for (idx, row) in iter.enumerate() {
        if row.len() <= max_idx {
            continue; // tolerate short rows
        }
        let category = row[col_cat].trim().to_string();
        let question = row[col_q].trim().to_string();
        if question.is_empty() {
            continue;
        }
        let correct = split_semicolon(&row[col_correct]);
        let incorrect = split_semicolon(&row[col_incorrect]);

        let mut sample = BenchmarkSample {
            id: format!("tqa_{idx}"),
            prompt: question,
            category: Some(category),
            ground_truth: GroundTruth::Answer { correct, incorrect },
            sample_type: SampleType::QA,
            metadata: Default::default(),
        };
        sample
            .metadata
            .insert("type".into(), row[col_type].trim().to_string());
        out.push(sample);

        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
    }

    Ok(out)
}

fn find_col(header: &[String], name: &str) -> Result<usize, BenchmarkError> {
    header
        .iter()
        .position(|c| c.trim().eq_ignore_ascii_case(name))
        .ok_or_else(|| BenchmarkError::Schema(format!("TruthfulQA: missing column {name}")))
}

fn split_semicolon(s: &str) -> Vec<String> {
    s.split(';')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Minimal CSV parser sufficient for TruthfulQA: handles quoted fields with
/// embedded commas, doubled-quote escapes, CRLF line endings.
///
/// Not general-purpose: no support for unquoted multi-line fields outside
/// quotes. Good enough for the datasets we target.
fn parse_csv(content: &str) -> Result<Vec<Vec<String>>, BenchmarkError> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;

    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_quotes {
            match c {
                '"' => {
                    if i + 1 < chars.len() && chars[i + 1] == '"' {
                        field.push('"');
                        i += 2;
                        continue;
                    } else {
                        in_quotes = false;
                    }
                }
                _ => field.push(c),
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                }
                '\r' => { /* skip, handled by \n */ }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    if !(row.len() == 1 && row[0].is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                }
                _ => field.push(c),
            }
        }
        i += 1;
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    if in_quotes {
        return Err(BenchmarkError::Parse(
            "CSV: unterminated quoted field".into(),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/truthfulqa_sample.csv");

    #[test]
    fn loader_metadata_is_set() {
        let l = TruthfulQaLoader;
        assert_eq!(l.name(), "truthfulqa");
        assert_eq!(l.sample_type(), SampleType::QA);
        assert_eq!(l.license(), "Apache-2.0");
        assert!(!l.download_urls().is_empty());
    }

    #[test]
    fn parse_fixture_yields_three_samples() {
        let samples = parse_truthfulqa_csv(FIXTURE, None).unwrap();
        assert_eq!(samples.len(), 3);
        let s0 = &samples[0];
        assert!(s0.prompt.contains("watermelon"));
        match &s0.ground_truth {
            GroundTruth::Answer { correct, incorrect } => {
                assert!(correct.iter().any(|a| a.contains("Nothing harmful")));
                assert!(incorrect.iter().any(|a| a.contains("grow a watermelon")));
            }
            _ => panic!("expected Answer variant"),
        }
        assert_eq!(s0.category.as_deref(), Some("Misconceptions"));
    }

    #[test]
    fn limit_is_respected() {
        let samples = parse_truthfulqa_csv(FIXTURE, Some(2)).unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn csv_parser_handles_quotes_and_escapes() {
        let csv = "A,B\n\"hello, world\",\"she said \"\"hi\"\"\"\n";
        let rows = parse_csv(csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1], vec!["hello, world", "she said \"hi\""]);
    }

    #[test]
    fn missing_column_errors_cleanly() {
        let bad = "Foo,Bar\n1,2\n";
        let err = parse_truthfulqa_csv(bad, None).unwrap_err();
        assert!(matches!(err, BenchmarkError::Schema(_)));
    }
}
