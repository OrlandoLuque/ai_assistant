//! Common types for dataset-based hallucination/faithfulness benchmarks.
//!
//! This module defines the data model shared by every benchmark loader:
//! a single `BenchmarkSample` with a typed `GroundTruth` variant, and the
//! `BenchmarkLoader` trait that each dataset (TruthfulQA, HaluEval, FActScore,
//! RAGAS, FEVER, ...) implements.
//!
//! See `src/benchmark.rs` for the *performance* benchmarking harness; this
//! module is about *evaluation quality* (accuracy, faithfulness, grounding).

use std::path::{Path, PathBuf};

/// Errors that can occur while downloading or parsing a benchmark dataset.
#[derive(Debug)]
#[non_exhaustive]
pub enum BenchmarkError {
    /// Could not reach the remote URL.
    Network(String),
    /// HTTP status was not success.
    Http { status: u16, url: String },
    /// Local filesystem error (read/write/create dir).
    Io(String),
    /// Parse error in the downloaded file.
    Parse(String),
    /// Dataset requires user opt-in (license acceptance) and it has not been given.
    LicenseOptIn(String),
    /// Dataset payload did not match the expected schema.
    Schema(String),
    /// Downloaded file size is suspiciously small/large (bomb or empty).
    SizeCheck(String),
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(s) => write!(f, "network error: {s}"),
            Self::Http { status, url } => write!(f, "HTTP {status} fetching {url}"),
            Self::Io(s) => write!(f, "I/O error: {s}"),
            Self::Parse(s) => write!(f, "parse error: {s}"),
            Self::LicenseOptIn(s) => write!(f, "license opt-in required: {s}"),
            Self::Schema(s) => write!(f, "schema mismatch: {s}"),
            Self::SizeCheck(s) => write!(f, "size check failed: {s}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

/// Kind of sample a benchmark produces. Used by the runner to pick the
/// right scoring path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    /// Open-ended Q&A with correct + incorrect reference answers (TruthfulQA).
    QA,
    /// Pair of two candidate responses, one hallucinated, one correct (HaluEval).
    HallucinationPair,
    /// Atomic claim list with supported/unsupported labels (FActScore).
    AtomicClaims,
    /// A claim plus evidence passages with a Supports/Refutes label (FEVER).
    ClaimVsEvidence,
    /// Context + question + reference answer for faithfulness scoring (RAGAS).
    ContextualQA,
}

/// FEVER-style three-way label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    /// Evidence supports the claim.
    Supports,
    /// Evidence refutes the claim.
    Refutes,
    /// There is not enough information in the evidence.
    NotEnoughInfo,
}

/// Ground-truth annotation attached to a `BenchmarkSample`.
#[derive(Debug, Clone)]
pub enum GroundTruth {
    /// Open-ended answers: a list of correct and incorrect reference answers.
    /// Scoring: the model answer is correct if it matches any of `correct` by
    /// the configured similarity metric.
    Answer {
        /// Reference correct answers.
        correct: Vec<String>,
        /// Reference incorrect / hallucinated answers.
        incorrect: Vec<String>,
    },
    /// Two candidate responses, one known-correct and one known-hallucinated.
    /// Scoring: the model should prefer `right` over `hallucinated`.
    HallucinationPair {
        /// The correct response.
        right: String,
        /// The known-hallucinated response.
        hallucinated: String,
    },
    /// Decomposition into atomic claims, each labelled `true` if supported.
    /// Scoring: FActScore = fraction of atomic claims labelled `true`.
    AtomicClaims(Vec<(String, bool)>),
    /// A three-way label plus a list of evidence passages.
    /// Scoring: match predicted label against `label` after the verifier runs
    /// over the evidence.
    SupportsRefutes {
        /// The gold label.
        label: Label,
        /// Evidence passages (concatenated as context at scoring time).
        evidence: Vec<String>,
    },
    /// Context (passages) + reference answer for faithfulness-style scoring.
    /// Scoring: compute faithfulness of the model answer vs. `context`, and
    /// similarity vs. `reference`.
    ContextualReference {
        /// Retrieved / ground-truth context passages.
        context: Vec<String>,
        /// Reference answer.
        reference: String,
    },
}

/// A single sample from a benchmark dataset.
#[derive(Debug, Clone)]
pub struct BenchmarkSample {
    /// Stable identifier within the dataset (index, question id, ...).
    pub id: String,
    /// The prompt / question to send to the model.
    pub prompt: String,
    /// Optional category (e.g., TruthfulQA category, HaluEval task).
    pub category: Option<String>,
    /// Ground truth for scoring.
    pub ground_truth: GroundTruth,
    /// Which sample shape this is — runner uses this to dispatch.
    pub sample_type: SampleType,
    /// Free-form metadata (source URL, original index, ...).
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl BenchmarkSample {
    /// Convenience: create a minimal QA sample.
    pub fn qa(
        id: impl Into<String>,
        prompt: impl Into<String>,
        correct: Vec<String>,
        incorrect: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            category: None,
            ground_truth: GroundTruth::Answer { correct, incorrect },
            sample_type: SampleType::QA,
            metadata: Default::default(),
        }
    }
}

/// Interface implemented by every benchmark dataset.
///
/// Loaders are pure (no mutation of shared state). They know:
///  * where to fetch the raw dataset,
///  * how to parse it into `BenchmarkSample`s,
///  * what license / opt-in applies.
///
/// Running the benchmark (querying the model, scoring) is done by the
/// `runner` module, not by the loader.
pub trait BenchmarkLoader: Send + Sync {
    /// Short machine-friendly name (e.g., "truthfulqa", "halueval_qa").
    fn name(&self) -> &'static str;
    /// One-line human description.
    fn description(&self) -> &'static str;
    /// License string as it appears on the upstream release.
    fn license(&self) -> &'static str;
    /// Citation / BibTeX key or DOI for academic attribution.
    fn citation(&self) -> &'static str;
    /// Candidate download URLs, tried in order. The first 2xx wins.
    fn download_urls(&self) -> &'static [&'static str];
    /// Expected file size in bytes (sanity-check after download). `None` = skip check.
    fn expected_size_bytes(&self) -> Option<u64> {
        None
    }
    /// What shape of sample this dataset produces.
    fn sample_type(&self) -> SampleType;
    /// Whether the user must pass an explicit opt-in flag before download.
    fn requires_opt_in(&self) -> bool {
        false
    }
    /// Download the raw dataset into `cache_dir`, returning the local path.
    /// Idempotent: if already present and size-matches, should be a no-op.
    fn download(&self, cache_dir: &Path) -> Result<PathBuf, BenchmarkError>;
    /// Parse the raw file into samples. `limit` caps the number of samples
    /// returned (useful for smoke-tests / CI).
    fn load(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qa_helper_builds_sample() {
        let s = BenchmarkSample::qa(
            "q1",
            "What is 2+2?",
            vec!["4".into()],
            vec!["5".into(), "22".into()],
        );
        assert_eq!(s.id, "q1");
        assert_eq!(s.sample_type, SampleType::QA);
        match &s.ground_truth {
            GroundTruth::Answer { correct, incorrect } => {
                assert_eq!(correct, &vec!["4".to_string()]);
                assert_eq!(incorrect.len(), 2);
            }
            _ => panic!("expected Answer"),
        }
    }

    #[test]
    fn error_display_is_informative() {
        let e = BenchmarkError::Http {
            status: 404,
            url: "http://x".into(),
        };
        assert!(format!("{e}").contains("404"));
        assert!(format!("{e}").contains("http://x"));
    }

    #[test]
    fn label_equality() {
        assert_eq!(Label::Supports, Label::Supports);
        assert_ne!(Label::Supports, Label::Refutes);
    }
}
