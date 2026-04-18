//! HaluEval loader — stub, to be filled in by V90.5.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, SampleType};

/// HaluEval QA subset loader (placeholder).
#[derive(Debug, Default, Clone)]
pub struct HaluEvalLoader;

impl BenchmarkLoader for HaluEvalLoader {
    fn name(&self) -> &'static str {
        "halueval_qa"
    }
    fn description(&self) -> &'static str {
        "HaluEval QA: right/hallucinated answer pairs (10k)."
    }
    fn license(&self) -> &'static str {
        "MIT"
    }
    fn citation(&self) -> &'static str {
        "Li et al. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. EMNLP 2023."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        &[]
    }
    fn sample_type(&self) -> SampleType {
        SampleType::HallucinationPair
    }
    fn download(&self, _cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        Err(BenchmarkError::Network(
            "HaluEvalLoader::download not yet implemented (V90.5)".into(),
        ))
    }
    fn load(
        &self,
        _path: &Path,
        _limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        Err(BenchmarkError::Parse(
            "HaluEvalLoader::load not yet implemented (V90.5)".into(),
        ))
    }
}
