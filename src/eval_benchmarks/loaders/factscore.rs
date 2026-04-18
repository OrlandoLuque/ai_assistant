//! FActScore loader — stub, to be filled in by V90.6.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, SampleType};

/// FActScore loader (placeholder).
#[derive(Debug, Default, Clone)]
pub struct FactScoreLoader;

impl BenchmarkLoader for FactScoreLoader {
    fn name(&self) -> &'static str {
        "factscore"
    }
    fn description(&self) -> &'static str {
        "FActScore: atomic-fact decomposition with supported/unsupported labels."
    }
    fn license(&self) -> &'static str {
        "MIT"
    }
    fn citation(&self) -> &'static str {
        "Min et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long-Form Text Generation. EMNLP 2023."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        &[]
    }
    fn sample_type(&self) -> SampleType {
        SampleType::AtomicClaims
    }
    fn download(&self, _cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        Err(BenchmarkError::Network(
            "FactScoreLoader::download not yet implemented (V90.6)".into(),
        ))
    }
    fn load(
        &self,
        _path: &Path,
        _limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        Err(BenchmarkError::Parse(
            "FactScoreLoader::load not yet implemented (V90.6)".into(),
        ))
    }
}
