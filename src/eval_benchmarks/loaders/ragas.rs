//! RAGAS WikiQA loader — stub, to be filled in by V90.7.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, SampleType};

/// RAGAS WikiQA loader (placeholder).
#[derive(Debug, Default, Clone)]
pub struct RagasLoader;

impl BenchmarkLoader for RagasLoader {
    fn name(&self) -> &'static str {
        "ragas_wikiqa"
    }
    fn description(&self) -> &'static str {
        "RAGAS WikiQA: question + contexts + ground-truth answer for faithfulness scoring."
    }
    fn license(&self) -> &'static str {
        "Apache-2.0"
    }
    fn citation(&self) -> &'static str {
        "Es et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        &[]
    }
    fn sample_type(&self) -> SampleType {
        SampleType::ContextualQA
    }
    fn download(&self, _cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        Err(BenchmarkError::Network(
            "RagasLoader::download not yet implemented (V90.7)".into(),
        ))
    }
    fn load(
        &self,
        _path: &Path,
        _limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        Err(BenchmarkError::Parse(
            "RagasLoader::load not yet implemented (V90.7)".into(),
        ))
    }
}
