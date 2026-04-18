//! FEVER loader (claim-only mode, no wiki pages fetched) — stub for V90.8.
//!
//! Legal note: FEVER is CC-BY-SA 3.0. We only *fetch* the dataset on explicit
//! user action; we do not redistribute. Loader requires an opt-in flag.

use std::path::{Path, PathBuf};

use crate::eval_benchmarks::types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, SampleType};

/// FEVER claim-only loader (placeholder).
#[derive(Debug, Default, Clone)]
pub struct FeverLoader;

impl BenchmarkLoader for FeverLoader {
    fn name(&self) -> &'static str {
        "fever"
    }
    fn description(&self) -> &'static str {
        "FEVER: 185k claim vs. evidence with Supports/Refutes/NotEnoughInfo labels (claim-only subset)."
    }
    fn license(&self) -> &'static str {
        "CC-BY-SA-3.0 (fetch-only, not redistributed)"
    }
    fn citation(&self) -> &'static str {
        "Thorne et al. FEVER: A Large-Scale Dataset for Fact Extraction and VERification. NAACL 2018."
    }
    fn download_urls(&self) -> &'static [&'static str] {
        &[]
    }
    fn sample_type(&self) -> SampleType {
        SampleType::ClaimVsEvidence
    }
    fn requires_opt_in(&self) -> bool {
        true
    }
    fn download(&self, _cache_dir: &Path) -> Result<PathBuf, BenchmarkError> {
        Err(BenchmarkError::Network(
            "FeverLoader::download not yet implemented (V90.8)".into(),
        ))
    }
    fn load(
        &self,
        _path: &Path,
        _limit: Option<usize>,
    ) -> Result<Vec<BenchmarkSample>, BenchmarkError> {
        Err(BenchmarkError::Parse(
            "FeverLoader::load not yet implemented (V90.8)".into(),
        ))
    }
}
