//! Dataset-based hallucination & faithfulness benchmarks.
//!
//! This module hosts the *evaluation quality* harness: standard community
//! benchmarks (TruthfulQA, HaluEval, FActScore, RAGAS, FEVER, ...) exposed
//! through a uniform `BenchmarkLoader` trait plus a runner + calibration
//! helpers.
//!
//! Not to be confused with `crate::benchmark` (performance micro-benchmarks
//! measuring latency / throughput of internal operations).

pub mod cache;
pub mod http;
pub mod types;

pub use cache::BenchmarkCache;
pub use http::{download_file, DownloadOptions, MAX_DOWNLOAD_BYTES};
pub use types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, Label, SampleType};
