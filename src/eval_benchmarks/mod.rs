//! Dataset-based hallucination & faithfulness benchmarks.
//!
//! This module hosts the *evaluation quality* harness: standard community
//! benchmarks (TruthfulQA, HaluEval, FActScore, RAGAS, FEVER, ...) exposed
//! through a uniform `BenchmarkLoader` trait plus a runner + calibration
//! helpers.
//!
//! Not to be confused with `crate::benchmark` (performance micro-benchmarks
//! measuring latency / throughput of internal operations).

pub mod types;

pub use types::{BenchmarkError, BenchmarkLoader, BenchmarkSample, GroundTruth, Label, SampleType};
