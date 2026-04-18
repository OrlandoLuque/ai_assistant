//! Dataset-specific loaders.
//!
//! Each submodule implements [`BenchmarkLoader`](super::types::BenchmarkLoader)
//! for one community benchmark. The runner picks one by name at dispatch time.

pub mod factscore;
pub mod fever;
pub mod halueval;
pub mod ragas;
pub mod truthfulqa;

pub use factscore::FactScoreLoader;
pub use fever::FeverLoader;
pub use halueval::HaluEvalLoader;
pub use ragas::RagasLoader;
pub use truthfulqa::TruthfulQaLoader;

use super::types::BenchmarkLoader;

/// Return every loader known at compile time. Order is stable.
///
/// The `ai_cli benchmark list` subcommand walks this iterator to print
/// available benchmarks.
pub fn all_loaders() -> Vec<Box<dyn BenchmarkLoader>> {
    vec![
        Box::new(TruthfulQaLoader::default()),
        Box::new(HaluEvalLoader::default()),
        Box::new(FactScoreLoader::default()),
        Box::new(RagasLoader::default()),
        Box::new(FeverLoader::default()),
    ]
}

/// Look up a loader by its `.name()`.
pub fn get_loader(name: &str) -> Option<Box<dyn BenchmarkLoader>> {
    all_loaders().into_iter().find(|l| l.name() == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_loaders_has_five_entries() {
        let loaders = all_loaders();
        assert_eq!(loaders.len(), 5);
        let names: Vec<_> = loaders.iter().map(|l| l.name()).collect();
        assert!(names.contains(&"truthfulqa"));
        assert!(names.contains(&"halueval_qa"));
        assert!(names.contains(&"factscore"));
        assert!(names.contains(&"ragas_wikiqa"));
        assert!(names.contains(&"fever"));
    }

    #[test]
    fn get_loader_by_name() {
        assert!(get_loader("truthfulqa").is_some());
        assert!(get_loader("nonsense").is_none());
    }

    #[test]
    fn all_loaders_report_license() {
        for l in all_loaders() {
            assert!(!l.license().is_empty(), "{} missing license", l.name());
            assert!(!l.citation().is_empty(), "{} missing citation", l.name());
        }
    }
}
