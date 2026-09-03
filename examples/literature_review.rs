//! The literature-review pipeline, end to end.
//!
//! This example exists because the same code was published on the website as a snippet
//! that had never been compiled — it named a `query` field the config does not have and
//! awaited a method that does not exist. An example is compiled by CI on every push, so
//! it cannot drift the way a snippet in an HTML file can.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example literature_review --features "full"
//! ```
//!
//! It hits the live arXiv and OpenAlex APIs and writes `review.md` and `review.bib` in the
//! current directory. **No model is involved**: the pipeline searches, groups and
//! structures, so this runs on a machine with no local LLM and no API key.

use ai_assistant::academic_search::{AcademicSearchEngine, ArxivProvider, OpenAlexProvider};
use ai_assistant::literature_review::{LiteratureReviewConfig, LiteratureReviewPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let topic = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reinforcement learning from human feedback".to_string());

    // Providers are added explicitly: the engine searches the ones you give it and
    // deduplicates the results by DOI, which is the only identifier that survives the same
    // paper being found through two of them.
    let mut engine = AcademicSearchEngine::new();
    engine.add_provider(Box::new(ArxivProvider::new()));
    engine.add_provider(Box::new(OpenAlexProvider::new()));

    // `quick()` and `systematic()` are presets over public fields — max_papers,
    // search_depth, synthesis_style, year_range, fields_of_study.
    let config = LiteratureReviewConfig::systematic();

    println!("Searching for: {topic}");
    let review = LiteratureReviewPipeline::new(engine, config).execute(&topic);

    println!(
        "{} of {} papers, {} words, {} sections",
        review.papers_included,
        review.papers_found,
        review.total_word_count(),
        review.sections.len()
    );

    std::fs::write("review.md", review.to_markdown())?;
    std::fs::write("review.bib", &review.bibtex)?;
    println!("Wrote review.md and review.bib");

    Ok(())
}
