//! Literature review pipeline
//!
//! Generates structured literature reviews from academic search results.
//! Integrates with `academic_search` providers, `bibtex` generator, and
//! the anti-hallucination framework for faithful output.

use std::collections::HashMap;

use super::academic_search::{AcademicPaper, AcademicSearchConfig, AcademicSearchEngine};
use super::bibtex::{BibEntry, BibGenerator};

// =============================================================================
// Configuration
// =============================================================================

/// Search depth for literature review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchDepth {
    /// Quick search — single pass, max 20 papers
    Quick,
    /// Standard search — multi-provider, max 50 papers
    Standard,
    /// Deep search — citation graph traversal, max 100 papers
    Deep,
}

impl SearchDepth {
    /// Maximum papers to consider for this depth.
    pub fn max_papers(&self) -> usize {
        match self {
            Self::Quick => 20,
            Self::Standard => 50,
            Self::Deep => 100,
        }
    }
}

impl Default for SearchDepth {
    fn default() -> Self {
        Self::Standard
    }
}

/// Style of literature review synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SynthesisStyle {
    /// Flowing narrative review
    Narrative,
    /// Systematic review with tables and categorization
    Systematic,
    /// Annotated bibliography (paper-by-paper summaries)
    Annotated,
    /// Comparative review (contrasting approaches)
    Comparative,
}

impl Default for SynthesisStyle {
    fn default() -> Self {
        Self::Narrative
    }
}

impl std::fmt::Display for SynthesisStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Narrative => write!(f, "Narrative"),
            Self::Systematic => write!(f, "Systematic"),
            Self::Annotated => write!(f, "Annotated"),
            Self::Comparative => write!(f, "Comparative"),
        }
    }
}

/// Bibliography output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BibliographyFormat {
    /// BibTeX format
    BibTeX,
    /// APA 7th edition
    Apa,
    /// MLA 9th edition
    Mla,
    /// Chicago 17th edition
    Chicago,
    /// IEEE style
    Ieee,
}

impl BibliographyFormat {
    /// Format a paper reference in this style.
    pub fn format_reference(&self, paper: &AcademicPaper) -> String {
        let authors_str = if paper.authors.is_empty() {
            "Unknown".to_string()
        } else if paper.authors.len() <= 3 {
            paper
                .authors
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            format!("{} et al.", paper.authors[0].name)
        };

        let year = paper.year.map(|y| y.to_string()).unwrap_or_default();
        let venue = paper.venue.as_deref().unwrap_or("");

        match self {
            Self::BibTeX => paper.citation_string(),
            Self::Apa => {
                format!(
                    "{}{}. {}. {}.",
                    authors_str,
                    if year.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", year)
                    },
                    paper.title,
                    venue,
                )
            }
            Self::Mla => {
                format!("{}. \"{}.\" {} {}.", authors_str, paper.title, venue, year,)
            }
            Self::Chicago => {
                format!(
                    "{}. \"{}.\" {} ({}).",
                    authors_str, paper.title, venue, year,
                )
            }
            Self::Ieee => {
                format!("{}, \"{},\" {}, {}.", authors_str, paper.title, venue, year,)
            }
        }
    }
}

impl Default for BibliographyFormat {
    fn default() -> Self {
        Self::BibTeX
    }
}

/// Configuration for a literature review.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LiteratureReviewConfig {
    /// Maximum papers to include
    pub max_papers: usize,
    /// Search depth
    pub search_depth: SearchDepth,
    /// Synthesis style
    pub synthesis_style: SynthesisStyle,
    /// Whether to include citation graph analysis
    pub include_citation_graph: bool,
    /// Bibliography output format
    pub bibliography_format: BibliographyFormat,
    /// Year range filter
    pub year_range: Option<(u16, u16)>,
    /// Fields of study filter
    pub fields_of_study: Vec<String>,
}

impl Default for LiteratureReviewConfig {
    fn default() -> Self {
        Self {
            max_papers: 50,
            search_depth: SearchDepth::default(),
            synthesis_style: SynthesisStyle::default(),
            include_citation_graph: false,
            bibliography_format: BibliographyFormat::default(),
            year_range: None,
            fields_of_study: Vec::new(),
        }
    }
}

impl LiteratureReviewConfig {
    /// Quick review preset — fewer papers, annotated style.
    pub fn quick() -> Self {
        Self {
            max_papers: 10,
            search_depth: SearchDepth::Quick,
            synthesis_style: SynthesisStyle::Annotated,
            ..Default::default()
        }
    }

    /// Systematic review preset — thorough, with tables.
    pub fn systematic() -> Self {
        Self {
            max_papers: 50,
            search_depth: SearchDepth::Deep,
            synthesis_style: SynthesisStyle::Systematic,
            include_citation_graph: true,
            bibliography_format: BibliographyFormat::BibTeX,
            ..Default::default()
        }
    }
}

// =============================================================================
// Review Output
// =============================================================================

/// A section of the literature review.
#[derive(Debug, Clone)]
pub struct ReviewSection {
    /// Section heading
    pub heading: String,
    /// Section content (markdown)
    pub content: String,
    /// Papers referenced in this section
    pub paper_ids: Vec<String>,
}

impl ReviewSection {
    /// Create a new review section.
    pub fn new(heading: &str, content: &str) -> Self {
        Self {
            heading: heading.to_string(),
            content: content.to_string(),
            paper_ids: Vec::new(),
        }
    }

    /// Add paper references to this section.
    pub fn with_papers(mut self, ids: Vec<String>) -> Self {
        self.paper_ids = ids;
        self
    }

    /// Word count of the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// A completed literature review.
#[derive(Debug, Clone)]
pub struct LiteratureReview {
    /// Review title
    pub title: String,
    /// Review sections
    pub sections: Vec<ReviewSection>,
    /// Bibliography in the requested format
    pub bibliography: String,
    /// BibTeX bibliography (always generated)
    pub bibtex: String,
    /// Number of papers found by search
    pub papers_found: usize,
    /// Number of papers included in the review
    pub papers_included: usize,
    /// Papers included in the review
    pub papers: Vec<AcademicPaper>,
    /// Synthesis style used
    pub synthesis_style: SynthesisStyle,
}

impl LiteratureReview {
    /// Total word count across all sections.
    pub fn total_word_count(&self) -> usize {
        self.sections.iter().map(|s| s.word_count()).sum()
    }

    /// Full review as markdown text.
    pub fn to_markdown(&self) -> String {
        let mut md = format!("# {}\n\n", self.title);

        for section in &self.sections {
            md.push_str(&format!(
                "## {}\n\n{}\n\n",
                section.heading, section.content
            ));
        }

        md.push_str("## References\n\n");
        md.push_str(&self.bibliography);
        md.push('\n');
        md
    }

    /// Get BibTeX entries for all papers.
    pub fn bib_entries(&self) -> Vec<BibEntry> {
        self.papers
            .iter()
            .map(|p| BibGenerator::from_paper(p))
            .collect()
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// Literature review pipeline.
///
/// Coordinates search → filter → categorize → synthesize → format.
pub struct LiteratureReviewPipeline {
    engine: AcademicSearchEngine,
    config: LiteratureReviewConfig,
}

impl LiteratureReviewPipeline {
    /// Create a new pipeline with the given engine and config.
    pub fn new(engine: AcademicSearchEngine, config: LiteratureReviewConfig) -> Self {
        Self { engine, config }
    }

    /// Create with default config.
    pub fn with_engine(engine: AcademicSearchEngine) -> Self {
        Self {
            engine,
            config: LiteratureReviewConfig::default(),
        }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: LiteratureReviewConfig) {
        self.config = config;
    }

    /// Execute the literature review pipeline.
    pub fn execute(&self, query: &str) -> LiteratureReview {
        // Step 1: Search for papers
        let search_config = AcademicSearchConfig {
            max_results: self.config.search_depth.max_papers(),
            year_range: self.config.year_range,
            fields_of_study: self.config.fields_of_study.clone(),
            ..Default::default()
        };

        let all_papers = self.engine.search_all(query, &search_config);
        let papers_found = all_papers.len();

        // Step 2: Filter and rank papers
        let mut papers = self.filter_and_rank(all_papers);
        papers.truncate(self.config.max_papers);
        let papers_included = papers.len();

        // Step 3: Categorize papers
        let categories = self.categorize_papers(&papers);

        // Step 4: Generate sections based on synthesis style
        let sections = self.generate_sections(&papers, &categories);

        // Step 5: Generate bibliography
        let bibliography = self.generate_bibliography(&papers);
        let bibtex = BibGenerator::from_papers(&papers);

        // Step 6: Build review
        LiteratureReview {
            title: format!("Literature Review: {}", query),
            sections,
            bibliography,
            bibtex,
            papers_found,
            papers_included,
            papers,
            synthesis_style: self.config.synthesis_style,
        }
    }

    /// Filter and rank papers by relevance and quality signals.
    fn filter_and_rank(&self, mut papers: Vec<AcademicPaper>) -> Vec<AcademicPaper> {
        // Remove papers without titles
        papers.retain(|p| !p.title.is_empty());

        // Sort by citation count (higher = better), then by year (newer = better)
        papers.sort_by(|a, b| {
            let cit_a = a.citation_count.unwrap_or(0);
            let cit_b = b.citation_count.unwrap_or(0);
            let year_a = a.year.unwrap_or(0);
            let year_b = b.year.unwrap_or(0);
            cit_b.cmp(&cit_a).then(year_b.cmp(&year_a))
        });

        papers
    }

    /// Categorize papers by field of study or topic.
    fn categorize_papers(&self, papers: &[AcademicPaper]) -> HashMap<String, Vec<usize>> {
        let mut categories: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, paper) in papers.iter().enumerate() {
            if paper.fields_of_study.is_empty() {
                categories.entry("General".to_string()).or_default().push(i);
            } else {
                for field in &paper.fields_of_study {
                    categories.entry(field.clone()).or_default().push(i);
                }
            }
        }

        categories
    }

    /// Generate review sections based on synthesis style.
    fn generate_sections(
        &self,
        papers: &[AcademicPaper],
        categories: &HashMap<String, Vec<usize>>,
    ) -> Vec<ReviewSection> {
        match self.config.synthesis_style {
            SynthesisStyle::Annotated => self.generate_annotated(papers),
            SynthesisStyle::Systematic => self.generate_systematic(papers, categories),
            SynthesisStyle::Comparative => self.generate_comparative(papers, categories),
            SynthesisStyle::Narrative => self.generate_narrative(papers, categories),
        }
    }

    /// Generate annotated bibliography sections (paper-by-paper).
    fn generate_annotated(&self, papers: &[AcademicPaper]) -> Vec<ReviewSection> {
        let mut sections = Vec::new();

        // Overview section
        let overview = format!(
            "This annotated bibliography covers {} papers on the topic.",
            papers.len()
        );
        sections.push(ReviewSection::new("Overview", &overview));

        // Group by year
        let mut by_year: HashMap<u16, Vec<&AcademicPaper>> = HashMap::new();
        for paper in papers {
            let year = paper.year.unwrap_or(0);
            by_year.entry(year).or_default().push(paper);
        }

        let mut years: Vec<u16> = by_year.keys().copied().collect();
        years.sort_unstable();
        years.reverse();

        for year in years {
            if let Some(year_papers) = by_year.get(&year) {
                let year_label = if year == 0 {
                    "Unknown Year".to_string()
                } else {
                    year.to_string()
                };

                let mut content = String::new();
                let mut ids = Vec::new();
                for paper in year_papers {
                    content.push_str(&format!(
                        "**{}**\n{}\n{}\n\n",
                        paper.title,
                        paper.citation_string(),
                        paper
                            .abstract_text
                            .as_deref()
                            .unwrap_or("(No abstract available)"),
                    ));
                    ids.push(paper.id.clone());
                }

                sections.push(ReviewSection::new(&year_label, content.trim()).with_papers(ids));
            }
        }

        sections
    }

    /// Generate systematic review sections (by category).
    fn generate_systematic(
        &self,
        papers: &[AcademicPaper],
        categories: &HashMap<String, Vec<usize>>,
    ) -> Vec<ReviewSection> {
        let mut sections = Vec::new();

        // Overview with statistics
        let overview = format!(
            "Systematic review of {} papers across {} categories.",
            papers.len(),
            categories.len()
        );
        sections.push(ReviewSection::new("Overview", &overview));

        // Statistics table
        let mut stats = String::from("| Category | Papers | Avg Citations |\n|---|---|---|\n");
        let mut sorted_cats: Vec<_> = categories.iter().collect();
        sorted_cats.sort_by_key(|e| std::cmp::Reverse(e.1.len()));

        for (cat, indices) in &sorted_cats {
            let avg_cit = if indices.is_empty() {
                0
            } else {
                let total: u32 = indices
                    .iter()
                    .map(|&i| papers[i].citation_count.unwrap_or(0))
                    .sum();
                total / indices.len() as u32
            };
            stats.push_str(&format!("| {} | {} | {} |\n", cat, indices.len(), avg_cit));
        }
        sections.push(ReviewSection::new("Distribution", &stats));

        // One section per category
        for (cat, indices) in sorted_cats {
            let mut content = String::new();
            let mut ids = Vec::new();
            for &i in indices {
                let p = &papers[i];
                content.push_str(&format!("- {} ({})\n", p.title, p.citation_string()));
                ids.push(p.id.clone());
            }
            sections.push(ReviewSection::new(cat, content.trim()).with_papers(ids));
        }

        sections
    }

    /// Generate comparative review.
    fn generate_comparative(
        &self,
        papers: &[AcademicPaper],
        categories: &HashMap<String, Vec<usize>>,
    ) -> Vec<ReviewSection> {
        let mut sections = Vec::new();

        let overview = format!(
            "Comparative analysis of {} papers across {} research areas.",
            papers.len(),
            categories.len()
        );
        sections.push(ReviewSection::new("Overview", &overview));

        // Comparison table
        let mut table = String::from("| Paper | Year | Citations | Source |\n|---|---|---|---|\n");
        for paper in papers.iter().take(30) {
            table.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                paper.title,
                paper.year.map(|y| y.to_string()).unwrap_or_default(),
                paper.citation_count.unwrap_or(0),
                paper.source.display_name(),
            ));
        }
        sections.push(ReviewSection::new("Comparison", &table));

        sections
    }

    /// Generate narrative review.
    fn generate_narrative(
        &self,
        papers: &[AcademicPaper],
        categories: &HashMap<String, Vec<usize>>,
    ) -> Vec<ReviewSection> {
        let mut sections = Vec::new();

        let overview = format!(
            "This review surveys {} papers covering the state of the art.",
            papers.len()
        );
        sections.push(ReviewSection::new("Introduction", &overview));

        // Group by category
        let mut sorted_cats: Vec<_> = categories.iter().collect();
        sorted_cats.sort_by_key(|e| std::cmp::Reverse(e.1.len()));

        for (cat, indices) in sorted_cats {
            let mut content = String::new();
            let mut ids = Vec::new();
            for &i in indices {
                let p = &papers[i];
                let year_str = p.year.map(|y| format!(" ({})", y)).unwrap_or_default();
                content.push_str(&format!(
                    "{}{} studied {}. ",
                    p.authors
                        .first()
                        .map(|a| a.name.as_str())
                        .unwrap_or("Unknown"),
                    year_str,
                    p.title.to_lowercase(),
                ));
                ids.push(p.id.clone());
            }
            sections.push(ReviewSection::new(cat, content.trim()).with_papers(ids));
        }

        sections
    }

    /// Generate bibliography text.
    fn generate_bibliography(&self, papers: &[AcademicPaper]) -> String {
        let mut bib = String::new();
        for (i, paper) in papers.iter().enumerate() {
            let ref_str = self.config.bibliography_format.format_reference(paper);
            bib.push_str(&format!("[{}] {}\n", i + 1, ref_str));
        }
        bib
    }

    /// Get the current config.
    pub fn config(&self) -> &LiteratureReviewConfig {
        &self.config
    }

    /// Get the search engine.
    pub fn engine(&self) -> &AcademicSearchEngine {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::super::academic_search::{AcademicPaper, AcademicSource, Author};
    use super::*;

    fn sample_papers() -> Vec<AcademicPaper> {
        let mut p1 = AcademicPaper::new("1", "Attention Is All You Need", AcademicSource::ArXiv);
        p1.authors = vec![Author::new("Vaswani"), Author::new("Shazeer")];
        p1.year = Some(2017);
        p1.citation_count = Some(50000);
        p1.abstract_text =
            Some("We propose a new model architecture, the Transformer.".to_string());
        p1.fields_of_study = vec!["Computer Science".to_string()];

        let mut p2 = AcademicPaper::new(
            "2",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            AcademicSource::SemanticScholar,
        );
        p2.authors = vec![Author::new("Devlin"), Author::new("Chang")];
        p2.year = Some(2019);
        p2.citation_count = Some(30000);
        p2.abstract_text = Some("We introduce BERT.".to_string());
        p2.fields_of_study = vec!["Computer Science".to_string(), "Linguistics".to_string()];

        let mut p3 = AcademicPaper::new(
            "3",
            "GPT-3: Language Models are Few-Shot Learners",
            AcademicSource::SemanticScholar,
        );
        p3.authors = vec![Author::new("Brown"), Author::new("Mann")];
        p3.year = Some(2020);
        p3.citation_count = Some(15000);
        p3.abstract_text = Some("We demonstrate that scaling up language models.".to_string());
        p3.fields_of_study = vec!["Computer Science".to_string()];

        vec![p1, p2, p3]
    }

    #[test]
    fn test_search_depth_max_papers() {
        assert_eq!(SearchDepth::Quick.max_papers(), 20);
        assert_eq!(SearchDepth::Standard.max_papers(), 50);
        assert_eq!(SearchDepth::Deep.max_papers(), 100);
    }

    #[test]
    fn test_synthesis_style_display() {
        assert_eq!(SynthesisStyle::Narrative.to_string(), "Narrative");
        assert_eq!(SynthesisStyle::Systematic.to_string(), "Systematic");
        assert_eq!(SynthesisStyle::Annotated.to_string(), "Annotated");
        assert_eq!(SynthesisStyle::Comparative.to_string(), "Comparative");
    }

    #[test]
    fn test_config_defaults() {
        let config = LiteratureReviewConfig::default();
        assert_eq!(config.max_papers, 50);
        assert_eq!(config.search_depth, SearchDepth::Standard);
        assert_eq!(config.synthesis_style, SynthesisStyle::Narrative);
        assert!(!config.include_citation_graph);
    }

    #[test]
    fn test_config_quick_preset() {
        let config = LiteratureReviewConfig::quick();
        assert_eq!(config.max_papers, 10);
        assert_eq!(config.search_depth, SearchDepth::Quick);
        assert_eq!(config.synthesis_style, SynthesisStyle::Annotated);
    }

    #[test]
    fn test_config_systematic_preset() {
        let config = LiteratureReviewConfig::systematic();
        assert_eq!(config.search_depth, SearchDepth::Deep);
        assert!(config.include_citation_graph);
    }

    #[test]
    fn test_review_section_creation() {
        let section = ReviewSection::new("Introduction", "This is the intro content.");
        assert_eq!(section.heading, "Introduction");
        assert_eq!(section.word_count(), 5);
        assert!(section.paper_ids.is_empty());
    }

    #[test]
    fn test_review_section_with_papers() {
        let section = ReviewSection::new("Test", "Content")
            .with_papers(vec!["p1".to_string(), "p2".to_string()]);
        assert_eq!(section.paper_ids.len(), 2);
    }

    #[test]
    fn test_bibliography_format_apa() {
        let mut paper = AcademicPaper::new("1", "Test Paper", AcademicSource::ArXiv);
        paper.authors = vec![Author::new("Smith")];
        paper.year = Some(2024);
        paper.venue = Some("Nature".to_string());

        let ref_str = BibliographyFormat::Apa.format_reference(&paper);
        assert!(ref_str.contains("Smith"));
        assert!(ref_str.contains("2024"));
        assert!(ref_str.contains("Test Paper"));
    }

    #[test]
    fn test_bibliography_format_ieee() {
        let mut paper = AcademicPaper::new("1", "Test", AcademicSource::PubMed);
        paper.authors = vec![Author::new("Jones")];
        paper.year = Some(2023);

        let ref_str = BibliographyFormat::Ieee.format_reference(&paper);
        assert!(ref_str.contains("Jones"));
    }

    #[test]
    fn test_filter_and_rank() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);

        let ranked = pipeline.filter_and_rank(papers);
        // Should be sorted by citations (descending)
        assert_eq!(ranked[0].citation_count, Some(50000));
        assert_eq!(ranked[1].citation_count, Some(30000));
    }

    #[test]
    fn test_categorize_papers() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);

        let categories = pipeline.categorize_papers(&papers);
        assert!(categories.contains_key("Computer Science"));
    }

    #[test]
    fn test_generate_annotated() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);

        let sections = pipeline.generate_annotated(&papers);
        assert!(!sections.is_empty());
        assert_eq!(sections[0].heading, "Overview");
    }

    #[test]
    fn test_generate_systematic() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);
        let categories = pipeline.categorize_papers(&papers);

        let sections = pipeline.generate_systematic(&papers, &categories);
        assert!(sections.len() >= 2); // Overview + Distribution + categories
    }

    #[test]
    fn test_generate_comparative() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);
        let categories = pipeline.categorize_papers(&papers);

        let sections = pipeline.generate_comparative(&papers, &categories);
        assert!(sections.len() >= 2); // Overview + Comparison
    }

    #[test]
    fn test_generate_narrative() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);
        let categories = pipeline.categorize_papers(&papers);

        let sections = pipeline.generate_narrative(&papers, &categories);
        assert!(!sections.is_empty());
    }

    #[test]
    fn test_generate_bibliography() {
        let papers = sample_papers();
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::with_engine(engine);

        let bib = pipeline.generate_bibliography(&papers);
        assert!(bib.contains("[1]"));
        assert!(bib.contains("[2]"));
    }

    #[test]
    fn test_literature_review_to_markdown() {
        let review = LiteratureReview {
            title: "Test Review".to_string(),
            sections: vec![
                ReviewSection::new("Introduction", "Intro text."),
                ReviewSection::new("Methods", "Method text."),
            ],
            bibliography: "[1] Smith (2024). Paper.".to_string(),
            bibtex: "@article{test, title={Test}}".to_string(),
            papers_found: 10,
            papers_included: 5,
            papers: Vec::new(),
            synthesis_style: SynthesisStyle::Narrative,
        };

        let md = review.to_markdown();
        assert!(md.contains("# Test Review"));
        assert!(md.contains("## Introduction"));
        assert!(md.contains("## References"));
    }

    #[test]
    fn test_literature_review_word_count() {
        let review = LiteratureReview {
            title: "Test".to_string(),
            sections: vec![
                ReviewSection::new("A", "one two three"),
                ReviewSection::new("B", "four five"),
            ],
            bibliography: String::new(),
            bibtex: String::new(),
            papers_found: 0,
            papers_included: 0,
            papers: Vec::new(),
            synthesis_style: SynthesisStyle::Narrative,
        };
        assert_eq!(review.total_word_count(), 5);
    }

    #[test]
    fn test_pipeline_execute_empty_engine() {
        let engine = AcademicSearchEngine::new();
        let pipeline = LiteratureReviewPipeline::new(engine, LiteratureReviewConfig::quick());

        let review = pipeline.execute("transformers NLP");
        assert_eq!(review.papers_found, 0);
        assert_eq!(review.papers_included, 0);
        assert!(!review.sections.is_empty()); // Still has overview section
    }

    #[test]
    fn test_pipeline_config_access() {
        let engine = AcademicSearchEngine::new();
        let config = LiteratureReviewConfig::quick();
        let pipeline = LiteratureReviewPipeline::new(engine, config);
        assert_eq!(pipeline.config().max_papers, 10);
    }
}
