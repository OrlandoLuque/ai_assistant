//! Ingesting academic papers into the RAG index.
//!
//! Research and RAG both existed and did not speak to each other: you could find fifty
//! papers and be left holding a list. This is the missing step — putting them in the
//! index so they can be *asked* afterwards, which is what turns a literature search into
//! something you work with.
//!
//! Two decisions shape everything here:
//!
//! * **The source key is the DOI when there is one.** `RagDb::index_document` keys on
//!   `source` and re-indexes when the content hash changes, so a stable key is what makes
//!   ingesting the same paper twice a no-op instead of a duplicate. The DOI is the only
//!   identifier that survives being found through a different provider — the same paper
//!   from arXiv and from Semantic Scholar carries different provider IDs and the same DOI.
//! * **Metadata goes into the text, not beside it.** The index stores chunks of text, so
//!   authors, year and venue are written as a small header inside the document. Without
//!   it, "who wrote the 2024 paper about X" cannot be answered from a chunk that contains
//!   only the abstract.

use crate::academic_search::AcademicPaper;

/// What an ingestion run did, kept per-paper rather than as one total.
///
/// `skipped` is not a failure: `index_document` returns 0 chunks when the content is
/// unchanged, which is the normal outcome of re-running a search. Reporting it separately
/// from `failed` is the difference between "nothing to do" and "something went wrong".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestReport {
    /// Papers indexed, and the chunks they produced.
    pub indexed: usize,
    /// Total chunks written across all indexed papers.
    pub chunks: usize,
    /// Papers already in the index with identical content.
    pub skipped: usize,
    /// Papers that could not be indexed, with the reason.
    pub failed: Vec<(String, String)>,
}

impl IngestReport {
    /// Papers seen, whatever happened to them.
    pub fn total(&self) -> usize {
        self.indexed + self.skipped + self.failed.len()
    }
}

/// The key a paper is stored under.
///
/// DOI first because it is provider-independent (see the module note); then the paper URL;
/// then a provider-qualified id. The last resort is deliberately qualified rather than a
/// bare id: two providers can both use `12345`, and an unqualified key would silently
/// overwrite one paper with another.
pub fn paper_source_key(paper: &AcademicPaper) -> String {
    if let Some(doi) = paper.doi.as_ref().filter(|d| !d.trim().is_empty()) {
        return format!("doi:{}", normalise_doi(doi));
    }
    if let Some(url) = paper.url.as_ref().filter(|u| !u.trim().is_empty()) {
        return url.clone();
    }
    format!(
        "{}:{}",
        paper.source.display_name().to_lowercase(),
        paper.id
    )
}

/// DOIs are case-insensitive and travel with assorted prefixes; comparing the raw strings
/// would file `10.1/ABC`, `10.1/abc` and `https://doi.org/10.1/abc` as three papers.
fn normalise_doi(doi: &str) -> String {
    doi.trim()
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/")
        .trim_start_matches("doi:")
        .to_lowercase()
}

/// Render a paper as the text that gets indexed.
///
/// The abstract carries the meaning; the header carries everything a question might ask
/// about *which* paper. Papers without an abstract still produce a useful document —
/// title, authors and venue are enough to find it again — so they are ingested rather
/// than skipped.
pub fn paper_to_document(paper: &AcademicPaper) -> String {
    let mut doc = format!("# {}\n\n", paper.title);

    if !paper.authors.is_empty() {
        let names: Vec<&str> = paper.authors.iter().map(|a| a.name.as_str()).collect();
        doc.push_str(&format!("Authors: {}\n", names.join(", ")));
    }
    if let Some(year) = paper.year {
        doc.push_str(&format!("Year: {year}\n"));
    }
    if let Some(venue) = &paper.venue {
        doc.push_str(&format!("Venue: {venue}\n"));
    }
    if let Some(doi) = &paper.doi {
        doc.push_str(&format!("DOI: {doi}\n"));
    }
    doc.push_str(&format!("Source: {}\n", paper.source.display_name()));
    if !paper.fields_of_study.is_empty() {
        doc.push_str(&format!("Fields: {}\n", paper.fields_of_study.join(", ")));
    }

    match &paper.abstract_text {
        Some(text) if !text.trim().is_empty() => {
            doc.push_str("\n## Abstract\n\n");
            doc.push_str(text.trim());
            doc.push('\n');
        }
        _ => doc.push_str("\n(No abstract available.)\n"),
    }
    doc
}

#[cfg(feature = "rag")]
mod ingest {
    use super::*;
    use crate::rag::RagDb;

    /// Ingest papers into `db`, keyed so that re-running a search does not duplicate.
    ///
    /// A paper that fails to index does not stop the rest: a literature search returns
    /// dozens of papers and losing all of them because one has a malformed field would be
    /// the wrong trade. Failures are collected and reported.
    pub fn ingest_papers(db: &RagDb, papers: &[AcademicPaper]) -> IngestReport {
        let mut report = IngestReport::default();
        for paper in papers {
            let key = paper_source_key(paper);
            let doc = paper_to_document(paper);
            match db.index_document(&key, &doc) {
                Ok(0) => report.skipped += 1,
                Ok(n) => {
                    report.indexed += 1;
                    report.chunks += n;
                }
                Err(e) => report.failed.push((key, e.to_string())),
            }
        }
        report
    }
}

#[cfg(feature = "rag")]
pub use ingest::ingest_papers;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::academic_search::{AcademicSource, Author};

    fn paper() -> AcademicPaper {
        let mut p = AcademicPaper::new(
            "2301.01234",
            "Attention Is Some Of What You Need",
            AcademicSource::ArXiv,
        );
        p.authors = vec![Author::new("A. Researcher"), Author::new("B. Coauthor")];
        p.year = Some(2024);
        p.venue = Some("NeurIPS".to_string());
        p.abstract_text = Some("We show that some attention suffices.".to_string());
        p
    }

    #[test]
    fn the_doi_is_the_key_when_there_is_one() {
        let mut p = paper();
        p.doi = Some("10.1234/ABC".to_string());
        assert_eq!(paper_source_key(&p), "doi:10.1234/abc");
    }

    #[test]
    fn the_same_paper_from_two_providers_gets_one_key() {
        // The whole reason the DOI comes first: provider ids differ, the DOI does not.
        // Without this, a search across arXiv and Semantic Scholar indexes every paper
        // that both know about twice.
        let mut from_arxiv = paper();
        from_arxiv.doi = Some("10.1234/abc".to_string());
        let mut from_s2 = AcademicPaper::new(
            "s2-999",
            "Attention Is Some Of What You Need",
            AcademicSource::SemanticScholar,
        );
        from_s2.doi = Some("https://doi.org/10.1234/ABC".to_string());
        assert_eq!(paper_source_key(&from_arxiv), paper_source_key(&from_s2));
    }

    #[test]
    fn without_a_doi_the_url_is_used_and_then_a_qualified_id() {
        let mut p = paper();
        p.doi = None;
        p.url = Some("http://arxiv.org/abs/2301.01234".to_string());
        assert_eq!(paper_source_key(&p), "http://arxiv.org/abs/2301.01234");

        p.url = None;
        // Qualified, not bare: two providers can both call a paper "2301.01234", and a
        // bare id would file one on top of the other.
        assert_eq!(paper_source_key(&p), "arxiv:2301.01234");
    }

    #[test]
    fn an_empty_doi_does_not_win_over_a_usable_url() {
        let mut p = paper();
        p.doi = Some("   ".to_string());
        p.url = Some("http://arxiv.org/abs/2301.01234".to_string());
        assert_eq!(paper_source_key(&p), "http://arxiv.org/abs/2301.01234");
    }

    #[test]
    fn the_document_carries_what_a_question_might_ask_about() {
        let doc = paper_to_document(&paper());
        assert!(doc.contains("Attention Is Some Of What You Need"));
        assert!(doc.contains("A. Researcher, B. Coauthor"));
        assert!(doc.contains("Year: 2024"));
        assert!(doc.contains("NeurIPS"));
        assert!(doc.contains("some attention suffices"));
    }

    #[test]
    fn a_paper_with_no_abstract_is_still_worth_indexing() {
        // Title, authors and venue are enough to find it again, so skipping it would
        // lose a paper the search legitimately returned.
        let mut p = paper();
        p.abstract_text = None;
        let doc = paper_to_document(&p);
        assert!(doc.contains("Attention Is Some Of What You Need"));
        assert!(doc.contains("No abstract available"));
    }

    #[test]
    fn the_report_separates_nothing_to_do_from_something_went_wrong() {
        let report = IngestReport {
            indexed: 2,
            chunks: 7,
            skipped: 3,
            failed: vec![("doi:x".into(), "disk full".into())],
        };
        assert_eq!(report.total(), 6);
        // Re-running a search skips everything unchanged; that is the normal path and
        // must not read as failure.
        assert!(report.skipped > 0 && report.failed.len() == 1);
    }

    /// The claim this module exists to make is "ingest twice, index once". Asserting it on
    /// the key strings alone would only test my own helper; this drives a real `RagDb` and
    /// checks what actually lands in it.
    #[cfg(feature = "rag")]
    #[test]
    fn ingesting_the_same_search_twice_does_not_duplicate() {
        let path =
            std::env::temp_dir().join(format!("research_rag_ingest_{}.db", uuid::Uuid::new_v4()));
        let db = crate::rag::RagDb::open(&path).expect("open temp rag db");

        let mut arxiv = paper();
        arxiv.doi = Some("10.1234/abc".to_string());
        let other = AcademicPaper::new("2302.00001", "A Second Paper", AcademicSource::ArXiv);

        let first = ingest_papers(&db, &[arxiv.clone(), other.clone()]);
        assert_eq!(first.indexed, 2, "both papers are new: {first:?}");
        assert_eq!(first.skipped, 0);
        assert!(first.failed.is_empty(), "unexpected failures: {first:?}");
        assert!(first.chunks >= 2, "each paper produced a chunk: {first:?}");

        // The same two papers again: unchanged content, so nothing is written.
        let again = ingest_papers(&db, &[arxiv.clone(), other.clone()]);
        assert_eq!(again.indexed, 0, "nothing changed: {again:?}");
        assert_eq!(again.skipped, 2, "both already present: {again:?}");

        // Now the first paper as Semantic Scholar would return it — different provider id,
        // same DOI. It lands on the same key, so it *replaces* rather than duplicating:
        // the rendering differs (Source: and the DOI form), so `index_document` re-indexes,
        // which deletes the old chunks first. One document, not two.
        let mut from_s2 = AcademicPaper::new(
            "s2-999",
            "Attention Is Some Of What You Need",
            AcademicSource::SemanticScholar,
        );
        from_s2.doi = Some("https://doi.org/10.1234/ABC".to_string());
        from_s2.authors = arxiv.authors.clone();
        from_s2.year = arxiv.year;
        from_s2.venue = arxiv.venue.clone();
        from_s2.abstract_text = arxiv.abstract_text.clone();

        let third = ingest_papers(&db, &[from_s2]);
        assert_eq!(third.indexed, 1, "re-rendered, so re-indexed: {third:?}");
        assert!(
            db.is_document_indexed("doi:10.1234/abc")
                .expect("query the doi key"),
            "the paper is still filed under its DOI"
        );
        assert!(
            !db.is_document_indexed("semantic scholar:s2-999")
                .expect("query the provider key"),
            "and NOT a second time under the Semantic Scholar id — that is the duplicate \
             this module exists to prevent"
        );

        let _ = std::fs::remove_file(&path);
    }
}
