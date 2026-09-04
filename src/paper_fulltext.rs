//! Downloading a paper's PDF and turning it into text.
//!
//! The gap this closes: `paper_metadata` could already pull structure out of a paper's
//! text, and `document_parsing` could already turn a PDF into text — but nothing joined
//! them to a search result. So the subsystem could *find* a paper and never read it, and
//! [`crate::research_rag`] indexed a title and an abstract when the whole paper was one
//! HTTP request away.
//!
//! Three things decide the shape of this module, and all three are about refusing to
//! guess:
//!
//! * **Most search results have no PDF.** `pdf_url` is populated by arXiv and OpenAlex and
//!   is `None` for most Crossref and PubMed records. That is not a failure and is not
//!   reported as one — it is counted separately, because "there is no open PDF" and "the
//!   download broke" call for different reactions.
//! * **A paywall answers 200 OK with HTML.** Publishers return a login page, not an error,
//!   so the status code says nothing. The bytes are checked for the `%PDF-` magic before
//!   anything is parsed; feeding an HTML login form to a PDF parser produces plausible
//!   garbage, which is worse than a clean skip.
//! * **Papers can be enormous.** Supplementary material runs to hundreds of megabytes. The
//!   response is capped rather than streamed into memory unbounded.
//!
//! Fetching reuses [`crate::academic_search`]'s retry policy, so a throttled host is
//! handled the same way here as everywhere else rather than by a second, divergent copy.

use crate::academic_search::{AcademicPaper, AcademicSearchError};

/// Largest PDF we will pull into memory.
///
/// Typical papers are under 5 MB. The cap exists for the supplementary-material case,
/// where a single "paper" can be a few hundred megabytes of data tables — downloading that
/// to extract prose is never what the caller meant.
pub const MAX_PDF_BYTES: usize = 32 * 1024 * 1024;

/// What a full-text run did, per paper.
///
/// The three "did not happen" counters are separate on purpose. Collapsing them into one
/// `skipped` would hide the difference between a corpus that is mostly closed-access and a
/// network that is broken, and those need opposite responses from the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FullTextReport {
    /// Papers whose PDF was fetched and parsed.
    pub extracted: usize,
    /// Characters of text recovered in total.
    pub chars: usize,
    /// Papers with no `pdf_url` at all — normal for Crossref and PubMed.
    pub no_pdf_url: usize,
    /// Fetches that returned something that was not a PDF, almost always a paywall page.
    pub not_a_pdf: usize,
    /// Fetches that exceeded [`MAX_PDF_BYTES`].
    pub too_large: usize,
    /// Real failures, with the reason.
    pub failed: Vec<(String, String)>,
}

impl FullTextReport {
    /// Papers seen, whatever happened to them.
    pub fn total(&self) -> usize {
        self.extracted + self.no_pdf_url + self.not_a_pdf + self.too_large + self.failed.len()
    }

    /// Papers that yielded no text for a reason that is not an error.
    ///
    /// Useful for the common report line: "40 papers, 12 read, 28 without an open PDF".
    pub fn unavailable(&self) -> usize {
        self.no_pdf_url + self.not_a_pdf + self.too_large
    }
}

/// Do these bytes begin a PDF?
///
/// The spec allows leading junk before `%PDF-`, and some publishers prepend a byte-order
/// mark or stray whitespace, so the marker is looked for in the first few hundred bytes
/// rather than only at offset 0. Anything else — an HTML login page, a JSON error, an
/// empty body — is not a PDF and must not reach the parser.
pub fn looks_like_pdf(bytes: &[u8]) -> bool {
    let window = &bytes[..bytes.len().min(1024)];
    window.windows(5).any(|w| w == b"%PDF-")
}

/// Download a PDF, refusing anything that is not one.
///
/// Errors are [`AcademicSearchError`] so a caller already handling academic search does not
/// need a second error type; in particular a throttled host still surfaces as
/// `RateLimit` rather than as a generic network failure.
pub fn fetch_pdf(url: &str, timeout: std::time::Duration) -> Result<Vec<u8>, AcademicSearchError> {
    if url.trim().is_empty() {
        return Err(AcademicSearchError::InvalidQuery(
            "empty PDF URL".to_string(),
        ));
    }

    let response = crate::academic_search::get_with_retry(|| {
        ureq::get(url)
            .timeout(timeout)
            .set("User-Agent", crate::academic_search::USER_AGENT)
            .set("Accept", "application/pdf")
    })?;

    // `Content-Length` is a hint, not a guarantee — it is absent on chunked responses and
    // some servers lie. It is used to bail out early when present, and the real enforcement
    // is on the bytes actually read.
    if let Some(len) = response
        .header("Content-Length")
        .and_then(|v| v.parse::<usize>().ok())
    {
        if len > MAX_PDF_BYTES {
            return Err(AcademicSearchError::Parse(format!(
                "PDF is {len} bytes, over the {MAX_PDF_BYTES} cap"
            )));
        }
    }

    use std::io::Read as _;
    let mut buf = Vec::new();
    response
        .into_reader()
        .take((MAX_PDF_BYTES + 1) as u64)
        .read_to_end(&mut buf)
        .map_err(|e| AcademicSearchError::Network(e.to_string()))?;

    if buf.len() > MAX_PDF_BYTES {
        return Err(AcademicSearchError::Parse(format!(
            "PDF exceeds the {MAX_PDF_BYTES} byte cap"
        )));
    }
    if !looks_like_pdf(&buf) {
        // Deliberately not a Network error: the request succeeded. The server chose to
        // answer with something else, which is usually a login wall.
        return Err(AcademicSearchError::ProviderUnavailable(format!(
            "response from {url} is not a PDF ({} bytes) — usually a paywall or login page",
            buf.len()
        )));
    }
    Ok(buf)
}

#[cfg(feature = "documents")]
mod extract {
    use super::*;
    use crate::document_parsing::{DocumentFormat, DocumentParser};

    /// Turn PDF bytes into plain text.
    pub fn pdf_to_text(bytes: &[u8]) -> Result<String, AcademicSearchError> {
        let parser = DocumentParser::new(Default::default());
        let parsed = parser
            .parse_bytes(bytes, DocumentFormat::Pdf)
            .map_err(|e| AcademicSearchError::Parse(e.to_string()))?;
        Ok(parsed.text)
    }

    /// Fetch a paper's PDF and return its text.
    ///
    /// `Ok(None)` means the paper has no `pdf_url` — a fact about the record, not a
    /// failure, and the caller usually wants to count it rather than log it.
    pub fn fetch_fulltext(
        paper: &AcademicPaper,
        timeout: std::time::Duration,
    ) -> Result<Option<String>, AcademicSearchError> {
        let Some(url) = paper.pdf_url.as_ref().filter(|u| !u.trim().is_empty()) else {
            return Ok(None);
        };
        let bytes = fetch_pdf(url, timeout)?;
        pdf_to_text(&bytes).map(Some)
    }

    /// Read as many of these papers as are actually readable.
    ///
    /// Never stops on a failure: a literature search returns dozens of papers and half of
    /// them will be closed-access, so aborting on the first one would make the whole thing
    /// useless. Everything is counted in the [`FullTextReport`].
    pub fn fetch_fulltexts(
        papers: &[AcademicPaper],
        timeout: std::time::Duration,
    ) -> (Vec<(String, String)>, FullTextReport) {
        let mut out = Vec::new();
        let mut report = FullTextReport::default();

        for paper in papers {
            let key = crate::research_rag::paper_source_key(paper);
            match fetch_fulltext(paper, timeout) {
                Ok(Some(text)) => {
                    report.extracted += 1;
                    report.chars += text.len();
                    out.push((key, text));
                }
                Ok(None) => report.no_pdf_url += 1,
                Err(AcademicSearchError::ProviderUnavailable(_)) => report.not_a_pdf += 1,
                Err(AcademicSearchError::Parse(msg)) if msg.contains("cap") => {
                    report.too_large += 1
                }
                Err(e) => report.failed.push((key, e.to_string())),
            }
        }
        (out, report)
    }
}

#[cfg(feature = "documents")]
pub use extract::{fetch_fulltext, fetch_fulltexts, pdf_to_text};

#[cfg(all(feature = "documents", feature = "rag"))]
mod ingest {
    use super::*;
    use crate::rag::RagDb;
    use crate::research_rag::{paper_source_key, IngestReport};

    /// Index the papers' **full text** instead of their abstracts.
    ///
    /// Keyed with [`paper_source_key`], the same function
    /// [`crate::research_rag::ingest_papers`] uses. That is the point: a paper already
    /// indexed from its abstract is *replaced* by its full text rather than stored twice,
    /// because `index_document` deletes the previous chunks for a source before writing.
    /// Running the cheap abstract pass first and this one later is therefore a safe
    /// upgrade, not a duplication.
    pub fn ingest_papers_fulltext(
        db: &RagDb,
        papers: &[AcademicPaper],
        timeout: std::time::Duration,
    ) -> (IngestReport, FullTextReport) {
        let (texts, full) = fetch_fulltexts(papers, timeout);
        let mut ingest = IngestReport::default();

        for (key, text) in texts {
            // The header keeps the paper identifiable inside a chunk, exactly as the
            // abstract-only path does — a chunk from page 9 must still say which paper it
            // came from.
            let paper = papers.iter().find(|p| paper_source_key(p) == key);
            let doc = match paper {
                Some(p) => format!("{}\n\n## Full text\n\n{}", header_for(p), text.trim()),
                None => text,
            };
            match db.index_document(&key, &doc) {
                Ok(0) => ingest.skipped += 1,
                Ok(n) => {
                    ingest.indexed += 1;
                    ingest.chunks += n;
                }
                Err(e) => ingest.failed.push((key, e.to_string())),
            }
        }
        (ingest, full)
    }

    /// Title, authors, year, venue, DOI — the part of
    /// [`crate::research_rag::paper_to_document`] that is not the abstract.
    fn header_for(paper: &AcademicPaper) -> String {
        let rendered = crate::research_rag::paper_to_document(paper);
        match rendered.find("\n## Abstract") {
            Some(i) => rendered[..i].trim_end().to_string(),
            None => match rendered.find("\n(No abstract available.)") {
                Some(i) => rendered[..i].trim_end().to_string(),
                None => rendered,
            },
        }
    }
}

#[cfg(all(feature = "documents", feature = "rag"))]
pub use ingest::ingest_papers_fulltext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_real_pdf_is_recognised() {
        assert!(looks_like_pdf(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n1 0 obj"));
    }

    #[test]
    fn a_paywall_page_is_not_a_pdf() {
        // This is the case that matters: the request succeeded, the status was 200, and
        // the body is a login form. Without the magic-byte check it reaches the parser and
        // comes out as plausible nonsense.
        let html = b"<!DOCTYPE html>\n<html><head><title>Sign in to continue</title></head>";
        assert!(!looks_like_pdf(html));
    }

    #[test]
    fn leading_junk_before_the_marker_is_tolerated() {
        // The spec allows bytes before %PDF-, and some servers prepend a BOM.
        let mut bytes = vec![0xEF, 0xBB, 0xBF, b'\n', b' '];
        bytes.extend_from_slice(b"%PDF-1.4");
        assert!(looks_like_pdf(&bytes));
    }

    #[test]
    fn an_empty_or_tiny_body_is_not_a_pdf() {
        assert!(!looks_like_pdf(b""));
        assert!(!looks_like_pdf(b"%PD"));
    }

    #[test]
    fn the_marker_is_not_hunted_through_the_whole_file() {
        // Only the first 1 KB is searched. A file whose *content* happens to mention
        // "%PDF-" far in is not thereby a PDF, and scanning 32 MB for a 5-byte string on
        // every download would be a silly cost.
        let mut bytes = vec![b'x'; 4096];
        bytes.extend_from_slice(b"%PDF-1.4");
        assert!(!looks_like_pdf(&bytes));
    }

    #[test]
    fn an_empty_url_is_refused_without_a_request() {
        assert!(matches!(
            fetch_pdf("   ", std::time::Duration::from_secs(1)),
            Err(AcademicSearchError::InvalidQuery(_))
        ));
    }

    #[test]
    fn the_report_separates_closed_access_from_broken() {
        // A corpus that is mostly paywalled and a network that is down produce very
        // different numbers here, and the caller should be able to tell which they have.
        let report = FullTextReport {
            extracted: 4,
            chars: 120_000,
            no_pdf_url: 20,
            not_a_pdf: 6,
            too_large: 1,
            failed: vec![("doi:x".into(), "connection refused".into())],
        };
        assert_eq!(report.total(), 32);
        assert_eq!(report.unavailable(), 27);
        assert_eq!(report.failed.len(), 1);
    }
}
