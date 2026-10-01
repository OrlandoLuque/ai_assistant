//! Running a retriever over a judged corpus to produce a scorable run.
//!
//! [`crate::retrieval_metrics`] scores a run. It does not produce one: it reads a
//! `RunFile` that *something else* wrote, and until this module there was no
//! something else. That is the gap this closes — the step that turns five metric
//! functions into a measurement.
//!
//! ```text
//!   corpus + queries + qrels        (this module: load_beir_dir)
//!               │
//!               ▼
//!   run_corpus(corpus, retriever)   (this module)
//!               │
//!               ▼
//!            RunFile  ──▶  retrieval_metrics::summarise  ──▶  RetrievalReport
//! ```
//!
//! # The format
//!
//! BEIR's layout, because BEIR, NanoBEIR, MIRACL and MessIRve all publish it and
//! a loader that reads it reads all four:
//!
//! | file | one line is |
//! |---|---|
//! | `corpus.jsonl` | `{"_id": "d1", "title": "…", "text": "…"}` |
//! | `queries.jsonl` | `{"_id": "q1", "text": "…"}` |
//! | `qrels/test.tsv` | `query-id⇥corpus-id⇥score`, with a header row |
//!
//! Parsing is separate from acquiring on purpose. These functions take `&str` and
//! are pure, so the whole loader is testable with no network and no files. That is
//! also the answer to the design note left in N108: `BenchmarkLoader` folds
//! *download* and *parse* into one trait, which is why a dataset that is
//! **generated** rather than downloaded does not fit it. Here, where the bytes come
//! from is the caller's problem.
//!
//! # The hazard this module exists to prevent
//!
//! **A retriever that returns ids from a different id space scores 0.0, and 0.0
//! looks like a result.**
//!
//! It is not hypothetical. The crate's lexical retrieval runs over SQLite FTS5, and
//! `KnowledgeChunk::id` is an autoincrementing `i64` row id — not the corpus's
//! `doc_id`. An adapter that returns row ids produces a run where every retrieved
//! id is unknown to the qrels, so every query scores zero, and the report reads
//! "recall@10 = 0.000" as confidently as it would report a real failure.
//!
//! So [`run_corpus`] **refuses** a run in which no returned id is one the corpus
//! contains, and counts the unknown ids otherwise. A measurement instrument whose
//! most likely wiring error is indistinguishable from a bad score is not an
//! instrument.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::retrieval_metrics::{QueryRun, RunFile};

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// One document of the collection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusDoc {
    /// The corpus's own identifier. This is what qrels and runs refer to.
    pub id: String,
    /// Title, where the dataset has one. Often empty.
    pub title: String,
    /// Body text.
    pub text: String,
}

impl CorpusDoc {
    /// Title and body joined, which is what BEIR baselines index.
    ///
    /// Indexing the body alone is a legitimate choice and a *different* one; it
    /// usually scores lower. Whichever an adapter picks, it must say so, because
    /// comparing a title+body run against a body-only run compares two corpora.
    pub fn indexable_text(&self) -> String {
        if self.title.is_empty() {
            self.text.clone()
        } else {
            format!("{}\n\n{}", self.title, self.text)
        }
    }
}

/// One query of the evaluation set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusQuery {
    /// The query's own identifier, as used in qrels.
    pub id: String,
    /// The query text handed to the retriever.
    pub text: String,
}

/// A document collection, a query set, and the judgements that connect them.
#[derive(Debug, Clone, Default)]
pub struct RetrievalCorpus {
    /// Name for reports. Not used in any computation.
    pub name: String,
    /// The documents to index.
    pub docs: Vec<CorpusDoc>,
    /// The queries to run.
    pub queries: Vec<CorpusQuery>,
    /// `query_id -> doc_id -> grade`. A pair that is absent is not judged, which
    /// is **not** the same as judged irrelevant — see [`RetrievalCorpus::unjudged_queries`].
    pub qrels: HashMap<String, HashMap<String, f64>>,
}

impl RetrievalCorpus {
    /// Queries with no judged-relevant document.
    ///
    /// These cannot be scored and are excluded from every average by
    /// [`crate::retrieval_metrics::summarise`]. Worth looking at before the
    /// scores: a corpus where most queries are unjudged produces a report about
    /// the judgements rather than about the retriever.
    pub fn unjudged_queries(&self) -> Vec<&str> {
        self.queries
            .iter()
            .filter(|q| {
                self.qrels
                    .get(&q.id)
                    .is_none_or(|g| !g.values().any(|v| *v > 0.0))
            })
            .map(|q| q.id.as_str())
            .collect()
    }

    /// Judged pairs whose `corpus-id` is not in `docs`.
    ///
    /// A non-empty answer means the qrels and the corpus file disagree — usually a
    /// partial download, or qrels from one split against the corpus of another.
    /// Those documents can never be retrieved, so recall is capped below 1.0 and
    /// the cap is invisible in the report.
    pub fn qrels_outside_corpus(&self) -> Vec<(&str, &str)> {
        let ids: HashSet<&str> = self.docs.iter().map(|d| d.id.as_str()).collect();
        let mut out = Vec::new();
        for (qid, grades) in &self.qrels {
            for (did, grade) in grades {
                if *grade > 0.0 && !ids.contains(did.as_str()) {
                    out.push((qid.as_str(), did.as_str()));
                }
            }
        }
        out.sort_unstable();
        out
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a corpus could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusError {
    /// A line was not JSON, or lacked `_id`/`text`. Carries the 1-based line
    /// number, because a 500 000-line file needs one.
    MalformedLine { line: usize, reason: String },
    /// A qrels row did not have three tab-separated columns.
    MalformedQrel { line: usize, reason: String },
    /// Two documents or two queries share an id.
    ///
    /// Rejected rather than last-wins: a duplicate silently changes which text is
    /// indexed, and the file still looks fine.
    DuplicateId(String),
    /// The file parsed and was empty.
    Empty(&'static str),
    /// A file could not be read from disk.
    Io { path: String, reason: String },
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedLine { line, reason } => write!(f, "line {line}: {reason}"),
            Self::MalformedQrel { line, reason } => write!(f, "qrels line {line}: {reason}"),
            Self::DuplicateId(id) => write!(
                f,
                "id {id:?} appears more than once; a duplicate would change which \
                 text is indexed and the file would still look fine"
            ),
            Self::Empty(what) => write!(f, "{what} parsed but contains no entries"),
            Self::Io { path, reason } => write!(f, "{path}: {reason}"),
        }
    }
}

impl std::error::Error for CorpusError {}

/// Why a run could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The retriever failed on a query.
    Retriever { query_id: String, reason: String },
    /// Not one returned id was a document of this corpus.
    ///
    /// Almost always an id-space mismatch — an adapter returning internal row ids
    /// instead of corpus ids — and the reason this is an error and not a score of
    /// zero. Carries one example so the mismatch is visible at a glance.
    IdSpaceMismatch {
        /// How many ids were returned in total.
        returned: usize,
        /// One of them, to compare against the corpus's own ids by eye.
        example_returned: String,
        /// One corpus id, for the same comparison.
        example_expected: String,
    },
    /// The corpus has no queries to run.
    NoQueries,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Retriever { query_id, reason } => {
                write!(f, "retriever failed on query {query_id:?}: {reason}")
            }
            Self::IdSpaceMismatch {
                returned,
                example_returned,
                example_expected,
            } => write!(
                f,
                "none of the {returned} returned ids is a document of this corpus \
                 (got {example_returned:?}, corpus ids look like {example_expected:?}). \
                 Every query would score zero and the report would not say why"
            ),
            Self::NoQueries => write!(f, "the corpus has no queries"),
        }
    }
}

impl std::error::Error for RunError {}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawDoc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Deserialize)]
struct RawQuery {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// Parse `corpus.jsonl`.
pub fn parse_corpus_jsonl(text: &str) -> Result<Vec<CorpusDoc>, CorpusError> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawDoc = serde_json::from_str(line).map_err(|e| CorpusError::MalformedLine {
            line: i + 1,
            reason: e.to_string(),
        })?;
        if !seen.insert(raw.id.clone()) {
            return Err(CorpusError::DuplicateId(raw.id));
        }
        out.push(CorpusDoc {
            id: raw.id,
            title: raw.title,
            text: raw.text,
        });
    }
    if out.is_empty() {
        return Err(CorpusError::Empty("corpus"));
    }
    Ok(out)
}

/// Parse `queries.jsonl`.
pub fn parse_queries_jsonl(text: &str) -> Result<Vec<CorpusQuery>, CorpusError> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawQuery = serde_json::from_str(line).map_err(|e| CorpusError::MalformedLine {
            line: i + 1,
            reason: e.to_string(),
        })?;
        if !seen.insert(raw.id.clone()) {
            return Err(CorpusError::DuplicateId(raw.id));
        }
        out.push(CorpusQuery {
            id: raw.id,
            text: raw.text,
        });
    }
    if out.is_empty() {
        return Err(CorpusError::Empty("queries"));
    }
    Ok(out)
}

/// Parse a qrels TSV into `query_id -> doc_id -> grade`.
///
/// The first row is skipped when it is BEIR's `query-id⇥corpus-id⇥score` header.
/// Detected by content rather than position: a file without a header would
/// otherwise lose its first judgement, which lowers recall by one document per
/// corpus and never looks like a bug.
pub fn parse_qrels_tsv(text: &str) -> Result<HashMap<String, HashMap<String, f64>>, CorpusError> {
    let mut out: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        if i == 0
            && cols
                .first()
                .is_some_and(|c| c.eq_ignore_ascii_case("query-id"))
        {
            continue;
        }
        if cols.len() < 3 {
            return Err(CorpusError::MalformedQrel {
                line: i + 1,
                reason: format!("expected 3 tab-separated columns, found {}", cols.len()),
            });
        }
        let grade: f64 = cols[2].parse().map_err(|_| CorpusError::MalformedQrel {
            line: i + 1,
            reason: format!("score {:?} is not a number", cols[2]),
        })?;
        out.entry(cols[0].to_string())
            .or_default()
            .insert(cols[1].to_string(), grade);
    }
    if out.is_empty() {
        return Err(CorpusError::Empty("qrels"));
    }
    Ok(out)
}

/// Load a BEIR-layout directory: `corpus.jsonl`, `queries.jsonl`, and the first
/// of `qrels/test.tsv`, `qrels/dev.tsv`, `qrels/train.tsv` that exists.
///
/// The split order is deliberate and worth knowing: `test` is the one papers
/// report, so a directory carrying both would otherwise be scored on whichever
/// the filesystem happened to list first.
pub fn load_beir_dir(dir: &Path) -> Result<RetrievalCorpus, CorpusError> {
    let read = |p: std::path::PathBuf| -> Result<String, CorpusError> {
        std::fs::read_to_string(&p).map_err(|e| CorpusError::Io {
            path: p.display().to_string(),
            reason: e.to_string(),
        })
    };
    let docs = parse_corpus_jsonl(&read(dir.join("corpus.jsonl"))?)?;
    let queries = parse_queries_jsonl(&read(dir.join("queries.jsonl"))?)?;

    let mut qrels_path = None;
    for split in ["test.tsv", "dev.tsv", "train.tsv"] {
        let p = dir.join("qrels").join(split);
        if p.is_file() {
            qrels_path = Some(p);
            break;
        }
    }
    let qrels_path = qrels_path.ok_or_else(|| CorpusError::Io {
        path: dir.join("qrels").display().to_string(),
        reason: "no test.tsv, dev.tsv or train.tsv".to_string(),
    })?;
    let qrels = parse_qrels_tsv(&read(qrels_path)?)?;

    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "corpus".to_string());
    Ok(RetrievalCorpus {
        name,
        docs,
        queries,
        qrels,
    })
}

// ---------------------------------------------------------------------------
// Running a retriever
// ---------------------------------------------------------------------------

/// Anything that turns a query into a ranked list of **corpus** document ids.
///
/// One method, and the whole contract is in the return type: *corpus* ids, best
/// first. An implementation that has its own internal ids must map them back
/// before returning — see the hazard in this module's header.
pub trait Retriever {
    /// Up to `k` document ids, best first. Fewer is fine; none is fine.
    fn retrieve(&self, query: &str, k: usize) -> Result<Vec<String>, String>;

    /// Name for the report, e.g. `"bm25-fts5"` or `"rrf(bm25,tfidf)"`.
    ///
    /// Include the configuration that would change the numbers. "hybrid" names
    /// four different retrievers depending on the weights.
    fn describe(&self) -> String {
        "unnamed retriever".to_string()
    }
}

/// A produced run, plus what went wrong quietly while producing it.
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusRun {
    /// The run, ready for [`crate::retrieval_metrics::summarise`].
    pub file: RunFile,
    /// What produced it.
    pub retriever: String,
    /// Returned ids that are not documents of this corpus.
    ///
    /// Non-zero means *some* of the retriever's output cannot be judged. Every
    /// one of those slots costs precision and can never earn recall, so a run
    /// with a high count is measuring a wiring problem.
    pub unknown_ids: usize,
    /// Queries for which the retriever returned nothing at all.
    pub empty_results: usize,
}

/// Run `retriever` over every query of `corpus` and build a scorable run.
///
/// Errors with [`RunError::IdSpaceMismatch`] when not one returned id belongs to
/// the corpus, rather than handing back a run that scores zero everywhere.
pub fn run_corpus(
    corpus: &RetrievalCorpus,
    retriever: &dyn Retriever,
    k: usize,
) -> Result<CorpusRun, RunError> {
    if corpus.queries.is_empty() {
        return Err(RunError::NoQueries);
    }
    let known: HashSet<&str> = corpus.docs.iter().map(|d| d.id.as_str()).collect();

    let mut queries = Vec::with_capacity(corpus.queries.len());
    let (mut returned, mut unknown_ids, mut empty_results) = (0usize, 0usize, 0usize);
    let mut example_returned: Option<String> = None;

    for q in &corpus.queries {
        let retrieved = retriever
            .retrieve(&q.text, k)
            .map_err(|reason| RunError::Retriever {
                query_id: q.id.clone(),
                reason,
            })?;
        if retrieved.is_empty() {
            empty_results += 1;
        }
        for id in &retrieved {
            returned += 1;
            if !known.contains(id.as_str()) {
                unknown_ids += 1;
                if example_returned.is_none() {
                    example_returned = Some(id.clone());
                }
            }
        }
        queries.push(QueryRun {
            query_id: q.id.clone(),
            retrieved,
            grades: corpus.qrels.get(&q.id).cloned().unwrap_or_default(),
        });
    }

    // Every single returned id unknown: the two id spaces are different. A
    // retriever can rank badly, but it cannot invent ids outside the collection
    // it indexed. Guarded only when something WAS returned -- a retriever that
    // found nothing has no ids to be wrong about, and that is `empty_results`.
    if returned > 0 && unknown_ids == returned {
        return Err(RunError::IdSpaceMismatch {
            returned,
            example_returned: example_returned.unwrap_or_default(),
            example_expected: corpus
                .docs
                .first()
                .map(|d| d.id.clone())
                .unwrap_or_default(),
        });
    }

    Ok(CorpusRun {
        file: RunFile { queries },
        retriever: retriever.describe(),
        unknown_ids,
        empty_results,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = r#"{"_id": "d1", "title": "Cats", "text": "Cats purr."}
{"_id": "d2", "title": "", "text": "Dogs bark."}
{"_id": "d3", "title": "Birds", "text": "Birds sing."}"#;

    const QUERIES: &str = r#"{"_id": "q1", "text": "purring animal"}
{"_id": "q2", "text": "barking animal"}"#;

    const QRELS: &str = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n";

    fn corpus() -> RetrievalCorpus {
        RetrievalCorpus {
            name: "toy".into(),
            docs: parse_corpus_jsonl(CORPUS).expect("corpus parses"),
            queries: parse_queries_jsonl(QUERIES).expect("queries parse"),
            qrels: parse_qrels_tsv(QRELS).expect("qrels parse"),
        }
    }

    /// Returns a fixed list per query, so a test can state exactly what the
    /// retriever did and assert on the consequence.
    struct Scripted(HashMap<String, Vec<String>>);

    impl Retriever for Scripted {
        fn retrieve(&self, query: &str, k: usize) -> Result<Vec<String>, String> {
            Ok(self
                .0
                .get(query)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(k)
                .collect())
        }
        fn describe(&self) -> String {
            "scripted".into()
        }
    }

    fn scripted(pairs: &[(&str, &[&str])]) -> Scripted {
        Scripted(
            pairs
                .iter()
                .map(|(q, ids)| {
                    (
                        (*q).to_string(),
                        ids.iter().map(|s| (*s).to_string()).collect(),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn parses_the_three_files() {
        let c = corpus();
        assert_eq!(c.docs.len(), 3);
        assert_eq!(c.queries.len(), 2);
        assert_eq!(c.qrels["q1"]["d1"], 1.0);
        // A missing title is empty, not absent: `#[serde(default)]`, since BEIR
        // omits the key entirely on some datasets.
        assert_eq!(c.docs[1].title, "");
    }

    #[test]
    fn the_qrels_header_is_detected_by_content_not_position() {
        // Without a header, the first judgement must survive. Skipping row 0 by
        // position would drop q1/d1 and lower recall by one document per corpus.
        let no_header = "q1\td1\t1\nq2\td2\t1\n";
        let parsed = parse_qrels_tsv(no_header).expect("parses");
        assert_eq!(parsed["q1"]["d1"], 1.0, "the first row must not be eaten");
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn indexable_text_joins_title_and_body_only_when_there_is_a_title() {
        let c = corpus();
        assert_eq!(c.docs[0].indexable_text(), "Cats\n\nCats purr.");
        assert_eq!(c.docs[1].indexable_text(), "Dogs bark.");
    }

    #[test]
    fn a_duplicate_id_is_refused() {
        let dup = "{\"_id\": \"d1\", \"text\": \"a\"}\n{\"_id\": \"d1\", \"text\": \"b\"}";
        assert_eq!(
            parse_corpus_jsonl(dup),
            Err(CorpusError::DuplicateId("d1".into()))
        );
    }

    #[test]
    fn a_malformed_line_names_its_line_number() {
        let bad = "{\"_id\": \"d1\", \"text\": \"a\"}\nnot json\n";
        match parse_corpus_jsonl(bad) {
            Err(CorpusError::MalformedLine { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected line 2 to be named, got {other:?}"),
        }
    }

    #[test]
    fn a_run_carries_the_grades_of_each_query() {
        let r = scripted(&[
            ("purring animal", &["d1", "d2"]),
            ("barking animal", &["d3", "d2"]),
        ]);
        let run = run_corpus(&corpus(), &r, 10).expect("runs");
        assert_eq!(run.file.queries.len(), 2);
        assert_eq!(run.file.queries[0].retrieved, vec!["d1", "d2"]);
        assert_eq!(run.file.queries[0].grades["d1"], 1.0);
        assert_eq!(run.unknown_ids, 0);
        assert_eq!(run.empty_results, 0);
        assert_eq!(run.retriever, "scripted");
    }

    #[test]
    fn the_run_scores_what_the_retriever_actually_did() {
        // The point of the whole module: the produced run must be scorable, and
        // the numbers must follow from the scripted ranking. q1's relevant doc is
        // at rank 1 and q2's at rank 2, so MRR = (1 + 1/2) / 2 = 0.75.
        let r = scripted(&[
            ("purring animal", &["d1", "d3"]),
            ("barking animal", &["d3", "d2"]),
        ]);
        let run = run_corpus(&corpus(), &r, 10).expect("runs");
        let report = crate::retrieval_metrics::summarise(&run.file.queries, 10);
        assert_eq!(report.scored, 2);
        assert_eq!(report.skipped, 0);
        assert!(
            (report.mrr - 0.75).abs() < 1e-9,
            "MRR should be (1 + 1/2)/2 = 0.75, got {}",
            report.mrr
        );
        assert!((report.recall_at_k - 1.0).abs() < 1e-9);
    }

    #[test]
    fn every_id_unknown_is_an_error_and_not_a_score_of_zero() {
        // The hazard: an adapter returning SQLite row ids instead of corpus ids.
        // Without this guard the report would read recall@10 = 0.000 and look
        // exactly like a retriever that works and finds nothing.
        let r = scripted(&[
            ("purring animal", &["17", "42"]),
            ("barking animal", &["99"]),
        ]);
        match run_corpus(&corpus(), &r, 10) {
            Err(RunError::IdSpaceMismatch {
                returned,
                example_returned,
                example_expected,
            }) => {
                assert_eq!(returned, 3);
                assert_eq!(example_returned, "17");
                assert_eq!(example_expected, "d1");
            }
            other => panic!("expected IdSpaceMismatch, got {other:?}"),
        }
    }

    #[test]
    fn some_ids_unknown_is_counted_and_not_refused() {
        // A partially wrong run is still a run: the mismatch guard must not fire
        // on it, or a retriever that returns one stale id becomes unmeasurable.
        let r = scripted(&[
            ("purring animal", &["d1", "ghost"]),
            ("barking animal", &["d2"]),
        ]);
        let run = run_corpus(&corpus(), &r, 10).expect("must not be refused");
        assert_eq!(run.unknown_ids, 1);
    }

    #[test]
    fn finding_nothing_is_not_an_id_mismatch() {
        // Zero returned ids means zero ids to be wrong about. Folding this into
        // the mismatch guard would turn "the retriever matched nothing" into
        // "your wiring is broken", which sends the reader to the wrong place.
        let r = scripted(&[]);
        let run = run_corpus(&corpus(), &r, 10).expect("an empty run is still a run");
        assert_eq!(run.empty_results, 2);
        assert_eq!(run.unknown_ids, 0);
    }

    #[test]
    fn k_is_honoured() {
        let r = scripted(&[("purring animal", &["d1", "d2", "d3"])]);
        let run = run_corpus(&corpus(), &r, 2).expect("runs");
        assert_eq!(run.file.queries[0].retrieved.len(), 2);
    }

    #[test]
    fn a_retriever_failure_names_the_query() {
        struct Broken;
        impl Retriever for Broken {
            fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<String>, String> {
                Err("index closed".into())
            }
        }
        match run_corpus(&corpus(), &Broken, 10) {
            Err(RunError::Retriever { query_id, reason }) => {
                assert_eq!(query_id, "q1");
                assert_eq!(reason, "index closed");
            }
            other => panic!("expected the failing query to be named, got {other:?}"),
        }
    }

    #[test]
    fn unjudged_queries_are_listed_before_anyone_reads_the_scores() {
        let mut c = corpus();
        c.queries.push(CorpusQuery {
            id: "q3".into(),
            text: "singing animal".into(),
        });
        assert_eq!(c.unjudged_queries(), vec!["q3"]);
        // And a judgement of grade 0 counts as unjudged-relevant, not as judged:
        // it caps nothing and must not make the query look scorable.
        c.qrels
            .insert("q3".into(), HashMap::from([("d3".to_string(), 0.0)]));
        assert_eq!(c.unjudged_queries(), vec!["q3"]);
    }

    #[test]
    fn qrels_pointing_outside_the_corpus_are_reported() {
        // A partial download: the judgements name a document the corpus file does
        // not contain, so recall is capped below 1.0 and nothing in the report
        // says why.
        let mut c = corpus();
        c.qrels
            .get_mut("q1")
            .expect("q1 exists")
            .insert("d_missing".into(), 1.0);
        assert_eq!(c.qrels_outside_corpus(), vec![("q1", "d_missing")]);
    }

    #[test]
    fn an_empty_corpus_is_refused_per_file() {
        assert_eq!(
            parse_corpus_jsonl("\n\n"),
            Err(CorpusError::Empty("corpus"))
        );
        assert_eq!(parse_queries_jsonl(""), Err(CorpusError::Empty("queries")));
        assert_eq!(parse_qrels_tsv(""), Err(CorpusError::Empty("qrels")));
        // A qrels file with only a header is empty too -- and this is the case a
        // `lines().count() > 0` check would have called fine.
        assert_eq!(
            parse_qrels_tsv("query-id\tcorpus-id\tscore\n"),
            Err(CorpusError::Empty("qrels"))
        );
    }

    #[test]
    fn a_non_numeric_grade_is_refused() {
        match parse_qrels_tsv("q1\td1\tvery\n") {
            Err(CorpusError::MalformedQrel { line, reason }) => {
                assert_eq!(line, 1);
                assert!(reason.contains("not a number"), "{reason}");
            }
            other => panic!("expected a malformed grade, got {other:?}"),
        }
    }
}
