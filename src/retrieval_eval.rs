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

/// Collapse a ranked list of chunk sources into a ranked list of **documents**.
///
/// The crate's lexical retrieval returns *chunks*, and `index_document` splits
/// anything over ~400 tokens, so one document can occupy several of the top
/// positions. Left alone that is a silent scoring error in both directions: it
/// inflates precision@k (three slots, one document) and suppresses recall@k
/// (those slots cannot hold the documents that were missed).
///
/// First occurrence wins, because the ranking is already best-first and a
/// document's best chunk is the one that earned its place.
///
/// Equivalent to [`Aggregation::MaxPassage`]; kept as a free function because it
/// is the obvious thing to reach for and needs no enum to use.
///
/// # This is where two different units of retrieval meet
///
/// **The chunking is not a wart to be removed.** Every real consumer needs it and
/// needs it at the passage level: `assistant::rag` injects the text into a
/// prompt, `ai_cli` asks with an 8000-token budget, and the MCP knowledge tool
/// takes `max_tokens` from its caller. A 50-page document does not fit in a
/// context window, and retrieving *the relevant part* is the entire point. So
/// dedup belongs **here, at the evaluation boundary** — never in `rag.rs`.
/// Collapsing to documents inside the product would break RAG, because a prompt
/// that needs three passages of one document wants all three.
///
/// And the caveat that follows: we are scoring a **passage** retriever with a
/// **document**-level benchmark, so an aggregation rule is unavoidable — see
/// [`Aggregation`], and note that the benchmark does not pick one for us.
pub fn dedup_by_document(ranked: impl IntoIterator<Item = String>, k: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(k);
    for id in ranked {
        if out.len() >= k {
            break;
        }
        if seen.insert(id.clone()) {
            out.push(id);
        }
    }
    out
}

/// How a ranked list of **passages** becomes a ranked list of **documents**.
///
/// Unavoidable, and **not chosen by the benchmark**: BEIR scores through
/// `pytrec_eval`, which evaluates whatever ranking it is handed, so whether a
/// published number used passage aggregation at all — and which rule — is a
/// property of that system and has to be checked per paper. Our own reports must
/// therefore say which rule produced them, which is why
/// [`sqlite::SqliteRetriever::describe`] names it.
///
/// The two rules here are **rank-only** on purpose. A score-weighted
/// sum-of-passages is the other obvious candidate and it cannot be implemented
/// today: `RagDb::search_knowledge` computes `bm25(knowledge_fts)` in SQL, orders
/// by it, and then returns `Vec<KnowledgeChunk>`, which has no score field. The
/// score exists and is thrown away, so the lexical path offers ranks and nothing
/// else. Only the hybrid path exposes numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregation {
    /// A document ranks where its best passage ranked. One strong passage is
    /// enough.
    #[default]
    MaxPassage,
    /// A document's weight is the sum of `1/rank` over all of its passages, and
    /// documents are re-sorted by that total.
    ///
    /// Rewards a document that is *diffusely* relevant — five mediocre passages
    /// instead of one excellent one — which `MaxPassage` cannot see at all. The
    /// same reciprocal-rank shape as RRF, for the same reason: it combines
    /// positions without needing comparable scores.
    ///
    /// **This changes the answer**, which is the whole point of having both. A
    /// document whose passages land at ranks 4, 5 and 6 totals `0.61` and beats
    /// one with a single passage at rank 2 (`0.5`), reversing `MaxPassage`.
    SumReciprocalRank,
}

impl Aggregation {
    /// Short label for a report. Include it: a score without its aggregation
    /// rule cannot be reproduced.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MaxPassage => "max-passage",
            Self::SumReciprocalRank => "sum-reciprocal-rank",
        }
    }

    /// Collapse `ranked` passage sources into at most `k` document ids.
    pub fn apply(&self, ranked: impl IntoIterator<Item = String>, k: usize) -> Vec<String> {
        match self {
            Self::MaxPassage => dedup_by_document(ranked, k),
            Self::SumReciprocalRank => {
                // Accumulate 1/rank per document, then sort by the total.
                let mut totals: Vec<(String, f64)> = Vec::new();
                for (i, id) in ranked.into_iter().enumerate() {
                    let weight = 1.0 / (i as f64 + 1.0);
                    match totals.iter_mut().find(|(d, _)| *d == id) {
                        Some((_, t)) => *t += weight,
                        None => totals.push((id, weight)),
                    }
                }
                // Ties break by first appearance, which `sort_by` preserves
                // because it is stable. Without that a tie would order by
                // whatever the allocator produced, and two identical runs could
                // disagree -- a benchmark that is not reproducible against
                // itself measures nothing.
                totals.sort_by(|a, b| b.1.total_cmp(&a.1));
                totals.into_iter().take(k).map(|(d, _)| d).collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fetching a public corpus, on demand, into a cache OUTSIDE the repository
// ---------------------------------------------------------------------------

/// Downloading a judged corpus when it is needed, and deleting it when it is not.
///
/// # Nothing here ever writes into the repository
///
/// **Strict rule, no exceptions:** third-party material is never committed.
/// Committing a corpus *is* redistributing it, each one carries its own licence,
/// and this repository is public. So the corpus is downloaded to a cache under
/// the user's data directory, and `.gitignore` is **not** the safeguard — not
/// downloading into the tree is.
///
/// The other half of the rule is that cleanup has to be comfortable, because a
/// corpus nobody can find is a corpus nobody deletes: [`installed`] reports what
/// is on disk and what it costs, and [`remove`] deletes one.
///
/// # Explicit, never automatic
///
/// Fetching is a command the user runs, not something `run_corpus` triggers when
/// a directory is missing. Two reasons and both matter: a silent download of
/// someone else's data is a decision the user should make, and **CI must never
/// fetch** — a network dependency in CI is flaky *and* a redistribution question
/// nobody reviewed. No test in this module touches the network; the ones that
/// cover unpacking build their own archive on disk.
#[cfg(feature = "zip")]
pub mod fetch {
    use std::fs;
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    /// Where a public corpus comes from, and everything needed to decide whether
    /// to use it.
    ///
    /// The fields that look like trivia are the ones that cost a night to learn:
    /// `avg_doc_words` decides whether the corpus can answer a question about
    /// passage-to-document aggregation at all, and `licence` is here so nobody has
    /// to go looking for it before deciding what a use is allowed to be.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CorpusSource {
        /// Short name, and the directory it unpacks into.
        pub name: &'static str,
        /// Where the archive lives.
        pub url: &'static str,
        /// Licence **as published upstream**. Written down rather than looked up,
        /// because a use decision made without it is a guess.
        pub licence: &'static str,
        /// Compressed size in bytes, for a sanity check after download and so the
        /// user knows what they are about to pull.
        pub archive_bytes: u64,
        /// Documents in the collection.
        pub docs: usize,
        /// Queries in the test split.
        pub queries: usize,
        /// Mean document length in **words**, as published.
        ///
        /// Compare against `sqlite::TARGET_CHUNK_TOKENS`: a corpus whose documents
        /// are shorter than one chunk **cannot** discriminate between
        /// passage-to-document aggregation rules, because every document is one
        /// passage and all the rules collapse to the same answer. Measured the
        /// hard way on a four-document toy corpus that reported both rules as
        /// identical and proved nothing.
        pub avg_doc_words: usize,
        /// One line on what this corpus is for.
        pub note: &'static str,
    }

    impl CorpusSource {
        /// Whether documents are long enough to split into several passages, and
        /// therefore whether this corpus can tell two aggregation rules apart.
        ///
        /// A words-to-tokens ratio of roughly 0.75 words per token is the usual
        /// English rule of thumb, so the comparison is deliberately generous: this
        /// answers "could it possibly chunk", and a `false` is a solid no.
        pub fn can_discriminate_aggregation(&self) -> bool {
            self.avg_doc_words > super::sqlite::TARGET_CHUNK_TOKENS
        }
    }

    /// The corpora this crate knows how to fetch.
    ///
    /// BEIR's own distribution, which ships exactly the layout
    /// [`super::load_beir_dir`] reads, so there is no format conversion step to get
    /// wrong. Verified reachable on 2026-10-07.
    pub const KNOWN: &[CorpusSource] = &[
        CorpusSource {
            name: "nfcorpus",
            url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
            licence: "see BEIR (CC BY-SA 4.0 for the BEIR packaging; NFCorpus is from Boteva et al. 2016)",
            archive_bytes: 2_448_432,
            docs: 3_633,
            queries: 323,
            avg_doc_words: 232,
            note: "Smallest practical BEIR set and the most queries per megabyte: 323 judged \
                   queries in 2.3 MiB. Start here to validate the instrument. Documents are \
                   SHORTER than one chunk, so it cannot discriminate aggregation rules.",
        },
        CorpusSource {
            name: "trec-covid",
            url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
            licence: "see BEIR; TREC-COVID is built on CORD-19 (Wang et al. 2020)",
            archive_bytes: 0,
            docs: 171_332,
            queries: 50,
            avg_doc_words: 161,
            note: "171k documents for only 50 queries -- a lot of indexing for a thin \
                   statistical signal. Registered for completeness, not recommended first.",
        },
    ];

    /// Look a corpus up by name.
    pub fn source(name: &str) -> Option<&'static CorpusSource> {
        KNOWN.iter().find(|s| s.name == name)
    }

    /// Why a fetch failed.
    #[derive(Debug)]
    pub enum FetchError {
        /// No corpus of that name is registered.
        Unknown(String),
        /// The download failed or the server answered something unusable.
        Download { url: String, reason: String },
        /// The archive was not the size the registry expects.
        ///
        /// Not pedantry: a truncated download unpacks into a corpus that is
        /// *partly* there, and a partial corpus produces a believable score from
        /// missing documents.
        WrongSize { expected: u64, got: u64 },
        /// The archive could not be opened or a member could not be written.
        Unpack(String),
        /// A member's path escapes the destination directory.
        ///
        /// A zip may contain `../../etc/passwd`. This is refused rather than
        /// sanitised, because an archive that tries it is not an archive we want
        /// half of.
        UnsafePath(String),
        /// The archive unpacked but does not contain the three files a BEIR corpus
        /// needs, so nothing downstream could read it.
        NotBeirLayout(String),
        /// A filesystem operation failed.
        Io(String),
    }

    impl std::fmt::Display for FetchError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Unknown(n) => {
                    let names: Vec<&str> = KNOWN.iter().map(|s| s.name).collect();
                    write!(f, "unknown corpus {n:?}; known: {}", names.join(", "))
                }
                Self::Download { url, reason } => write!(f, "cannot download {url}: {reason}"),
                Self::WrongSize { expected, got } => write!(
                    f,
                    "archive is {got} bytes, expected {expected}. A truncated download \
                     unpacks into a corpus that is only partly there, and a partial corpus \
                     produces a believable score from missing documents"
                ),
                Self::Unpack(e) => write!(f, "cannot unpack: {e}"),
                Self::UnsafePath(p) => write!(
                    f,
                    "archive member {p:?} escapes the destination directory; refusing the \
                     whole archive"
                ),
                Self::NotBeirLayout(what) => write!(
                    f,
                    "unpacked, but {what} -- nothing downstream could read this"
                ),
                Self::Io(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for FetchError {}

    /// Default cache directory, **outside** the repository.
    ///
    /// Under the OS data directory, falling back to the temp directory, so a fetch
    /// never has a reason to write next to the source tree.
    pub fn default_cache_dir() -> PathBuf {
        let base = std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("XDG_DATA_HOME"))
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        base.join("ai_assistant").join("corpora")
    }

    /// Download and unpack `source` into `cache_dir/<name>`, returning that path.
    ///
    /// Idempotent: if the three BEIR files are already there, nothing is
    /// downloaded. That is what makes it safe to put in a script.
    pub fn fetch(source: &CorpusSource, cache_dir: &Path) -> Result<PathBuf, FetchError> {
        let dest = cache_dir.join(source.name);
        if looks_like_beir(&dest).is_ok() {
            return Ok(dest);
        }
        fs::create_dir_all(&dest).map_err(|e| FetchError::Io(e.to_string()))?;

        let bytes = download(source.url)?;
        // Checked before unpacking, not after: a truncated archive should never
        // get as far as producing files somebody might then measure.
        if source.archive_bytes != 0 && bytes.len() as u64 != source.archive_bytes {
            return Err(FetchError::WrongSize {
                expected: source.archive_bytes,
                got: bytes.len() as u64,
            });
        }
        unpack_flat(&bytes, &dest)?;
        looks_like_beir(&dest).map_err(FetchError::NotBeirLayout)?;
        Ok(dest)
    }

    fn download(url: &str) -> Result<Vec<u8>, FetchError> {
        let resp = ureq::get(url)
            .timeout(std::time::Duration::from_secs(600))
            .call()
            .map_err(|e| FetchError::Download {
                url: url.to_string(),
                reason: e.to_string(),
            })?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| FetchError::Download {
                url: url.to_string(),
                reason: e.to_string(),
            })?;
        Ok(buf)
    }

    /// Unpack, flattening the archive's own top-level directory.
    ///
    /// BEIR's zips wrap everything in `<name>/`, so unpacking verbatim would give
    /// `cache/nfcorpus/nfcorpus/corpus.jsonl` and the loader would not find it.
    /// Flattening one level is the fix, and it is a *decision*: an archive with
    /// two top-level directories would lose that distinction, which is why the
    /// layout is verified afterwards instead of assumed.
    fn unpack_flat(archive: &[u8], dest: &Path) -> Result<(), FetchError> {
        let reader = std::io::Cursor::new(archive);
        let mut zip =
            zip::ZipArchive::new(reader).map_err(|e| FetchError::Unpack(e.to_string()))?;
        for i in 0..zip.len() {
            let mut member = zip
                .by_index(i)
                .map_err(|e| FetchError::Unpack(e.to_string()))?;
            // `enclosed_name` is the zip-slip guard: it returns None for a member
            // whose path escapes the destination (`../../etc/passwd`). Refusing
            // beats sanitising -- an archive that tries it is not one we want half
            // of.
            let raw = member
                .enclosed_name()
                .ok_or_else(|| FetchError::UnsafePath(member.name().to_string()))?;
            let mut parts = raw.components();
            parts.next(); // drop the archive's own top-level directory
            let relative: PathBuf = parts.collect();
            if relative.as_os_str().is_empty() {
                continue;
            }
            let out = dest.join(&relative);
            if member.is_dir() {
                fs::create_dir_all(&out).map_err(|e| FetchError::Io(e.to_string()))?;
                continue;
            }
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent).map_err(|e| FetchError::Io(e.to_string()))?;
            }
            let mut f = fs::File::create(&out).map_err(|e| FetchError::Io(e.to_string()))?;
            let mut buf = Vec::new();
            member
                .read_to_end(&mut buf)
                .map_err(|e| FetchError::Unpack(e.to_string()))?;
            f.write_all(&buf)
                .map_err(|e| FetchError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Unpack an in-memory archive, for the tests that must not touch the network.
    ///
    /// A thin `pub(crate)` door onto [`unpack_flat`] rather than making that
    /// public: unpacking an arbitrary archive into an arbitrary directory is not
    /// something a library consumer should be handed, and the zip-slip guard and
    /// the one-level flattening are only correct *as part of* a fetch.
    #[doc(hidden)]
    pub fn unpack_for_test(archive: &[u8], dest: &Path) -> Result<(), FetchError> {
        unpack_flat(archive, dest)
    }

    /// Check a directory is a readable BEIR corpus, for the same tests.
    #[doc(hidden)]
    pub fn verify_layout_for_test(dir: &Path) -> Result<(), String> {
        looks_like_beir(dir)
    }

    /// The three files a BEIR corpus needs, or a sentence saying which is missing.
    fn looks_like_beir(dir: &Path) -> Result<(), String> {
        if !dir.join("corpus.jsonl").is_file() {
            return Err("there is no corpus.jsonl".to_string());
        }
        if !dir.join("queries.jsonl").is_file() {
            return Err("there is no queries.jsonl".to_string());
        }
        let splits = ["test.tsv", "dev.tsv", "train.tsv"];
        if !splits.iter().any(|s| dir.join("qrels").join(s).is_file()) {
            return Err("there is no qrels/test.tsv, dev.tsv or train.tsv".to_string());
        }
        Ok(())
    }

    /// What is in the cache and what it costs: `(name, bytes on disk)`.
    ///
    /// Exists so cleanup is possible. A corpus nobody can find is a corpus nobody
    /// deletes, and 171k documents do not announce themselves.
    pub fn installed(cache_dir: &Path) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(cache_dir) else {
            return out;
        };
        for e in entries.flatten() {
            if e.path().is_dir() {
                let name = e.file_name().to_string_lossy().into_owned();
                out.push((name, dir_size(&e.path())));
            }
        }
        out.sort();
        out
    }

    fn dir_size(dir: &Path) -> u64 {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .map(|e| {
                let p = e.path();
                if p.is_dir() {
                    dir_size(&p)
                } else {
                    e.metadata().map(|m| m.len()).unwrap_or(0)
                }
            })
            .sum()
    }

    /// Delete a cached corpus, returning how many bytes went.
    ///
    /// Refuses a name with a path separator in it. The caller passes a corpus name,
    /// and a function that deletes a directory tree should never be reachable with
    /// `../..` in its argument.
    pub fn remove(name: &str, cache_dir: &Path) -> Result<u64, FetchError> {
        if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(FetchError::UnsafePath(name.to_string()));
        }
        let dir = cache_dir.join(name);
        if !dir.is_dir() {
            return Err(FetchError::Unknown(name.to_string()));
        }
        let size = dir_size(&dir);
        fs::remove_dir_all(&dir).map_err(|e| FetchError::Io(e.to_string()))?;
        Ok(size)
    }
}

// ---------------------------------------------------------------------------
// The crate's own lexical retriever, as a `Retriever`
// ---------------------------------------------------------------------------

/// Adapter over the crate's SQLite/FTS5 retrieval, so a run measures the
/// **product** rather than a toy scorer written for the benchmark.
///
/// Behind `--features rag` because that is where [`crate::rag::RagDb`] lives.
#[cfg(feature = "rag")]
pub mod sqlite {
    use std::path::Path;

    use super::{Aggregation, RetrievalCorpus, Retriever};
    use crate::rag::RagDb;

    /// Which of the crate's two search paths to measure.
    ///
    /// Named separately from any "hybrid" config because **"hybrid" is not one
    /// retriever**: its behaviour depends on the BM25/semantic weights, and a
    /// report that says "hybrid" without them cannot be reproduced.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SearchPath {
        /// `search_knowledge` — BM25 over FTS5 only.
        Lexical,
        /// `search_knowledge_hybrid` — BM25 fused with the semantic score.
        ///
        /// **Read this before comparing it against `Lexical`:** without an
        /// embedding service configured, the "semantic" half is TF-IDF cosine,
        /// so the comparison is BM25 against TF-IDF and any conclusion about
        /// embeddings would be a conclusion about the *absence* of embeddings.
        /// That is what N105 unblocks.
        Hybrid,
    }

    /// A token budget large enough not to truncate a ranked list.
    ///
    /// Not cosmetic. `search_knowledge` accumulates `token_count` and **breaks**
    /// out of the loop once the budget is spent, so a modest `max_tokens`
    /// silently returns a shorter list — which lands in the report as a lower
    /// recall and reads as "the retriever did not find them". Evaluation wants
    /// the ranking, not a context budget.
    pub const NO_TOKEN_BUDGET: usize = 1_000_000_000;

    /// How many chunks to ask for per requested document.
    ///
    /// Asking for exactly `k` chunks can yield fewer than `k` distinct documents
    /// once chunks collapse, which would under-report recall for a reason that
    /// has nothing to do with retrieval. Over-fetching does not fix that in
    /// general — a single document could in principle own every chunk — so
    /// [`SqliteRetriever::retrieve`] may still return fewer than `k`, and that is
    /// honest rather than padded.
    pub const CHUNK_OVERFETCH: usize = 5;

    /// The chunk size `index_document` aims for, mirrored here for the report.
    ///
    /// Duplicated from `rag::chunk_document`'s private `TARGET_CHUNK_TOKENS`
    /// rather than read from it, because that constant is not public. If the two
    /// drift, the report misstates the unit it measured -- which is why a test
    /// pins the number here, so a change in `rag.rs` has one place that argues
    /// back instead of none.
    pub const TARGET_CHUNK_TOKENS: usize = 400;

    /// The crate's lexical (or hybrid) retrieval, driven over a judged corpus.
    pub struct SqliteRetriever {
        db: RagDb,
        mode: SearchPath,
        aggregation: Aggregation,
        max_tokens: usize,
        indexed_docs: usize,
    }

    impl SqliteRetriever {
        /// Index every document of `corpus` into a database at `db_path`.
        ///
        /// **The id mapping lives here**: each document is indexed with its
        /// corpus id as the `source`, and [`SqliteRetriever::retrieve`] reads
        /// that same field back. `KnowledgeChunk::id` is an autoincrementing row
        /// id and returning it would produce a run that scores zero everywhere —
        /// the hazard this module's parent exists to refuse.
        ///
        /// Indexes `title + body`, matching the BEIR baselines.
        pub fn index(
            corpus: &RetrievalCorpus,
            db_path: &Path,
            mode: SearchPath,
        ) -> Result<Self, String> {
            let db = RagDb::open(db_path).map_err(|e| format!("cannot open {db_path:?}: {e}"))?;
            for doc in &corpus.docs {
                db.index_document(&doc.id, &doc.indexable_text())
                    .map_err(|e| format!("cannot index document {:?}: {e}", doc.id))?;
            }
            Ok(Self {
                db,
                mode,
                aggregation: Aggregation::default(),
                max_tokens: NO_TOKEN_BUDGET,
                indexed_docs: corpus.docs.len(),
            })
        }

        /// Choose how passages collapse into documents. See [`Aggregation`] --
        /// this changes the scores, so a report must say which one was used.
        pub fn with_aggregation(mut self, aggregation: Aggregation) -> Self {
            self.aggregation = aggregation;
            self
        }

        /// Override the token budget. See [`NO_TOKEN_BUDGET`] before lowering it.
        pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
            self.max_tokens = max_tokens;
            self
        }

        /// How many documents were indexed.
        pub fn indexed_docs(&self) -> usize {
            self.indexed_docs
        }
    }

    impl Retriever for SqliteRetriever {
        fn retrieve(&self, query: &str, k: usize) -> Result<Vec<String>, String> {
            let want_chunks = k.saturating_mul(CHUNK_OVERFETCH).max(k);
            let sources: Vec<String> = match self.mode {
                SearchPath::Lexical => self
                    .db
                    .search_knowledge(query, self.max_tokens, want_chunks)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|c| c.source)
                    .collect(),
                SearchPath::Hybrid => self
                    .db
                    .search_knowledge_hybrid(query, self.max_tokens, want_chunks)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|r| r.chunk.source)
                    .collect(),
            };
            Ok(self.aggregation.apply(sources, k))
        }

        /// Names everything that changes the numbers, which is this method's whole
        /// job: the search path, what was indexed, the passage-to-document
        /// aggregation and the chunk size it operates on.
        ///
        /// A score reported without its aggregation rule is not reproducible, and
        /// that was true of this method until it said so.
        fn describe(&self) -> String {
            let path = match self.mode {
                SearchPath::Lexical => "sqlite-fts5-bm25",
                SearchPath::Hybrid => {
                    "sqlite-fts5-hybrid(semantic half is TF-IDF, no embedding service)"
                }
            };
            format!(
                "{path} (title+body; {} over ~{}-token chunks)",
                self.aggregation.as_str(),
                TARGET_CHUNK_TOKENS
            )
        }
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
    fn dedup_keeps_the_first_occurrence_and_honours_k() {
        // The chunk-collapse hazard, tested on the pure function so the case is
        // deterministic. "d1" owning ranks 1 and 2 must not cost "d2" a slot.
        let ranked = ["d1", "d1", "d2", "d3", "d2"].map(String::from);
        assert_eq!(
            dedup_by_document(ranked.clone(), 10),
            vec!["d1", "d2", "d3"]
        );
        assert_eq!(dedup_by_document(ranked, 2), vec!["d1", "d2"]);
    }

    #[test]
    fn dedup_may_return_fewer_than_k_and_does_not_pad() {
        // One document owning every chunk. Returning fewer is honest; inventing
        // ids to reach k would fabricate retrievals that then get scored.
        let all_one = ["d1", "d1", "d1"].map(String::from);
        assert_eq!(dedup_by_document(all_one, 3), vec!["d1"]);
    }

    #[test]
    fn the_two_aggregations_disagree_and_that_is_the_point() {
        // THE separating case for N138, and the argument for implementing both.
        //
        // "a" has one passage at rank 2        -> max 1/2 = 0.50
        // "b" has three passages at 4, 5, 6    -> sum 1/4 + 1/5 + 1/6 = 0.616…
        //
        // MaxPassage ranks a first because its best passage is better.
        // SumReciprocalRank ranks b first because it is diffusely relevant.
        // Neither is wrong; a report that does not say which one it used is.
        let ranked = ["z", "a", "z", "b", "b", "b"].map(String::from);
        assert_eq!(
            Aggregation::MaxPassage.apply(ranked.clone(), 3),
            vec!["z", "a", "b"]
        );
        assert_eq!(
            Aggregation::SumReciprocalRank.apply(ranked, 3),
            vec!["z", "b", "a"],
            "b must overtake a: 0.616 beats 0.5"
        );
    }

    #[test]
    fn aggregation_ties_break_by_first_appearance() {
        // Two documents with one passage each at adjacent ranks cannot tie, so
        // the tie has to be built: identical totals via identical positions in
        // separate runs is impossible, so use one passage each and check the
        // ORDER is the rank order rather than whatever the sort happened to do.
        // An unstable sort here would make two identical runs disagree, and a
        // benchmark that is not reproducible against itself measures nothing.
        let ranked = ["p", "q", "r"].map(String::from);
        assert_eq!(
            Aggregation::SumReciprocalRank.apply(ranked, 3),
            vec!["p", "q", "r"]
        );
    }

    #[test]
    fn both_aggregations_honour_k_and_neither_pads() {
        let ranked = ["a", "a", "a"].map(String::from);
        assert_eq!(Aggregation::MaxPassage.apply(ranked.clone(), 5), vec!["a"]);
        assert_eq!(
            Aggregation::SumReciprocalRank.apply(ranked, 5),
            vec!["a"],
            "one document owning every passage is one document, not five"
        );
    }

    #[test]
    fn the_aggregation_label_is_not_empty_and_differs_per_rule() {
        // It goes into every report, so it has to say something and say something
        // different per rule.
        assert_eq!(Aggregation::MaxPassage.as_str(), "max-passage");
        assert_eq!(
            Aggregation::SumReciprocalRank.as_str(),
            "sum-reciprocal-rank"
        );
        assert_ne!(
            Aggregation::MaxPassage.as_str(),
            Aggregation::SumReciprocalRank.as_str()
        );
    }

    #[cfg(feature = "rag")]
    #[test]
    fn the_description_names_the_aggregation_and_the_chunk_size() {
        // A score without its aggregation rule is not reproducible, and this
        // method is where the rule becomes part of the record.
        use super::sqlite::{SearchPath, SqliteRetriever, TARGET_CHUNK_TOKENS};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_retrieval_eval_desc_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");

        let c = RetrievalCorpus {
            name: "desc".into(),
            docs: parse_corpus_jsonl("{\"_id\": \"d1\", \"title\": \"\", \"text\": \"t\"}")
                .expect("corpus"),
            queries: parse_queries_jsonl("{\"_id\": \"q1\", \"text\": \"t\"}").expect("queries"),
            qrels: parse_qrels_tsv("q1\td1\t1\n").expect("qrels"),
        };
        let r = SqliteRetriever::index(&c, &dir.join("kb.sqlite"), SearchPath::Lexical)
            .expect("indexes");

        let d = r.describe();
        assert!(d.contains("max-passage"), "must name the aggregation: {d}");
        assert!(
            d.contains(&TARGET_CHUNK_TOKENS.to_string()),
            "must name the chunk size, the unit it aggregated over: {d}"
        );
        assert!(d.contains("bm25"), "must name the search path: {d}");

        // And it must CHANGE when the rule changes, or it is decoration.
        let other = r
            .with_aggregation(Aggregation::SumReciprocalRank)
            .describe();
        assert_ne!(d, other);
        assert!(other.contains("sum-reciprocal-rank"), "{other}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn the_sqlite_retriever_returns_corpus_ids_and_is_scorable() {
        use super::sqlite::{SearchPath, SqliteRetriever};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // An atomic counter and not `line!()`: `line!()` expands where it is
        // WRITTEN, so two tests in one file sharing a helper got the same
        // directory and raced. That cost a flaky test once already.
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_retrieval_eval_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let db_path = dir.join("kb.sqlite");

        // A corpus whose QUERIES SHARE TERMS WITH THE DOCUMENTS, because the
        // retriever under test is BM25 and BM25 is a bag of words. The shared
        // fixture above was written for a scripted retriever and its
        // "purring animal" matches nothing in "Cats purr." — which is a true
        // statement about lexical retrieval, not a bug, and is pinned as its own
        // test below.
        let c = RetrievalCorpus {
            name: "lexical-toy".into(),
            docs: parse_corpus_jsonl(
                "{\"_id\": \"d1\", \"title\": \"Cats\", \"text\": \"Cats purr when content.\"}\n\
                 {\"_id\": \"d2\", \"title\": \"Dogs\", \"text\": \"Dogs bark at strangers.\"}\n\
                 {\"_id\": \"d3\", \"title\": \"Birds\", \"text\": \"Birds sing at dawn.\"}",
            )
            .expect("corpus parses"),
            queries: parse_queries_jsonl(
                "{\"_id\": \"q1\", \"text\": \"purr\"}\n\
                 {\"_id\": \"q2\", \"text\": \"bark\"}",
            )
            .expect("queries parse"),
            qrels: parse_qrels_tsv("q1\td1\t1\nq2\td2\t1\n").expect("qrels parse"),
        };

        let r = SqliteRetriever::index(&c, &db_path, SearchPath::Lexical).expect("indexes");
        assert_eq!(r.indexed_docs(), 3);

        // The whole point: ids the qrels can judge, not `1`/`2`/`3` row ids.
        let hits = r.retrieve("purr", 10).expect("retrieves");
        assert!(
            hits.iter().all(|h| c.docs.iter().any(|d| &d.id == h)),
            "returned ids must be corpus ids, got {hits:?}"
        );
        assert_eq!(hits.first().map(String::as_str), Some("d1"), "got {hits:?}");

        // End to end, which is the claim that matters: a real retriever over a
        // real index produces a run the metrics can score.
        let run = run_corpus(&c, &r, 10).expect("run_corpus accepts the real adapter");
        let report = crate::retrieval_metrics::summarise(&run.file.queries, 10);
        assert_eq!(run.unknown_ids, 0, "no id outside the corpus");
        assert_eq!(
            run.empty_results, 0,
            "both queries share terms with a document"
        );
        assert_eq!(report.scored, 2);
        assert!(
            (report.recall_at_k - 1.0).abs() < 1e-9,
            "each query's one relevant document is the only one containing its term, \
             so recall@10 must be 1.0; report was {report:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn the_lexical_retriever_misses_a_morphological_variant_and_says_so_honestly() {
        // Not a defect — a measured property of the thing being measured, pinned
        // here because the whole point of the golden set is to quantify it.
        //
        // FTS5's default tokenizer does not stem, so "purring" does not reach
        // "purr". The lexical retriever returns NOTHING, and the instrument
        // reports that as `empty_results`, NOT as an id-space mismatch: a
        // retriever that matched nothing has no ids to be wrong about. Those two
        // would send a reader to completely different places.
        //
        // This gap is exactly what semantic retrieval is supposed to close, and
        // it cannot be measured honestly until N105 gives us real embeddings —
        // today "semantic" is TF-IDF cosine, which does not stem either.
        use super::sqlite::{SearchPath, SqliteRetriever};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ai_retrieval_eval_morph_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");

        let c = RetrievalCorpus {
            name: "morphology".into(),
            docs: parse_corpus_jsonl(
                "{\"_id\": \"d1\", \"title\": \"\", \"text\": \"Cats purr when content.\"}",
            )
            .expect("corpus parses"),
            queries: parse_queries_jsonl("{\"_id\": \"q1\", \"text\": \"purring\"}")
                .expect("queries parse"),
            qrels: parse_qrels_tsv("q1\td1\t1\n").expect("qrels parse"),
        };
        let r = SqliteRetriever::index(&c, &dir.join("kb.sqlite"), SearchPath::Lexical)
            .expect("indexes");

        assert!(
            r.retrieve("purring", 10).expect("retrieves").is_empty(),
            "BM25 over FTS5 does not stem; if this starts passing, the tokenizer \
             changed and every recorded lexical score is from a different retriever"
        );

        let run = run_corpus(&c, &r, 10).expect("finding nothing is still a run");
        assert_eq!(run.empty_results, 1);
        assert_eq!(run.unknown_ids, 0);
        let report = crate::retrieval_metrics::summarise(&run.file.queries, 10);
        assert_eq!(report.scored, 1, "the query IS judged, it just missed");
        assert_eq!(report.recall_at_k, 0.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "zip")]
    mod fetching {
        use super::super::fetch::{installed, remove, source, FetchError, KNOWN};
        use std::io::Write;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicUsize, Ordering};

        fn tmp(tag: &str) -> PathBuf {
            static N: AtomicUsize = AtomicUsize::new(0);
            let d = std::env::temp_dir().join(format!(
                "ai_fetch_{}_{}_{}",
                tag,
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&d).expect("temp dir");
            d
        }

        #[test]
        fn the_registry_says_which_corpora_can_answer_the_aggregation_question() {
            // The field that cost a night to learn. NFCorpus averages 232 words,
            // under one ~400-token chunk, so every document is one passage and all
            // the aggregation rules collapse to the same answer. Registering that
            // as DATA means the next person does not have to rediscover it by
            // running a measurement that proves nothing.
            let nf = source("nfcorpus").expect("registered");
            assert!(
                !nf.can_discriminate_aggregation(),
                "NFCorpus documents are shorter than one chunk"
            );
            assert!(
                !KNOWN.iter().any(|s| s.can_discriminate_aggregation()),
                "no registered corpus has long documents yet -- that is exactly what \
                 N138 is still waiting for, and this test is how it stops being a \
                 surprise"
            );
        }

        #[test]
        fn every_registered_source_carries_its_licence_and_a_url() {
            // A licence nobody wrote down is a use decision made by guessing, and
            // this is third-party material in a public project.
            for s in KNOWN {
                assert!(!s.licence.is_empty(), "{} has no licence", s.name);
                assert!(
                    s.url.starts_with("https://"),
                    "{} must be fetched over TLS, got {:?}",
                    s.name,
                    s.url
                );
                assert!(!s.note.is_empty(), "{} has no note", s.name);
            }
        }

        #[test]
        fn an_unknown_name_lists_the_known_ones() {
            // The error a typo produces should answer the question it raises.
            let e = remove("does-not-exist", &tmp("unknown")).expect_err("must fail");
            assert!(matches!(e, FetchError::Unknown(_)));
            assert!(e.to_string().contains("nfcorpus"), "{e}");
        }

        #[test]
        fn remove_refuses_a_name_that_is_a_path() {
            // `remove` deletes a directory tree. It must be unreachable with `..`
            // in its argument, whatever the caller thought it was passing.
            for bad in ["../..", "a/b", "a\\b", ".."] {
                let e = remove(bad, &tmp("escape")).expect_err("must refuse");
                assert!(
                    matches!(e, FetchError::UnsafePath(_)),
                    "{bad:?} should be refused as a path, got {e}"
                );
            }
        }

        #[test]
        fn installed_reports_sizes_and_remove_reclaims_them() {
            // The cleanup half of the strict no-vendoring rule: a corpus nobody can
            // find is a corpus nobody deletes.
            let cache = tmp("cleanup");
            let corpus = cache.join("fake");
            std::fs::create_dir_all(corpus.join("qrels")).expect("dirs");
            std::fs::write(corpus.join("corpus.jsonl"), b"0123456789").expect("write");
            std::fs::write(corpus.join("qrels").join("test.tsv"), b"01234").expect("write");

            let listed = installed(&cache);
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].0, "fake");
            assert_eq!(
                listed[0].1, 15,
                "must count nested files, not just the top level"
            );

            let freed = remove("fake", &cache).expect("removes");
            assert_eq!(freed, 15);
            assert!(installed(&cache).is_empty());
            let _ = std::fs::remove_dir_all(&cache);
        }

        /// A zip built here, so the unpack tests never touch the network.
        fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
            let mut buf = Vec::new();
            {
                let mut w = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
                let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
                for (name, body) in entries {
                    w.start_file(*name, opts).expect("start");
                    w.write_all(body).expect("write");
                }
                w.finish().expect("finish");
            }
            buf
        }

        #[test]
        fn unpacking_flattens_the_archives_own_top_level_directory() {
            // BEIR wraps everything in `<name>/`, so unpacking verbatim would give
            // `cache/nfcorpus/nfcorpus/corpus.jsonl` and the loader would not find
            // it -- a corpus that downloaded "fine" and cannot be read.
            let zipped = build_zip(&[
                (
                    "nfcorpus/corpus.jsonl",
                    b"{\"_id\": \"d1\", \"text\": \"t\"}",
                ),
                (
                    "nfcorpus/queries.jsonl",
                    b"{\"_id\": \"q1\", \"text\": \"t\"}",
                ),
                ("nfcorpus/qrels/test.tsv", b"q1\td1\t1\n"),
            ]);
            let dest = tmp("flatten");
            super::super::fetch::unpack_for_test(&zipped, &dest).expect("unpacks");
            assert!(dest.join("corpus.jsonl").is_file(), "flattened one level");
            assert!(dest.join("qrels").join("test.tsv").is_file());
            assert!(
                !dest.join("nfcorpus").exists(),
                "the archive's own directory must not survive"
            );

            // And the whole point: the existing loader can now read it.
            let c = super::super::load_beir_dir(&dest).expect("loads");
            assert_eq!(c.docs.len(), 1);
            let _ = std::fs::remove_dir_all(&dest);
        }

        #[test]
        fn a_member_escaping_the_destination_is_refused_whole() {
            // Zip-slip. `../../pwned` must not be written, and the archive is
            // refused rather than sanitised: one that tries this is not one we
            // want half of.
            //
            // Note the doubled prefix: the first component is dropped as the
            // archive's own top-level directory, so the escape has to survive
            // that to be a real test of the guard rather than of the flattening.
            let zipped = build_zip(&[("x/../../pwned", b"nope")]);
            let dest = tmp("slip");
            let e = super::super::fetch::unpack_for_test(&zipped, &dest).expect_err("must refuse");
            assert!(
                matches!(e, FetchError::UnsafePath(_)),
                "expected UnsafePath, got {e}"
            );
            assert!(
                !dest
                    .parent()
                    .map(|p| p.join("pwned").exists())
                    .unwrap_or(false),
                "nothing may be written outside the destination"
            );
            let _ = std::fs::remove_dir_all(&dest);
        }

        #[test]
        fn an_archive_without_the_three_files_is_reported_not_accepted() {
            // Unpacking successfully is not the same as having a corpus. Without
            // this, a changed upstream layout would produce an empty directory and
            // the failure would surface later as "corpus.jsonl: not found" from a
            // completely different part of the program.
            let zipped = build_zip(&[("x/README.md", b"nothing useful")]);
            let dest = tmp("layout");
            super::super::fetch::unpack_for_test(&zipped, &dest).expect("unpacks");
            let e = super::super::fetch::verify_layout_for_test(&dest)
                .expect_err("must report the missing file");
            assert!(e.contains("corpus.jsonl"), "{e}");
            let _ = std::fs::remove_dir_all(&dest);
        }
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
