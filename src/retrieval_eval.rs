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

/// Which BEIR judgement file to score against.
///
/// A BEIR directory ships **one `queries.jsonl` holding every split's queries**
/// and a separate `qrels/<split>.tsv` per split. NFCorpus, for instance, has
/// 3,237 queries in that one file and 323 judged by `test`. Picking the split
/// therefore picks the query set too — see [`load_beir_dir_split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// What published numbers report, and the default.
    Test,
    /// Development / validation.
    Dev,
    /// Training. Scoring on it is a different claim — say so if you report it.
    Train,
}

impl Split {
    /// The file name inside `qrels/`.
    pub fn file_name(self) -> &'static str {
        match self {
            Split::Test => "test.tsv",
            Split::Dev => "dev.tsv",
            Split::Train => "train.tsv",
        }
    }

    /// The name as written on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Test => "test",
            Split::Dev => "dev",
            Split::Train => "train",
        }
    }
}

impl std::str::FromStr for Split {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "test" => Ok(Split::Test),
            "dev" | "validation" | "val" => Ok(Split::Dev),
            "train" | "training" => Ok(Split::Train),
            other => Err(format!(
                "unknown split '{other}' -- expected test, dev or train"
            )),
        }
    }
}

/// Load a BEIR-layout directory, scoring against the first of
/// `qrels/test.tsv`, `qrels/dev.tsv`, `qrels/train.tsv` that exists.
///
/// The order is deliberate: `test` is the one papers report, so a directory
/// carrying both would otherwise be scored on whichever the filesystem listed
/// first. Use [`load_beir_dir_split`] to say which one explicitly.
pub fn load_beir_dir(dir: &Path) -> Result<RetrievalCorpus, CorpusError> {
    load_beir_dir_split(dir, None)
}

/// Load a BEIR-layout directory, scoring against a named split.
///
/// `None` keeps [`load_beir_dir`]'s behaviour of taking the first split present.
///
/// **The queries are narrowed to the ones the chosen split judges.** This is not
/// an optimisation bolted on afterwards: `queries.jsonl` holds every split's
/// queries, so running the whole file means retrieving for queries that no
/// judgement covers, and [`crate::retrieval_metrics::summarise`] drops them from
/// every average anyway (counting them in `skipped`). On NFCorpus that is 3,237
/// retrievals to score 323 — a tenfold cost for an identical number.
///
/// It belongs here rather than in [`run_corpus`] because this is where the BEIR
/// quirk lives. `run_corpus` runs the queries it is handed; a caller that builds
/// a [`RetrievalCorpus`] by hand may well mean to run unjudged ones.
pub fn load_beir_dir_split(
    dir: &Path,
    split: Option<Split>,
) -> Result<RetrievalCorpus, CorpusError> {
    let read = |p: std::path::PathBuf| -> Result<String, CorpusError> {
        std::fs::read_to_string(&p).map_err(|e| CorpusError::Io {
            path: p.display().to_string(),
            reason: e.to_string(),
        })
    };
    let docs = parse_corpus_jsonl(&read(dir.join("corpus.jsonl"))?)?;
    let all_queries = parse_queries_jsonl(&read(dir.join("queries.jsonl"))?)?;

    let qrels_path = match split {
        // Asked for by name: it is there or this is an error. Falling back to
        // another split would answer a different question under the same name.
        Some(s) => {
            let p = dir.join("qrels").join(s.file_name());
            if !p.is_file() {
                return Err(CorpusError::Io {
                    path: p.display().to_string(),
                    reason: format!("the '{}' split is not in this corpus", s.as_str()),
                });
            }
            p
        }
        None => [Split::Test, Split::Dev, Split::Train]
            .iter()
            .map(|s| dir.join("qrels").join(s.file_name()))
            .find(|p| p.is_file())
            .ok_or_else(|| CorpusError::Io {
                path: dir.join("qrels").display().to_string(),
                reason: "no test.tsv, dev.tsv or train.tsv".to_string(),
            })?,
    };
    let qrels = parse_qrels_tsv(&read(qrels_path)?)?;

    // Keep only what this split judges. `qrels` may name a query with every
    // grade 0 -- judged irrelevant is a judgement, and `summarise` is the one
    // that decides such a query cannot be scored, not this function.
    let queries: Vec<CorpusQuery> = all_queries
        .into_iter()
        .filter(|q| qrels.contains_key(&q.id))
        .collect();
    if queries.is_empty() {
        return Err(CorpusError::Empty("queries"));
    }

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
/// corpus nobody can find is a corpus nobody deletes: [`fetch::installed`]
/// reports what is on disk and what it costs, and [`fetch::remove`] deletes one.
///
/// Qualified with the module name on purpose. This is an **outer** doc comment on
/// `pub mod fetch`, so rustdoc resolves its links in the PARENT scope — a bare
/// `[`installed`]` dangles and renders as dead text. Same family as the trap
/// recorded in V352, where four lines of courtesy on a `pub mod x;` broke eight
/// links inside that module at once.
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

    /// How a registered corpus is packaged upstream.
    ///
    /// Added when MLDR turned out not to ship a BEIR zip. It is an enum rather
    /// than a second fetch function so that `KNOWN` stays one list: a corpus
    /// you cannot find in the registry is a corpus nobody will use.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Layout {
        /// One zip already containing `corpus.jsonl`, `queries.jsonl` and
        /// `qrels/<split>.tsv`. BEIR's own packaging.
        BeirZip,
        /// MLDR: **three** separately-downloaded files, two of them gzipped, with
        /// different field names and a four-column TREC qrels file.
        ///
        /// Converted to BEIR layout on the way in, because the alternative is a
        /// second reader for every consumer downstream.
        MldrGzip {
            /// Language code, e.g. `es`. Picks the directory upstream.
            lang: &'static str,
        },
    }

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
        /// How it is packaged upstream.
        pub layout: Layout,
        /// Where the archive lives. For [`Layout::MldrGzip`] this is the
        /// repository root the three files hang off, not a single archive.
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
            layout: Layout::BeirZip,
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
            layout: Layout::BeirZip,
            url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
            licence: "see BEIR; TREC-COVID is built on CORD-19 (Wang et al. 2020)",
            archive_bytes: 0,
            docs: 171_332,
            queries: 50,
            avg_doc_words: 161,
            note: "171k documents for only 50 queries -- a lot of indexing for a thin \
                   statistical signal. Registered for completeness, not recommended first.",
        },
        CorpusSource {
            name: "mldr-es",
            layout: Layout::MldrGzip { lang: "es" },
            url: "https://huggingface.co/datasets/Shitao/MLDR/resolve/main",
            licence: "MIT (declared on the dataset card). Built from Wikipedia, mC4 and \
                      Pile by Chen et al. (BGE-M3, 2024)",
            // The corpus file only. The queries (8 MiB) and the qrels (5 KiB) are
            // not size-checked: a truncated 5 KiB TSV produces fewer judgements,
            // which `corpus` reports as unjudged queries, where a truncated
            // corpus silently removes documents a query needed.
            // Measured from the server, not estimated. The first value written
            // here was a mistyped 127_040_171 and the guard refused the real
            // file -- which is the guard working: it cannot tell a truncated
            // download from a wrong constant, and both deserve to stop.
            archive_bytes: 126_690_465,
            // Counted after downloading, not taken from the paper: the figure
            // first written here was 200,000, which is MLDR's total across all
            // thirteen languages and not what the Spanish file contains.
            docs: 9_551,
            queries: 200,
            // Mean over ALL 9,551 documents. Two corrections live in this one
            // number: the published ~4,737 tokens converts to ~3,550 words,
            // which is low; and a 300-document sample said 2,084, which is less
            // than half the truth, because the file is not shuffled. Measure the
            // whole thing or do not quote a mean.
            //
            // Median 5,597, p10 948, p90 10,485, and **9,550 of 9,551 documents
            // are longer than one ~300-word chunk** -- which is the entire point
            // of this entry. See `can_discriminate_aggregation`.
            avg_doc_words: 5_991,
            note: "The only registered corpus with documents LONG enough to split into \
                   several passages (23 chunks each), so the only one that can tell two \
                   aggregation rules apart (N138). Spanish, 121 MiB -- one of MLDR's \
                   smaller languages (English is 939 MiB). Not a BEIR zip: three gzipped \
                   files with other field names, converted on the way in. \
                   KNOWN-ITEM, NOT AD-HOC: exactly one relevant document per query and \
                   binary grades, so MAP equals MRR by construction and the natural \
                   metric is MRR. And the queries were GENERATED BY GPT-3.5 from the \
                   documents' own paragraphs, so they share vocabulary with the answer by \
                   construction -- a high lexical score here measures the task, not the \
                   retriever. Use it for the aggregation question; use NFCorpus (many \
                   graded relevant documents, real queries) to judge retrieval quality.",
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

        match source.layout {
            Layout::BeirZip => {
                let bytes = download(source.url)?;
                // Checked before unpacking, not after: a truncated archive should
                // never get as far as producing files somebody might then measure.
                if source.archive_bytes != 0 && bytes.len() as u64 != source.archive_bytes {
                    return Err(FetchError::WrongSize {
                        expected: source.archive_bytes,
                        got: bytes.len() as u64,
                    });
                }
                unpack_flat(&bytes, &dest)?;
            }
            Layout::MldrGzip { lang } => fetch_mldr(source, lang, &dest)?,
        }
        looks_like_beir(&dest).map_err(FetchError::NotBeirLayout)?;
        Ok(dest)
    }

    /// Download MLDR's three files and write them out as a BEIR directory.
    ///
    /// Converting here rather than teaching every consumer a second format: the
    /// loader, the CLI and the tests all speak BEIR, and one more dialect is one
    /// more thing that can be read slightly wrong somewhere.
    fn fetch_mldr(source: &CorpusSource, lang: &str, dest: &Path) -> Result<(), FetchError> {
        let base = source.url.trim_end_matches('/');

        let corpus_gz = download(&format!("{base}/mldr-v1.0-{lang}/corpus.jsonl.gz"))?;
        if source.archive_bytes != 0 && corpus_gz.len() as u64 != source.archive_bytes {
            return Err(FetchError::WrongSize {
                expected: source.archive_bytes,
                got: corpus_gz.len() as u64,
            });
        }
        let queries_gz = download(&format!("{base}/mldr-v1.0-{lang}/test.jsonl.gz"))?;
        let qrels = download(&format!("{base}/qrels/qrels.mldr-v1.0-{lang}-test.tsv"))?;

        let corpus = gunzip(&corpus_gz)?;
        let queries = gunzip(&queries_gz)?;
        let qrels = String::from_utf8_lossy(&qrels).into_owned();

        fs::create_dir_all(dest.join("qrels")).map_err(|e| FetchError::Io(e.to_string()))?;
        let write = |p: PathBuf, s: String| -> Result<(), FetchError> {
            fs::write(&p, s).map_err(|e| FetchError::Io(format!("{}: {e}", p.display())))
        };
        write(dest.join("corpus.jsonl"), mldr_corpus_to_beir(&corpus)?)?;
        write(dest.join("queries.jsonl"), mldr_queries_to_beir(&queries)?)?;
        write(
            dest.join("qrels").join("test.tsv"),
            trec_qrels_to_beir(&qrels)?,
        )?;
        Ok(())
    }

    fn gunzip(bytes: &[u8]) -> Result<String, FetchError> {
        use std::io::Read;
        let mut out = String::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_string(&mut out)
            .map_err(|e| FetchError::Unpack(format!("gzip: {e}")))?;
        Ok(out)
    }

    /// `{"docid": …, "text": …}` -> `{"_id": …, "title": "", "text": …}`.
    ///
    /// `pub` so a test can check the shape without the network, which is the only
    /// way any of this is checked: CI must never download 121 MiB.
    pub fn mldr_corpus_to_beir(jsonl: &str) -> Result<String, FetchError> {
        let mut out = String::with_capacity(jsonl.len());
        for (n, line) in jsonl.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| FetchError::Unpack(format!("corpus line {}: {e}", n + 1)))?;
            let id = v
                .get("docid")
                .and_then(|x| x.as_str())
                .ok_or_else(|| FetchError::Unpack(format!("corpus line {}: no `docid`", n + 1)))?;
            let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
            out.push_str(&serde_json::json!({ "_id": id, "title": "", "text": text }).to_string());
            out.push('\n');
        }
        if out.is_empty() {
            return Err(FetchError::Unpack("corpus is empty".into()));
        }
        Ok(out)
    }

    /// `{"query_id": …, "query": …, "positive_passages": […]}` -> `{"_id": …, "text": …}`.
    ///
    /// The passages are dropped on purpose. They are MLDR's *training* signal;
    /// keeping them would let a retriever be scored against documents handed to
    /// it with the question, which is not retrieval. The judgements come from
    /// the qrels file, like every other corpus here.
    pub fn mldr_queries_to_beir(jsonl: &str) -> Result<String, FetchError> {
        let mut out = String::new();
        for (n, line) in jsonl.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| FetchError::Unpack(format!("queries line {}: {e}", n + 1)))?;
            let id = v.get("query_id").and_then(|x| x.as_str()).ok_or_else(|| {
                FetchError::Unpack(format!("queries line {}: no `query_id`", n + 1))
            })?;
            let text = v.get("query").and_then(|x| x.as_str()).unwrap_or("");
            out.push_str(&serde_json::json!({ "_id": id, "text": text }).to_string());
            out.push('\n');
        }
        if out.is_empty() {
            return Err(FetchError::Unpack("queries are empty".into()));
        }
        Ok(out)
    }

    /// Four-column TREC qrels -> the three BEIR expects.
    ///
    /// `q-es-8  Q0  doc-es-64  1` becomes `q-es-8  doc-es-64  1`, dropping the
    /// `Q0` column that TREC carries and BEIR does not.
    ///
    /// **Not cosmetic, and the danger is narrower than it first looks — which
    /// is worth knowing precisely.** Handed the four-column file,
    /// [`super::parse_qrels_tsv`] reads `Q0` as the document and the document
    /// as the grade. For MLDR that *fails loudly*, because `doc-es-64` is not a
    /// number. But **TREC collections commonly use numeric document ids**, and
    /// with those the same line parses cleanly into a judgement that is
    /// completely wrong, with no error anywhere. Both cases are pinned by a
    /// test; the first was measured after I had assumed the second applied to
    /// everything.
    pub fn trec_qrels_to_beir(tsv: &str) -> Result<String, FetchError> {
        let mut out = String::new();
        for (n, line) in tsv.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            let (q, d, rel) = match cols.as_slice() {
                [q, _q0, d, rel] => (*q, *d, *rel),
                [q, d, rel] => (*q, *d, *rel),
                _ => {
                    return Err(FetchError::Unpack(format!(
                        "qrels line {}: expected 3 or 4 tab-separated columns, found {}",
                        n + 1,
                        cols.len()
                    )))
                }
            };
            out.push_str(&format!("{q}\t{d}\t{rel}\n"));
        }
        if out.is_empty() {
            return Err(FetchError::Unpack("qrels are empty".into()));
        }
        Ok(out)
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

            // This assertion used to read "no registered corpus has long
            // documents yet -- that is what N138 is waiting for, and this test
            // is how it stops being a surprise". It failed the moment MLDR was
            // registered, which is the test doing its job: the gap closed, and
            // the assertion now records WHICH corpus closed it rather than
            // being deleted.
            let long: Vec<&str> = KNOWN
                .iter()
                .filter(|s| s.can_discriminate_aggregation())
                .map(|s| s.name)
                .collect();
            assert_eq!(
                long,
                vec!["mldr-es"],
                "exactly one registered corpus has documents longer than a chunk; \
                 if that changes, say which and why here"
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

    // -----------------------------------------------------------------------
    // Splits
    // -----------------------------------------------------------------------

    /// A BEIR directory whose `queries.jsonl` carries BOTH splits' queries,
    /// which is the layout the real ones ship and the reason this exists.
    fn write_two_split_corpus(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ai_assistant_split_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("qrels")).expect("mkdir");
        std::fs::write(dir.join("corpus.jsonl"), CORPUS).expect("corpus");
        std::fs::write(
            dir.join("queries.jsonl"),
            "{\"_id\": \"q1\", \"text\": \"purring animal\"}\n\
             {\"_id\": \"q2\", \"text\": \"barking animal\"}\n\
             {\"_id\": \"q3\", \"text\": \"singing animal\"}\n",
        )
        .expect("queries");
        // test judges q1 only; dev judges q2 and q3.
        std::fs::write(dir.join("qrels").join("test.tsv"), "q1\td1\t1\n").expect("test qrels");
        std::fs::write(dir.join("qrels").join("dev.tsv"), "q2\td2\t1\nq3\td3\t1\n")
            .expect("dev qrels");
        dir
    }

    #[test]
    fn a_split_selects_its_own_queries_not_the_whole_file() {
        let dir = write_two_split_corpus("select");

        let test = load_beir_dir_split(&dir, Some(Split::Test)).expect("test split loads");
        assert_eq!(test.queries.len(), 1, "test judges one query");
        assert_eq!(test.queries[0].id, "q1");

        let dev = load_beir_dir_split(&dir, Some(Split::Dev)).expect("dev split loads");
        assert_eq!(dev.queries.len(), 2, "dev judges two");

        // All three documents stay: narrowing the QUERIES must not narrow the
        // collection being searched, or the task gets easier and the score moves.
        assert_eq!(test.docs.len(), 3);
        assert_eq!(dev.docs.len(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The point of the change is that it is free: the same retriever must score
    /// exactly the same with and without the unjudged queries, because
    /// `summarise` was already dropping them. If this ever fails, the narrowing
    /// is not an optimisation and the published number has to be re-derived.
    #[test]
    fn narrowing_to_the_split_does_not_move_the_score() {
        use crate::retrieval_metrics::summarise;

        let dir = write_two_split_corpus("score");
        let narrowed = load_beir_dir_split(&dir, Some(Split::Test)).expect("narrowed");

        // The same corpus with every query, which is what the loader used to do.
        let mut everything = narrowed.clone();
        everything.queries = parse_queries_jsonl(
            "{\"_id\": \"q1\", \"text\": \"purring animal\"}\n\
             {\"_id\": \"q2\", \"text\": \"barking animal\"}\n\
             {\"_id\": \"q3\", \"text\": \"singing animal\"}\n",
        )
        .expect("queries");

        let r = scripted(&[
            ("purring animal", &["d1", "d2"]),
            ("barking animal", &["d2"]),
            ("singing animal", &["d3"]),
        ]);

        let a = summarise(
            &run_corpus(&narrowed, &r, 10)
                .expect("narrowed run")
                .file
                .queries,
            10,
        );
        let b = summarise(
            &run_corpus(&everything, &r, 10)
                .expect("full run")
                .file
                .queries,
            10,
        );

        assert_eq!(a.ndcg_at_k, b.ndcg_at_k, "nDCG must not move");
        assert_eq!(a.recall_at_k, b.recall_at_k, "recall must not move");
        assert_eq!(a.mrr, b.mrr, "MRR must not move");
        // What DOES change is how much work was done, and how many queries the
        // report had to throw away to get there.
        assert_eq!(a.skipped, 0, "nothing to skip once narrowed");
        assert_eq!(
            b.skipped, 2,
            "the two dev queries were retrieved for nothing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn asking_for_a_split_that_is_not_there_is_an_error_not_a_fallback() {
        let dir = write_two_split_corpus("missing");
        // `train` does not exist here. Quietly scoring `test` instead would
        // answer a different question under the name the caller asked for.
        match load_beir_dir_split(&dir, Some(Split::Train)) {
            Err(CorpusError::Io { reason, .. }) => {
                assert!(reason.contains("train"), "{reason}");
            }
            other => panic!("expected a missing-split error, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_split_given_prefers_test_which_is_what_papers_report() {
        let dir = write_two_split_corpus("default");
        let c = load_beir_dir(&dir).expect("loads");
        assert_eq!(c.queries.len(), 1, "test, not dev");
        assert_eq!(c.queries[0].id, "q1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_names_parse_the_way_a_person_would_type_them() {
        use std::str::FromStr;
        assert_eq!(Split::from_str("test"), Ok(Split::Test));
        assert_eq!(Split::from_str(" TEST "), Ok(Split::Test));
        assert_eq!(Split::from_str("validation"), Ok(Split::Dev));
        assert_eq!(Split::from_str("training"), Ok(Split::Train));
        assert!(Split::from_str("teest").is_err());
    }

    // -----------------------------------------------------------------------
    // MLDR -> BEIR
    //
    // No network anywhere: the inputs are the real upstream shapes, copied from
    // one line of each file, so the conversion is checked without CI ever
    // downloading 121 MiB.
    // -----------------------------------------------------------------------

    #[cfg(feature = "zip")]
    mod mldr {
        use super::super::fetch::{mldr_corpus_to_beir, mldr_queries_to_beir, trec_qrels_to_beir};
        use super::super::{parse_corpus_jsonl, parse_qrels_tsv, parse_queries_jsonl};

        /// Upstream shape, abridged: MLDR keeps the passages it trained on in
        /// the same record as the query.
        const MLDR_QUERIES: &str = r#"{"query_id": "q-es-8", "query": "deidad solar eslava", "positive_passages": [{"docid": "doc-es-64", "text": "largo"}], "negative_passages": [{"docid": "doc-es-65", "text": "otro"}]}
{"query_id": "q-es-9", "query": "ciclo anual", "positive_passages": [], "negative_passages": []}"#;

        const MLDR_CORPUS: &str = r#"{"docid": "doc-es-64", "text": "Mitologia eslava. Fuentes."}
{"docid": "doc-es-65", "text": "Sancho Perez de Paz."}"#;

        /// Four columns, which is what TREC writes and BEIR does not.
        const MLDR_QRELS: &str = "q-es-8\tQ0\tdoc-es-64\t1\nq-es-9\tQ0\tdoc-es-65\t1\n";

        #[test]
        fn the_three_files_convert_into_something_the_loader_can_read() {
            let corpus = mldr_corpus_to_beir(MLDR_CORPUS).expect("corpus converts");
            let docs = parse_corpus_jsonl(&corpus).expect("and parses as BEIR");
            assert_eq!(docs.len(), 2);
            assert_eq!(docs[0].id, "doc-es-64");
            assert!(docs[0].text.contains("Mitologia"));

            let queries = mldr_queries_to_beir(MLDR_QUERIES).expect("queries convert");
            let qs = parse_queries_jsonl(&queries).expect("and parse as BEIR");
            assert_eq!(qs.len(), 2);
            assert_eq!(qs[0].id, "q-es-8");
            assert_eq!(qs[0].text, "deidad solar eslava");

            let qrels = trec_qrels_to_beir(MLDR_QRELS).expect("qrels convert");
            let judged = parse_qrels_tsv(&qrels).expect("and parse as BEIR");
            assert_eq!(judged["q-es-8"]["doc-es-64"], 1.0);
        }

        /// The reason `trec_qrels_to_beir` exists, and the shape of the danger
        /// measured rather than assumed.
        ///
        /// I expected the unconverted four-column file to be read silently as
        /// three columns -- `Q0` as the document id, the real id as the score.
        /// It is not, **for MLDR**: the ids are strings, so the score column
        /// fails to parse and the loader refuses the file. Good.
        ///
        /// But that safety is a property of MLDR's ids, not of the parser.
        /// **TREC collections routinely use numeric document ids**, and with
        /// those the same mistake parses cleanly into judgements that are
        /// entirely wrong, with no error anywhere. Both halves are asserted
        /// here so the next person reads the real boundary instead of my first
        /// guess at it.
        #[test]
        fn the_unconverted_trec_column_is_caught_for_string_ids_and_not_for_numeric_ones() {
            // MLDR: refused, because "doc-es-64" is not a score.
            let e = parse_qrels_tsv(MLDR_QRELS).expect_err("string ids save us here");
            assert!(format!("{e:?}").contains("not a number"), "{e:?}");

            // Numeric ids: parses, and every judgement is nonsense. This is the
            // case the conversion is actually protecting against.
            let numeric = "q1\tQ0\t12345\t1\n";
            let poisoned = parse_qrels_tsv(numeric).expect("it parses, which is the problem");
            assert_eq!(
                poisoned["q1"]["Q0"], 12345.0,
                "the literal Q0 became the document, and the document became the grade"
            );
            assert!(!poisoned["q1"].contains_key("12345"));

            // Converted, both are right.
            for raw in [MLDR_QRELS, numeric] {
                let ok =
                    parse_qrels_tsv(&trec_qrels_to_beir(raw).expect("converts")).expect("parses");
                assert!(ok.values().all(|m| !m.contains_key("Q0")), "{ok:?}");
            }
            let ok =
                parse_qrels_tsv(&trec_qrels_to_beir(numeric).expect("converts")).expect("parses");
            assert_eq!(ok["q1"]["12345"], 1.0);
        }

        /// The passages MLDR ships with each query are dropped, and that is a
        /// correctness requirement, not tidiness: scoring a retriever against
        /// documents that were handed to it with the question is not retrieval.
        #[test]
        fn the_queries_keep_no_trace_of_the_passages_they_shipped_with() {
            let out = mldr_queries_to_beir(MLDR_QUERIES).expect("converts");
            assert!(!out.contains("positive_passages"), "{out}");
            assert!(!out.contains("negative_passages"), "{out}");
            assert!(
                !out.contains("doc-es-64"),
                "a passage id leaked into the queries"
            );
        }

        #[test]
        fn a_three_column_qrels_file_still_works() {
            // BEIR's own format, so the converter is safe to run on anything.
            let out = trec_qrels_to_beir("q1\td1\t2\n").expect("converts");
            assert_eq!(out, "q1\td1\t2\n");
        }

        #[test]
        fn a_malformed_qrels_line_is_refused_rather_than_guessed_at() {
            let e = trec_qrels_to_beir("q1\td1\n").expect_err("two columns is not a judgement");
            assert!(format!("{e}").contains("3 or 4"), "{e}");
        }

        #[test]
        fn a_record_without_an_id_is_an_error_not_a_skipped_line() {
            // Silently dropping it would shrink the corpus and move the score.
            let e = mldr_corpus_to_beir(r#"{"text": "no id here"}"#)
                .expect_err("a document with no id cannot be judged");
            assert!(format!("{e}").contains("docid"), "{e}");
        }

        /// MLDR is registered as the one corpus that can answer N138, so the
        /// registry entry has to actually say that.
        #[test]
        fn the_registered_entry_is_the_one_that_can_discriminate_aggregation() {
            let s = super::super::fetch::source("mldr-es").expect("registered");
            assert!(
                s.can_discriminate_aggregation(),
                "mldr-es exists precisely because the BEIR pair cannot"
            );
            for other in ["nfcorpus", "trec-covid"] {
                let o = super::super::fetch::source(other).expect("registered");
                assert!(
                    !o.can_discriminate_aggregation(),
                    "{other} has documents shorter than one chunk"
                );
            }
        }
    }
}
