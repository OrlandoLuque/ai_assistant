// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Retrieval quality metrics: recall@k, precision@k, MRR, MAP and nDCG.
//!
//! # Why this module exists
//!
//! Before it, this crate had rank fusion, three rerankers, keyword/semantic
//! weights, a configurable RRF `k`, relevance thresholds and 46 `RagFeatures`
//! flags — and **no way to tell whether any of those settings helped**. Every
//! decision about retrieval had been argued rather than measured.
//!
//! These functions are the instrument. They have no dependencies and no feature
//! gate, because a measuring device that is only available in some builds
//! cannot be used to compare builds.
//!
//! # Which metric answers which question
//!
//! | metric | question |
//! |---|---|
//! | [`recall_at_k`] | of the relevant documents that exist, how many reached the top k? |
//! | [`precision_at_k`] | of the top k, how many were relevant? |
//! | [`reciprocal_rank`] | how far down was the *first* hit? |
//! | [`average_precision`] | precision averaged over every hit's position |
//! | [`ndcg_at_k`] | the whole ordering, with graded relevance and position discount |
//!
//! **A reranker cannot raise recall.** It reorders the list it was given, so
//! recall@k is identical before and after. Measure retrieval with recall@k and
//! reranking with MRR, MAP or nDCG. Evaluating a reranker with recall produces
//! a tie every time and concludes, wrongly, that reranking does nothing.
//! The test `rerankers_cannot_change_recall_only_order` asserts exactly that, so
//! the rule survives as a test rather than as a paragraph. Named in plain text
//! and not linked: it lives behind `#[cfg(test)]`, so it does not exist in a
//! documentation build, and a link to a test renders as dead words.
//!
//! # Two decisions worth knowing about
//!
//! **Duplicates are collapsed.** A retrieved list naming the same document
//! twice counts it once, keeping the earlier position. Fusing two ranked lists
//! is precisely how duplicates appear, and counting one document twice inflates
//! recall — the metric would reward the bug it exists to catch.
//!
//! **A query with no relevant document returns `None`, never `0.0`.** Recall of
//! nothing is undefined, and zero is a score. Folding the undefined case into a
//! bad score makes a corpus with missing judgements look like a failing
//! retriever. [`summarise`] counts those queries in
//! [`RetrievalReport::skipped`] instead of averaging them in.
//!
//! ```
//! use std::collections::HashMap;
//! use ai_assistant::retrieval_metrics::{ndcg_at_k, reciprocal_rank};
//!
//! let retrieved = vec!["d2".to_string(), "d1".to_string()];
//! let relevant: std::collections::HashSet<String> =
//!     ["d1".to_string()].into_iter().collect();
//!
//! // The only hit is second, so the reciprocal rank is 1/2.
//! assert!((reciprocal_rank(&retrieved, &relevant) - 0.5).abs() < 1e-12);
//!
//! let grades: HashMap<String, f64> =
//!     [("d1".to_string(), 1.0)].into_iter().collect();
//! assert!(ndcg_at_k(&retrieved, &grades, 2).is_some());
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collapse repeats, keeping the first occurrence, and truncate to `k`.
///
/// `k == 0` means "no cut", which is what the rank-wide metrics (MRR, MAP)
/// need; the `@k` metrics always pass a real `k`.
fn top_k_unique(retrieved: &[String], k: usize) -> Vec<&String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out: Vec<&String> = Vec::new();
    for id in retrieved {
        if seen.insert(id.as_str()) {
            out.push(id);
            if k > 0 && out.len() == k {
                break;
            }
        }
    }
    out
}

/// The documents a grade map considers relevant: grade strictly above zero.
///
/// Graded relevance uses zero for "judged and not relevant", which is a
/// different statement from "not judged" — but both are non-relevant for the
/// binary metrics, so both are excluded here.
fn relevant_from_grades(grades: &HashMap<String, f64>) -> HashSet<String> {
    grades
        .iter()
        .filter(|(_, g)| **g > 0.0)
        .map(|(id, _)| id.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Per-query metrics
// ---------------------------------------------------------------------------

/// Fraction of the relevant documents that appear in the top `k`.
///
/// Returns `None` when `relevant` is empty: recall is `0/0` there, and
/// answering `0.0` would report an unjudged query as a failed one.
///
/// Insensitive to the order *within* the top k — that is the point, and the
/// reason a reranker cannot move this number.
pub fn recall_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> Option<f64> {
    if relevant.is_empty() {
        return None;
    }
    let hits = top_k_unique(retrieved, k)
        .into_iter()
        .filter(|id| relevant.contains(*id))
        .count();
    Some(hits as f64 / relevant.len() as f64)
}

/// Fraction of **what came back** that is relevant. **Not the TREC definition.**
///
/// The denominator is how many distinct documents were actually returned, not
/// `k` — a retriever that returns three documents for `k = 10` is not penalised
/// for the seven it never claimed.
///
/// That is a deliberate choice and it is **not comparable with published
/// numbers**. BEIR scores through `pytrec_eval`, which follows the TREC
/// convention: the denominator is always `k`, *even on a shorter run*. The two
/// disagree exactly when a retriever returns fewer than `k` documents, and they
/// disagree a lot — one relevant hit out of one returned document at `k = 5` is
/// `1.0` here and `0.2` there.
///
/// Use [`precision_at_k_strict`] for anything compared against a paper, and this
/// one to ask "of what it gave me, how much was useful". `recall@k`, MRR, MAP and
/// nDCG@k have no such divergence — they are comparable as they stand.
///
/// Returns `None` for `k == 0` and for an empty retrieved list: dividing by the
/// size of nothing is undefined.
pub fn precision_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> Option<f64> {
    if k == 0 {
        return None;
    }
    let top = top_k_unique(retrieved, k);
    if top.is_empty() {
        return None;
    }
    let hits = top.iter().filter(|id| relevant.contains(**id)).count();
    Some(hits as f64 / top.len() as f64)
}

/// Fraction of `k` that is relevant — **the TREC and BEIR definition**.
///
/// The denominator is `k`, flat, even when the retriever returned fewer. That is
/// what `pytrec_eval` computes and therefore what every published BEIR P@k means,
/// so this is the one to quote next to somebody else's number.
///
/// Returns `None` only for `k == 0`. An empty retrieved list is `0.0` here and
/// **not** `None`, unlike [`precision_at_k`]: with a fixed denominator, returning
/// nothing is a score of zero rather than an undefined quantity. That difference
/// matters more than it looks — a `None` is dropped from the averages, so a
/// retriever that found nothing for half the queries would be flattered by being
/// judged only on the half where it answered.
pub fn precision_at_k_strict(
    retrieved: &[String],
    relevant: &HashSet<String>,
    k: usize,
) -> Option<f64> {
    if k == 0 {
        return None;
    }
    let top = top_k_unique(retrieved, k);
    let hits = top.iter().filter(|id| relevant.contains(**id)).count();
    Some(hits as f64 / k as f64)
}

/// `1 / position` of the first relevant document, counting from one.
///
/// Zero when nothing relevant was retrieved, which is a genuine score and not
/// an undefined one — the retriever was asked and returned nothing useful.
pub fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    for (i, id) in top_k_unique(retrieved, 0).into_iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Precision measured at every position that holds a hit, averaged.
///
/// Unlike [`reciprocal_rank`], which stops at the first hit, this rewards
/// getting *all* the relevant documents high. Returns `None` when there is
/// nothing relevant to find.
pub fn average_precision(retrieved: &[String], relevant: &HashSet<String>) -> Option<f64> {
    if relevant.is_empty() {
        return None;
    }
    let mut hits = 0usize;
    let mut sum = 0.0f64;
    for (i, id) in top_k_unique(retrieved, 0).into_iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum += hits as f64 / (i as f64 + 1.0);
        }
    }
    // Divided by the number of relevant documents that EXIST, not by the number
    // found. A retriever that returns one perfect hit out of five relevant
    // documents scores 0.2, not 1.0.
    Some(sum / relevant.len() as f64)
}

/// Discounted cumulative gain of an already-ordered list of gains.
///
/// `gain_i / log2(i + 1)`, positions counting from one, so the first position
/// is undiscounted.
pub fn dcg_at_k(gains: &[f64], k: usize) -> f64 {
    let limit = if k == 0 {
        gains.len()
    } else {
        k.min(gains.len())
    };
    gains
        .iter()
        .take(limit)
        .enumerate()
        .map(|(i, g)| g / ((i as f64 + 2.0).log2()))
        .sum()
}

/// Normalised DCG at `k`, with graded relevance.
///
/// `grades` maps a document id to its relevance; anything absent counts as
/// zero. Returns `None` when no document has a positive grade.
///
/// The ideal DCG comes from **every graded document**, sorted by grade and cut
/// at `k` — not from a reordering of what was retrieved. Building the ideal
/// from the retrieved list is the classic implementation bug: a retriever that
/// missed the best document entirely would score 1.0 for ordering the
/// leftovers correctly.
pub fn ndcg_at_k(retrieved: &[String], grades: &HashMap<String, f64>, k: usize) -> Option<f64> {
    let mut ideal: Vec<f64> = grades.values().copied().filter(|g| *g > 0.0).collect();
    if ideal.is_empty() {
        return None;
    }
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let actual: Vec<f64> = top_k_unique(retrieved, k)
        .into_iter()
        .map(|id| grades.get(id).copied().unwrap_or(0.0))
        .collect();

    let idcg = dcg_at_k(&ideal, k);
    if idcg <= 0.0 {
        return None;
    }
    Some(dcg_at_k(&actual, k) / idcg)
}

// ---------------------------------------------------------------------------
// Aggregating a run
// ---------------------------------------------------------------------------

/// One query's retrieved list together with its relevance judgements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryRun {
    /// Identifies the query in reports. Not used in any computation.
    pub query_id: String,
    /// Document ids, best first.
    pub retrieved: Vec<String>,
    /// Graded relevance. Absent ids count as zero.
    pub grades: HashMap<String, f64>,
}

impl QueryRun {
    /// A query judged with binary relevance: listed ids are relevant, grade 1.
    pub fn binary(
        query_id: impl Into<String>,
        retrieved: Vec<String>,
        relevant: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            retrieved,
            grades: relevant.into_iter().map(|id| (id, 1.0)).collect(),
        }
    }

    /// A query judged with graded relevance.
    pub fn graded(
        query_id: impl Into<String>,
        retrieved: Vec<String>,
        grades: HashMap<String, f64>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            retrieved,
            grades,
        }
    }
}

// ---------------------------------------------------------------------------
// Reading a run from a file
// ---------------------------------------------------------------------------

/// A whole run, as it is written to disk.
///
/// ```json
/// {
///   "queries": [
///     {
///       "query_id": "q1",
///       "retrieved": ["doc-3", "doc-1", "doc-7"],
///       "grades": { "doc-1": 1.0, "doc-9": 2.0 }
///     }
///   ]
/// }
/// ```
///
/// `retrieved` is what a retriever returned, best first. `grades` is the
/// judgement: which documents are relevant and how much. A document in `grades`
/// but not in `retrieved` is one the retriever missed — which is the whole point
/// of measuring recall, so those entries matter more than the ones that match.
///
/// An object rather than a bare array so the format can grow a header (which
/// retriever, which corpus, which configuration) without breaking readers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunFile {
    /// Every query in the run.
    pub queries: Vec<QueryRun>,
}

/// Why a run file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunFileError {
    /// Not JSON, or not this shape.
    Malformed(String),
    /// Two queries share a `query_id`.
    ///
    /// Not tolerated: a duplicated id silently double-weights one query in every
    /// average, and the file still looks fine.
    DuplicateQueryId(String),
    /// The file parsed and contained no queries.
    Empty,
}

impl std::fmt::Display for RunFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(e) => write!(f, "not a retrieval run file: {e}"),
            Self::DuplicateQueryId(id) => write!(
                f,
                "query id {id:?} appears more than once; a duplicate would be \
                 averaged in twice and the file would still look fine"
            ),
            Self::Empty => write!(f, "the file parsed but contains no queries"),
        }
    }
}

impl std::error::Error for RunFileError {}

/// Parse a run from JSON text, rejecting duplicate ids and empty runs.
pub fn parse_run_file(json: &str) -> Result<RunFile, RunFileError> {
    let file: RunFile =
        serde_json::from_str(json).map_err(|e| RunFileError::Malformed(e.to_string()))?;
    if file.queries.is_empty() {
        return Err(RunFileError::Empty);
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for q in &file.queries {
        if !seen.insert(q.query_id.as_str()) {
            return Err(RunFileError::DuplicateQueryId(q.query_id.clone()));
        }
    }
    Ok(file)
}

/// Averages over a set of queries, plus how many could not be scored.
///
/// Not called `RunSummary`, which would have been the obvious name: that one is
/// already taken by `eval_suite::ablation::RunSummary`, re-exported at the crate
/// root. Two public types with one name is how the three copies of RRF started.
///
/// Deliberately *not* an intra-doc link to it. That type lives behind
/// `--features eval-suite` and this module has no feature gate at all, so the
/// link would resolve here and dangle in any build without that feature — and
/// the documentation gate runs with it on, so it would never notice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalReport {
    /// The `k` the `@k` metrics were computed at.
    pub k: usize,
    /// Queries that contributed to the averages.
    pub scored: usize,
    /// Queries with no relevant document, excluded from every average.
    ///
    /// Read this before the scores. A run where most queries were skipped is
    /// a statement about the judgements, not about the retriever.
    pub skipped: usize,
    /// Mean recall@k.
    pub recall_at_k: f64,
    /// Mean precision@k with **our** denominator: how many documents came back.
    ///
    /// Not comparable with a paper. See [`precision_at_k`] for why, and
    /// [`RetrievalReport::precision_at_k_strict`] for the one that is.
    pub precision_at_k: f64,
    /// Mean precision@k with the **TREC/BEIR** denominator: `k`, flat.
    ///
    /// This is the number to put next to somebody else's. It is always lower than
    /// or equal to the one above, and strictly lower whenever the retriever
    /// returned fewer than `k` documents.
    pub precision_at_k_strict: f64,
    /// Mean reciprocal rank.
    pub mrr: f64,
    /// Mean average precision.
    pub map: f64,
    /// Mean nDCG@k.
    pub ndcg_at_k: f64,
}

impl RetrievalReport {
    /// Did every query get scored?
    pub fn is_complete(&self) -> bool {
        self.skipped == 0
    }
}

/// Score a whole run at a given `k`.
///
/// Queries with no relevant document are counted in [`RetrievalReport::skipped`] and
/// left out of the averages, rather than averaged in as zero.
pub fn summarise(queries: &[QueryRun], k: usize) -> RetrievalReport {
    let mut scored = 0usize;
    let mut skipped = 0usize;
    let (mut recall, mut precision, mut mrr, mut map, mut ndcg) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut precision_strict = 0.0;

    for q in queries {
        let relevant = relevant_from_grades(&q.grades);
        if relevant.is_empty() {
            skipped += 1;
            continue;
        }
        scored += 1;
        recall += recall_at_k(&q.retrieved, &relevant, k).unwrap_or(0.0);
        precision += precision_at_k(&q.retrieved, &relevant, k).unwrap_or(0.0);
        precision_strict += precision_at_k_strict(&q.retrieved, &relevant, k).unwrap_or(0.0);
        mrr += reciprocal_rank(&q.retrieved, &relevant);
        map += average_precision(&q.retrieved, &relevant).unwrap_or(0.0);
        ndcg += ndcg_at_k(&q.retrieved, &q.grades, k).unwrap_or(0.0);
    }

    let n = if scored == 0 { 1.0 } else { scored as f64 };
    RetrievalReport {
        k,
        scored,
        skipped,
        recall_at_k: recall / n,
        precision_at_k: precision / n,
        precision_at_k_strict: precision_strict / n,
        mrr: mrr / n,
        map: map / n,
        ndcg_at_k: ndcg / n,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    fn rel(v: &[&str]) -> HashSet<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn recall_counts_how_many_of_the_relevant_arrived() {
        let retrieved = ids(&["d1", "d9", "d2", "d8", "d3"]);
        let relevant = rel(&["d1", "d2", "d3", "d4"]);
        // Three of four relevant documents are in the top five.
        assert!(close(
            recall_at_k(&retrieved, &relevant, 5).expect("judged"),
            0.75
        ));
        // Only one of them is in the top two.
        assert!(close(
            recall_at_k(&retrieved, &relevant, 2).expect("judged"),
            0.25
        ));
    }

    #[test]
    fn precision_divides_by_what_was_returned_not_by_k() {
        let retrieved = ids(&["d1", "d2", "d9"]);
        let relevant = rel(&["d1", "d2"]);
        // Three returned, two relevant — not 2/10.
        assert!(close(
            precision_at_k(&retrieved, &relevant, 10).expect("judged"),
            2.0 / 3.0
        ));
    }

    #[test]
    fn reciprocal_rank_is_the_position_of_the_first_hit() {
        let relevant = rel(&["gold"]);
        assert!(close(
            reciprocal_rank(&ids(&["gold", "a", "b"]), &relevant),
            1.0
        ));
        assert!(close(
            reciprocal_rank(&ids(&["a", "gold", "b"]), &relevant),
            0.5
        ));
        assert!(close(
            reciprocal_rank(&ids(&["a", "b", "gold"]), &relevant),
            1.0 / 3.0
        ));
        // Asked and found nothing: a real zero, not an undefined value.
        assert!(close(reciprocal_rank(&ids(&["a", "b"]), &relevant), 0.0));
    }

    #[test]
    fn average_precision_divides_by_the_relevant_that_exist() {
        // One perfect hit, but four relevant documents were never returned.
        let retrieved = ids(&["d1"]);
        let relevant = rel(&["d1", "d2", "d3", "d4", "d5"]);
        assert!(close(
            average_precision(&retrieved, &relevant).expect("judged"),
            0.2
        ));
    }

    #[test]
    fn ndcg_matches_the_textbook_worked_example() {
        // The canonical worked example: grades 3,2,3,0,1,2 in retrieved order.
        // Computed by hand, term by term, so this asserts against arithmetic
        // rather than against a number this implementation once printed:
        //
        //   DCG  = 3/log2(2) + 2/log2(3) + 3/log2(4)
        //        + 0/log2(5) + 1/log2(6) + 2/log2(7)
        //        = 3 + 1.261859507 + 1.5 + 0 + 0.386852807 + 0.712414374
        //        = 6.861126688
        //
        //   ideal order is 3,3,2,2,1,0, so
        //   IDCG = 3 + 3/log2(3) + 2/log2(4) + 2/log2(5) + 1/log2(6) + 0
        //        = 3 + 1.892789261 + 1 + 0.861353116 + 0.386852807
        //        = 7.140995184
        //
        //   nDCG = 6.861126688 / 7.140995184 = 0.9608081943
        //
        // The tolerance is tight on purpose: this is a closed-form value, so
        // anything beyond float noise means the formula changed.
        let retrieved = ids(&["d1", "d2", "d3", "d4", "d5", "d6"]);
        let grades: HashMap<String, f64> = [
            ("d1".to_string(), 3.0),
            ("d2".to_string(), 2.0),
            ("d3".to_string(), 3.0),
            ("d4".to_string(), 0.0),
            ("d5".to_string(), 1.0),
            ("d6".to_string(), 2.0),
        ]
        .into_iter()
        .collect();

        // And assert the two halves separately, so a failure says WHICH one
        // moved instead of only that the ratio did.
        let dcg = dcg_at_k(&[3.0, 2.0, 3.0, 0.0, 1.0, 2.0], 6);
        assert!(
            (dcg - 6.861126688_f64).abs() < 1e-9,
            "DCG changed: expected 6.861126688, got {dcg}"
        );
        let idcg = dcg_at_k(&[3.0, 3.0, 2.0, 2.0, 1.0, 0.0], 6);
        assert!(
            (idcg - 7.140995184_f64).abs() < 1e-9,
            "IDCG changed: expected 7.140995184, got {idcg}"
        );

        let got = ndcg_at_k(&retrieved, &grades, 6).expect("judged");
        assert!(
            (got - 0.9608081943_f64).abs() < 1e-9,
            "expected 0.9608081943 from the hand computation above, got {got}"
        );
    }

    #[test]
    fn the_ideal_comes_from_every_graded_document_not_from_what_was_returned() {
        // The best document (grade 3) was never retrieved. A perfect ordering
        // of the leftovers must NOT score 1.0 — that is the bug this guards.
        let retrieved = ids(&["b", "c"]);
        let grades: HashMap<String, f64> = [
            ("a".to_string(), 3.0), // missed entirely
            ("b".to_string(), 2.0),
            ("c".to_string(), 1.0),
        ]
        .into_iter()
        .collect();

        let got = ndcg_at_k(&retrieved, &grades, 3).expect("judged");
        assert!(
            got < 0.85,
            "missing the best document must cost real score, got {got}"
        );
    }

    #[test]
    fn the_two_precisions_differ_exactly_when_the_list_is_short() {
        // THE separating case, and the reason both exist. One relevant hit out of
        // one returned document at k = 5:
        //   ours   = 1/1 = 1.0   "of what it gave me, all of it was useful"
        //   TREC   = 1/5 = 0.2   "of the five slots asked for, one was useful"
        // A number compared against a paper must be the second one.
        let retrieved = ids(&["d1"]);
        let relevant = rel(&["d1"]);
        assert_eq!(precision_at_k(&retrieved, &relevant, 5), Some(1.0));
        assert_eq!(precision_at_k_strict(&retrieved, &relevant, 5), Some(0.2));

        // And they agree when the list is full, which is why the divergence went
        // unnoticed: every example with k documents returned looks identical.
        let full = ids(&["d1", "d2", "d3", "d4", "d5"]);
        assert_eq!(
            precision_at_k(&full, &relevant, 5),
            precision_at_k_strict(&full, &relevant, 5)
        );
    }

    #[test]
    fn an_empty_list_is_zero_strictly_and_undefined_loosely() {
        // Not a quirk — it decides whether a query counts. `None` is dropped from
        // the averages, so a retriever that found nothing for half the queries
        // would be judged only on the half where it answered, and flattered.
        // With a fixed denominator, finding nothing is a score of zero.
        let empty: Vec<String> = Vec::new();
        let relevant = rel(&["d1"]);
        assert_eq!(precision_at_k(&empty, &relevant, 5), None);
        assert_eq!(precision_at_k_strict(&empty, &relevant, 5), Some(0.0));
        // k == 0 is undefined for both: there is no "top zero".
        assert_eq!(precision_at_k_strict(&ids(&["d1"]), &relevant, 0), None);
    }

    #[test]
    fn strict_precision_is_never_above_the_loose_one() {
        // The invariant worth pinning: same numerator, denominator k >= returned,
        // so strict <= loose always. If this ever flips, one of them is wrong.
        let relevant = rel(&["d1", "d3"]);
        for n in 1..=6usize {
            let retrieved: Vec<String> = (1..=n).map(|i| format!("d{i}")).collect();
            let loose = precision_at_k(&retrieved, &relevant, 5).unwrap_or(0.0);
            let strict = precision_at_k_strict(&retrieved, &relevant, 5).unwrap_or(0.0);
            assert!(
                strict <= loose + 1e-12,
                "with {n} returned: strict {strict} must not exceed loose {loose}"
            );
        }
    }

    #[test]
    fn the_report_carries_both_precisions() {
        // So nobody has to recompute one of them from the other, which is
        // impossible anyway without knowing how many documents came back.
        let runs = vec![QueryRun::binary("q1", ids(&["d1"]), ["d1".to_string()])];
        let r = summarise(&runs, 5);
        assert_eq!(r.precision_at_k, 1.0);
        assert_eq!(r.precision_at_k_strict, 0.2);
    }

    #[test]
    fn rerankers_cannot_change_recall_only_order() {
        // The rule this module exists to make checkable. Same five documents,
        // two different orderings: recall is identical, MRR and nDCG are not.
        let relevant = rel(&["d3", "d5"]);
        let grades: HashMap<String, f64> = [("d3".to_string(), 1.0), ("d5".to_string(), 1.0)]
            .into_iter()
            .collect();

        let before = ids(&["d1", "d2", "d3", "d4", "d5"]);
        let after = ids(&["d3", "d5", "d1", "d2", "d4"]);

        let r_before = recall_at_k(&before, &relevant, 5).expect("judged");
        let r_after = recall_at_k(&after, &relevant, 5).expect("judged");
        assert!(
            close(r_before, r_after),
            "reordering changed recall@5 ({r_before} -> {r_after}); either the \
             metric is wrong or the list is not a permutation"
        );

        let mrr_before = reciprocal_rank(&before, &relevant);
        let mrr_after = reciprocal_rank(&after, &relevant);
        assert!(
            mrr_after > mrr_before,
            "a strictly better ordering must raise MRR: {mrr_before} -> {mrr_after}"
        );

        let n_before = ndcg_at_k(&before, &grades, 5).expect("judged");
        let n_after = ndcg_at_k(&after, &grades, 5).expect("judged");
        assert!(
            n_after > n_before,
            "a strictly better ordering must raise nDCG: {n_before} -> {n_after}"
        );
        assert!(close(n_after, 1.0), "the ideal ordering must score 1.0");
    }

    #[test]
    fn a_duplicate_cannot_inflate_recall() {
        // Fusing two ranked lists is exactly how the same id ends up twice.
        // Counting it twice would reward the defect these metrics exist to
        // catch.
        let retrieved = ids(&["d1", "d1", "d1", "d1"]);
        let relevant = rel(&["d1", "d2", "d3", "d4"]);
        assert!(
            close(recall_at_k(&retrieved, &relevant, 4).expect("judged"), 0.25),
            "one distinct hit out of four relevant is 0.25, whatever the repeats"
        );
    }

    #[test]
    fn an_unjudged_query_is_none_and_never_zero() {
        let retrieved = ids(&["d1"]);
        let empty = rel(&[]);
        assert!(recall_at_k(&retrieved, &empty, 5).is_none());
        assert!(average_precision(&retrieved, &empty).is_none());
        assert!(ndcg_at_k(&retrieved, &HashMap::new(), 5).is_none());
    }

    #[test]
    fn nothing_retrieved_scores_zero_and_does_not_panic() {
        let relevant = rel(&["d1"]);
        assert!(close(recall_at_k(&[], &relevant, 5).expect("judged"), 0.0));
        assert!(close(reciprocal_rank(&[], &relevant), 0.0));
        assert!(precision_at_k(&[], &relevant, 5).is_none());
    }

    #[test]
    fn k_larger_than_the_list_uses_what_is_there() {
        let retrieved = ids(&["d1", "d2"]);
        let relevant = rel(&["d1", "d2"]);
        assert!(close(
            recall_at_k(&retrieved, &relevant, 1000).expect("judged"),
            1.0
        ));
    }

    #[test]
    fn summarise_skips_unjudged_queries_instead_of_averaging_them_as_zero() {
        let queries = vec![
            QueryRun::binary("q1", ids(&["a", "b"]), [String::from("a")]),
            // No judgements at all: must not drag the average down.
            QueryRun::binary("q2", ids(&["c", "d"]), Vec::<String>::new()),
        ];
        let s = summarise(&queries, 2);
        assert_eq!(s.scored, 1);
        assert_eq!(s.skipped, 1);
        assert!(!s.is_complete());
        assert!(
            close(s.recall_at_k, 1.0),
            "the one scored query had perfect recall; got {}",
            s.recall_at_k
        );
        assert!(close(s.mrr, 1.0));
    }

    #[test]
    fn summarise_with_no_scorable_query_reports_zero_without_dividing_by_zero() {
        let queries = vec![QueryRun::binary("q1", ids(&["a"]), Vec::<String>::new())];
        let s = summarise(&queries, 5);
        assert_eq!(s.scored, 0);
        assert_eq!(s.skipped, 1);
        assert!(s.recall_at_k.is_finite(), "must not be NaN");
        assert!(close(s.recall_at_k, 0.0));
    }

    #[test]
    fn a_run_file_round_trips_and_scores_the_same() {
        let json = r#"{
            "queries": [
                {"query_id": "q1",
                 "retrieved": ["d1", "d9"],
                 "grades": {"d1": 1.0, "d7": 1.0}},
                {"query_id": "q2",
                 "retrieved": ["d4", "d2", "d3"],
                 "grades": {"d2": 2.0, "d3": 1.0}}
            ]
        }"#;
        let file = parse_run_file(json).expect("well-formed");
        assert_eq!(file.queries.len(), 2);

        let direct = summarise(&file.queries, 3);
        // q1: one of two relevant in the top 3 -> 0.5. q2: two of two -> 1.0.
        assert!(
            close(direct.recall_at_k, 0.75),
            "got {}",
            direct.recall_at_k
        );
        assert_eq!(direct.skipped, 0);

        // And serialising then reading it back must score identically — if the
        // representation loses the grades, every metric silently drops.
        let round = parse_run_file(&serde_json::to_string(&file).expect("serialises"))
            .expect("round trips");
        assert_eq!(summarise(&round.queries, 3), direct);
    }

    #[test]
    fn a_duplicated_query_id_is_rejected_not_averaged_twice() {
        // The file would look perfectly fine and quietly give one query double
        // weight in every average.
        let json = r#"{"queries": [
            {"query_id": "same", "retrieved": ["a"], "grades": {"a": 1.0}},
            {"query_id": "same", "retrieved": ["b"], "grades": {"z": 1.0}}
        ]}"#;
        assert_eq!(
            parse_run_file(json),
            Err(RunFileError::DuplicateQueryId("same".to_string()))
        );
    }

    #[test]
    fn an_empty_or_malformed_run_file_is_an_error() {
        assert_eq!(
            parse_run_file(r#"{"queries": []}"#),
            Err(RunFileError::Empty)
        );
        assert!(matches!(
            parse_run_file("not json at all"),
            Err(RunFileError::Malformed(_))
        ));
        // A plausible-but-wrong shape: a bare array instead of the object.
        assert!(matches!(
            parse_run_file(r#"[{"query_id": "q", "retrieved": [], "grades": {}}]"#),
            Err(RunFileError::Malformed(_))
        ));
    }

    #[test]
    fn dcg_discounts_by_position() {
        // First position undiscounted, second divided by log2(3).
        assert!(close(dcg_at_k(&[1.0], 1), 1.0));
        assert!(close(dcg_at_k(&[0.0, 1.0], 2), 1.0 / 3.0_f64.log2()));
        // A better ordering of the same gains scores higher.
        assert!(dcg_at_k(&[3.0, 1.0], 2) > dcg_at_k(&[1.0, 3.0], 2));
    }
}
