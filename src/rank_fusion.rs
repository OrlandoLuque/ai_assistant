//! Reciprocal Rank Fusion, in one place.
//!
//! # Why this module exists
//!
//! The same algorithm was written three times:
//!
//! | where | k | identity | normalised? |
//! |---|---|---|---|
//! | [`crate::reranker::ReciprocalRankFusion`] | constructor, default 60 | the `content` string | no |
//! | [`crate::rag_methods::RrfFusion`] | configurable | a caller-supplied closure | no |
//! | `RagPipeline::reciprocal_rank_fusion` | `60.0`, hard-coded | `chunk_id` | **yes** |
//!
//! The formula was identical in all three — `1 / (k + rank)` with the rank
//! starting at 1. What differed was the *adapter*: what counts as the identity
//! of a document, and what shape the input arrives in. Those are real
//! differences and they stay with their callers. The arithmetic is one
//! function, here.
//!
//! # The normalisation is not an option, it is the fix
//!
//! RRF scores encode **rank, not relevance**. With `k = 60` the best attainable
//! value is `1/61 ≈ 0.0164`, or `≈0.033` for a document ranked first by two
//! lists. Every other stage of this library works on a 0-1 relevance scale, and
//! `min_relevance_score` — 0.1 by default — is applied to whatever `score`
//! holds. So raw RRF output is *always* under the floor.
//!
//! That was found once (V316) and fixed in exactly one of the three copies:
//! `RagTier::Semantic`, the only tier that fuses without a reranker afterwards,
//! was discarding every chunk of every query. The tiers that rerank had their
//! scores rewritten by the reranker and never noticed. The other two copies
//! still carried the defect, waiting for a caller.
//!
//! Hence [`RrfOptions::normalise`](crate::rank_fusion::RrfOptions::normalise)
//! defaults to `true`, and turning it off is
//! something a caller has to ask for on purpose.

use std::collections::HashMap;

/// The `k` from the original paper. Dampens how much a low rank contributes.
pub const DEFAULT_K: f32 = 60.0;

/// How to fuse.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RrfOptions {
    /// Dampening constant. Larger values flatten the difference between ranks.
    ///
    /// 60 is the value from the paper and the default everywhere in this
    /// library. It was chosen for web-scale result lists; with the short lists
    /// a local search produces it flattens more than it probably should, which
    /// is measured work rather than something to guess at here.
    pub k: f32,
    /// Divide by the best score so the output lands on 0-1.
    ///
    /// Defaults to `true`. See the module documentation for why this is not a
    /// neutral choice: raw RRF values sit under this library's relevance floor,
    /// so an un-normalised list fed to any score threshold comes back empty.
    /// Dividing by the maximum preserves the ordering exactly — which is all
    /// RRF determines — and puts the values back on the scale the rest of the
    /// pipeline uses.
    pub normalise: bool,
}

with_setters!(RrfOptions {
    with_normalise -> normalise: bool,
});

impl Default for RrfOptions {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            normalise: true,
        }
    }
}

impl RrfOptions {
    /// Options with a given `k`, still normalised.
    pub fn with_k(k: f32) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Raw RRF values, on the `1/(k+rank)` scale.
    ///
    /// Only for a caller that does its own scaling and says so. Anything that
    /// hands these to a 0-1 threshold will find every document below it.
    pub fn raw(k: f32) -> Self {
        Self {
            k,
            normalise: false,
        }
    }
}

/// Fuse ranked lists of identifiers.
///
/// Each inner list is one ranking, best first. An identifier appearing in
/// several lists accumulates their contributions — that is the whole point of
/// the algorithm: agreement between independent rankings outweighs a strong
/// showing in one.
///
/// Returns `(identifier, score)` sorted best first. Ties keep the order in
/// which identifiers were first seen, so the result is deterministic rather
/// than whatever a hash map happened to yield.
///
/// Takes **N** lists, not two. The pipeline's copy had "keyword and semantic"
/// baked into its shape — two filters and two copied loops — so adding a third
/// source meant rewriting the function rather than passing another argument.
///
/// # Examples
///
/// ```
/// use ai_assistant::rank_fusion::{fuse_ranked_ids, RrfOptions};
///
/// let keyword = vec!["a".to_string(), "b".to_string()];
/// let semantic = vec!["b".to_string(), "c".to_string()];
///
/// let fused = fuse_ranked_ids(&[keyword, semantic], &RrfOptions::default());
///
/// // "b" is in both lists, so it wins even though neither ranked it first.
/// assert_eq!(fused[0].0, "b");
/// // Normalised by default, so the best score is exactly 1.0.
/// assert!((fused[0].1 - 1.0).abs() < f32::EPSILON);
/// ```
pub fn fuse_ranked_ids(lists: &[Vec<String>], options: &RrfOptions) -> Vec<(String, f32)> {
    // Insertion order is kept so ties are broken the same way every run. A
    // plain HashMap made the output depend on hashing, which is the kind of
    // instability that gets blamed on the model.
    let mut totals: HashMap<&str, f32> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();

    // `k` is sanitised HERE, where it is consumed, and not in the constructors.
    // `RrfOptions::k` is a public field: `#[non_exhaustive]` stops an outside
    // caller writing a struct literal, but nothing stops
    // `let mut o = RrfOptions::default(); o.k = -1.0;`. Guarding the two
    // constructors would be two implementations of one rule, and the one a
    // caller can route around.
    //
    // The domain is `k >= 0` and finite. `k` is a smoothing constant; a
    // negative one has no meaning in `1/(k + rank)`. Out-of-domain values are
    // clamped rather than rejected because the function returns no error type
    // and a panic in a fusion step is worse than a sane default.
    //
    // **The `+ 1.0` below is the DEFINITION, not a guard.** RRF counts ranks
    // from one, so index 0 is rank 1. It happens to keep `k == 0` safe, and
    // V392 leaned on that — which was imprecise, because it stops working one
    // step further out: at `k == -1.0` the denominator is exactly zero
    // (infinite scores) and below that it is negative, which **inverts the
    // ordering** while every score still looks like a score.
    let k = if options.k.is_finite() {
        options.k.max(0.0)
    } else {
        DEFAULT_K
    };

    for list in lists {
        for (rank_from_zero, id) in list.iter().enumerate() {
            let contribution = 1.0 / (k + rank_from_zero as f32 + 1.0);
            match totals.get_mut(id.as_str()) {
                Some(total) => *total += contribution,
                None => {
                    totals.insert(id.as_str(), contribution);
                    order.push(id.as_str());
                }
            }
        }
    }

    let mut fused: Vec<(String, f32)> = order
        .into_iter()
        .map(|id| (id.to_string(), totals.get(id).copied().unwrap_or(0.0)))
        .collect();

    // Stable, so equal scores keep first-seen order.
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if options.normalise {
        if let Some(best) = fused.first().map(|(_, score)| *score) {
            if best > 0.0 {
                for (_, score) in &mut fused {
                    *score /= best;
                }
            }
        }
    }

    fused
}

/// The same, as a lookup from identifier to score.
///
/// For a caller that already holds its documents and only needs the new score
/// to write back into them.
pub fn fuse_ranked_ids_to_map(lists: &[Vec<String>], options: &RrfOptions) -> HashMap<String, f32> {
    fuse_ranked_ids(lists, options).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn agreement_between_lists_beats_a_strong_showing_in_one() {
        // The property RRF exists for. "b" is second in both lists and wins
        // against "a", which is first in one and absent from the other.
        let fused = fuse_ranked_ids(
            &[ids(&["a", "b", "c"]), ids(&["d", "b", "e"])],
            &RrfOptions::default(),
        );
        assert_eq!(fused[0].0, "b", "{fused:?}");
    }

    #[test]
    fn the_scores_land_on_the_scale_the_rest_of_the_library_uses() {
        // THE defect this module exists for. Raw RRF with k=60 tops out at
        // 1/61 ≈ 0.0164, and `min_relevance_score` is 0.1, so an un-normalised
        // list is discarded in full by the very next stage.
        let raw = fuse_ranked_ids(&[ids(&["a", "b"])], &RrfOptions::raw(DEFAULT_K));
        assert!(
            raw[0].1 < 0.1,
            "raw RRF is supposed to be under the relevance floor; if it is not, \
             the reason this module normalises has changed: {raw:?}"
        );

        let fused = fuse_ranked_ids(&[ids(&["a", "b"])], &RrfOptions::default());
        assert!((fused[0].1 - 1.0).abs() < f32::EPSILON, "{fused:?}");
        assert!(
            fused.iter().all(|(_, s)| (0.0..=1.0).contains(s)),
            "{fused:?}"
        );
    }

    #[test]
    fn the_canonical_formula_is_pinned_to_its_exact_values() {
        // V392 measured that NOT ONE of the three modules pinned an absolute
        // RRF value, so the formula could be reparameterised and no test would
        // object. The consumers asserting only ORDER is right -- the output is
        // normalised and the raw scale is an internal detail -- but *nobody*
        // asserting it is not. The canonical implementation owns that job.
        //
        // `1 / (k + rank)`, ranks from ONE. With k = 0 that is the harmonic
        // series, which makes the expected values readable instead of a
        // magic constant: 1/1, 1/2, 1/3, 1/4.
        let raw = fuse_ranked_ids(&[ids(&["a", "b", "c", "d"])], &RrfOptions::raw(0.0));
        for (i, (_, score)) in raw.iter().enumerate() {
            let want = 1.0 / (i as f32 + 1.0);
            assert!(
                (score - want).abs() < 1e-6,
                "rank {} should contribute exactly {want}, got {score} -- {raw:?}",
                i + 1
            );
        }

        // And at the documented default, the exact paper value: 1/(60+1).
        let at_default = fuse_ranked_ids(&[ids(&["a"])], &RrfOptions::raw(DEFAULT_K));
        assert!(
            (at_default[0].1 - 1.0 / 61.0).abs() < 1e-9,
            "k = 60 rank 1 is 1/61, got {:?}",
            at_default[0].1
        );

        // Two lists agreeing on one document sum, they do not max.
        let both = fuse_ranked_ids(&[ids(&["a"]), ids(&["a"])], &RrfOptions::raw(DEFAULT_K));
        assert!(
            (both[0].1 - 2.0 / 61.0).abs() < 1e-9,
            "agreement adds: 2/61, got {:?}",
            both[0].1
        );
    }

    #[test]
    fn a_k_outside_its_domain_is_clamped_rather_than_dividing_by_zero() {
        // The author's objection, and it was right: leaning on the `+ 1.0` to
        // keep the denominator off zero only works at exactly k == 0 and breaks
        // one step further out. These are the inputs that used to produce
        // infinities and inverted orderings, and `RrfOptions::k` is a public
        // field, so they are reachable by assignment even though
        // `#[non_exhaustive]` blocks a struct literal from outside.
        let wanted = ids(&["a", "b", "c"]);
        for bad in [
            -1.0f32,
            -2.0,
            -60.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ] {
            let mut options = RrfOptions::raw(DEFAULT_K);
            options.k = bad; // the route a constructor guard would not cover
            let got = fuse_ranked_ids(std::slice::from_ref(&wanted), &options);

            assert_eq!(got.len(), 3, "k = {bad} lost documents: {got:?}");
            assert!(
                got.iter().all(|(_, s)| s.is_finite() && *s > 0.0),
                "k = {bad} produced a non-finite or non-positive score: {got:?}"
            );
            // Order must still be the input order: a negative denominator used
            // to invert it while every score still looked like a score.
            let order: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(order, vec!["a", "b", "c"], "k = {bad} reordered: {got:?}");
        }
    }

    #[test]
    fn a_k_of_zero_is_finite_because_ranks_start_at_one() {
        // Found by a mutation (V392): dropping the `+ 1.0` from
        // `1 / (k + rank + 1)` survived every test in this module AND in both
        // consumers, because it is a monotone shift — it changes every absolute
        // score and reorders nothing, the output is normalised so the top is
        // 1.0 either way, and the one raw assertion is an inequality with a 6x
        // margin (0.0164 against a 0.1 floor).
        //
        // There is exactly one input where the `+ 1.0` is not cosmetic: `k = 0`,
        // where removing it divides by zero. `with_k` and `raw` are public and
        // take any `f32`, so that is reachable from outside the crate — and
        // nothing pinned it.
        let raw = fuse_ranked_ids(&[ids(&["a", "b", "c"])], &RrfOptions::raw(0.0));
        assert!(
            raw.iter().all(|(_, s)| s.is_finite()),
            "k = 0 must stay finite: ranks count from one, so the smallest \
             denominator is 1 and not 0 — {raw:?}"
        );
        // The first rank is the undiscounted one: 1 / (0 + 0 + 1) = 1.0.
        assert!(
            (raw[0].1 - 1.0).abs() < f32::EPSILON,
            "at k = 0 the first rank contributes exactly 1.0, got {raw:?}"
        );
        // And the shape still decays: 1/1, 1/2, 1/3.
        assert!((raw[1].1 - 0.5).abs() < f32::EPSILON, "{raw:?}");
        assert!((raw[2].1 - 1.0 / 3.0).abs() < 1e-6, "{raw:?}");
    }

    #[test]
    fn normalising_does_not_reorder() {
        // All RRF determines is the ordering. Dividing by a positive constant
        // must leave it untouched, or the fix would be changing the answer
        // rather than its scale.
        let lists = [ids(&["a", "b", "c"]), ids(&["c", "a", "d"])];
        let raw = fuse_ranked_ids(&lists, &RrfOptions::raw(DEFAULT_K));
        let normalised = fuse_ranked_ids(&lists, &RrfOptions::default());

        let raw_order: Vec<&str> = raw.iter().map(|(id, _)| id.as_str()).collect();
        let normalised_order: Vec<&str> = normalised.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(raw_order, normalised_order);
    }

    #[test]
    fn it_takes_as_many_lists_as_it_is_given() {
        // Not two. The pipeline's copy had "keyword and semantic" baked into
        // its shape, so a third source meant rewriting it.
        let fused = fuse_ranked_ids(
            &[
                ids(&["x", "a"]),
                ids(&["y", "a"]),
                ids(&["z", "a"]),
                ids(&["w", "a"]),
            ],
            &RrfOptions::default(),
        );
        assert_eq!(fused[0].0, "a", "four lists agreed and it lost: {fused:?}");
        assert_eq!(fused.len(), 5);
    }

    #[test]
    fn a_larger_k_flattens_the_difference_between_ranks() {
        // What k is for, pinned so nobody "simplifies" it into a constant with
        // no meaning.
        let lists = [ids(&["first", "second"])];
        let spread = |k: f32| {
            let f = fuse_ranked_ids(&lists, &RrfOptions::raw(k));
            f[0].1 - f[1].1
        };
        assert!(
            spread(5.0) > spread(60.0),
            "a small k should separate the ranks more than a large one"
        );
    }

    #[test]
    fn ties_are_broken_the_same_way_every_time() {
        // Two identifiers with identical scores must not swap between runs.
        // Ordering that depends on hashing gets blamed on the model.
        let lists = [ids(&["a", "b"]), ids(&["a", "b"])];
        let first = fuse_ranked_ids(&lists, &RrfOptions::default());
        for _ in 0..20 {
            assert_eq!(fuse_ranked_ids(&lists, &RrfOptions::default()), first);
        }
    }

    #[test]
    fn nothing_in_produces_nothing_out_rather_than_a_panic() {
        assert!(fuse_ranked_ids(&[], &RrfOptions::default()).is_empty());
        assert!(fuse_ranked_ids(&[Vec::new()], &RrfOptions::default()).is_empty());
    }

    #[test]
    fn a_repeated_id_within_one_list_does_not_count_twice_as_a_new_document() {
        // A caller that hands in a list with duplicates gets one entry, not
        // two: the identifier is the document.
        let fused = fuse_ranked_ids(&[ids(&["a", "a", "b"])], &RrfOptions::default());
        assert_eq!(fused.len(), 2, "{fused:?}");
    }

    #[test]
    fn the_map_form_agrees_with_the_list_form() {
        let lists = [ids(&["a", "b"]), ids(&["b", "c"])];
        let list = fuse_ranked_ids(&lists, &RrfOptions::default());
        let map = fuse_ranked_ids_to_map(&lists, &RrfOptions::default());
        assert_eq!(list.len(), map.len());
        for (id, score) in list {
            assert_eq!(map.get(&id), Some(&score));
        }
    }
}
