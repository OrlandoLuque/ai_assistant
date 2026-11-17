//! Per-instance cache of the detected context window, keyed by the model it
//! was detected for.
//!
//! The third cluster pulled out of the god-struct in Phase 5 of
//! `ai_assistant_plans/HEXAGONAL_PLAN.md`, after `KnowledgeStore` (V459) and
//! `ProceduralMemory` (V460). Both fields were **private**, so like the
//! procedural cluster this one costs no API break: the facade's
//! `detect_model_context_size`, `get_model_context_size` and
//! `invalidate_context_cache` keep their names and signatures and delegate.
//!
//! # Why this is a component and not two fields
//!
//! The facade held `detected_context_size: Option<usize>` next to
//! `detected_context_model: Option<String>`, and the rule between them — *the
//! size is only valid for the model it was detected for* — was enforced by
//! whoever read them. Four sites read them and **two of the four did not check
//! the model**: `context_budget_status` and the context-budget allocation in
//! `assistant/memory.rs` both took `self.detected_context_size` on its own,
//! because they are `&self` methods and the key-checking path
//! (`get_model_context_size`) needs `&mut self`.
//!
//! That was reachable: `config` is a public field, seven of this crate's own
//! binaries assign `config.selected_model` at runtime, and
//! `invalidate_context_cache` — whose documentation says "call when model
//! changes" — had **zero callers outside tests**. So after a model switch those
//! two sites answered with the previous model's window: a 4K model could be
//! told it had 128K to spend, and the Butler advisor's "context N% full"
//! recommendation was computed on that number.
//!
//! Here the pair is **one** `Option<(String, usize)>`, so it cannot be half
//! set, and the only way to read it is [`ContextSizeCache::get`], which takes
//! the model and returns `None` when it does not match. The stale read is not
//! fixed in two places; it is unreachable.

/// The context window detected for one model, or nothing.
///
/// One field, not two: "a size for a model" is a single fact, and splitting it
/// across two `Option`s is what let half of it be read alone.
#[derive(Debug, Clone, Default)]
pub(crate) struct ContextSizeCache {
    /// The model the size was detected for, and the size. `None` means nothing
    /// has been detected yet, or the cache was invalidated.
    entry: Option<(String, usize)>,
}

impl ContextSizeCache {
    /// An empty cache.
    pub(crate) fn new() -> Self {
        Self { entry: None }
    }

    /// The cached size **for this model**, or `None` if the cache is empty or
    /// holds another model's size.
    ///
    /// The comparison is exact, as it was when this lived on the facade. A
    /// case-only difference is a miss here and a hit in the process-wide cache
    /// of `crate::context::get_model_context_size_cached`, which lowercases its
    /// key — so the caller gets the same number, one map lookup later, and
    /// never a size belonging to a different model.
    pub(crate) fn get(&self, model: &str) -> Option<usize> {
        match &self.entry {
            Some((cached_model, size)) if cached_model == model => Some(*size),
            _ => None,
        }
    }

    /// Remember `size` as the window of `model`, replacing whatever was there.
    pub(crate) fn store(&mut self, model: String, size: usize) {
        self.entry = Some((model, size));
    }

    /// Forget the cached size.
    ///
    /// Not needed for correctness any more — [`get`](Self::get) cannot return
    /// another model's size — but it still forces the next detection to go
    /// through the provider API instead of answering from this instance.
    pub(crate) fn invalidate(&mut self) {
        self.entry = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_cache_answers_nothing_for_any_model() {
        let c = ContextSizeCache::new();
        assert_eq!(c.get("llama3.2:latest"), None);
        assert_eq!(c.get(""), None);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(ContextSizeCache::default().get("llama3.2:latest"), None);
    }

    #[test]
    fn a_stored_size_is_returned_for_its_own_model() {
        let mut c = ContextSizeCache::new();
        c.store("llama3.2:latest".to_string(), 128_000);
        assert_eq!(c.get("llama3.2:latest"), Some(128_000));
    }

    #[test]
    fn another_models_size_is_never_returned() {
        // The invariant this type exists for. On the facade this check was
        // written at two of four read sites, and the two that skipped it
        // reported the previous model's window after a model switch.
        let mut c = ContextSizeCache::new();
        c.store("llama3.2:latest".to_string(), 128_000);
        assert_eq!(
            c.get("llama2:7b"),
            None,
            "a size detected for another model must not be reachable"
        );
    }

    #[test]
    fn the_comparison_is_exact_so_a_case_difference_misses() {
        // Documented behaviour, not an accident: a miss here falls through to
        // the process-wide cache, which lowercases and answers correctly.
        let mut c = ContextSizeCache::new();
        c.store("Llama3.2:Latest".to_string(), 128_000);
        assert_eq!(c.get("llama3.2:latest"), None);
    }

    #[test]
    fn storing_replaces_the_previous_model() {
        let mut c = ContextSizeCache::new();
        c.store("llama3.2:latest".to_string(), 128_000);
        c.store("llama2:7b".to_string(), 4_096);
        assert_eq!(c.get("llama2:7b"), Some(4_096));
        assert_eq!(
            c.get("llama3.2:latest"),
            None,
            "the cache holds one model, so the old one must be gone"
        );
    }

    #[test]
    fn invalidating_empties_it() {
        let mut c = ContextSizeCache::new();
        c.store("llama3.2:latest".to_string(), 128_000);
        c.invalidate();
        assert_eq!(c.get("llama3.2:latest"), None);
    }
}
