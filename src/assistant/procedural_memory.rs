//! Procedural-memory state for the assistant facade.
//!
//! The second cluster pulled out of the god-struct in Phase 5 of
//! `ai_assistant_plans/HEXAGONAL_PLAN.md`, after `KnowledgeStore` (V459). All
//! three fields were **private**, so unlike the RAG cluster this one costs no
//! API break: the facade's public methods keep their names and signatures and
//! simply delegate.
//!
//! The reason to extract it is the invariant, not the field count. "Procedural
//! memory is on" means **two** fields are `Some` together: a `ProceduralStore`
//! to hold the procedures and a `ProcedureEvolver` to track their confidence
//! from outcomes. That pairing was kept by hand in two separate places on the
//! facade, each building the evolver with its own copy of
//! `ProcedureEvolver::new(EvolutionConfig::default())` — and one of them
//! (`load_procedures`) had to remember to check whether the evolver already
//! existed. Here it is one method, so there is one place that can be wrong.

use crate::advanced_memory::{EvolutionConfig, ProceduralStore, ProcedureEvolver};

/// Procedures the assistant injects into the system prompt, and their feedback.
///
/// Fields are `pub(crate)` because the methods that read them are
/// `impl AiAssistant` blocks in sibling modules; see `KnowledgeStore` for the
/// same reasoning. What is *not* `pub(crate)` is the pairing rule: the only way
/// to turn this on is [`enable`](Self::enable) or [`adopt`](Self::adopt), and
/// both leave the store and the evolver consistent.
pub(crate) struct ProceduralMemory {
    /// Workflow procedures, checklists and methodologies. `Some` means
    /// procedural memory is enabled.
    pub(crate) store: Option<ProceduralStore>,
    /// Confidence tracking for those procedures via outcome feedback. `Some`
    /// whenever `store` is `Some`.
    pub(crate) evolver: Option<ProcedureEvolver>,
    /// IDs of the procedures injected in the current turn, for outcome
    /// tracking. Empty when nothing was injected, and cleared on `disable`.
    pub(crate) active_ids: Vec<String>,
}

impl Default for ProceduralMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ProceduralMemory {
    /// Disabled: no store, no evolver, nothing injected.
    pub(crate) fn new() -> Self {
        Self {
            store: None,
            evolver: None,
            active_ids: Vec::new(),
        }
    }

    /// Enable with a fresh, empty store of the given capacity.
    pub(crate) fn enable(&mut self, max_procedures: usize) {
        self.store = Some(ProceduralStore::new(max_procedures));
        self.ensure_evolver();
    }

    /// Enable with a store that was built elsewhere (e.g. loaded from a file).
    ///
    /// Separate from [`enable`](Self::enable) because the caller already has
    /// the procedures and must not lose them to a fresh store.
    pub(crate) fn adopt(&mut self, store: ProceduralStore) {
        self.store = Some(store);
        self.ensure_evolver();
    }

    /// Discard everything, including the ids injected this turn.
    ///
    /// The `active_ids` clear is the part that is easy to forget: leaving them
    /// behind would let an outcome report arrive for a procedure that no longer
    /// exists.
    pub(crate) fn disable(&mut self) {
        self.store = None;
        self.evolver = None;
        self.active_ids.clear();
    }

    /// Whether procedural memory is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        self.store.is_some()
    }

    /// The one place that builds an evolver, and the one that decides it must
    /// exist whenever a store does.
    fn ensure_evolver(&mut self) {
        if self.evolver.is_none() {
            self.evolver = Some(ProcedureEvolver::new(EvolutionConfig::default()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_component_is_disabled() {
        let p = ProceduralMemory::new();
        assert!(!p.is_enabled());
        assert!(p.store.is_none());
        assert!(p.evolver.is_none());
        assert!(p.active_ids.is_empty());
    }

    #[test]
    fn enabling_pairs_the_store_with_an_evolver() {
        // The invariant this type exists for: "enabled" is never half-enabled.
        let mut p = ProceduralMemory::new();
        p.enable(8);
        assert!(p.is_enabled());
        assert!(
            p.evolver.is_some(),
            "a store without an evolver cannot take outcome feedback"
        );
    }

    #[test]
    fn adopting_a_store_keeps_it_and_still_pairs_an_evolver() {
        let mut p = ProceduralMemory::new();
        let mut store = ProceduralStore::new(4);
        store.add(crate::advanced_memory::Procedure {
            id: "p1".to_string(),
            name: "Deploy".to_string(),
            condition: "deploy".to_string(),
            steps: vec!["step one".to_string()],
            ..Default::default()
        });
        p.adopt(store);
        assert!(p.evolver.is_some());
        assert_eq!(
            p.store.as_ref().map(|s| s.len()),
            Some(1),
            "adopt must not replace the caller's procedures with an empty store"
        );
    }

    #[test]
    fn adopting_twice_does_not_replace_a_live_evolver() {
        // `load_procedures` can be called on an already-enabled assistant. The
        // evolver holds accumulated confidence, so rebuilding it would silently
        // throw that away — which is why `ensure_evolver` checks first.
        let mut p = ProceduralMemory::new();
        p.enable(4);
        let first = p.evolver.as_ref().map(|e| e as *const _);
        p.adopt(ProceduralStore::new(4));
        let second = p.evolver.as_ref().map(|e| e as *const _);
        assert_eq!(first, second, "the evolver must be the same instance");
    }

    #[test]
    fn disabling_clears_the_active_ids_too() {
        // Not cosmetic: an outcome report for a procedure that no longer exists
        // is the bug this clear prevents.
        let mut p = ProceduralMemory::new();
        p.enable(4);
        p.active_ids = vec!["p1".to_string(), "p2".to_string()];
        p.disable();
        assert!(!p.is_enabled());
        assert!(p.evolver.is_none());
        assert!(
            p.active_ids.is_empty(),
            "ids of procedures that no longer exist must not survive"
        );
    }

    #[test]
    fn default_is_disabled_like_new() {
        let p = ProceduralMemory::default();
        assert!(!p.is_enabled());
        assert!(p.active_ids.is_empty());
    }
}
