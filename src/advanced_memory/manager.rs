//! Unified manager that owns all three memory stores plus the consolidator.

use crate::error::AiError;
use super::episodic::{Episode, EpisodicStore};
use super::procedural::{Procedure, ProceduralStore};
use super::entity::{EntityRecord, EntityStore};
use super::consolidation::{ConsolidationResult, MemoryConsolidator};

/// Unified manager that owns all three memory stores plus the consolidator.
pub struct AdvancedMemoryManager {
    pub episodic: EpisodicStore,
    pub procedural: ProceduralStore,
    pub entities: EntityStore,
    pub consolidator: MemoryConsolidator,
}

impl AdvancedMemoryManager {
    /// Create a manager with sensible defaults (1000 episodes, 500 procedures,
    /// 0.001 decay).
    pub fn new() -> Self {
        Self {
            episodic: EpisodicStore::new(1000, 0.001),
            procedural: ProceduralStore::new(500),
            entities: EntityStore::new(),
            consolidator: MemoryConsolidator::new(),
        }
    }

    /// Create a manager with custom capacity and decay configuration.
    pub fn with_config(episodic_max: usize, procedural_max: usize, decay: f64) -> Self {
        Self {
            episodic: EpisodicStore::new(episodic_max, decay),
            procedural: ProceduralStore::new(procedural_max),
            entities: EntityStore::new(),
            consolidator: MemoryConsolidator::new(),
        }
    }

    /// Add an episode to the episodic store.
    pub fn add_episode(&mut self, episode: Episode) {
        self.episodic.add(episode);

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::EpisodicMemory,
            self.episodic.len(),
        );
    }

    /// Add a procedure to the procedural store.
    pub fn add_procedure(&mut self, procedure: Procedure) {
        self.procedural.add(procedure);
    }

    /// Add an entity record.
    pub fn add_entity(&mut self, record: EntityRecord) -> Result<(), AiError> {
        let result = self.entities.add(record);

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::EntityStore,
            self.entities.len(),
        );

        result
    }

    /// Run consolidation on all current episodes and add resulting procedures
    /// to the procedural store.
    pub fn consolidate(&mut self) -> ConsolidationResult {
        let episodes = self.episodic.all().to_vec();
        let result = self.consolidator.consolidate(&episodes);
        for proc in &result.procedures_created {
            self.procedural.add(proc.clone());
        }
        result
    }

    /// Recall episodes by embedding similarity.
    pub fn recall_episodes(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<Episode> {
        self.episodic.recall(query_embedding, top_k)
    }

    /// Find procedures matching a context.
    pub fn find_procedures(&self, context: &str) -> Vec<&Procedure> {
        self.procedural.find_by_condition(context)
    }

    /// Find an entity by name.
    pub fn find_entity(&self, name: &str) -> Option<&EntityRecord> {
        self.entities.find_by_name(name)
    }
}
