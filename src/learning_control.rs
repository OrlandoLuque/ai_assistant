//! Learning Control — freeze, validate, and manage learning subsystems.
//!
//! Allows freezing/unfreezing learning for each subsystem independently.
//! When frozen, the subsystem accepts queries but rejects updates.

use serde::{Deserialize, Serialize};

/// Configuration for freezing learning subsystems.
///
/// When a subsystem is frozen, it continues to serve queries (read)
/// but rejects all updates (write). This is useful for:
/// - Production: freeze after validating learned state
/// - Security: prevent poisoning attacks on learning
/// - Debug: reproducible behavior
/// - Import: freeze before importing untrusted data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LearningFreezeConfig {
    /// Freeze all learning (overrides individual flags).
    pub freeze_all: bool,
    /// StrategyBandit — don't update arm rewards.
    pub freeze_bandit: bool,
    /// ProcedureEvolver — don't modify confidence.
    pub freeze_procedures: bool,
    /// Entity FactStore — don't reinforce or evict.
    pub freeze_entity_facts: bool,
    /// Semantic FactStore (consolidation) — don't accumulate confidence.
    pub freeze_semantic_facts: bool,
    /// Distributed reputation — don't change scores.
    pub freeze_reputation: bool,
    /// Auto model selector — don't record outcomes.
    pub freeze_model_selection: bool,
    /// MultiLayerGraph User layer — read-only beliefs.
    pub freeze_user_beliefs: bool,
    /// RAG tier auto-selection — don't auto-adjust.
    pub freeze_rag_tiers: bool,
    /// V96 F1: Skill Forge — don't register new skills or change status.
    pub freeze_skill_forge: bool,
    /// V96 F2: Fragment Synthesis — FragmentBandit accepts queries but
    /// rejects reward updates; cluster manager rejects new clusters.
    pub freeze_fragment_synthesis: bool,
    /// V96 F3: Feedback Loop — dispatcher drops new records (counts in
    /// metrics) without feeding any downstream sink.
    pub freeze_feedback_loop: bool,
}

impl LearningFreezeConfig {
    /// Everything unfrozen (default — learning is active).
    pub fn all_active() -> Self {
        Self {
            freeze_all: false,
            freeze_bandit: false,
            freeze_procedures: false,
            freeze_entity_facts: false,
            freeze_semantic_facts: false,
            freeze_reputation: false,
            freeze_model_selection: false,
            freeze_user_beliefs: false,
            freeze_rag_tiers: false,
            freeze_skill_forge: false,
            freeze_fragment_synthesis: false,
            freeze_feedback_loop: false,
        }
    }

    /// Everything frozen — no learning at all.
    pub fn all_frozen() -> Self {
        Self {
            freeze_all: true,
            ..Self::all_active()
        }
    }

    /// Check if a specific subsystem is frozen.
    pub fn is_frozen(&self, subsystem: LearningSubsystem) -> bool {
        if self.freeze_all {
            return true;
        }
        match subsystem {
            LearningSubsystem::Bandit => self.freeze_bandit,
            LearningSubsystem::Procedures => self.freeze_procedures,
            LearningSubsystem::EntityFacts => self.freeze_entity_facts,
            LearningSubsystem::SemanticFacts => self.freeze_semantic_facts,
            LearningSubsystem::Reputation => self.freeze_reputation,
            LearningSubsystem::ModelSelection => self.freeze_model_selection,
            LearningSubsystem::UserBeliefs => self.freeze_user_beliefs,
            LearningSubsystem::RagTiers => self.freeze_rag_tiers,
            LearningSubsystem::SkillForge => self.freeze_skill_forge,
            LearningSubsystem::FragmentSynthesis => self.freeze_fragment_synthesis,
            LearningSubsystem::FeedbackLoop => self.freeze_feedback_loop,
        }
    }

    /// Freeze a specific subsystem.
    pub fn freeze(&mut self, subsystem: LearningSubsystem) {
        match subsystem {
            LearningSubsystem::Bandit => self.freeze_bandit = true,
            LearningSubsystem::Procedures => self.freeze_procedures = true,
            LearningSubsystem::EntityFacts => self.freeze_entity_facts = true,
            LearningSubsystem::SemanticFacts => self.freeze_semantic_facts = true,
            LearningSubsystem::Reputation => self.freeze_reputation = true,
            LearningSubsystem::ModelSelection => self.freeze_model_selection = true,
            LearningSubsystem::UserBeliefs => self.freeze_user_beliefs = true,
            LearningSubsystem::RagTiers => self.freeze_rag_tiers = true,
            LearningSubsystem::SkillForge => self.freeze_skill_forge = true,
            LearningSubsystem::FragmentSynthesis => self.freeze_fragment_synthesis = true,
            LearningSubsystem::FeedbackLoop => self.freeze_feedback_loop = true,
        }
    }

    /// Unfreeze a specific subsystem.
    pub fn unfreeze(&mut self, subsystem: LearningSubsystem) {
        match subsystem {
            LearningSubsystem::Bandit => self.freeze_bandit = false,
            LearningSubsystem::Procedures => self.freeze_procedures = false,
            LearningSubsystem::EntityFacts => self.freeze_entity_facts = false,
            LearningSubsystem::SemanticFacts => self.freeze_semantic_facts = false,
            LearningSubsystem::Reputation => self.freeze_reputation = false,
            LearningSubsystem::ModelSelection => self.freeze_model_selection = false,
            LearningSubsystem::UserBeliefs => self.freeze_user_beliefs = false,
            LearningSubsystem::RagTiers => self.freeze_rag_tiers = false,
            LearningSubsystem::SkillForge => self.freeze_skill_forge = false,
            LearningSubsystem::FragmentSynthesis => self.freeze_fragment_synthesis = false,
            LearningSubsystem::FeedbackLoop => self.freeze_feedback_loop = false,
        }
    }

    /// Count of frozen subsystems.
    pub fn frozen_count(&self) -> usize {
        if self.freeze_all {
            return 11;
        }
        [
            self.freeze_bandit,
            self.freeze_procedures,
            self.freeze_entity_facts,
            self.freeze_semantic_facts,
            self.freeze_reputation,
            self.freeze_model_selection,
            self.freeze_user_beliefs,
            self.freeze_rag_tiers,
            self.freeze_skill_forge,
            self.freeze_fragment_synthesis,
            self.freeze_feedback_loop,
        ]
        .iter()
        .filter(|&&f| f)
        .count()
    }

    /// List of frozen subsystem names.
    pub fn frozen_list(&self) -> Vec<&'static str> {
        let all = [
            (self.freeze_bandit, "bandit"),
            (self.freeze_procedures, "procedures"),
            (self.freeze_entity_facts, "entity_facts"),
            (self.freeze_semantic_facts, "semantic_facts"),
            (self.freeze_reputation, "reputation"),
            (self.freeze_model_selection, "model_selection"),
            (self.freeze_user_beliefs, "user_beliefs"),
            (self.freeze_rag_tiers, "rag_tiers"),
            (self.freeze_skill_forge, "skill_forge"),
            (self.freeze_fragment_synthesis, "fragment_synthesis"),
            (self.freeze_feedback_loop, "feedback_loop"),
        ];
        if self.freeze_all {
            return all.iter().map(|(_, name)| *name).collect();
        }
        all.iter()
            .filter(|(frozen, _)| *frozen)
            .map(|(_, name)| *name)
            .collect()
    }
}

impl Default for LearningFreezeConfig {
    fn default() -> Self {
        Self::all_active()
    }
}

/// Identifiers for learning subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LearningSubsystem {
    Bandit,
    Procedures,
    EntityFacts,
    SemanticFacts,
    Reputation,
    ModelSelection,
    UserBeliefs,
    RagTiers,
    /// V96 F1: Skill Forge.
    SkillForge,
    /// V96 F2: Fragment Synthesis.
    FragmentSynthesis,
    /// V96 F3: Feedback Loop.
    FeedbackLoop,
}

impl std::fmt::Display for LearningSubsystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bandit => write!(f, "bandit"),
            Self::Procedures => write!(f, "procedures"),
            Self::EntityFacts => write!(f, "entity_facts"),
            Self::SemanticFacts => write!(f, "semantic_facts"),
            Self::Reputation => write!(f, "reputation"),
            Self::ModelSelection => write!(f, "model_selection"),
            Self::UserBeliefs => write!(f, "user_beliefs"),
            Self::RagTiers => write!(f, "rag_tiers"),
            Self::SkillForge => write!(f, "skill_forge"),
            Self::FragmentSynthesis => write!(f, "fragment_synthesis"),
            Self::FeedbackLoop => write!(f, "feedback_loop"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_all_active() {
        let config = LearningFreezeConfig::default();
        assert!(!config.is_frozen(LearningSubsystem::Bandit));
        assert!(!config.is_frozen(LearningSubsystem::Procedures));
        assert_eq!(config.frozen_count(), 0);
    }

    #[test]
    fn test_all_frozen() {
        let config = LearningFreezeConfig::all_frozen();
        assert!(config.is_frozen(LearningSubsystem::Bandit));
        assert!(config.is_frozen(LearningSubsystem::Procedures));
        assert!(config.is_frozen(LearningSubsystem::Reputation));
        assert!(config.is_frozen(LearningSubsystem::SkillForge));
        assert!(config.is_frozen(LearningSubsystem::FragmentSynthesis));
        assert!(config.is_frozen(LearningSubsystem::FeedbackLoop));
        assert_eq!(config.frozen_count(), 11);
    }

    #[test]
    fn test_freeze_individual() {
        let mut config = LearningFreezeConfig::default();
        config.freeze(LearningSubsystem::Bandit);
        config.freeze(LearningSubsystem::Procedures);

        assert!(config.is_frozen(LearningSubsystem::Bandit));
        assert!(config.is_frozen(LearningSubsystem::Procedures));
        assert!(!config.is_frozen(LearningSubsystem::Reputation));
        assert_eq!(config.frozen_count(), 2);
    }

    #[test]
    fn test_unfreeze() {
        let mut config = LearningFreezeConfig::all_frozen();
        // freeze_all overrides — unfreezing individual has no effect
        config.unfreeze(LearningSubsystem::Bandit);
        assert!(config.is_frozen(LearningSubsystem::Bandit)); // still frozen due to freeze_all

        // Clear freeze_all first
        config.freeze_all = false;
        config.unfreeze(LearningSubsystem::Bandit);
        assert!(!config.is_frozen(LearningSubsystem::Bandit));
    }

    #[test]
    fn test_frozen_list() {
        let mut config = LearningFreezeConfig::default();
        config.freeze(LearningSubsystem::Bandit);
        config.freeze(LearningSubsystem::EntityFacts);

        let list = config.frozen_list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"bandit"));
        assert!(list.contains(&"entity_facts"));
    }

    #[test]
    fn test_frozen_list_all() {
        let config = LearningFreezeConfig::all_frozen();
        assert_eq!(config.frozen_list().len(), 11);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut config = LearningFreezeConfig::default();
        config.freeze(LearningSubsystem::Bandit);
        config.freeze(LearningSubsystem::Reputation);

        let json = serde_json::to_string(&config).unwrap();
        let restored: LearningFreezeConfig = serde_json::from_str(&json).unwrap();

        assert!(restored.is_frozen(LearningSubsystem::Bandit));
        assert!(restored.is_frozen(LearningSubsystem::Reputation));
        assert!(!restored.is_frozen(LearningSubsystem::Procedures));
    }
}
