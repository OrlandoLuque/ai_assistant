//! Prompt Synthesis — contextual bandit over prompt-fragment combinations.
//!
//! Queries are embedded and bucketed into `IntentCluster`s. For each cluster
//! the system keeps a set of `PromptArm`s (candidate fragment compositions).
//! `FragmentBandit` selects an arm using Bayesian UCB with a 5% ε-random
//! safety floor, and records the outcome against a hash-chained ledger.
//!
//! Arms are segmented by `provider_fingerprint` so the system does not mix
//! reward signal from a fast local model with a slow cloud one.
//!
//! # Freeze
//!
//! When `LearningFreezeConfig::freeze_fragment_synthesis` is true, the bandit
//! keeps serving selection (queries) but rejects reward updates and new
//! cluster/arm creation. See `learning_control::LearningSubsystem::FragmentSynthesis`.
//!
//! # Opt-in feature
//!
//! Behind `prompt-synthesis`. Not in `full`. See `Cargo.toml` for the
//! dependency surface (`prompt-fragments`, `embeddings`, `advanced-memory`).

pub mod arm;
pub mod bandit;
pub mod exploration;
pub mod intent;
pub mod ledger;
pub mod reward;

pub use arm::{PromptArm, PromptArmId, ProviderFingerprint};
pub use bandit::{
    ArmSelection, BanditError, BanditStats, FragmentBandit, FragmentBanditConfig, SelectionReason,
};
pub use exploration::{
    ArmProposer, EpsilonRandomProposer, ExplorationControl, ExplorationError, RejectProposer,
};
pub use intent::{
    IntentCluster, IntentClusterId, IntentClusterManager, IntentClusterManagerConfig,
    IntentEmbedding,
};
pub use ledger::{FragmentEvent, FragmentEventKind, FragmentLedger, FragmentLedgerError};
pub use reward::{RewardPolicy, RewardPolicyConfig, RewardSignal};

/// Default tuning constants. Copied from `V96_self_learning_3phases.md`.
pub mod defaults {
    /// Minimum clusters kept by the adaptive manager.
    pub const MIN_CLUSTERS: usize = 1;
    /// Maximum clusters kept by the adaptive manager.
    pub const MAX_CLUSTERS: usize = 64;
    /// ε-random exploration floor — safety net vs local minima.
    pub const EPSILON_RANDOM: f32 = 0.05;
    /// Default Bayesian prior strength (α=β=1, uniform on [0,1]).
    pub const PRIOR_ALPHA: f32 = 1.0;
    pub const PRIOR_BETA: f32 = 1.0;
    /// UCB exploration constant.
    pub const UCB_C: f32 = 1.4142;
}
