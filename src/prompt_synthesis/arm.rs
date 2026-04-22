//! `PromptArm` — a candidate fragment composition for a given cluster.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable identifier for a prompt arm. Typically a short slug like
/// `arm_base_concise_v2` — opaque to the runtime, used only for ledger
/// correlation and display.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PromptArmId(pub(crate) String);

impl PromptArmId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PromptArmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Model+provider fingerprint used to segment arms. Reward from
/// `ollama/mistral:7b` should not be averaged with `claude-opus-4-7`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderFingerprint(String);

impl ProviderFingerprint {
    pub fn new(provider: &str, model: &str) -> Self {
        Self(format!("{provider}/{model}"))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A candidate prompt composition. `fragments` is an ordered list of
/// fragment identifiers the caller will resolve against their registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptArm {
    /// Stable id for ledger correlation.
    pub id: PromptArmId,
    /// Ordered fragment identifiers composing this prompt.
    pub fragments: Vec<String>,
    /// Segments reward: `ollama/mistral:7b` vs `claude-opus-4-7` etc.
    pub provider: ProviderFingerprint,
    /// Number of reward samples recorded so far. Incremented on each
    /// `FragmentBandit::record_reward`. Capped by `u64::MAX` in practice.
    pub samples: u64,
    /// Sum of reward signals. Average = `reward_sum / samples` when samples>0.
    pub reward_sum: f64,
    /// Origin of this arm — manual, proposed by LLM, or via ε-random.
    pub origin: ArmOrigin,
    /// Marks an arm as retired. Retired arms are not selected but are kept
    /// for ledger/audit trails.
    pub retired: bool,
}

impl PromptArm {
    /// Convenience constructor for a fresh, unretired arm.
    pub fn new(
        id: PromptArmId,
        fragments: Vec<String>,
        provider: ProviderFingerprint,
        origin: ArmOrigin,
    ) -> Self {
        Self {
            id,
            fragments,
            provider,
            samples: 0,
            reward_sum: 0.0,
            origin,
            retired: false,
        }
    }

    /// Mean reward in `[0, 1]` under the assumption reward is normalized.
    /// Returns 0 if no samples have been recorded yet.
    pub fn mean_reward(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.reward_sum / self.samples as f64
        }
    }
}

/// Where an arm came from. Used by exploration policy + ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ArmOrigin {
    /// Curated by a human or seeded at deployment.
    Manual,
    /// Proposed by the LLM on-the-fly (via `ArmProposer`).
    LlmProposed,
    /// Generated via the ε-random floor — random combination of existing
    /// fragments. Kept as a separate origin so analytics can distinguish
    /// "real" exploration from safety-floor noise.
    EpsilonRandom,
}

impl fmt::Display for ArmOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manual => f.write_str("manual"),
            Self::LlmProposed => f.write_str("llm_proposed"),
            Self::EpsilonRandom => f.write_str("epsilon_random"),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_id_display_roundtrip() {
        let id = PromptArmId::new("arm_alpha");
        assert_eq!(id.as_str(), "arm_alpha");
        assert_eq!(id.to_string(), "arm_alpha");
    }

    #[test]
    fn provider_fingerprint_combines_fields() {
        let fp = ProviderFingerprint::new("ollama", "mistral:7b");
        assert_eq!(fp.as_str(), "ollama/mistral:7b");
    }

    #[test]
    fn new_arm_has_zero_samples() {
        let arm = PromptArm::new(
            PromptArmId::new("a"),
            vec!["f1".into()],
            ProviderFingerprint::new("o", "m"),
            ArmOrigin::Manual,
        );
        assert_eq!(arm.samples, 0);
        assert_eq!(arm.reward_sum, 0.0);
        assert_eq!(arm.mean_reward(), 0.0);
        assert!(!arm.retired);
    }

    #[test]
    fn mean_reward_handles_zero_samples() {
        let mut arm = PromptArm::new(
            PromptArmId::new("a"),
            vec![],
            ProviderFingerprint::new("o", "m"),
            ArmOrigin::Manual,
        );
        assert_eq!(arm.mean_reward(), 0.0);
        arm.samples = 4;
        arm.reward_sum = 2.0;
        assert!((arm.mean_reward() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn arm_origin_display_stable() {
        assert_eq!(ArmOrigin::Manual.to_string(), "manual");
        assert_eq!(ArmOrigin::LlmProposed.to_string(), "llm_proposed");
        assert_eq!(ArmOrigin::EpsilonRandom.to_string(), "epsilon_random");
    }
}
