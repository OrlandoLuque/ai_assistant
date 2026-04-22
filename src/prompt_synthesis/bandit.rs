//! `FragmentBandit` — Bayesian UCB over `PromptArm`s, segmented by cluster
//! and `ProviderFingerprint`.
//!
//! Algorithm: for each arm we keep a Beta(α, β) posterior over its expected
//! reward. Selection uses UCB1-style score
//!     score = μ + UCB_C · sqrt(ln(total_samples) / samples)
//! with `μ` being the Beta posterior mean. Unpulled arms get `score = +∞` so
//! every arm is tried once before exploitation kicks in.
//!
//! ε-random is applied from outside (`ExplorationControl`) — the bandit is
//! deterministic given its inputs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::arm::{PromptArm, PromptArmId, ProviderFingerprint};
use super::defaults::{PRIOR_ALPHA, PRIOR_BETA, UCB_C};
use super::intent::IntentClusterId;

/// Configuration for `FragmentBandit`.
#[derive(Debug, Clone)]
pub struct FragmentBanditConfig {
    pub prior_alpha: f32,
    pub prior_beta: f32,
    pub ucb_c: f32,
    /// If true, reward updates are rejected with `BanditError::Frozen`.
    /// Mirrors `LearningFreezeConfig::freeze_fragment_synthesis`. Use
    /// [`FragmentBandit::set_frozen`] to toggle this at runtime without
    /// reconstructing the bandit.
    pub frozen: bool,
}

impl Default for FragmentBanditConfig {
    fn default() -> Self {
        Self {
            prior_alpha: PRIOR_ALPHA,
            prior_beta: PRIOR_BETA,
            ucb_c: UCB_C,
            frozen: false,
        }
    }
}

/// Why a specific arm was chosen. Useful for ledger events + UI telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SelectionReason {
    /// Best UCB score, arm had prior samples.
    Ucb,
    /// Arm had no samples — explored it by policy.
    FirstPull,
    /// Only one eligible arm.
    OnlyOption,
    /// Caller's `ExplorationControl` forced this arm (e.g. ε-random).
    ExternalOverride,
}

/// Result of `select_arm`.
#[derive(Debug, Clone)]
pub struct ArmSelection {
    pub cluster: IntentClusterId,
    pub provider: ProviderFingerprint,
    pub arm: PromptArmId,
    pub score: f32,
    pub reason: SelectionReason,
}

/// Errors raised by `FragmentBandit`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BanditError {
    /// No arms available for the (cluster, provider) combo.
    NoArms {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
    },
    /// Write blocked because the bandit is frozen.
    Frozen,
    /// The given arm id is unknown within the (cluster, provider) combo.
    UnknownArm {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
        arm: PromptArmId,
    },
    /// Lock poisoned — last writer panicked.
    Poisoned,
}

impl std::fmt::Display for BanditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoArms { cluster, provider } => {
                write!(f, "no arms for {cluster}/{provider}")
            }
            Self::Frozen => f.write_str("fragment synthesis is frozen"),
            Self::UnknownArm {
                cluster,
                provider,
                arm,
            } => write!(f, "unknown arm {arm} in {cluster}/{provider}"),
            Self::Poisoned => f.write_str("bandit lock poisoned"),
        }
    }
}

impl std::error::Error for BanditError {}

/// Snapshot stats useful for audit/telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditStats {
    pub total_arms: usize,
    pub total_samples: u64,
    pub total_reward_sum: f64,
    pub retired_arms: usize,
}

// =============================================================================
// Storage
// =============================================================================

type Key = (IntentClusterId, ProviderFingerprint);

#[derive(Debug, Default)]
struct Inner {
    arms: HashMap<Key, Vec<PromptArm>>,
}

/// Bayesian UCB bandit over prompt arms. Thread-safe (internal `RwLock`).
#[derive(Debug, Clone)]
pub struct FragmentBandit {
    cfg: FragmentBanditConfig,
    inner: Arc<RwLock<Inner>>,
}

impl FragmentBandit {
    pub fn new(cfg: FragmentBanditConfig) -> Self {
        Self {
            cfg,
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Update the frozen flag. Read-paths (selection, stats) remain active.
    pub fn set_frozen(&mut self, frozen: bool) {
        self.cfg.frozen = frozen;
    }

    pub fn is_frozen(&self) -> bool {
        self.cfg.frozen
    }

    /// Insert a new arm under `(cluster, provider)`. Duplicate ids are
    /// rejected by returning false — the caller is expected to treat that
    /// as a no-op.
    pub fn add_arm(&self, cluster: IntentClusterId, arm: PromptArm) -> Result<bool, BanditError> {
        if self.cfg.frozen {
            return Err(BanditError::Frozen);
        }
        let mut inner = self.inner.write().map_err(|_| BanditError::Poisoned)?;
        let key: Key = (cluster, arm.provider.clone());
        let slot = inner.arms.entry(key).or_default();
        if slot.iter().any(|a| a.id == arm.id) {
            return Ok(false);
        }
        slot.push(arm);
        Ok(true)
    }

    /// Retire an arm. Retired arms stay in the store (for audit) but are
    /// skipped during selection.
    pub fn retire_arm(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
        arm: &PromptArmId,
    ) -> Result<(), BanditError> {
        if self.cfg.frozen {
            return Err(BanditError::Frozen);
        }
        let mut inner = self.inner.write().map_err(|_| BanditError::Poisoned)?;
        let key: Key = (cluster, provider.clone());
        let slot = inner
            .arms
            .get_mut(&key)
            .ok_or_else(|| BanditError::UnknownArm {
                cluster,
                provider: provider.clone(),
                arm: arm.clone(),
            })?;
        let found =
            slot.iter_mut()
                .find(|a| &a.id == arm)
                .ok_or_else(|| BanditError::UnknownArm {
                    cluster,
                    provider: provider.clone(),
                    arm: arm.clone(),
                })?;
        found.retired = true;
        Ok(())
    }

    /// Select the best arm for `(cluster, provider)` using UCB. Returns
    /// `NoArms` if none are eligible (empty slot, or all retired).
    pub fn select(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
    ) -> Result<ArmSelection, BanditError> {
        let inner = self.inner.read().map_err(|_| BanditError::Poisoned)?;
        let key: Key = (cluster, provider.clone());
        let slot = inner.arms.get(&key).ok_or_else(|| BanditError::NoArms {
            cluster,
            provider: provider.clone(),
        })?;
        let eligible: Vec<&PromptArm> = slot.iter().filter(|a| !a.retired).collect();
        if eligible.is_empty() {
            return Err(BanditError::NoArms {
                cluster,
                provider: provider.clone(),
            });
        }
        if eligible.len() == 1 {
            let a = eligible[0];
            return Ok(ArmSelection {
                cluster,
                provider: provider.clone(),
                arm: a.id.clone(),
                score: f32::INFINITY,
                reason: SelectionReason::OnlyOption,
            });
        }
        let total_samples: u64 = eligible.iter().map(|a| a.samples).sum();
        let total_f = (total_samples.max(1)) as f32;

        let mut best: Option<(&PromptArm, f32, SelectionReason)> = None;
        for a in &eligible {
            let (score, reason) = if a.samples == 0 {
                (f32::INFINITY, SelectionReason::FirstPull)
            } else {
                let mean = self.posterior_mean(a);
                let explor = self.cfg.ucb_c * (total_f.ln() / (a.samples as f32)).sqrt();
                (mean + explor, SelectionReason::Ucb)
            };
            match best {
                None => best = Some((*a, score, reason)),
                Some((_, best_score, _)) if score > best_score => best = Some((*a, score, reason)),
                _ => {}
            }
        }
        let (arm, score, reason) = best.ok_or_else(|| BanditError::NoArms {
            cluster,
            provider: provider.clone(),
        })?;
        Ok(ArmSelection {
            cluster,
            provider: provider.clone(),
            arm: arm.id.clone(),
            score,
            reason,
        })
    }

    /// Record a reward in `[0, 1]` for the named arm. Rejected when frozen.
    pub fn record_reward(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
        arm: &PromptArmId,
        reward: f32,
    ) -> Result<(), BanditError> {
        if self.cfg.frozen {
            return Err(BanditError::Frozen);
        }
        let reward = reward.clamp(0.0, 1.0) as f64;
        let mut inner = self.inner.write().map_err(|_| BanditError::Poisoned)?;
        let key: Key = (cluster, provider.clone());
        let slot = inner
            .arms
            .get_mut(&key)
            .ok_or_else(|| BanditError::UnknownArm {
                cluster,
                provider: provider.clone(),
                arm: arm.clone(),
            })?;
        let found =
            slot.iter_mut()
                .find(|a| &a.id == arm)
                .ok_or_else(|| BanditError::UnknownArm {
                    cluster,
                    provider: provider.clone(),
                    arm: arm.clone(),
                })?;
        found.samples = found.samples.saturating_add(1);
        found.reward_sum += reward;
        Ok(())
    }

    /// Snapshot of all arms for a (cluster, provider). Cloned — safe to
    /// iterate without holding the lock.
    pub fn arms_for(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
    ) -> Vec<PromptArm> {
        let Ok(inner) = self.inner.read() else {
            return Vec::new();
        };
        let key: Key = (cluster, provider.clone());
        inner.arms.get(&key).cloned().unwrap_or_default()
    }

    /// Global stats across all slots. Snapshot.
    pub fn stats(&self) -> BanditStats {
        let Ok(inner) = self.inner.read() else {
            return BanditStats {
                total_arms: 0,
                total_samples: 0,
                total_reward_sum: 0.0,
                retired_arms: 0,
            };
        };
        let mut total_arms = 0usize;
        let mut total_samples = 0u64;
        let mut total_reward_sum = 0.0f64;
        let mut retired = 0usize;
        for slot in inner.arms.values() {
            for a in slot {
                total_arms += 1;
                total_samples = total_samples.saturating_add(a.samples);
                total_reward_sum += a.reward_sum;
                if a.retired {
                    retired += 1;
                }
            }
        }
        BanditStats {
            total_arms,
            total_samples,
            total_reward_sum,
            retired_arms: retired,
        }
    }

    /// All (cluster, provider) keys currently populated.
    pub fn keys(&self) -> Vec<(IntentClusterId, ProviderFingerprint)> {
        let Ok(inner) = self.inner.read() else {
            return Vec::new();
        };
        inner.arms.keys().cloned().collect()
    }

    /// Beta posterior mean for an arm.
    fn posterior_mean(&self, a: &PromptArm) -> f32 {
        let successes = a.reward_sum as f32;
        let failures = a.samples as f32 - successes;
        let alpha = self.cfg.prior_alpha + successes;
        let beta = self.cfg.prior_beta + failures.max(0.0);
        alpha / (alpha + beta).max(f32::EPSILON)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::arm::{ArmOrigin, PromptArm, PromptArmId};
    use super::*;

    fn arm(id: &str, provider: ProviderFingerprint) -> PromptArm {
        PromptArm::new(
            PromptArmId::new(id),
            vec!["f1".into(), "f2".into()],
            provider,
            ArmOrigin::Manual,
        )
    }

    fn provider() -> ProviderFingerprint {
        ProviderFingerprint::new("ollama", "mistral:7b")
    }

    #[test]
    fn add_arm_rejects_duplicate() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        assert!(b.add_arm(c, arm("a", provider())).unwrap());
        assert!(!b.add_arm(c, arm("a", provider())).unwrap());
    }

    #[test]
    fn select_no_arms_errors() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let err = b.select(IntentClusterId(0), &provider());
        assert!(matches!(err, Err(BanditError::NoArms { .. })));
    }

    #[test]
    fn select_first_pull_has_infinite_score() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("a", provider())).unwrap();
        b.add_arm(c, arm("b", provider())).unwrap();
        let s = b.select(c, &provider()).unwrap();
        assert_eq!(s.reason, SelectionReason::FirstPull);
        assert!(s.score.is_infinite());
    }

    #[test]
    fn select_only_option_is_labelled() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("solo", provider())).unwrap();
        let s = b.select(c, &provider()).unwrap();
        assert_eq!(s.reason, SelectionReason::OnlyOption);
        assert_eq!(s.arm.as_str(), "solo");
    }

    #[test]
    fn record_reward_updates_samples() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("a", provider())).unwrap();
        b.record_reward(c, &provider(), &PromptArmId::new("a"), 1.0)
            .unwrap();
        let arms = b.arms_for(c, &provider());
        assert_eq!(arms[0].samples, 1);
        assert!((arms[0].reward_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn record_reward_clamps_out_of_range() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("a", provider())).unwrap();
        b.record_reward(c, &provider(), &PromptArmId::new("a"), 9.0)
            .unwrap();
        b.record_reward(c, &provider(), &PromptArmId::new("a"), -1.0)
            .unwrap();
        let arms = b.arms_for(c, &provider());
        // Clamped: 1.0 then 0.0 = sum 1.0, samples 2.
        assert_eq!(arms[0].samples, 2);
        assert!((arms[0].reward_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn frozen_blocks_writes_but_not_reads() {
        let mut b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("a", provider())).unwrap();
        b.set_frozen(true);
        assert!(matches!(
            b.add_arm(c, arm("b", provider())),
            Err(BanditError::Frozen)
        ));
        assert!(matches!(
            b.record_reward(c, &provider(), &PromptArmId::new("a"), 0.5),
            Err(BanditError::Frozen)
        ));
        // Read still works.
        let s = b.select(c, &provider()).unwrap();
        assert_eq!(s.arm.as_str(), "a");
    }

    #[test]
    fn provider_isolation() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        let p1 = ProviderFingerprint::new("ollama", "mistral:7b");
        let p2 = ProviderFingerprint::new("anthropic", "claude-opus-4-7");
        b.add_arm(c, arm("a", p1.clone())).unwrap();
        assert!(matches!(b.select(c, &p2), Err(BanditError::NoArms { .. })));
        assert!(b.select(c, &p1).is_ok());
    }

    #[test]
    fn retired_arm_skipped_in_selection() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("a", provider())).unwrap();
        b.add_arm(c, arm("b", provider())).unwrap();
        b.retire_arm(c, &provider(), &PromptArmId::new("a"))
            .unwrap();
        let s = b.select(c, &provider()).unwrap();
        assert_eq!(s.arm.as_str(), "b");
    }

    #[test]
    fn stats_aggregates_across_keys() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c0 = IntentClusterId(0);
        let c1 = IntentClusterId(1);
        b.add_arm(c0, arm("a", provider())).unwrap();
        b.add_arm(c1, arm("b", provider())).unwrap();
        let s = b.stats();
        assert_eq!(s.total_arms, 2);
    }

    #[test]
    fn ucb_prefers_higher_mean_after_samples() {
        let b = FragmentBandit::new(FragmentBanditConfig::default());
        let c = IntentClusterId(0);
        b.add_arm(c, arm("hi", provider())).unwrap();
        b.add_arm(c, arm("lo", provider())).unwrap();
        // Seed both arms with one pull each so UCB starts from samples>0.
        b.record_reward(c, &provider(), &PromptArmId::new("hi"), 0.9)
            .unwrap();
        b.record_reward(c, &provider(), &PromptArmId::new("lo"), 0.1)
            .unwrap();
        // Many more pulls of both at the same rates to amortize the UCB term.
        for _ in 0..20 {
            b.record_reward(c, &provider(), &PromptArmId::new("hi"), 0.9)
                .unwrap();
            b.record_reward(c, &provider(), &PromptArmId::new("lo"), 0.1)
                .unwrap();
        }
        let s = b.select(c, &provider()).unwrap();
        assert_eq!(s.arm.as_str(), "hi");
    }
}
