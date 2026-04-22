//! Exploration control — ε-random safety net plus LLM-proposed arms.
//!
//! Composition: caller invokes `FragmentBandit::select` for the normal
//! Bayesian UCB arm. Separately it consults `ExplorationControl::should_override`
//! to decide whether to override with either (a) a brand-new LLM-proposed arm
//! or (b) a random existing arm (ε-random floor). The control itself is
//! stateless apart from a simple deterministic PRNG.
//!
//! Per the plan, ε=5% is the floor. This protects against local minima even
//! when Bayesian UCB converges aggressively on a well-performing arm.

use std::sync::atomic::{AtomicU64, Ordering};

use super::arm::{ArmOrigin, PromptArm, PromptArmId, ProviderFingerprint};
use super::defaults::EPSILON_RANDOM;
use super::intent::IntentClusterId;

/// Errors raised by exploration logic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExplorationError {
    /// The proposer did not return a usable arm.
    ProposerFailed(String),
    /// Invalid configuration (ε out of [0,1]).
    InvalidConfig(String),
}

impl std::fmt::Display for ExplorationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProposerFailed(s) => write!(f, "proposer failed: {s}"),
            Self::InvalidConfig(s) => write!(f, "invalid exploration config: {s}"),
        }
    }
}

impl std::error::Error for ExplorationError {}

/// Callback that proposes a new `PromptArm` for a given (cluster, provider).
/// Typically backed by an LLM call. Must be side-effect free — the caller is
/// responsible for persisting the returned arm.
pub trait ArmProposer: Send + Sync {
    fn propose(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
        existing: &[PromptArm],
    ) -> Result<PromptArm, ExplorationError>;
}

/// Proposer that always refuses. Useful as a default (no LLM configured).
pub struct RejectProposer;

impl ArmProposer for RejectProposer {
    fn propose(
        &self,
        _cluster: IntentClusterId,
        _provider: &ProviderFingerprint,
        _existing: &[PromptArm],
    ) -> Result<PromptArm, ExplorationError> {
        Err(ExplorationError::ProposerFailed(
            "no proposer configured".into(),
        ))
    }
}

/// Proposer that builds an ε-random arm by remixing existing fragments.
/// Deterministic given the seed — does not call into the LLM.
pub struct EpsilonRandomProposer {
    /// Seed for the fragment-shuffle PRNG. Lets tests pin ordering.
    seed: AtomicU64,
}

impl EpsilonRandomProposer {
    pub fn new(seed: u64) -> Self {
        Self {
            seed: AtomicU64::new(seed),
        }
    }
}

impl Default for EpsilonRandomProposer {
    fn default() -> Self {
        Self::new(0xC0FFEE)
    }
}

impl ArmProposer for EpsilonRandomProposer {
    fn propose(
        &self,
        _cluster: IntentClusterId,
        provider: &ProviderFingerprint,
        existing: &[PromptArm],
    ) -> Result<PromptArm, ExplorationError> {
        if existing.is_empty() {
            return Err(ExplorationError::ProposerFailed(
                "no fragments to remix".into(),
            ));
        }
        // Collect a candidate pool of all fragments seen across arms.
        let mut pool: Vec<String> = existing
            .iter()
            .flat_map(|a| a.fragments.iter().cloned())
            .collect();
        pool.sort();
        pool.dedup();
        if pool.is_empty() {
            return Err(ExplorationError::ProposerFailed(
                "no fragments to remix".into(),
            ));
        }
        // Simple xorshift* so we don't pull `rand` here.
        let mut s = self.seed.fetch_add(1, Ordering::Relaxed);
        let shuffle = |seed: &mut u64| -> u64 {
            *seed ^= *seed >> 12;
            *seed ^= *seed << 25;
            *seed ^= *seed >> 27;
            seed.wrapping_mul(0x2545F4914F6CDD1D)
        };
        // Pick between 1 and pool.len() fragments, in a permuted order.
        let count = ((shuffle(&mut s) as usize) % pool.len()) + 1;
        let mut indices: Vec<usize> = (0..pool.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = (shuffle(&mut s) as usize) % (i + 1);
            indices.swap(i, j);
        }
        let fragments: Vec<String> = indices
            .into_iter()
            .take(count)
            .map(|i| pool[i].clone())
            .collect();
        let arm_id = PromptArmId::new(format!("eps_rand_{:016x}", shuffle(&mut s)));
        Ok(PromptArm::new(
            arm_id,
            fragments,
            provider.clone(),
            ArmOrigin::EpsilonRandom,
        ))
    }
}

/// Wrapper that carries ε + a proposer. Simple wrapper type — the interesting
/// logic is in `should_override`.
pub struct ExplorationControl {
    epsilon: f32,
    proposer: Box<dyn ArmProposer>,
    /// Deterministic "coin" — overridden in tests via `set_coin`.
    coin: AtomicU64,
}

impl ExplorationControl {
    pub fn new(epsilon: f32, proposer: Box<dyn ArmProposer>) -> Result<Self, ExplorationError> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(ExplorationError::InvalidConfig(format!(
                "epsilon={epsilon} out of [0,1]"
            )));
        }
        Ok(Self {
            epsilon,
            proposer,
            coin: AtomicU64::new(0x9E3779B97F4A7C15),
        })
    }

    /// Build a default control with ε=5% floor and no LLM proposer. Use
    /// `with_proposer` to swap in a real one later.
    pub fn default_floor() -> Self {
        Self {
            epsilon: EPSILON_RANDOM,
            proposer: Box::new(RejectProposer),
            coin: AtomicU64::new(0x9E3779B97F4A7C15),
        }
    }

    pub fn with_proposer(mut self, p: Box<dyn ArmProposer>) -> Self {
        self.proposer = p;
        self
    }

    /// Manually pin the PRNG state — tests only.
    #[doc(hidden)]
    pub fn set_coin(&self, state: u64) {
        self.coin.store(state, Ordering::Relaxed);
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns true when the caller should override the UCB pick with a
    /// random/proposed arm. Deterministic given the internal PRNG state.
    pub fn should_override(&self) -> bool {
        let mut s = self.coin.load(Ordering::Relaxed);
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let next = s.wrapping_mul(0x2545F4914F6CDD1D);
        self.coin.store(next, Ordering::Relaxed);
        // Map to [0,1).
        let u = (next >> 11) as f64 / (1u64 << 53) as f64;
        u < self.epsilon as f64
    }

    /// Run the proposer to propose a fresh arm — typically called after
    /// `should_override()` returned true. Propagates errors from the proposer.
    pub fn propose_override(
        &self,
        cluster: IntentClusterId,
        provider: &ProviderFingerprint,
        existing: &[PromptArm],
    ) -> Result<PromptArm, ExplorationError> {
        self.proposer.propose(cluster, provider, existing)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> ProviderFingerprint {
        ProviderFingerprint::new("ollama", "m")
    }

    #[test]
    fn invalid_epsilon_rejected() {
        assert!(ExplorationControl::new(-0.1, Box::new(RejectProposer)).is_err());
        assert!(ExplorationControl::new(1.1, Box::new(RejectProposer)).is_err());
    }

    #[test]
    fn default_floor_uses_epsilon_constant() {
        let c = ExplorationControl::default_floor();
        assert!((c.epsilon() - EPSILON_RANDOM).abs() < 1e-6);
    }

    #[test]
    fn reject_proposer_errors() {
        let p = RejectProposer;
        let err = p.propose(IntentClusterId(0), &provider(), &[]);
        assert!(matches!(err, Err(ExplorationError::ProposerFailed(_))));
    }

    #[test]
    fn epsilon_proposer_fails_without_fragments() {
        let p = EpsilonRandomProposer::new(1);
        let err = p.propose(IntentClusterId(0), &provider(), &[]);
        assert!(err.is_err());
    }

    #[test]
    fn epsilon_proposer_builds_arm_from_pool() {
        let existing = vec![PromptArm::new(
            PromptArmId::new("a"),
            vec!["frag1".into(), "frag2".into()],
            provider(),
            ArmOrigin::Manual,
        )];
        let p = EpsilonRandomProposer::new(0xDEAD_BEEF);
        let arm = p
            .propose(IntentClusterId(0), &provider(), &existing)
            .unwrap();
        assert_eq!(arm.origin, ArmOrigin::EpsilonRandom);
        assert!(!arm.fragments.is_empty());
        // Every returned fragment was in the pool.
        for f in &arm.fragments {
            assert!(f == "frag1" || f == "frag2");
        }
    }

    #[test]
    fn override_rate_roughly_matches_epsilon() {
        let c = ExplorationControl::new(0.25, Box::new(RejectProposer)).unwrap();
        let mut hits = 0usize;
        for _ in 0..10_000 {
            if c.should_override() {
                hits += 1;
            }
        }
        let rate = hits as f32 / 10_000.0;
        // Loose bound — the PRNG is deterministic, just check order of magnitude.
        assert!(rate > 0.18 && rate < 0.32, "got rate={rate}");
    }

    #[test]
    fn override_never_fires_at_zero() {
        let c = ExplorationControl::new(0.0, Box::new(RejectProposer)).unwrap();
        for _ in 0..1_000 {
            assert!(!c.should_override());
        }
    }

    #[test]
    fn override_always_fires_at_one() {
        let c = ExplorationControl::new(1.0, Box::new(RejectProposer)).unwrap();
        for _ in 0..1_000 {
            assert!(c.should_override());
        }
    }

    #[test]
    fn with_proposer_swaps_successfully() {
        let c = ExplorationControl::default_floor()
            .with_proposer(Box::new(EpsilonRandomProposer::new(42)));
        let existing = vec![PromptArm::new(
            PromptArmId::new("x"),
            vec!["a".into()],
            provider(),
            ArmOrigin::Manual,
        )];
        let arm = c
            .propose_override(IntentClusterId(0), &provider(), &existing)
            .unwrap();
        assert_eq!(arm.origin, ArmOrigin::EpsilonRandom);
    }
}
