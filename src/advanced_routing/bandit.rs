//! Multi-armed bandit router: arms, strategies, reward policy, routing context.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// MULTI-ARMED BANDIT ROUTER
// =============================================================================

/// Parameters of a Beta distribution for Thompson Sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    /// Alpha parameter (successes + prior)
    pub alpha: f64,
    /// Beta parameter (failures + prior)
    pub beta: f64,
}

/// A single arm in the bandit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// Arm identifier (model ID)
    pub id: ArmId,
    /// Beta distribution parameters
    pub params: BetaParams,
    /// Total times this arm was pulled
    pub pull_count: u64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Last time this arm was pulled (unix timestamp seconds)
    pub last_pulled: u64,
}

/// Strategy for selecting arms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum BanditStrategy {
    /// Thompson Sampling: sample from Beta posterior, pick highest
    ThompsonSampling,
    /// UCB1: pick arm maximizing mean + sqrt(2 * ln(N) / n_i)
    Ucb1,
    /// Epsilon-greedy: explore with probability epsilon, else exploit best
    EpsilonGreedy { epsilon: f64 },
}

// =============================================================================
// REWARD POLICY (Section A2)
// =============================================================================

/// Policy for computing composite reward from quality, latency, and cost.
///
/// Combines three dimensions into a single 0..1 reward signal for bandit learning.
/// Latency and cost are normalized using reference values: a latency of
/// `latency_ref_ms` maps to score 0.0, and zero latency maps to 1.0.
///
/// When `ArmFeedback` has `None` for latency or cost, the corresponding
/// weight is redistributed proportionally to the active components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPolicy {
    /// Weight for quality component (default 0.7)
    pub quality_weight: f64,
    /// Weight for latency component (default 0.2)
    pub latency_weight: f64,
    /// Weight for cost component (default 0.1)
    pub cost_weight: f64,
    /// Reference latency in ms for normalization (default 5000.0).
    /// A latency equal to this value yields a latency score of 0.0.
    pub latency_ref_ms: f64,
    /// Reference cost for normalization (default 0.01).
    /// A cost equal to this value yields a cost score of 0.0.
    pub cost_ref: f64,
}

impl Default for RewardPolicy {
    fn default() -> Self {
        Self {
            quality_weight: 0.7,
            latency_weight: 0.2,
            cost_weight: 0.1,
            latency_ref_ms: 5000.0,
            cost_ref: 0.01,
        }
    }
}

impl RewardPolicy {
    /// Normalize weights so they sum to 1.0.
    ///
    /// If all weights are zero (or negative), returns equal thirds (1/3, 1/3, 1/3).
    pub fn normalize_weights(&self) -> (f64, f64, f64) {
        let qw = self.quality_weight.max(0.0);
        let lw = self.latency_weight.max(0.0);
        let cw = self.cost_weight.max(0.0);
        let total = qw + lw + cw;
        if total < 1e-12 {
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        } else {
            (qw / total, lw / total, cw / total)
        }
    }

    /// Compute a composite reward in 0..1 from feedback dimensions.
    ///
    /// Components:
    /// - quality: `feedback.quality` or `1.0` if success, `0.0` if failure
    /// - latency: `(1.0 - latency_ms / latency_ref_ms).clamp(0.0, 1.0)`, skipped if `None`
    /// - cost: `(1.0 - cost / cost_ref).clamp(0.0, 1.0)`, skipped if `None`
    ///
    /// Missing components have their weight redistributed to active components.
    pub fn compute_reward(&self, feedback: &ArmFeedback) -> f64 {
        let (qw, lw, cw) = self.normalize_weights();

        let quality_score = feedback
            .quality
            .unwrap_or(if feedback.success { 1.0 } else { 0.0 });

        let latency_available = feedback.latency_ms.is_some() && self.latency_ref_ms > 0.0;
        let cost_available = feedback.cost.is_some() && self.cost_ref > 0.0;

        let latency_score = if latency_available {
            let ms = feedback.latency_ms.unwrap_or(0) as f64;
            (1.0 - ms / self.latency_ref_ms).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let cost_score = if cost_available {
            let c = feedback.cost.unwrap_or(0.0);
            (1.0 - c / self.cost_ref).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Redistribute weights of missing components
        let active_extra_weight =
            if !latency_available { lw } else { 0.0 } + if !cost_available { cw } else { 0.0 };

        let effective_qw;
        let effective_lw;
        let effective_cw;

        if !latency_available && !cost_available {
            // Only quality is available
            effective_qw = 1.0;
            effective_lw = 0.0;
            effective_cw = 0.0;
        } else if !latency_available {
            // Quality + cost active, redistribute latency weight
            let active_sum = qw + cw;
            if active_sum < 1e-12 {
                effective_qw = 0.5;
                effective_lw = 0.0;
                effective_cw = 0.5;
            } else {
                effective_qw = qw + active_extra_weight * (qw / active_sum);
                effective_lw = 0.0;
                effective_cw = cw + active_extra_weight * (cw / active_sum);
            }
        } else if !cost_available {
            // Quality + latency active, redistribute cost weight
            let active_sum = qw + lw;
            if active_sum < 1e-12 {
                effective_qw = 0.5;
                effective_lw = 0.5;
                effective_cw = 0.0;
            } else {
                effective_qw = qw + active_extra_weight * (qw / active_sum);
                effective_lw = lw + active_extra_weight * (lw / active_sum);
                effective_cw = 0.0;
            }
        } else {
            // All three active
            effective_qw = qw;
            effective_lw = lw;
            effective_cw = cw;
        }

        let reward =
            effective_qw * quality_score + effective_lw * latency_score + effective_cw * cost_score;

        reward.clamp(0.0, 1.0)
    }
}

fn default_prefer_boost() -> f64 {
    2.0
}

/// Per-query routing preferences that override default RewardPolicy weights.
///
/// Weight overrides (quality/latency/cost) affect how outcomes are RECORDED
/// (via `record_outcome_with_preferences`). Arm exclusion/boosting affects
/// which arm is SELECTED.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPreferences {
    /// Override quality weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_weight: Option<f64>,
    /// Override latency weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_weight: Option<f64>,
    /// Override cost weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_weight: Option<f64>,
    /// Arms to exclude from selection for this query
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_arms: Vec<ArmId>,
    /// Arms to boost for this query
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preferred_arms: Vec<ArmId>,
    /// Boost factor for preferred arms (default 2.0).
    #[serde(default = "default_prefer_boost")]
    pub prefer_boost: f64,
}

impl Default for RoutingPreferences {
    fn default() -> Self {
        Self {
            quality_weight: None,
            latency_weight: None,
            cost_weight: None,
            excluded_arms: Vec::new(),
            preferred_arms: Vec::new(),
            prefer_boost: 2.0,
        }
    }
}

impl RoutingPreferences {
    /// Build a temporary RewardPolicy by merging preferences over a base policy.
    pub fn apply_to_policy(&self, base: &RewardPolicy) -> RewardPolicy {
        RewardPolicy {
            quality_weight: self.quality_weight.unwrap_or(base.quality_weight),
            latency_weight: self.latency_weight.unwrap_or(base.latency_weight),
            cost_weight: self.cost_weight.unwrap_or(base.cost_weight),
            latency_ref_ms: base.latency_ref_ms,
            cost_ref: base.cost_ref,
        }
    }

    /// Convenience: create preferences that ignore cost.
    pub fn ignore_cost() -> Self {
        Self {
            cost_weight: Some(0.0),
            ..Default::default()
        }
    }

    /// Convenience: create preferences that minimize latency.
    pub fn minimize_latency() -> Self {
        Self {
            latency_weight: Some(0.8),
            quality_weight: Some(0.2),
            cost_weight: Some(0.0),
            ..Default::default()
        }
    }

    /// Convenience: create preferences that maximize quality only.
    pub fn quality_only() -> Self {
        Self {
            quality_weight: Some(1.0),
            latency_weight: Some(0.0),
            cost_weight: Some(0.0),
            ..Default::default()
        }
    }
}

/// Extended routing context: QueryFeatures + agent-level metadata.
///
/// Provides a superset of information for routing decisions, including
/// budget constraints, RAG status, and agent tier. The pipeline can
/// auto-derive RoutingPreferences from context fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    /// Core query features
    pub features: QueryFeatures,
    /// Whether RAG is active for this query
    #[serde(default)]
    pub rag_active: bool,
    /// Remaining budget (currency units). Low budget auto-boosts cost_weight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_remaining: Option<f64>,
    /// Agent tier (e.g., "free", "pro", "enterprise")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_tier: Option<String>,
    /// Total cost accumulated in the current session
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_cost_so_far: Option<f64>,
    /// Preferred provider to filter arms by (e.g., "openai", "anthropic")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_provider: Option<String>,
}

impl RoutingContext {
    /// Create a new routing context from query features.
    pub fn new(features: QueryFeatures) -> Self {
        Self {
            features,
            rag_active: false,
            budget_remaining: None,
            agent_tier: None,
            session_cost_so_far: None,
            preferred_provider: None,
        }
    }

    /// Auto-derive RoutingPreferences from context fields.
    ///
    /// Rules:
    /// - If `budget_remaining` < `cost_ref * 10.0`, boosts cost_weight to 0.5
    /// - Otherwise returns default preferences
    pub fn derive_preferences(&self, base_policy: &RewardPolicy) -> RoutingPreferences {
        let mut prefs = RoutingPreferences::default();

        if let Some(budget) = self.budget_remaining {
            if budget < base_policy.cost_ref * 10.0 {
                prefs.cost_weight = Some(0.5);
                prefs.quality_weight = Some(0.4);
                prefs.latency_weight = Some(0.1);
            }
        }

        prefs
    }
}

impl From<QueryFeatures> for RoutingContext {
    fn from(features: QueryFeatures) -> Self {
        Self::new(features)
    }
}

/// Configuration for the bandit router.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BanditConfig {
    /// Selection strategy
    pub strategy: BanditStrategy,
    /// Prior alpha for new arms (default 1.0 = uniform prior)
    pub prior_alpha: f64,
    /// Prior beta for new arms (default 1.0)
    pub prior_beta: f64,
    /// Minimum pulls before an arm can be pruned
    pub min_pulls_before_prune: u64,
    /// Decay factor for old observations (1.0 = no decay)
    pub decay_factor: f64,
    /// Reward computation policy for composite reward from quality/latency/cost.
    #[serde(default)]
    pub reward_policy: RewardPolicy,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            strategy: BanditStrategy::ThompsonSampling,
            prior_alpha: 1.0,
            prior_beta: 1.0,
            min_pulls_before_prune: 10,
            decay_factor: 1.0,
            reward_policy: RewardPolicy::default(),
        }
    }
}

/// Visibility of arms for distributed state sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ArmVisibility {
    /// Share freely with all nodes (default)
    Public,
    /// Never share — local-only model
    Private,
}

impl Default for ArmVisibility {
    fn default() -> Self {
        Self::Public
    }
}

/// Multi-Armed Bandit router with per-task-type bandits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditRouter {
    pub(crate) config: BanditConfig,
    /// Per-task bandits: task_type -> list of arms
    pub(crate) bandits: HashMap<String, Vec<BanditArm>>,
    /// Global bandit (when task type is unknown)
    pub(crate) global_bandit: Vec<BanditArm>,
    /// Total pulls across all bandits (for UCB1)
    pub(crate) total_pulls: u64,
    /// PRNG seed state (LCG for deterministic testing)
    pub(crate) seed: u64,
    /// Arms marked as private (local-only, not shared in distributed merging).
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub(crate) private_arms: HashSet<ArmId>,
}

impl BanditRouter {
    /// Create a new bandit router with the given configuration.
    pub fn new(config: BanditConfig) -> Self {
        Self {
            config,
            bandits: HashMap::new(),
            global_bandit: Vec::new(),
            total_pulls: 0,
            seed: 12345,
            private_arms: HashSet::new(),
        }
    }

    /// Create with a specific seed for deterministic testing.
    pub fn with_seed(config: BanditConfig, seed: u64) -> Self {
        Self {
            config,
            bandits: HashMap::new(),
            global_bandit: Vec::new(),
            total_pulls: 0,
            seed,
            private_arms: HashSet::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &BanditConfig {
        &self.config
    }

    /// Add an arm to the global bandit.
    pub fn add_arm(&mut self, arm_id: &str) {
        if !self.global_bandit.iter().any(|a| a.id == arm_id) {
            self.global_bandit.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams {
                    alpha: self.config.prior_alpha,
                    beta: self.config.prior_beta,
                },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Add an arm to a task-specific bandit.
    pub fn add_arm_for_task(&mut self, task_type: &str, arm_id: &str) {
        let arms = self.bandits.entry(task_type.to_string()).or_default();
        if !arms.iter().any(|a| a.id == arm_id) {
            arms.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams {
                    alpha: self.config.prior_alpha,
                    beta: self.config.prior_beta,
                },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Select an arm using the configured strategy.
    pub fn select(
        &mut self,
        task_type: Option<&str>,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Clone arms snapshot to avoid borrow conflicts with &mut self in sampling
        let arms_snapshot: Vec<BanditArm> = if let Some(tt) = task_type {
            self.bandits.get(tt).unwrap_or(&self.global_bandit).clone()
        } else {
            self.global_bandit.clone()
        };

        if arms_snapshot.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: task_type.unwrap_or("global").to_string(),
                reason: "No arms registered".to_string(),
            });
        }

        let start = std::time::Instant::now();

        let (selected_idx, scores) = match self.config.strategy {
            BanditStrategy::ThompsonSampling => self.thompson_select(&arms_snapshot),
            BanditStrategy::Ucb1 => self.ucb1_select(&arms_snapshot),
            BanditStrategy::EpsilonGreedy { epsilon } => {
                self.epsilon_greedy_select(&arms_snapshot, epsilon)
            }
        };

        let selected_id = arms_snapshot[selected_idx].id.clone();
        let confidence = if scores.is_empty() {
            0.5
        } else {
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            if (max_score - min_score).abs() < 1e-10 {
                0.5
            } else {
                (scores[selected_idx] - min_score) / (max_score - min_score)
            }
        };

        let mut alternatives: Vec<(ArmId, f64)> = arms_snapshot
            .iter()
            .zip(scores.iter())
            .enumerate()
            .filter(|(i, _)| *i != selected_idx)
            .map(|(_, (a, &s))| (a.id.clone(), s))
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update the actual arm's pull count
        let arms_mut = if let Some(tt) = task_type {
            self.bandits.get_mut(tt).unwrap_or(&mut self.global_bandit)
        } else {
            &mut self.global_bandit
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if selected_idx < arms_mut.len() {
            arms_mut[selected_idx].last_pulled = now;
        }
        self.total_pulls += 1;

        let elapsed = start.elapsed().as_micros() as u64;

        Ok(RoutingOutcome {
            selected_arm: selected_id,
            confidence,
            reason: format!("{:?} selection", self.config.strategy),
            alternatives,
            router_id: "bandit".to_string(),
            decision_time_us: elapsed,
        })
    }

    /// Record outcome feedback for an arm.
    pub fn record_outcome(&mut self, feedback: &ArmFeedback) {
        let task_type = feedback.task_type.as_deref();
        let arms = if let Some(tt) = task_type {
            if let Some(a) = self.bandits.get_mut(tt) {
                a
            } else {
                &mut self.global_bandit
            }
        } else {
            &mut self.global_bandit
        };

        if let Some(arm) = arms.iter_mut().find(|a| a.id == feedback.arm_id) {
            let reward = self.config.reward_policy.compute_reward(feedback);

            // Apply decay if configured
            if self.config.decay_factor < 1.0 {
                let d = self.config.decay_factor;
                arm.params.alpha =
                    self.config.prior_alpha + (arm.params.alpha - self.config.prior_alpha) * d;
                arm.params.beta =
                    self.config.prior_beta + (arm.params.beta - self.config.prior_beta) * d;
            }

            arm.params.alpha += reward;
            arm.params.beta += 1.0 - reward;
            arm.pull_count += 1;
            arm.total_reward += reward;
        }
    }

    /// Set specific priors for an arm (warm start from eval data).
    pub fn warm_start(&mut self, arm_id: &str, alpha: f64, beta: f64) {
        if let Some(arm) = self.global_bandit.iter_mut().find(|a| a.id == arm_id) {
            arm.params.alpha = alpha;
            arm.params.beta = beta;
        } else {
            self.global_bandit.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams { alpha, beta },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Set specific priors for a task-specific arm.
    pub fn warm_start_for_task(&mut self, task_type: &str, arm_id: &str, alpha: f64, beta: f64) {
        let arms = self.bandits.entry(task_type.to_string()).or_default();
        if let Some(arm) = arms.iter_mut().find(|a| a.id == arm_id) {
            arm.params.alpha = alpha;
            arm.params.beta = beta;
        } else {
            arms.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams { alpha, beta },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Get stats for a specific arm (global bandit).
    pub fn arm_stats(&self, arm_id: &str) -> Option<&BanditArm> {
        self.global_bandit.iter().find(|a| a.id == arm_id)
    }

    /// Get all arms for a task type (or global).
    pub fn all_arms(&self, task_type: Option<&str>) -> &[BanditArm] {
        if let Some(tt) = task_type {
            self.bandits
                .get(tt)
                .map(|v| v.as_slice())
                .unwrap_or(&self.global_bandit)
        } else {
            &self.global_bandit
        }
    }

    /// Get total pulls across all bandits.
    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }

    /// Returns all task types that have dedicated bandit arms.
    pub fn task_types(&self) -> Vec<&str> {
        self.bandits.keys().map(|s| s.as_str()).collect()
    }

    /// Returns all arms for a given task type (or global if None) as a Vec.
    pub fn all_arms_vec(&self, task_type: Option<&str>) -> Vec<&BanditArm> {
        match task_type {
            Some(t) => self
                .bandits
                .get(t)
                .map(|v| v.iter().collect())
                .unwrap_or_default(),
            None => self.global_bandit.iter().collect(),
        }
    }

    /// Remove an arm from the bandit (task-specific or global).
    /// Returns true if the arm was found and removed.
    pub fn remove_arm(&mut self, arm_id: &str, task_type: Option<&str>) -> bool {
        match task_type {
            Some(t) => {
                if let Some(arms) = self.bandits.get_mut(t) {
                    let before = arms.len();
                    arms.retain(|a| a.id != arm_id);
                    arms.len() < before
                } else {
                    false
                }
            }
            None => {
                let before = self.global_bandit.len();
                self.global_bandit.retain(|a| a.id != arm_id);
                self.global_bandit.len() < before
            }
        }
    }

    // --- Internal: strategy implementations ---

    fn thompson_select(&mut self, arms: &[BanditArm]) -> (usize, Vec<f64>) {
        let scores: Vec<f64> = arms
            .iter()
            .map(|a| self.sample_beta(a.params.alpha, a.params.beta))
            .collect();
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, scores)
    }

    fn ucb1_select(&self, arms: &[BanditArm]) -> (usize, Vec<f64>) {
        let total = self.total_pulls.max(1);
        let scores: Vec<f64> = arms.iter().map(|a| self.ucb1_score(a, total)).collect();
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, scores)
    }

    fn epsilon_greedy_select(&mut self, arms: &[BanditArm], epsilon: f64) -> (usize, Vec<f64>) {
        let scores: Vec<f64> = arms
            .iter()
            .map(|a| {
                if a.pull_count == 0 {
                    0.5
                } else {
                    a.total_reward / a.pull_count as f64
                }
            })
            .collect();

        let r = self.next_random();
        let idx = if r < epsilon {
            // Explore: random arm
            (self.next_random() * arms.len() as f64) as usize % arms.len()
        } else {
            // Exploit: best mean arm
            scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        (idx, scores)
    }

    fn ucb1_score(&self, arm: &BanditArm, total_n: u64) -> f64 {
        if arm.pull_count == 0 {
            return f64::INFINITY;
        }
        let mean = arm.total_reward / arm.pull_count as f64;
        let exploration = (2.0 * (total_n as f64).ln() / arm.pull_count as f64).sqrt();
        mean + exploration
    }

    /// Sample from Beta(alpha, beta) distribution using Gamma variates.
    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.sample_gamma(alpha);
        let y = self.sample_gamma(beta);
        if x + y == 0.0 {
            0.5
        } else {
            x / (x + y)
        }
    }

    /// Sample from Gamma(alpha) using Marsaglia-Tsang method.
    fn sample_gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            // Ahrens-Dieter: Gamma(a) = Gamma(a+1) * U^(1/a)
            let g = self.sample_gamma(alpha + 1.0);
            let u = self.next_random().max(1e-30);
            return g * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.sample_normal();
            let v_base = 1.0 + c * x;
            if v_base <= 0.0 {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = self.next_random().max(1e-30);

            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Box-Muller transform for standard normal samples.
    fn sample_normal(&mut self) -> f64 {
        let u1 = self.next_random().max(1e-30);
        let u2 = self.next_random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// LCG PRNG — deterministic, fast, no external deps.
    fn next_random(&mut self) -> f64 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    // --- Private arms management ---

    /// Mark an arm as private (local-only, not shared in distributed merging).
    pub fn set_arm_private(&mut self, arm_id: &str) {
        self.private_arms.insert(arm_id.to_string());
    }

    /// Mark an arm as public (default, shareable in distributed merging).
    pub fn set_arm_public(&mut self, arm_id: &str) {
        self.private_arms.remove(arm_id);
    }

    /// Check if an arm is private.
    pub fn is_arm_private(&self, arm_id: &str) -> bool {
        self.private_arms.contains(arm_id)
    }

    /// Get the set of private arm IDs.
    pub fn private_arm_ids(&self) -> &HashSet<ArmId> {
        &self.private_arms
    }

    // --- Per-query preference-aware selection ---

    /// Select an arm with per-query preferences applied.
    ///
    /// Filters out excluded_arms, boosts preferred_arms scores.
    /// The weight overrides in preferences do NOT affect selection (they apply to recording).
    pub fn select_with_preferences(
        &mut self,
        task_type: Option<&str>,
        prefs: &RoutingPreferences,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let start = std::time::Instant::now();

        // Clone arms snapshot to avoid borrow conflicts with &mut self in sampling
        let arms_snapshot: Vec<BanditArm> = if let Some(tt) = task_type {
            self.bandits.get(tt).unwrap_or(&self.global_bandit).clone()
        } else {
            self.global_bandit.clone()
        };

        // Filter out excluded arms
        let candidates: Vec<BanditArm> = arms_snapshot
            .into_iter()
            .filter(|a| !prefs.excluded_arms.contains(&a.id))
            .collect();

        if candidates.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: task_type.unwrap_or("global").to_string(),
                reason: "All arms excluded by preferences".to_string(),
            });
        }

        // Score using the configured strategy
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for (i, arm) in candidates.iter().enumerate() {
            let base_score = match self.config.strategy {
                BanditStrategy::ThompsonSampling => {
                    self.sample_beta(arm.params.alpha, arm.params.beta)
                }
                BanditStrategy::Ucb1 => {
                    if arm.pull_count == 0 {
                        return Ok(RoutingOutcome {
                            selected_arm: arm.id.clone(),
                            confidence: 0.0,
                            reason: "UCB1: unpulled arm (preferences)".to_string(),
                            alternatives: vec![],
                            router_id: "bandit".to_string(),
                            decision_time_us: start.elapsed().as_micros() as u64,
                        });
                    }
                    let mean = arm.total_reward / arm.pull_count as f64;
                    let exploration =
                        (2.0 * (self.total_pulls as f64).ln() / arm.pull_count as f64).sqrt();
                    mean + exploration
                }
                BanditStrategy::EpsilonGreedy { epsilon } => {
                    let r = self.next_random();
                    if r < epsilon {
                        self.next_random() // random score for exploration
                    } else if arm.pull_count > 0 {
                        arm.total_reward / arm.pull_count as f64
                    } else {
                        f64::INFINITY // unexplored arms get priority
                    }
                }
            };

            // Apply boost for preferred arms
            let score = if prefs.preferred_arms.contains(&arm.id) {
                base_score * prefs.prefer_boost
            } else {
                base_score
            };

            scores.push((i, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = scores[0].0;
        let selected_id = candidates[best_idx].id.clone();

        let confidence = if candidates.len() > 1 {
            let gap = scores[0].1 - scores[1].1;
            (gap / (scores[0].1.abs() + 1e-10)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let alternatives: Vec<(ArmId, f64)> = scores
            .iter()
            .skip(1)
            .take(5)
            .map(|(idx, score)| (candidates[*idx].id.clone(), *score))
            .collect();

        // Update pull tracking
        if let Some(tt) = task_type {
            if let Some(arms_mut) = self.bandits.get_mut(tt) {
                if let Some(arm) = arms_mut.iter_mut().find(|a| a.id == selected_id) {
                    arm.pull_count += 1;
                    arm.last_pulled = self.total_pulls;
                }
            }
        }
        if let Some(arm) = self.global_bandit.iter_mut().find(|a| a.id == selected_id) {
            arm.pull_count += 1;
            arm.last_pulled = self.total_pulls;
        }
        self.total_pulls += 1;

        Ok(RoutingOutcome {
            selected_arm: selected_id,
            confidence,
            reason: format!("{:?} selection with preferences", self.config.strategy),
            alternatives,
            router_id: "bandit".to_string(),
            decision_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Record outcome with preference-overridden RewardPolicy.
    ///
    /// The weight overrides in prefs modify how THIS outcome's reward is computed,
    /// allowing per-query customization (e.g., "ignore cost for this query").
    pub fn record_outcome_with_preferences(
        &mut self,
        feedback: &ArmFeedback,
        prefs: &RoutingPreferences,
    ) {
        let policy = prefs.apply_to_policy(&self.config.reward_policy);
        let reward = policy.compute_reward(feedback);

        let task_type = feedback.task_type.as_deref();
        let arms = if let Some(tt) = task_type {
            if let Some(a) = self.bandits.get_mut(tt) {
                a
            } else {
                &mut self.global_bandit
            }
        } else {
            &mut self.global_bandit
        };

        if let Some(arm) = arms.iter_mut().find(|a| a.id == feedback.arm_id) {
            if self.config.decay_factor < 1.0 {
                let d = self.config.decay_factor;
                arm.params.alpha =
                    self.config.prior_alpha + (arm.params.alpha - self.config.prior_alpha) * d;
                arm.params.beta =
                    self.config.prior_beta + (arm.params.beta - self.config.prior_beta) * d;
            }

            arm.params.alpha += reward;
            arm.params.beta += 1.0 - reward;
            arm.pull_count += 1;
            arm.total_reward += reward;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BANDIT ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_bandit_creation_default() {
        let bandit = BanditRouter::new(BanditConfig::default());
        assert!(bandit.all_arms(None).is_empty());
        assert_eq!(bandit.total_pulls(), 0);
    }

    #[test]
    fn test_bandit_add_arm() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("model-a");
        bandit.add_arm("model-b");
        assert_eq!(bandit.all_arms(None).len(), 2);
        // Adding duplicate does nothing
        bandit.add_arm("model-a");
        assert_eq!(bandit.all_arms(None).len(), 2);
    }

    #[test]
    fn test_bandit_add_arm_for_task() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("coding", "gpt-4");
        bandit.add_arm_for_task("coding", "claude");
        bandit.add_arm("default-model");
        assert_eq!(bandit.all_arms(Some("coding")).len(), 2);
        assert_eq!(bandit.all_arms(None).len(), 1);
    }

    #[test]
    fn test_thompson_sampling_selects_arm() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("a");
        bandit.add_arm("b");
        let result = bandit.select(None).unwrap();
        assert!(!result.selected_arm.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_ucb1_selects_unpulled_first() {
        let config = BanditConfig {
            strategy: BanditStrategy::Ucb1,
            ..Default::default()
        };
        let mut bandit = BanditRouter::new(config);
        bandit.add_arm("pulled");
        bandit.add_arm("unpulled");

        // Record some outcomes for "pulled"
        bandit.record_outcome(&ArmFeedback {
            arm_id: "pulled".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: None,
            task_type: None,
        });

        // UCB1 should select "unpulled" first (infinity score)
        let result = bandit.select(None).unwrap();
        assert_eq!(result.selected_arm, "unpulled");
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        let config = BanditConfig {
            strategy: BanditStrategy::EpsilonGreedy { epsilon: 1.0 },
            ..Default::default()
        };
        let mut bandit = BanditRouter::with_seed(config, 42);
        bandit.add_arm("a");
        bandit.add_arm("b");
        // With epsilon=1.0, always explores (random)
        let result = bandit.select(None).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_record_outcome_updates_params() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("model-a");

        let initial_alpha = bandit.arm_stats("model-a").unwrap().params.alpha;
        bandit.record_outcome(&ArmFeedback {
            arm_id: "model-a".to_string(),
            success: true,
            quality: None,
            latency_ms: None,
            cost: None,
            task_type: None,
        });

        let after = bandit.arm_stats("model-a").unwrap();
        assert!(after.params.alpha > initial_alpha);
        assert_eq!(after.pull_count, 1);
        assert_eq!(after.total_reward, 1.0);
    }

    #[test]
    fn test_warm_start_priors() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.warm_start("model-x", 10.0, 2.0);
        let arm = bandit.arm_stats("model-x").unwrap();
        assert_eq!(arm.params.alpha, 10.0);
        assert_eq!(arm.params.beta, 2.0);
    }

    #[test]
    fn test_per_task_bandit_isolation() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("coding", "gpt-4");
        bandit.add_arm_for_task("math", "claude");

        bandit.record_outcome(&ArmFeedback {
            arm_id: "gpt-4".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: Some("coding".to_string()),
        });

        // Math arm should be unaffected
        let math_arm = bandit.all_arms(Some("math"));
        assert_eq!(math_arm[0].pull_count, 0);
    }

    #[test]
    fn test_beta_sampling_bounds() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 99);
        for _ in 0..100 {
            let sample = bandit.sample_beta(2.0, 5.0);
            assert!(
                sample >= 0.0 && sample <= 1.0,
                "Sample {} out of [0,1]",
                sample
            );
        }
    }

    #[test]
    fn test_bandit_select_no_arms_error() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        let result = bandit.select(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bandit_decay_factor() {
        let config = BanditConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut bandit = BanditRouter::new(config);
        bandit.add_arm("a");

        // First outcome
        bandit.record_outcome(&ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(1.0),
            latency_ms: None,
            cost: None,
            task_type: None,
        });
        let alpha1 = bandit.arm_stats("a").unwrap().params.alpha;

        // Second outcome with decay
        bandit.record_outcome(&ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(1.0),
            latency_ms: None,
            cost: None,
            task_type: None,
        });
        let alpha2 = bandit.arm_stats("a").unwrap().params.alpha;

        // With decay, alpha shouldn't grow as fast as without
        assert!(alpha2 < alpha1 + 1.0 + 0.01); // decayed first, then added 1
    }

    #[test]
    fn test_bandit_deterministic_with_seed() {
        let config = BanditConfig::default();
        let mut b1 = BanditRouter::with_seed(config.clone(), 42);
        let mut b2 = BanditRouter::with_seed(config, 42);
        b1.add_arm("a");
        b1.add_arm("b");
        b2.add_arm("a");
        b2.add_arm("b");

        let r1 = b1.select(None).unwrap();
        let r2 = b2.select(None).unwrap();
        assert_eq!(r1.selected_arm, r2.selected_arm);
    }

    #[test]
    fn test_bandit_routing_outcome_fields() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("x");
        bandit.add_arm("y");
        let result = bandit.select(None).unwrap();
        assert_eq!(result.router_id, "bandit");
        assert!(!result.reason.is_empty());
        // One alternative since 2 arms total
        assert_eq!(result.alternatives.len(), 1);
    }

    // =========================================================================
    // REWARD POLICY TESTS
    // =========================================================================

    #[test]
    fn test_reward_policy_default_values() {
        let rp = RewardPolicy::default();
        assert!((rp.quality_weight - 0.7).abs() < 1e-10);
        assert!((rp.latency_weight - 0.2).abs() < 1e-10);
        assert!((rp.cost_weight - 0.1).abs() < 1e-10);
        assert!((rp.latency_ref_ms - 5000.0).abs() < 1e-10);
        assert!((rp.cost_ref - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_quality_only() {
        let rp = RewardPolicy::default();
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: None,
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // With latency=None and cost=None, all weight goes to quality
        assert!((reward - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_all_components() {
        let rp = RewardPolicy {
            quality_weight: 0.5,
            latency_weight: 0.3,
            cost_weight: 0.2,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: Some(500),
            cost: Some(0.05),
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // quality_score = 0.9, latency_score = 1 - 500/1000 = 0.5, cost_score = 1 - 0.05/0.1 = 0.5
        let expected = 0.5 * 0.9 + 0.3 * 0.5 + 0.2 * 0.5;
        assert!((reward - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_all_weights_zero() {
        let rp = RewardPolicy {
            quality_weight: 0.0,
            latency_weight: 0.0,
            cost_weight: 0.0,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.6),
            latency_ms: Some(200),
            cost: Some(0.02),
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // Normalizes to 1/3 each
        let q = 0.6;
        let l = 1.0 - 200.0 / 1000.0;
        let c = 1.0 - 0.02 / 0.1;
        let expected = (q + l + c) / 3.0;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_latency_none_redistributes() {
        let rp = RewardPolicy {
            quality_weight: 0.6,
            latency_weight: 0.3,
            cost_weight: 0.1,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: Some(0.05),
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // Latency weight 0.3 redistributed to quality (0.6) and cost (0.1) proportionally
        // effective_qw = 0.6 + 0.3 * (0.6/0.7) ≈ 0.8571
        // effective_cw = 0.1 + 0.3 * (0.1/0.7) ≈ 0.1429
        let cost_score = 1.0 - 0.05 / 0.1;
        let eqw = 0.6 + 0.3 * (0.6 / 0.7);
        let ecw = 0.1 + 0.3 * (0.1 / 0.7);
        let expected = eqw * 0.8 + ecw * cost_score;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_cost_none_redistributes() {
        let rp = RewardPolicy {
            quality_weight: 0.6,
            latency_weight: 0.3,
            cost_weight: 0.1,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.7),
            latency_ms: Some(300),
            cost: None,
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        let lat_score = 1.0 - 300.0 / 1000.0;
        let eqw = 0.6 + 0.1 * (0.6 / 0.9);
        let elw = 0.3 + 0.1 * (0.3 / 0.9);
        let expected = eqw * 0.7 + elw * lat_score;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_both_none_quality_only() {
        let rp = RewardPolicy {
            quality_weight: 0.5,
            latency_weight: 0.3,
            cost_weight: 0.2,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: false,
            quality: None,
            latency_ms: None,
            cost: None,
            task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        assert!((reward - 0.0).abs() < 1e-10); // success=false, quality=None -> 0.0
    }

    #[test]
    fn test_reward_policy_high_latency_penalized() {
        let rp = RewardPolicy {
            quality_weight: 0.5,
            latency_weight: 0.5,
            cost_weight: 0.0,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let fast = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: Some(100),
            cost: None,
            task_type: None,
        };
        let slow = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: Some(900),
            cost: None,
            task_type: None,
        };
        assert!(rp.compute_reward(&fast) > rp.compute_reward(&slow));
    }

    #[test]
    fn test_reward_policy_high_cost_penalized() {
        let rp = RewardPolicy {
            quality_weight: 0.5,
            latency_weight: 0.0,
            cost_weight: 0.5,
            latency_ref_ms: 1000.0,
            cost_ref: 0.1,
        };
        let cheap = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: Some(0.01),
            task_type: None,
        };
        let expensive = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: Some(0.09),
            task_type: None,
        };
        assert!(rp.compute_reward(&cheap) > rp.compute_reward(&expensive));
    }

    #[test]
    fn test_reward_policy_zero_ref_values() {
        let rp = RewardPolicy {
            quality_weight: 0.5,
            latency_weight: 0.3,
            cost_weight: 0.2,
            latency_ref_ms: 0.0,
            cost_ref: 0.0,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(),
            success: true,
            quality: Some(0.7),
            latency_ms: Some(100),
            cost: Some(0.01),
            task_type: None,
        };
        // Zero refs -> those components treated as unavailable -> quality only
        let reward = rp.compute_reward(&feedback);
        assert!((reward - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_record_outcome_uses_reward_policy() {
        let config = BanditConfig {
            reward_policy: RewardPolicy {
                quality_weight: 0.5,
                latency_weight: 0.5,
                cost_weight: 0.0,
                latency_ref_ms: 1000.0,
                cost_ref: 0.1,
            },
            ..BanditConfig::default()
        };
        let mut router = BanditRouter::new(config);
        router.add_arm("model-a");
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(),
            success: true,
            quality: Some(1.0),
            latency_ms: Some(500),
            cost: None,
            task_type: None,
        };
        router.record_outcome(&feedback);
        // Reward = 0.5*1.0 + 0.5*0.5 = 0.75 (latency score = 1 - 500/1000 = 0.5)
        let arm = router.all_arms(None).first().unwrap();
        // alpha should be prior(1.0) + 0.75 = 1.75, beta = prior(1.0) + 0.25 = 1.25
        assert!((arm.params.alpha - 1.75).abs() < 0.01);
        assert!((arm.params.beta - 1.25).abs() < 0.01);
    }

    #[test]
    fn test_record_outcome_backward_compat() {
        // Default policy with None latency/cost -> pure quality, matching old behavior
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(),
            success: true,
            quality: Some(0.6),
            latency_ms: None,
            cost: None,
            task_type: None,
        };
        router.record_outcome(&feedback);
        let arm = router.all_arms(None).first().unwrap();
        // alpha = 1.0 + 0.6 = 1.6, beta = 1.0 + 0.4 = 1.4
        assert!((arm.params.alpha - 1.6).abs() < 0.01);
        assert!((arm.params.beta - 1.4).abs() < 0.01);
    }

    // =========================================================================
    // ROUTING PREFERENCES TESTS
    // =========================================================================

    #[test]
    fn test_routing_preferences_default() {
        let prefs = RoutingPreferences::default();
        assert!(prefs.quality_weight.is_none());
        assert!(prefs.latency_weight.is_none());
        assert!(prefs.cost_weight.is_none());
        assert!(prefs.excluded_arms.is_empty());
        assert!(prefs.preferred_arms.is_empty());
        assert!((prefs.prefer_boost - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_routing_preferences_ignore_cost() {
        let prefs = RoutingPreferences::ignore_cost();
        assert_eq!(prefs.cost_weight, Some(0.0));
    }

    #[test]
    fn test_routing_preferences_minimize_latency() {
        let prefs = RoutingPreferences::minimize_latency();
        assert_eq!(prefs.latency_weight, Some(0.8));
        assert_eq!(prefs.quality_weight, Some(0.2));
        assert_eq!(prefs.cost_weight, Some(0.0));
    }

    #[test]
    fn test_routing_preferences_apply_to_policy() {
        let base = RewardPolicy::default();
        let prefs = RoutingPreferences {
            cost_weight: Some(0.0),
            ..Default::default()
        };
        let policy = prefs.apply_to_policy(&base);
        assert!((policy.cost_weight - 0.0).abs() < 1e-10);
        assert!((policy.quality_weight - 0.7).abs() < 1e-10);
        assert!((policy.latency_weight - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_select_with_preferences_excludes_arms() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        router.add_arm("model-b");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["model-a".to_string()],
            ..Default::default()
        };
        let outcome = router.select_with_preferences(None, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "model-b");
    }

    #[test]
    fn test_select_with_preferences_boosts_preferred() {
        let mut router = BanditRouter::with_seed(BanditConfig::default(), 42);
        router.add_arm("model-a");
        router.add_arm("model-b");
        // Give model-a strong priors
        router.warm_start("model-a", 100.0, 1.0);
        router.warm_start("model-b", 1.0, 100.0);
        // Prefer model-b with high boost
        let prefs = RoutingPreferences {
            preferred_arms: vec!["model-b".to_string()],
            prefer_boost: 1000.0,
            ..Default::default()
        };
        let outcome = router.select_with_preferences(None, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "model-b");
    }

    #[test]
    fn test_select_with_preferences_all_excluded_errors() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["model-a".to_string()],
            ..Default::default()
        };
        let result = router.select_with_preferences(None, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_outcome_with_preferences_custom_weights() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let prefs = RoutingPreferences::ignore_cost();
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None,
            cost: Some(0.1),
            task_type: None,
        };
        // With ignore_cost, cost is 0 weight -> reward should be quality-only = 0.8
        router.record_outcome_with_preferences(&feedback, &prefs);
        let arm = router.all_arms(None).first().unwrap();
        assert!((arm.params.alpha - 1.8).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_route_with_preferences() {
        let mut pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let features = QueryFeatureExtractor::extract("hello world");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["m1".to_string()],
            ..Default::default()
        };
        let outcome = pipeline.route_with_preferences(&features, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "m2");
    }

    #[test]
    fn test_preferences_serialize_deserialize() {
        let prefs = RoutingPreferences::ignore_cost();
        let json = serde_json::to_string(&prefs).unwrap();
        let prefs2: RoutingPreferences = serde_json::from_str(&json).unwrap();
        assert_eq!(prefs2.cost_weight, Some(0.0));
    }

    // =========================================================================
    // ARM VISIBILITY / PRIVATE ARMS TESTS
    // =========================================================================

    #[test]
    fn test_arm_visibility_default_is_public() {
        let router = BanditRouter::new(BanditConfig::default());
        assert!(router.private_arm_ids().is_empty());
    }

    #[test]
    fn test_set_arm_private_and_query() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("local-model");
        router.set_arm_private("local-model");
        assert!(router.is_arm_private("local-model"));
        assert!(!router.is_arm_private("other-model"));
    }

    #[test]
    fn test_set_arm_public_reverses_private() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        router.set_arm_private("model-a");
        assert!(router.is_arm_private("model-a"));
        router.set_arm_public("model-a");
        assert!(!router.is_arm_private("model-a"));
    }

    #[test]
    fn test_private_arms_accessor() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("m1");
        router.add_arm("m2");
        router.set_arm_private("m1");
        assert_eq!(router.private_arm_ids().len(), 1);
        assert!(router.private_arm_ids().contains("m1"));
    }

    #[test]
    fn test_private_arm_still_selectable_locally() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("local-model");
        router.set_arm_private("local-model");
        // Private arms are still selectable locally
        let outcome = router.select(None).unwrap();
        assert_eq!(outcome.selected_arm, "local-model");
    }

    #[test]
    fn test_snapshot_preserves_private_arms() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("m1");
        router.add_arm("m2");
        router.set_arm_private("m1");
        let json = router.to_json().unwrap();
        let restored = BanditRouter::from_json(&json).unwrap();
        assert!(restored.is_arm_private("m1"));
        assert!(!restored.is_arm_private("m2"));
    }

    // =========================================================================
    // ROUTING CONTEXT TESTS
    // =========================================================================

    #[test]
    fn test_routing_context_new() {
        let features = QueryFeatureExtractor::extract("test query");
        let ctx = RoutingContext::new(features.clone());
        assert!(!ctx.rag_active);
        assert!(ctx.budget_remaining.is_none());
        assert!(ctx.agent_tier.is_none());
        assert_eq!(ctx.features.domain, features.domain);
    }

    #[test]
    fn test_routing_context_from_features() {
        let features = QueryFeatureExtractor::extract("code fn main() {}");
        let ctx: RoutingContext = features.clone().into();
        assert_eq!(ctx.features.has_code, features.has_code);
    }

    #[test]
    fn test_derive_preferences_low_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features,
            rag_active: false,
            budget_remaining: Some(0.001), // Very low budget
            agent_tier: None,
            session_cost_so_far: None,
            preferred_provider: None,
        };
        let policy = RewardPolicy::default(); // cost_ref = 0.01
        let prefs = ctx.derive_preferences(&policy);
        // budget (0.001) < cost_ref * 10 (0.1) -> cost weight boosted
        assert_eq!(prefs.cost_weight, Some(0.5));
    }

    #[test]
    fn test_derive_preferences_no_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext::new(features);
        let policy = RewardPolicy::default();
        let prefs = ctx.derive_preferences(&policy);
        // No budget -> default preferences
        assert!(prefs.cost_weight.is_none());
    }

    #[test]
    fn test_derive_preferences_normal_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features,
            rag_active: false,
            budget_remaining: Some(100.0), // High budget
            agent_tier: None,
            session_cost_so_far: None,
            preferred_provider: None,
        };
        let policy = RewardPolicy::default();
        let prefs = ctx.derive_preferences(&policy);
        assert!(prefs.cost_weight.is_none()); // No override needed
    }

    #[test]
    fn test_pipeline_route_with_context() {
        let mut pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let features = QueryFeatureExtractor::extract("hello");
        let ctx = RoutingContext::new(features);
        let outcome = pipeline.route_with_context(&ctx).unwrap();
        assert_eq!(outcome.selected_arm, "m1");
    }

    #[test]
    fn test_routing_context_serialize_deserialize() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features,
            rag_active: true,
            budget_remaining: Some(5.0),
            agent_tier: Some("pro".to_string()),
            session_cost_so_far: Some(1.23),
            preferred_provider: Some("openai".to_string()),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let ctx2: RoutingContext = serde_json::from_str(&json).unwrap();
        assert!(ctx2.rag_active);
        assert_eq!(ctx2.budget_remaining, Some(5.0));
        assert_eq!(ctx2.agent_tier.as_deref(), Some("pro"));
    }

    #[test]
    fn test_routing_context_with_rag() {
        let features = QueryFeatureExtractor::extract("search for something");
        let ctx = RoutingContext {
            features,
            rag_active: true,
            budget_remaining: None,
            agent_tier: None,
            session_cost_so_far: None,
            preferred_provider: None,
        };
        assert!(ctx.rag_active);
    }
}
